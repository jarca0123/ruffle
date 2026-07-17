//! Verified AVM2 `Op` stream → [`Block`] CFG translation.
//!
//! The stream is split into basic blocks (leaders = index 0, every branch target, and the
//! fallthrough after every branch), each block a straight-line body plus a [`Term`] telling
//! how it hands off control. Straight-line bodies cover: local moves, primitive/string
//! constants, `pop`/`dup`/`swap`/`pushscope`, null-safe `getslot`, `getproperty`,
//! `callproperty`/`callpropvoid`, the arithmetic/comparison/coerce ops (via the `binop`
//! /`unop` helpers), in-place `inc/declocal`, and `returnvalue`/`returnvoid`. Control flow
//! covers `jump`/`iftrue`/`iffalse`; `emit` renders the blocks as a `br_table` dispatch loop.
//!
//! REQUIREMENT: the operand stack is empty at every block boundary (a `jump`/fallthrough
//! leaves it empty; a conditional leaves exactly the branch condition, which the terminator
//! consumes). This holds for statement-level `if`/`while`/`for`; expression-level control
//! (ternary, `&&`/`||` mid-expression) leaves a value across the edge → declines. Anything
//! unhandled (a `LookupSwitch`, an unsupported op) also declines → the interpreter runs it.

use std::collections::BTreeSet;

use crate::emit::{Block, JitOp, NumUnbox, Promotion, RegKind, Term};
use crate::helpers::{binop_code, mop_code, unop_code};
use crate::typed::{coerce_result_repr, param_repr, Repr};
use crate::value::to_bits;
use ruffle_core::avm2::script::Script;
use ruffle_core::avm2::{Class, Op, Value};

/// One operand-stack slot / local as tracked during translation: the declared class
/// (for return-coercion elision) plus the proven [`Repr`] (for guard/coerce elision and
/// the unboxed `f64` fast path). See [`crate::typed`].
#[derive(Clone, Copy)]
struct Ty<'gc> {
    class: Option<Class<'gc>>,
    repr: Repr,
}

impl<'gc> Ty<'gc> {
    /// The unknown/conservative slot — no class, `Repr::Boxed`. What every op whose
    /// result isn't provably numeric pushes.
    fn boxed() -> Self {
        Ty { class: None, repr: Repr::Boxed }
    }

    /// A slot known only by its repr (no class handle).
    fn of_repr(repr: Repr) -> Self {
        Ty { class: None, repr }
    }

    /// The seed for a param/local of declared `class` (see [`param_repr`]).
    fn param(class: Option<Class<'gc>>, canonical_params: bool) -> Self {
        Ty { class, repr: param_repr(class, canonical_params) }
    }

    /// Meet two slots across a control-flow merge: the class survives only if identical on both
    /// paths; the repr is [`Repr::meet`]. Used by the cross-block entry-repr fixpoint.
    fn meet(self, other: Ty<'gc>) -> Ty<'gc> {
        Ty {
            class: if self.class == other.class { self.class } else { None },
            repr: self.repr.meet(other.repr),
        }
    }
}

/// Translates `ops` into basic blocks, or `None` to decline the whole method.
///
/// - `null_safe`: parsed-code op indices whose `getslot` the verifier proved non-null.
/// - `local_types`: each local slot's declared class (`this`/untyped = `None`); slot `1 + i`
///   is param `i`'s type. Seeds the operand-type tracker (single-block methods only).
/// - `canonical_params`: whether a declared-`Number` param is a guaranteed canonical inline
///   `Number` (true unless the method is `unchecked`, where a missing param is `undefined`).
/// Returns the block CFG, the locals to promote to typed WASM registers (see [`Promotion`]:
/// `Number`→f64, `Int`→i32), and the local slots the WASM prologue must `undefined`-init
/// (non-param, non-promoted locals that are read-before-written — the caller now writes ONLY
/// `[this, params]` into the frame, not the full-width `undefined` padding). `None` declines.
///
/// - `nparams`: the method's parameter count (params occupy frame slots `[1, 1+nparams)`).
pub fn translate<'gc>(
    ops: &[Op<'gc>],
    null_safe: &[u32],
    number_slots: &[u32],
    int_slots: &[u32],
    local_types: &[Option<Class<'gc>>],
    canonical_params: bool,
    nparams: usize,
) -> Option<(Vec<Block>, Vec<Promotion>, Vec<u32>)> {
    reset_decline_reason();
    if ops.is_empty() {
        return None;
    }
    let leaders = compute_leaders(ops)?;
    let num_blocks = leaders.len();
    let block_of = |target: usize| -> Option<usize> { leaders.binary_search(&target).ok() };

    // Only maintain the runtime local scope stack for methods that actually READ it
    // (`getscopeobject`). Everything else keeps the free `getlocal0; pushscope` prologue
    // (pushscope → a plain `pop`).
    let needs_scopes = ops.iter().any(|op| matches!(op, Op::GetScopeObject { .. }));

    // Operand-stack depth entering each block. Propagated forward from the entry (depth 0):
    // each terminator hands its crossing depth to its successors. Reducible AVM2 control flow
    // reaches every block via a lower-index predecessor first, so a single index-order pass
    // sets each `entry_depth[b]` before block `b` is generated (a back-edge to an already-set
    // header must agree — the verifier guarantees consistent stack heights). Idempotent, so it
    // stays stable across the (re-run) repr fixpoint passes below.
    let mut entry_depth: Vec<Option<usize>> = vec![None; num_blocks];
    entry_depth[0] = Some(0);

    // Per-block ENTRY local reprs — the cross-block dataflow the Phase-1 wins (BinOpNum, coerce
    // elision) read for multi-block methods. A forward "numeric on EVERY predecessor path"
    // analysis: `None` = block not yet reached (top); block 0 seeds from the params. At a merge
    // we [`Ty::meet`] each predecessor's EXIT locals in. The generation loop below IS the
    // transfer function (it already threads `locals` through `translate_op`, the single source
    // of truth for each op's effect — no duplicate arity table), re-run to a fixpoint. `meet`
    // only lowers (toward `Boxed`) over a finite-height lattice, so this terminates; the final
    // (no-change) pass generated every block from its stable seed, so its `blocks` are correct.
    let nlocals = local_types.len();
    let param_seeds: Vec<Ty<'gc>> =
        local_types.iter().map(|&c| Ty::param(c, canonical_params)).collect();
    // PROMOTION ⇄ REPR ROUND-TRIP. The two feed each other: a promoted local's `GetLocal` yields
    // its register's repr EXACTLY (the emit re-boxes it), which is strictly better than the
    // conservative cross-block meet — and better operand reprs prove more arithmetic, which makes
    // more stores `Int`, which promotes more locals. So run the repr fixpoint, promote, then re-run
    // with those promotions known, until the promoted set stops growing.
    //
    // Terminates: `promoted_repr` only ever GAINS entries (a promotion is proven from readable
    // defs, and better reprs can only make more defs match), the set is bounded by `nlocals`, and
    // the round cap is a hard backstop. Each round is a full fixpoint, hence the small cap.
    let mut promoted_repr: Vec<Option<Repr>> = vec![None; nlocals];
    let mut blocks;
    let mut promoted: Vec<Promotion>;
    let mut round = 0;
    loop {

        let mut entry_locals: Vec<Option<Vec<Ty<'gc>>>> = vec![None; num_blocks];
        entry_locals[0] = Some(param_seeds.clone());

        // Every def of every local this pass: `(op index, local, stored repr)` — the promotion input.
        // Recomputed each pass (a store's repr depends on the seeds); after the fixpoint it holds the
        // final, stable verdict.
        let mut def_reprs: Vec<(usize, u32, Repr)> = Vec::new();

        blocks = Vec::with_capacity(num_blocks);
        loop {
            let mut changed = false;
            def_reprs.clear();
            blocks = Vec::with_capacity(num_blocks);
            for b in 0..num_blocks {
                let start = leaders[b];
                let end = *leaders.get(b + 1).unwrap_or(&ops.len());
                let ed = entry_depth[b].unwrap_or(0);

                let mut out = Vec::new();
                // Seed the operand-type stack with the `ed` values that crossed from a predecessor
                // (reprs unknown → conservative; `emit` reloads them from the spill locals).
                let mut tys: Vec<Ty<'gc>> = vec![Ty::boxed(); ed];
                // Seed locals from the fixpoint state; an unreached block (dead code, or not yet
                // reached this pass) is conservative (all `Boxed`).
                let mut locals: Vec<Ty<'gc>> = match &entry_locals[b] {
                    Some(l) => l.clone(),
                    None => vec![Ty::boxed(); nlocals],
                };

                let mut term: Option<Term> = None;
                // Successor block(s) + the operand depth crossing to each.
                let mut succ: Vec<(usize, usize)> = Vec::new();
                for i in start..end {
                    match &ops[i] {
                        Op::Jump { offset } => {
                            let t = block_of(*offset)?;
                            succ.push((t, tys.len())); // the whole operand stack crosses
                            term = Some(Term::Jump(t));
                            break;
                        }
                        Op::IfTrue { offset } => {
                            tys.pop()?; // the condition — consumed by `truthy`; the rest crosses
                            let d = tys.len();
                            let (ot, of) = (block_of(*offset)?, block_of(i + 1)?);
                            succ.push((ot, d));
                            succ.push((of, d));
                            term = Some(Term::Cond { on_true: ot, on_false: of });
                            break;
                        }
                        Op::IfFalse { offset } => {
                            tys.pop()?;
                            let d = tys.len();
                            let (ot, of) = (block_of(i + 1)?, block_of(*offset)?);
                            succ.push((ot, d));
                            succ.push((of, d));
                            term = Some(Term::Cond { on_true: ot, on_false: of });
                            break;
                        }
                        // Tail-call fusion: `return receiver.f(args)` — emitted as `Call*; BailIfError;
                        // <return the result unchanged>` — becomes a single `TailCall*` (a
                        // `return_call` on web, §8). Only for an UNTYPED return (`return_type: None` ⇒
                        // a plain `ReturnValue`, no post-call coercion — a `CoerceReturn` is real work
                        // after the call, not tail position) whose just-emitted op IS the fusable call.
                        Op::ReturnValue { return_type: None }
                            if matches!(out.last(), Some(JitOp::BailIfError))
                                && matches!(
                                    out.len().checked_sub(2).and_then(|k| out.get(k)),
                                    Some(JitOp::CallProperty(..) | JitOp::CallMethod(..))
                                ) =>
                        {
                            let n = out.len();
                            let tail = match out[n - 2] {
                                JitOp::CallProperty(mn, argc) => JitOp::TailCallProperty(mn, argc),
                                JitOp::CallMethod(disp, argc) => JitOp::TailCallMethod(disp, argc),
                                _ => unreachable!(),
                            };
                            out.truncate(n - 2); // drop the `Call*` + its `BailIfError`
                            out.push(tail);
                            tys.pop()?; // the returned call result
                            term = Some(Term::Return);
                            break;
                        }
                        op @ (Op::ReturnValue { .. } | Op::ReturnVoid { .. }) => {
                            translate_op(op, i, null_safe, number_slots, int_slots, needs_scopes, &promoted_repr, &mut out, &mut tys, &mut locals)?;
                            term = Some(Term::Return);
                            break;
                        }
                        // `throw` diverges — a terminator like `returnvalue`. Pop the thrown value;
                        // `JitOp::Throw` stashes the error + `Return`s (rest of the block is dead).
                        Op::Throw => {
                            tys.pop()?; // the thrown value (consumed by `throw_value`)
                            out.push(JitOp::Throw);
                            term = Some(Term::Return);
                            break;
                        }
                        op => {
                            translate_op(op, i, null_safe, number_slots, int_slots, needs_scopes, &promoted_repr, &mut out, &mut tys, &mut locals)?;
                            // Track store reprs for promotion: after `translate_op`, a store op's
                            // target `locals[idx]` holds the stored repr. Flag stores that aren't a
                            // canonical `Number` / a provable `Int` (each disqualifies that target).
                            for idx in store_targets(op) {
                                let r = locals.get(idx as usize).map(|t| t.repr);
                                // Record every def (op index → local, stored repr) for the web
                                // analysis below. Reset each pass, so after the fixpoint this is the
                                // stable verdict.
                                if let Some(r) = r {
                                    def_reprs.push((i, idx, r));
                                }
                                if r != Some(Repr::Int) {
                                    record_store_non_int(r, &out);
                                }
                            }
                        }
                    }
                }

                // No explicit terminator → implicit fallthrough to the next block; the whole stack
                // crosses the edge.
                let term = match term {
                    Some(t) => t,
                    None => {
                        let t = block_of(end)?; // `end` is the next leader (else off-the-end → None)
                        succ.push((t, tys.len()));
                        Term::Jump(t)
                    }
                };
                // Hand each successor its crossing depth; a disagreement (irreducible / verifier-
                // inconsistent) declines. Then meet this block's EXIT locals into each successor's
                // entry reprs (`None` → first info; `Some` → per-slot meet), flagging any change.
                for (s, d) in &succ {
                    match entry_depth[*s] {
                        None => entry_depth[*s] = Some(*d),
                        Some(existing) if existing == *d => {}
                        Some(_) => {
                            set_decline("stack@edge_conflict");
                            return None;
                        }
                    }
                    match &mut entry_locals[*s] {
                        None => {
                            entry_locals[*s] = Some(locals.clone());
                            changed = true;
                        }
                        Some(existing) => {
                            for (slot, &exit) in existing.iter_mut().zip(locals.iter()) {
                                let met = slot.meet(exit);
                                if met.repr != slot.repr || met.class != slot.class {
                                    *slot = met;
                                    changed = true;
                                }
                            }
                        }
                    }
                }
                blocks.push(Block { ops: out, term, entry_depth: ed });
            }
            if !changed {
                break;
            }
        }

        // Phase 3 (local promotion): lift locals out of the memory frame into typed WASM `f64`
        // locals (register-allocatable — no per-access `i64.load`/`store`; the loop-carried
        // accumulator's per-iteration memory round-trip is the headline win). Promote local `i` iff:
        //   every def of `i` whose value can be READ stores that repr, AND
        //   the entry value (a param, or `undefined`) is that repr if it can be read at all.
        // Then `i` holds that repr at every point a `GetLocal` can observe, so the register round-trip
        // is bit-identity. No writeback is needed: nothing reads the (now stale) frame slot — only
        // `GetLocal`, which the emit redirects, and reification helpers never read caller locals.
        let live_in = compute_liveness(ops, &leaders, &blocks, nlocals);
        let readable_defs = live_defs(ops, &leaders, &blocks, nlocals, &live_in);
        let pinned = frame_pinned_locals(ops, nlocals);
        // Is local `i` promotable to a `target` (`Number`→f64 / `Int`/`Bool`→i32) register?
        //
        // PRECISION: this replaced a conjunction of two SEPARATE analyses — "no store is off-repr"
        // (blind to whether the store is ever read) AND "the meet is `target` at every live-on-entry
        // block" (a forward repr meet ∧ a backward liveness, which lose each other's path
        // information). That pair rejected 556 locals SWF-wide whose every readable def IS `Int` —
        // e.g. one DEAD off-repr store poisoned the slot, or an off-repr value reached a block by a
        // path along which nothing read it. Reasoning about the DEFS A READ CAN OBSERVE is both
        // simpler and strictly more precise.
        //
        // SOUNDNESS: at a `GetLocal(i)` the register holds either (a) the prologue init, when no def
        // executed — then the entry value is readable, so `live_in[0][i]` and the entry-repr check
        // below cover it; or (b) the last def executed on the actual path — which is a def this read
        // can observe, i.e. a `readable_defs` entry, checked below. Dead defs write garbage into the
        // register but by definition no read observes it. `pinned` locals are excluded outright.
        //
        // Returns `init_from_frame`: load the entry value at the prologue only if it can be read.
        let promotable_as = |i: usize, target: Repr| -> Option<bool> {
            if pinned[i] {
                record_promo_fail(target, "frame_pinned(hasnext2)");
                return None;
            }
            let entry_read = live_in[0][i];
            if entry_read && entry_locals[0].as_ref().map(|l| l[i].repr) != Some(target) {
                record_promo_fail(target, "entry_value_read_and_off_repr");
                return None;
            }
            let mut any = entry_read;
            for &(op_idx, l, r) in &def_reprs {
                if l as usize == i && readable_defs.contains(&(op_idx, l)) {
                    any = true;
                    if r != target {
                        record_promo_fail(target, "readable_def_off_repr");
                        return None;
                    }
                }
            }
            // Never read anywhere → a register would buy nothing (and would burn one).
            if !any {
                return None;
            }
            Some(entry_read)
        };
        promoted = (0..nlocals)
            .filter_map(|i| {
                let local = i as u32;
                if let Some(init_from_frame) = promotable_as(i, Repr::Number) {
                    Some(Promotion { local, kind: RegKind::F64, init_from_frame })
                } else if let Some(init_from_frame) = promotable_as(i, Repr::Int) {
                    Some(Promotion { local, kind: RegKind::IntI32, init_from_frame })
                } else if let Some(init_from_frame) = promotable_as(i, Repr::Bool) {
                    Some(Promotion { local, kind: RegKind::BoolI32, init_from_frame })
                } else if !pinned[i]
                    && (live_in[0][i] || readable_defs.iter().any(|&(_, l)| l as usize == i))
                    // The entry value is only READABLE from the frame for `this`/params, which
                    // `try_enter` writes. A live-on-entry NON-param reads its `undefined` default,
                    // which lives in the frame ONLY when the slot is left unpromoted (see
                    // `undefined_init` below) — promoting it would load uninitialised memory. The
                    // typed kinds are shielded from this by their entry-repr check; this one has
                    // no repr check at all, so it must exclude the case explicitly.
                    && (!live_in[0][i] || i < 1 + nparams)
                {
                    // No repr proof needed: the register holds the box verbatim. Anything not
                    // frame-pinned and read somewhere qualifies — which is how `Boxed` (object)
                    // locals finally get out of linear memory.
                    Some(Promotion { local, kind: RegKind::BoxedI64, init_from_frame: live_in[0][i] })
                } else {
                    None
                }
            })
            .collect();

        // VERIFY (diagnostic) whether live-range splitting would actually unlock more registers than
        // the whole-slot analysis above. Runs only under `HISTO_MIN_OPS`.
        {
            let mut already = vec![false; nlocals];
            for p in &promoted {
                already[p.local as usize] = true;
            }
            let entry_repr0: Vec<Repr> = (0..nlocals)
                .map(|i| entry_locals[0].as_ref().map(|l| l[i].repr).unwrap_or(Repr::Boxed))
                .collect();
            report_webs(ops, &leaders, &blocks, nlocals, &def_reprs, &entry_repr0, &already);
        }

        // Slots the WASM prologue must `undefined`-init: a NON-param (`i ≥ 1+nparams`), NON-promoted
        // local that is LIVE-on-entry to block 0 — i.e. read before it is written on some path, so
        // its `undefined` default is observable. Everything else needs nothing: params/`this` are
        // written by `try_enter`; promoted locals live in registers; a local written-before-read has
        // its (stale) frame slot overwritten before any read. For the common case (locals assigned
        // before use) this set is EMPTY → zero per-call padding.
        let is_promoted: Vec<bool> = {
            let mut v = vec![false; nlocals];
            for p in &promoted {
                if let Some(s) = v.get_mut(p.local as usize) {
                    *s = true;
                }
            }
            v
        };

        // Feed the promotions back into the reprs and re-run — unless nothing new was proven.
        let mut next: Vec<Option<Repr>> = vec![None; nlocals];
        for p in &promoted {
            next[p.local as usize] = match p.kind {
                RegKind::F64 => Some(Repr::Number),
                RegKind::IntI32 => Some(Repr::Int),
                RegKind::BoolI32 => Some(Repr::Bool),
                // Proves nothing about the value — leave the repr to the meet, which may well
                // know more than `Boxed` (and claiming `Boxed` here would only lose information).
                RegKind::BoxedI64 => None,
            };
        }
        round += 1;
        if next == promoted_repr || round >= 3 {
            break;
        }
        promoted_repr = next;
    }

    // Recomputed against the FINAL blocks/promotions (both were rebuilt by the last round).
    let live_in = compute_liveness(ops, &leaders, &blocks, nlocals);
    let mut is_promoted = vec![false; nlocals];
    for p in &promoted {
        is_promoted[p.local as usize] = true;
    }
    let undefined_init: Vec<u32> = (1 + nparams..nlocals)
        .filter(|&i| !is_promoted[i] && live_in[0][i])
        .map(|i| i as u32)
        .collect();
    Some((blocks, promoted, undefined_init))
}

/// Local indices WRITTEN by `op` (`setlocal`/`storelocal`/`kill`/`inc·declocal`/`hasnext2`) —
/// for the promotion store-repr check and liveness `kill` sets. Conservative: an unrecognized
/// local-writing op must be added here (an unpromoted local is merely unoptimized; a MISSED
/// writer would make promotion of that local unsound).
fn store_targets(op: &Op) -> Vec<u32> {
    match op {
        Op::SetLocal { index }
        | Op::StoreLocal { index }
        | Op::Kill { index }
        | Op::IncLocal { index }
        | Op::DecLocal { index }
        | Op::IncLocalI { index }
        | Op::DecLocalI { index } => vec![*index],
        Op::HasNext2 { object_register, index_register } => vec![*object_register, *index_register],
        _ => vec![],
    }
}

/// Local indices READ by `op` (`getlocal`, the read-modify-write `inc·declocal`, `hasnext2`) —
/// liveness `use` (upward-exposed) sets.
fn local_reads(op: &Op) -> Vec<u32> {
    match op {
        Op::GetLocal { index }
        | Op::IncLocal { index }
        | Op::DecLocal { index }
        | Op::IncLocalI { index }
        | Op::DecLocalI { index } => vec![*index],
        Op::HasNext2 { object_register, index_register } => vec![*object_register, *index_register],
        _ => vec![],
    }
}

/// VERIFICATION for live-range splitting (diagnostic, gated by `HISTO_MIN_OPS`).
///
/// `promotable_as` rejects a local when its repr differs across blocks — but a single AVM2 local
/// SLOT is often several unrelated variables (FlasCC/Alchemy reuses a slot across C scopes), so the
/// meet reports a conflict that no single read can ever observe. The fix would be splitting the
/// slot into **webs** and promoting each independently. This measures whether that would actually
/// pay, BEFORE writing the transform.
///
/// A web = a maximal set of defs connected by sharing a use (classic register-allocator webs):
/// reaching-definitions, then union every def that reaches the same use. A web is promotable to
/// `target` iff EVERY def in it stores `target` — a read only ever observes its own web's defs, so
/// that makes the web that repr at every point it can be read. Def `0` is the pseudo-def of the
/// entry value (a param, or `undefined` for a non-param).
fn report_webs(
    ops: &[Op],
    leaders: &[usize],
    blocks: &[Block],
    nlocals: usize,
    def_reprs: &[(usize, u32, Repr)],
    entry_repr0: &[Repr],
    already_promoted: &[bool],
) {
    if crate::histo_min_ops() == 0 {
        return;
    }
    let n = blocks.len();
    let succs = |t: &Term| -> Vec<usize> {
        match *t {
            Term::Return => vec![],
            Term::Jump(s) => vec![s],
            Term::Cond { on_true, on_false } => vec![on_true, on_false],
        }
    };
    let block_of = |op_idx: usize| -> usize {
        leaders.partition_point(|&l| l <= op_idx).saturating_sub(1)
    };
    let (mut split_wins, mut whole_slot_ok, mut no_win) = (0usize, 0usize, 0usize);
    let mut extra_regs = 0usize;
    for i in 0..nlocals {
        if already_promoted[i] {
            continue;
        }
        // Defs of local `i`, in op order; index 0 is the entry pseudo-def.
        let mut defs: Vec<(Option<usize>, Repr)> = vec![(None, entry_repr0[i])];
        for &(op_idx, l, r) in def_reprs {
            if l as usize == i {
                defs.push((Some(op_idx), r));
            }
        }
        defs.sort_by_key(|(o, _)| o.unwrap_or(0));
        defs.dedup_by_key(|(o, _)| *o);
        let ndefs = defs.len();
        // Per-block LAST def and whether the block defines `i` at all.
        let mut r#gen = vec![None; n];
        for (d, (op_idx, _)) in defs.iter().enumerate().skip(1) {
            let b = block_of(op_idx.unwrap());
            r#gen[b] = Some(d); // later defs overwrite → ends up the last one in the block
        }
        // Reaching definitions.
        let mut reach_in = vec![vec![false; ndefs]; n];
        reach_in[0][0] = true;
        let mut changed = true;
        while changed {
            changed = false;
            let mut out = vec![vec![false; ndefs]; n];
            for b in 0..n {
                match r#gen[b] {
                    Some(d) => out[b][d] = true,
                    None => out[b].copy_from_slice(&reach_in[b]),
                }
            }
            for b in 0..n {
                for s in succs(&blocks[b].term) {
                    for d in 0..ndefs {
                        if out[b][d] && !reach_in[s][d] {
                            reach_in[s][d] = true;
                            changed = true;
                        }
                    }
                }
            }
        }
        // Union every def reaching each USE. `uf[d]` = union-find parent.
        let mut uf: Vec<usize> = (0..ndefs).collect();
        fn find(uf: &mut Vec<usize>, x: usize) -> usize {
            if uf[x] != x {
                let r = find(uf, uf[x]);
                uf[x] = r;
            }
            uf[x]
        }
        let mut used: Vec<bool> = vec![false; ndefs];
        for b in 0..n {
            let start = leaders[b];
            let end = leaders.get(b + 1).copied().unwrap_or(ops.len());
            // Walk the block, tracking the def in effect: `reach_in[b]` until a def in this block.
            let mut cur: Option<usize> = None; // Some(d) once a def in this block took effect
            for (j, op) in ops[start..end].iter().enumerate() {
                let op_idx = start + j;
                for r in local_reads(op) {
                    if r as usize != i {
                        continue;
                    }
                    let reaching: Vec<usize> = match cur {
                        Some(d) => vec![d],
                        None => (0..ndefs).filter(|&d| reach_in[b][d]).collect(),
                    };
                    let Some(&first) = reaching.first() else { continue };
                    for &d in &reaching {
                        used[d] = true;
                        let (ra, rb) = (find(&mut uf, first), find(&mut uf, d));
                        uf[ra] = rb;
                    }
                }
                if store_targets(op).into_iter().any(|t| t as usize == i) {
                    cur = defs.iter().position(|(o, _)| *o == Some(op_idx));
                }
            }
        }
        // A web is promotable iff every def in it is `Int` (and it is actually read).
        let mut web_ok: std::collections::BTreeMap<usize, (bool, bool)> = Default::default();
        for d in 0..ndefs {
            let root = find(&mut uf, d);
            let e = web_ok.entry(root).or_insert((true, false));
            if used[d] {
                e.1 = true;
                if defs[d].1 != Repr::Int {
                    e.0 = false;
                }
            }
        }
        let good = web_ok.values().filter(|(ok, used)| *ok && *used).count();
        let total = web_ok.values().filter(|(_, used)| *used).count();
        if good == 0 {
            no_win += 1;
        } else if good == total {
            whole_slot_ok += 1; // no split needed — the whole slot is Int (shouldn't reach here)
        } else {
            split_wins += 1;
            extra_regs += good;
        }
    }
    crate::runner::diag_log(&format!(
        "JIT3 WEBS: split_wins={split_wins} (would add {extra_regs} i32 regs) \
         whole_slot_ok={whole_slot_ok} no_win={no_win}"
    ));
}

/// The `(op index, local)` defs whose value can actually be READ — i.e. the local is LIVE
/// immediately after the def. A def NOT in this set is dead: no `getlocal` can ever observe it, so
/// its repr must not disqualify the local from register promotion.
///
/// "this def reaches some use" ⟺ "the local is live immediately after this def" — that IS
/// liveness's definition, so this gives reaching-definitions' answer to our question for one
/// backward scan over the ops, instead of a per-local reaching-defs fixpoint.
fn live_defs(
    ops: &[Op],
    leaders: &[usize],
    blocks: &[Block],
    nlocals: usize,
    live_in: &[Vec<bool>],
) -> BTreeSet<(usize, u32)> {
    let mut out = BTreeSet::new();
    for b in 0..blocks.len() {
        // live-out = ∪ live-in of the successors.
        let mut live = vec![false; nlocals];
        let succs: &[usize] = &match blocks[b].term {
            Term::Return => vec![],
            Term::Jump(s) => vec![s],
            Term::Cond { on_true, on_false } => vec![on_true, on_false],
        };
        for &s in succs {
            for i in 0..nlocals {
                live[i] |= live_in[s][i];
            }
        }
        // Walk the block BACKWARD, maintaining `live = (live − def) ∪ use` per op. A def is
        // readable exactly when the local is live at the point just after it.
        let start = leaders[b];
        let end = leaders.get(b + 1).copied().unwrap_or(ops.len());
        for (j, op) in ops[start..end].iter().enumerate().rev() {
            for w in store_targets(op) {
                if live.get(w as usize).copied().unwrap_or(false) {
                    out.insert((start + j, w));
                }
                if let Some(l) = live.get_mut(w as usize) {
                    *l = false; // the def kills what was live
                }
            }
            for r in local_reads(op) {
                if let Some(l) = live.get_mut(r as usize) {
                    *l = true;
                }
            }
        }
    }
    out
}

/// Locals that MUST stay in the memory frame: `hasnext2` hands its two register INDICES to a helper
/// that reads and writes `frame[reg]` directly (`helpers::has_next_2`), bypassing any WASM register
/// a promotion would put them in — so the helper's writes would be invisible and its reads stale.
/// (Every other local access goes through `GetLocal`/`SetLocal`, which the emit redirects.)
///
/// This exclusion used to be implicit: `hasnext2` stores `Boxed`, and the old promotion test
/// rejected a local if ANY store was off-repr. The readable-defs test below is deliberately
/// blind to DEAD stores, which would lose that protection — hence this explicit pin.
fn frame_pinned_locals(ops: &[Op], nlocals: usize) -> Vec<bool> {
    let mut pinned = vec![false; nlocals];
    for op in ops {
        if let Op::HasNext2 { object_register, index_register } = op {
            for r in [*object_register, *index_register] {
                if let Some(p) = pinned.get_mut(r as usize) {
                    *p = true;
                }
            }
        }
    }
    pinned
}

/// Live-on-entry locals per block: a standard backward liveness fixpoint. `uses[b]` =
/// upward-exposed uses (read before written in `b`); `kill[b]` = written in `b`. Successors come
/// from each block's [`Term`]. Used by promotion to check a local is `Number` wherever readable.
fn compute_liveness(ops: &[Op], leaders: &[usize], blocks: &[Block], nlocals: usize) -> Vec<Vec<bool>> {
    let n = blocks.len();
    let mut uses = vec![vec![false; nlocals]; n];
    let mut kill = vec![vec![false; nlocals]; n];
    for b in 0..n {
        let start = leaders[b];
        let end = leaders.get(b + 1).copied().unwrap_or(ops.len());
        for op in &ops[start..end] {
            for r in local_reads(op) {
                if !kill[b][r as usize] {
                    if let Some(g) = uses[b].get_mut(r as usize) {
                        *g = true;
                    }
                }
            }
            for w in store_targets(op) {
                if let Some(k) = kill[b].get_mut(w as usize) {
                    *k = true;
                }
            }
        }
    }
    let succs = |t: &Term| -> Vec<usize> {
        match *t {
            Term::Return => vec![],
            Term::Jump(s) => vec![s],
            Term::Cond { on_true, on_false } => vec![on_true, on_false],
        }
    };
    let mut live_in = vec![vec![false; nlocals]; n];
    let mut changed = true;
    while changed {
        changed = false;
        for b in 0..n {
            let mut live_out = vec![false; nlocals];
            for s in succs(&blocks[b].term) {
                for i in 0..nlocals {
                    live_out[i] |= live_in[s][i];
                }
            }
            for i in 0..nlocals {
                let v = uses[b][i] || (live_out[i] && !kill[b][i]);
                if v != live_in[b][i] {
                    live_in[b][i] = v;
                    changed = true;
                }
            }
        }
    }
    live_in
}

/// Basic-block leaders: index 0, every branch target and post-branch index. `None` if the
/// method has control flow this translator can't structure (`LookupSwitch`).
fn compute_leaders(ops: &[Op]) -> Option<Vec<usize>> {
    let mut set = BTreeSet::new();
    set.insert(0);
    for (i, op) in ops.iter().enumerate() {
        match op {
            Op::Jump { offset } | Op::IfTrue { offset } | Op::IfFalse { offset } => {
                if *offset > ops.len() {
                    return None;
                }
                set.insert(*offset);
                if i + 1 < ops.len() {
                    set.insert(i + 1);
                }
            }
            Op::LookupSwitch(_) => {
                record_decline(op); // multi-target — not structured here (logged for the report)
                return None;
            }
            _ => {}
        }
    }
    Some(set.into_iter().collect())
}

/// Translates one straight-line op into `out`, maintaining the operand-type stack `tys` and
/// the per-slot `locals`. `None` declines (and records the unsupported op for the log).
fn translate_op<'gc>(
    op: &Op<'gc>,
    i: usize,
    null_safe: &[u32],
    number_slots: &[u32],
    int_slots: &[u32],
    needs_scopes: bool,
    // `Some(repr)` for a local already PROVEN promotable to a register of that repr (see the
    // promotion round-trip in `translate`). A `GetLocal` on it reads the register and re-boxes,
    // so it yields that repr EXACTLY — regardless of what the conservative cross-block meet says.
    promoted_repr: &[Option<Repr>],
    out: &mut Vec<JitOp>,
    tys: &mut Vec<Ty<'gc>>,
    locals: &mut Vec<Ty<'gc>>,
) -> Option<()> {
    match op {
        Op::GetLocal { index } => {
            out.push(JitOp::GetLocal(*index));
            // A PROMOTED local's read is `local.get reg; i64.extend_i32_u; i64.or MARK` — it
            // reconstructs a box of the register's repr every time, so that repr is exact. Prefer
            // it over the meet, which is deliberately conservative and (being a forward meet ∧ a
            // backward liveness) loses path information the promotion proof does not.
            match promoted_repr.get(*index as usize).copied().flatten() {
                Some(r) => tys.push(Ty::of_repr(r)),
                None => tys.push(locals.get(*index as usize).copied().unwrap_or_else(Ty::boxed)),
            }
        }
        Op::SetLocal { index } => {
            out.push(JitOp::SetLocal(*index));
            let t = tys.pop()?;
            if let Some(slot) = locals.get_mut(*index as usize) {
                *slot = t;
            }
        }
        // `storelocal` PEEKS (keeps the value on the stack) — `dup; setlocal`. Net stack
        // unchanged; the local takes the top's type.
        Op::StoreLocal { index } => {
            let t = *tys.last()?;
            out.push(JitOp::Dup);
            out.push(JitOp::SetLocal(*index));
            if let Some(slot) = locals.get_mut(*index as usize) {
                *slot = t;
            }
        }
        Op::GetSlot { index } => {
            // Null-safe (verifier-proven) → the pure no-throw helper; otherwise the
            // null-checking helper (throws #1009 on a null/primitive receiver) + BailIfError.
            if null_safe.contains(&(i as u32)) {
                out.push(JitOp::GetSlot(*index as u32));
            } else {
                out.push(JitOp::GetSlotChecked(*index as u32));
                out.push(JitOp::BailIfError);
            }
            tys.pop()?;
            // The verifier proved these getslots resolve to `Number` (`number_slots`). Mark the
            // result `NumberBoxed` — "numeric, but possibly a lossless-boxed `Gc<f64>`": the codegen
            // then unboxes it with a runtime check (the same repr the mop `lf32`/`lf64` loads
            // produce), avoiding the `binop` helper's `valueOf` coercion on the numeric path.
            // NOT canonical `Repr::Number`: a `Number`-CLASS slot value can still be lossless-boxed
            // (e.g. a domainMemory float read stored via `SetSlotNoCoerce`, which skips the
            // canonicalizing coercion), and a blind `f64.reinterpret` on that pointer crashes.
            if number_slots.contains(&(i as u32)) {
                tys.push(Ty::of_repr(Repr::NumberBoxed));
            } else if int_slots.contains(&(i as u32)) {
                // The verifier proved this getslot resolves to `int`. Unlike `Number` above, an
                // `int`-CLASS value has exactly ONE representation — it is ALWAYS a `Value::Integer`
                // box — so the canonical `Repr::Int` is sound here (no box-safe middle ground is
                // needed). Every producer the verifier types `int` yields `Value::Integer`
                // (`pushint`/`coerce_i`/`*_i`/bitwise/shifts/`li*`/`sxi*`/an `int` param, whose
                // identity-coerce fast path requires an already-`TAG_INT` box); a coerced slot
                // write runs `coerce_to_i32().into()`; and `SetSlotNoCoerce` fires only on
                // `matches_type(int)`, whose sole non-class path (`contains_valid_integer`) is set
                // only by `pushint`/`pushuint <= i32::MAX` — both `Value::Integer` too. Crucially
                // `add`/`subtract`/`multiply` of two ints are typed `Number` by the verifier, NOT
                // `int`, so the overflow-to-`Number` case is not in this set. `uint` is excluded
                // (a `u32` >= 2^31 boxes as a `Number`).
                //
                // This is FlasCC's `li32`: it keeps its C globals and register variables in class
                // SLOTS, so `getslot` was the dominant remaining producer of `Repr::Boxed`.
                tys.push(Ty::of_repr(Repr::Int));
            } else {
                tys.push(Ty::boxed());
            }
        }
        // getproperty FAST (`obj[name]`, runtime name from the stack). The name is the single
        // runtime param; a runtime NAMESPACE (lazy ns) would need an extra stack slot → decline.
        Op::GetPropertyFast { multiname } => {
            if multiname.has_lazy_ns() {
                set_decline("lazy_multiname");
                return None;
            }
            let ptr = gc_arena::Gc::as_ptr(*multiname) as usize as u64;
            out.push(JitOp::GetPropertyFast(ptr));
            out.push(JitOp::BailIfError);
            tys.pop()?; // name
            tys.pop()?; // object
            tys.push(Ty::boxed());
        }
        Op::SetPropertyFast { multiname } => {
            if multiname.has_lazy_ns() {
                set_decline("lazy_multiname");
                return None;
            }
            let ptr = gc_arena::Gc::as_ptr(*multiname) as usize as u64;
            out.push(JitOp::SetPropertyFast(ptr));
            out.push(JitOp::BailIfError);
            out.push(JitOp::Pop); // discard the (undefined) result
            tys.pop()?; // value
            tys.pop()?; // name
            tys.pop()?; // object
        }
        // getproperty (static multiname). NB: only `Static` — `GetPropertyFast` uses a LAZY
        // (runtime-name) multiname whose name comes from the stack, which `get_property`
        // can't resolve (it panics). The lazy guard below is belt-and-suspenders.
        Op::GetPropertyStatic { multiname } => {
            if multiname.has_lazy_component() {
                set_decline("lazy_multiname");
                return None;
            }
            let ptr = gc_arena::Gc::as_ptr(*multiname) as usize as u64;
            out.push(JitOp::GetProperty(ptr));
            out.push(JitOp::BailIfError);
            tys.pop()?;
            tys.push(Ty::boxed());
        }
        Op::CallProperty { multiname, num_args } => {
            if multiname.has_lazy_component() {
                set_decline("lazy_multiname");
                return None;
            }
            let ptr = gc_arena::Gc::as_ptr(*multiname) as usize as u64;
            out.push(JitOp::CallProperty(ptr, *num_args));
            out.push(JitOp::BailIfError);
            for _ in 0..*num_args {
                tys.pop()?;
            }
            tys.pop()?;
            tys.push(Ty::boxed());
        }
        Op::CallPropVoid { multiname, num_args } => {
            if multiname.has_lazy_component() {
                set_decline("lazy_multiname");
                return None;
            }
            let ptr = gc_arena::Gc::as_ptr(*multiname) as usize as u64;
            out.push(JitOp::CallProperty(ptr, *num_args));
            out.push(JitOp::BailIfError);
            out.push(JitOp::Pop);
            for _ in 0..*num_args {
                tys.pop()?;
            }
            tys.pop()?;
        }
        // callnative: a baked native fn-pointer call; pops args + receiver, result if wanted.
        Op::CallNative { method, num_args, push_return_value } => {
            let ptr = *method as usize as u64;
            out.push(JitOp::CallNative(ptr, *num_args));
            out.push(JitOp::BailIfError);
            if !push_return_value {
                out.push(JitOp::Pop);
            }
            for _ in 0..*num_args {
                tys.pop()?;
            }
            tys.pop()?; // receiver
            if *push_return_value {
                tys.push(Ty::boxed());
            }
        }
        // callmethod (by disp-id): pops args + receiver; result pushed only if wanted.
        Op::CallMethod { index, num_args, push_return_value } => {
            out.push(JitOp::CallMethod(*index as u32, *num_args));
            out.push(JitOp::BailIfError);
            if !push_return_value {
                out.push(JitOp::Pop);
            }
            for _ in 0..*num_args {
                tys.pop()?;
            }
            tys.pop()?; // receiver
            if *push_return_value {
                tys.push(Ty::boxed());
            }
        }
        // newarray: pop `num_args` elements, push the Array (no throw).
        Op::NewArray { num_args } => {
            for _ in 0..*num_args {
                tys.pop()?;
            }
            out.push(JitOp::NewArray(*num_args));
            tys.push(Ty::boxed());
        }
        // istypelate / astypelate: pop `type`/`class` then `value`, push the result. Throwing.
        Op::IsTypeLate => {
            tys.pop()?; // type
            tys.pop()?; // value
            out.push(JitOp::IsTypeLate);
            out.push(JitOp::BailIfError);
            tys.push(Ty::of_repr(Repr::Bool)); // `is` → Boolean
        }
        Op::AsTypeLate => {
            tys.pop()?; // class
            tys.pop()?; // value
            out.push(JitOp::AsTypeLate);
            out.push(JitOp::BailIfError);
            tys.push(Ty::boxed());
        }
        // construct: pop args + ctor, push `new ctor(args)` (may throw).
        Op::Construct { num_args } => {
            out.push(JitOp::Construct(*num_args));
            out.push(JitOp::BailIfError);
            for _ in 0..*num_args {
                tys.pop()?;
            }
            tys.pop()?; // ctor
            tys.push(Ty::boxed());
        }
        // constructprop: `new source.<mn>(args)`; pops args + source, pushes the object.
        Op::ConstructProp { multiname, num_args } => {
            if multiname.has_lazy_component() {
                set_decline("lazy_multiname");
                return None;
            }
            let ptr = gc_arena::Gc::as_ptr(*multiname) as usize as u64;
            out.push(JitOp::ConstructProp(ptr, *num_args));
            out.push(JitOp::BailIfError);
            for _ in 0..*num_args {
                tys.pop()?;
            }
            tys.pop()?; // source
            tys.push(Ty::boxed());
        }
        // constructslot: `new (source.slot[index])(args)`; pops args + source, pushes the object.
        Op::ConstructSlot { index, num_args } => {
            out.push(JitOp::ConstructSlot(*index as u32, *num_args));
            out.push(JitOp::BailIfError);
            for _ in 0..*num_args {
                tys.pop()?;
            }
            tys.pop()?; // source
            tys.push(Ty::boxed());
        }
        // call: `function.call(receiver, args)`; pops args + receiver + function, pushes result.
        Op::Call { num_args } => {
            out.push(JitOp::Call(*num_args));
            out.push(JitOp::BailIfError);
            for _ in 0..*num_args {
                tys.pop()?;
            }
            tys.pop()?; // receiver
            tys.pop()?; // function
            tys.push(Ty::boxed());
        }
        // applytype: `base.<T…>` (e.g. `Vector.<int>`); pops types + base, pushes the applied type.
        Op::ApplyType { num_types } => {
            out.push(JitOp::ApplyType(*num_types));
            out.push(JitOp::BailIfError);
            for _ in 0..*num_types {
                tys.pop()?;
            }
            tys.pop()?; // base
            tys.push(Ty::boxed());
        }
        // newobject: build a dynamic object from `num_args` [name, value] pairs.
        Op::NewObject { num_args } => {
            out.push(JitOp::NewObject(*num_args));
            out.push(JitOp::BailIfError);
            for _ in 0..(*num_args * 2) {
                tys.pop()?;
            }
            tys.push(Ty::boxed());
        }
        // newfunction: build a closure from the baked method + current scope (no stack input).
        Op::NewFunction { method } => {
            out.push(JitOp::NewFunction(method.as_ptr() as usize as u64));
            tys.push(Ty::boxed());
        }
        // in: `name in value`; pops value + name, pushes a Boolean.
        Op::In => {
            tys.pop()?; // value
            tys.pop()?; // name
            out.push(JitOp::In);
            out.push(JitOp::BailIfError);
            tys.push(Ty::of_repr(Repr::Bool)); // `name in value` → Boolean
        }
        // nextvalue/nextname/hasnext: the for..in enumerant ops; pop index + value, push one.
        Op::NextValue => {
            tys.pop()?; // index
            tys.pop()?; // value
            out.push(JitOp::NextValue);
            out.push(JitOp::BailIfError);
            tys.push(Ty::boxed());
        }
        Op::NextName => {
            tys.pop()?;
            tys.pop()?;
            out.push(JitOp::NextName);
            out.push(JitOp::BailIfError);
            tys.push(Ty::boxed());
        }
        Op::HasNext => {
            tys.pop()?;
            tys.pop()?;
            out.push(JitOp::HasNext);
            out.push(JitOp::BailIfError);
            tys.push(Ty::of_repr(Repr::Bool)); // `hasnext` → Boolean
        }
        // MOP (domain-memory) loads + sign-extends: pop 1, push 1.
        Op::Li8 => mop_load_op(out, tys, mop_code::LI8)?,
        Op::Li16 => mop_load_op(out, tys, mop_code::LI16)?,
        Op::Li32 => mop_load_op(out, tys, mop_code::LI32)?,
        Op::Lf32 => mop_load_op(out, tys, mop_code::LF32)?,
        Op::Lf64 => mop_load_op(out, tys, mop_code::LF64)?,
        Op::Sxi1 => mop_load_op(out, tys, mop_code::SXI1)?,
        Op::Sxi8 => mop_load_op(out, tys, mop_code::SXI8)?,
        Op::Sxi16 => mop_load_op(out, tys, mop_code::SXI16)?,
        // MOP stores: pop value + address, no result.
        Op::Si8 => mop_store_op(out, tys, mop_code::SI8)?,
        Op::Si16 => mop_store_op(out, tys, mop_code::SI16)?,
        Op::Si32 => mop_store_op(out, tys, mop_code::SI32)?,
        Op::Sf32 => mop_store_op(out, tys, mop_code::SF32)?,
        Op::Sf64 => mop_store_op(out, tys, mop_code::SF64)?,
        // kill: set local `index` to `undefined` (`op_kill`). Push undefined + store; net
        // stack unchanged.
        Op::Kill { index } => {
            push_const(out, tys, to_bits(Value::Undefined), Repr::Boxed);
            out.push(JitOp::SetLocal(*index));
            tys.pop()?;
            if let Some(s) = locals.get_mut(*index as usize) {
                *s = Ty::boxed();
            }
        }
        // hasnext2: reads/writes locals `object_register`/`index_register` in the frame and
        // pushes a Boolean (no operand-stack input). Those two locals become int/object.
        Op::HasNext2 { object_register, index_register } => {
            out.push(JitOp::HasNext2(*object_register, *index_register));
            out.push(JitOp::BailIfError);
            if let Some(s) = locals.get_mut(*object_register as usize) {
                *s = Ty::boxed();
            }
            if let Some(s) = locals.get_mut(*index_register as usize) {
                *s = Ty::boxed();
            }
            tys.push(Ty::of_repr(Repr::Bool)); // `hasnext2` → Boolean
        }
        // getouterscope: push the index-th captured scope's object (no stack input, no throw).
        Op::GetOuterScope { index } => {
            out.push(JitOp::OuterScope(*index as u32));
            tys.push(Ty::boxed());
        }
        // getscriptglobals: push a baked script's globals (may throw on lazy init).
        Op::GetScriptGlobals { script } => {
            // SAFETY: `Script` is a single-`Gc` newtype live for the method's run; erased for
            // baking, reversed by `helpers::script_globals`.
            let ptr = unsafe { core::mem::transmute::<Script<'gc>, *const ()>(*script) };
            out.push(JitOp::ScriptGlobals(ptr as usize as u64));
            out.push(JitOp::BailIfError);
            tys.push(Ty::boxed());
        }
        // newactivation: push a fresh activation object for the baked class (no throw).
        Op::NewActivation { activation_class } => {
            // SAFETY: as `Op::Coerce` — a live single-`Gc` `Class` handle.
            let ptr = unsafe { core::mem::transmute::<Class<'gc>, *const ()>(*activation_class) };
            out.push(JitOp::NewActivation(ptr as usize as u64));
            tys.push(Ty::boxed());
        }
        // deleteproperty (static multiname): pop receiver, push Boolean. Lazy names decline.
        Op::DeleteProperty { multiname } => {
            if multiname.has_lazy_component() {
                set_decline("lazy_multiname");
                return None;
            }
            let ptr = gc_arena::Gc::as_ptr(*multiname) as usize as u64;
            out.push(JitOp::DeleteProperty(ptr));
            out.push(JitOp::BailIfError);
            tys.pop()?; // receiver
            tys.push(Ty::boxed()); // Boolean result
        }
        // super ops (static multiname): resolve against the bound superclass. Lazy → decline.
        Op::GetSuper { multiname } => {
            if multiname.has_lazy_component() {
                set_decline("lazy_multiname");
                return None;
            }
            let ptr = gc_arena::Gc::as_ptr(*multiname) as usize as u64;
            out.push(JitOp::GetSuper(ptr));
            out.push(JitOp::BailIfError);
            tys.pop()?; // receiver
            tys.push(Ty::boxed());
        }
        Op::SetSuper { multiname } => {
            if multiname.has_lazy_component() {
                set_decline("lazy_multiname");
                return None;
            }
            let ptr = gc_arena::Gc::as_ptr(*multiname) as usize as u64;
            out.push(JitOp::SetSuper(ptr));
            out.push(JitOp::BailIfPerr);
            tys.pop()?; // value
            tys.pop()?; // receiver
        }
        Op::CallSuper { multiname, num_args } => {
            if multiname.has_lazy_component() {
                set_decline("lazy_multiname");
                return None;
            }
            let ptr = gc_arena::Gc::as_ptr(*multiname) as usize as u64;
            out.push(JitOp::CallSuper(ptr, *num_args));
            out.push(JitOp::BailIfError);
            for _ in 0..*num_args {
                tys.pop()?;
            }
            tys.pop()?; // receiver
            tys.push(Ty::boxed());
        }
        Op::ConstructSuper { num_args } => {
            out.push(JitOp::ConstructSuper(*num_args));
            out.push(JitOp::BailIfError);
            out.push(JitOp::Pop); // discard the (undefined) result
            for _ in 0..*num_args {
                tys.pop()?;
            }
            tys.pop()?; // receiver (no result pushed)
        }
        // setproperty (static multiname): `object.set_property(mn, value)`. Pops value then
        // object; may throw (null receiver / setter) → BailIfError.
        Op::SetPropertyStatic { multiname } => {
            if multiname.has_lazy_component() {
                set_decline("lazy_multiname");
                return None;
            }
            let ptr = gc_arena::Gc::as_ptr(*multiname) as usize as u64;
            out.push(JitOp::SetProperty(ptr));
            out.push(JitOp::BailIfPerr);
            tys.pop()?; // value
            tys.pop()?; // object
        }
        // Slot writes (mode: 0 coerce / 1 coerce-int / 2 no-coerce).
        Op::SetSlot { index } => set_slot(out, tys, *index, 0)?,
        Op::SetSlotCoerceI { index } => set_slot(out, tys, *index, 1)?,
        Op::SetSlotNoCoerce { index } => set_slot(out, tys, *index, 2)?,
        Op::Add { .. } => bin(out, tys, binop_code::ADD)?,
        Op::Subtract { .. } => bin(out, tys, binop_code::SUBTRACT)?,
        Op::Multiply => bin(out, tys, binop_code::MULTIPLY)?,
        Op::Divide => bin(out, tys, binop_code::DIVIDE)?,
        Op::Modulo => bin(out, tys, binop_code::MODULO)?,
        Op::BitAnd => bin(out, tys, binop_code::BITAND)?,
        Op::BitOr => bin(out, tys, binop_code::BITOR)?,
        Op::BitXor => bin(out, tys, binop_code::BITXOR)?,
        Op::LShift => bin(out, tys, binop_code::LSHIFT)?,
        Op::RShift => bin(out, tys, binop_code::RSHIFT)?,
        Op::URShift => bin(out, tys, binop_code::URSHIFT)?,
        Op::Equals => bin(out, tys, binop_code::EQUALS)?,
        Op::StrictEquals => bin(out, tys, binop_code::STRICT_EQUALS)?,
        Op::LessThan => bin(out, tys, binop_code::LESS_THAN)?,
        Op::LessEquals => bin(out, tys, binop_code::LESS_EQUALS)?,
        Op::GreaterThan => bin(out, tys, binop_code::GREATER_THAN)?,
        Op::GreaterEquals => bin(out, tys, binop_code::GREATER_EQUALS)?,
        Op::AddI => bin(out, tys, binop_code::ADD_I)?,
        Op::SubtractI => bin(out, tys, binop_code::SUBTRACT_I)?,
        Op::MultiplyI => bin(out, tys, binop_code::MULTIPLY_I)?,
        Op::Negate => un(out, tys, unop_code::NEGATE)?,
        Op::Increment => un(out, tys, unop_code::INCREMENT)?,
        Op::Decrement => un(out, tys, unop_code::DECREMENT)?,
        Op::Not => un(out, tys, unop_code::NOT)?,
        Op::BitNot => un(out, tys, unop_code::BITNOT)?,
        Op::NegateI => un(out, tys, unop_code::NEGATE_I)?,
        Op::IncrementI => un(out, tys, unop_code::INCREMENT_I)?,
        Op::DecrementI => un(out, tys, unop_code::DECREMENT_I)?,
        Op::CoerceB => un(out, tys, unop_code::COERCE_B)?,
        Op::CoerceD => un(out, tys, unop_code::COERCE_D)?,
        Op::CoerceI => un(out, tys, unop_code::COERCE_I)?,
        Op::CoerceU => un(out, tys, unop_code::COERCE_U)?,
        Op::CoerceS => un(out, tys, unop_code::COERCE_S)?,
        Op::ConvertS => un(out, tys, unop_code::CONVERT_S)?,
        Op::CoerceO => un(out, tys, unop_code::COERCE_O)?,
        Op::CoerceA => {} // `coerce a` (to `*`) is the identity
        Op::Coerce { class } => {
            let top = *tys.last()?;
            // Elide a coerce to `Number` only when the operand is a canonical inline `Number`:
            // `Value::from(same f64)` is a bit-identity. NOT a `NumberBoxed` (the interpreter
            // canonicalizes it → eliding would diverge), and NOT an `int` box (`Numeric` — the
            // int→Number conversion is real).
            if class.is_builtin_number() && top.repr.is_canonical_number() {
                return Some(());
            }
            let result = Ty { class: Some(*class), repr: coerce_result_repr(*class) };
            *tys.last_mut()? = result;
            // SAFETY: `Class` is a live single-`Gc` handle for the method's run; erased for
            // baking, reversed by `helpers::coerce_return` (same as `CoerceReturn`).
            let ptr = unsafe { core::mem::transmute::<Class<'gc>, *const ()>(*class) };
            out.push(JitOp::Coerce(ptr as usize as u64));
            out.push(JitOp::BailIfError);
        }
        Op::Swap => {
            let b = tys.pop()?;
            let a = tys.pop()?;
            tys.push(b);
            tys.push(a);
            out.push(JitOp::Swap);
        }
        Op::IncLocal { index } => local_inplace(out, locals, *index, unop_code::INCREMENT)?,
        Op::DecLocal { index } => local_inplace(out, locals, *index, unop_code::DECREMENT)?,
        Op::IncLocalI { index } => local_inplace(out, locals, *index, unop_code::INCREMENT_I)?,
        Op::DecLocalI { index } => local_inplace(out, locals, *index, unop_code::DECREMENT_I)?,
        // `pushint`: an `int` box → `Int`. `pushuint`: a `uint` box → `Numeric` (a value ≥ 2^31
        // boxes as `Number`). `pushdouble`: `Value::pack` canonicalizes NaN → a canonical `Number`.
        Op::PushInt { value } => push_const(out, tys, to_bits(Value::from(*value)), Repr::Int),
        Op::PushUint { value } => push_const(out, tys, to_bits(Value::from(*value)), Repr::Numeric),
        Op::PushDouble { value } => push_const(out, tys, to_bits(Value::from(*value)), Repr::Number),
        // A boolean literal IS a `Bool` box (`VALUE_BOOL_MARK | b`) — the very bits pushed here.
        // `Repr::Bool` (not `Boxed`) keeps a local that only ever holds booleans promotable to a
        // `BoolI32` register; `Boxed` here silently disqualified every such local.
        Op::PushTrue => push_const(out, tys, to_bits(Value::from(true)), Repr::Bool),
        Op::PushFalse => push_const(out, tys, to_bits(Value::from(false)), Repr::Bool),
        Op::PushNull => push_const(out, tys, to_bits(Value::Null), Repr::Boxed),
        Op::PushUndefined => push_const(out, tys, to_bits(Value::Undefined), Repr::Boxed),
        Op::PushString { string } => push_const(out, tys, to_bits(Value::from(*string)), Repr::Boxed),
        Op::Pop => {
            out.push(JitOp::Pop);
            tys.pop()?;
        }
        // pushscope: pops the object. If the method reads scopes (`getscopeobject`), push it
        // onto the local scope stack (null-check + may throw). Otherwise the scope-stack side
        // effect is unobserved → a plain `pop` (the common `getlocal0; pushscope` prologue).
        Op::PushScope => {
            if needs_scopes {
                out.push(JitOp::ScopePush);
                out.push(JitOp::BailIfPerr);
            } else {
                out.push(JitOp::Pop);
            }
            tys.pop()?;
        }
        // popscope: no operand-stack effect. Maintain the scope stack only when it matters.
        Op::PopScope => {
            if needs_scopes {
                out.push(JitOp::ScopePop);
            }
        }
        // getscopeobject: push the index-th local scope's object (no throw). Requires the
        // scope stack to be maintained (`needs_scopes`); decline otherwise.
        Op::GetScopeObject { index } => {
            if !needs_scopes {
                set_decline("no_scopes");
                return None;
            }
            out.push(JitOp::GetScope(*index as u32));
            tys.push(Ty::boxed());
        }
        Op::Dup => {
            out.push(JitOp::Dup);
            let t = *tys.last()?;
            tys.push(t);
        }
        Op::ReturnValue { return_type } => {
            let top = tys.pop()?;
            match return_type {
                None => out.push(JitOp::ReturnValue),
                // The value already has the return type's class → no coercion.
                Some(rt) if top.class == Some(*rt) => out.push(JitOp::ReturnValue),
                // A `Number` return whose value is a canonical inline `Number`: the return
                // coercion is a bit-identity, drop it. (Not `NumberBoxed` — the interpreter's
                // coercion canonicalizes it, so eliding would diverge.)
                Some(rt) if rt.is_builtin_number() && top.repr.is_canonical_number() => {
                    out.push(JitOp::ReturnValue)
                }
                Some(rt) => {
                    // SAFETY: same as `Op::Coerce` — a live single-`Gc` `Class` handle.
                    let ptr = unsafe { core::mem::transmute::<Class<'gc>, *const ()>(*rt) };
                    out.push(JitOp::CoerceReturn(ptr as usize as u64));
                }
            }
        }
        Op::ReturnVoid { return_type } => {
            // A typed `returnvoid` yields the type's DEFAULT value (mirrors `return_void`):
            // numeric → int 0, boolean → false, void → undefined, else → null. Compute it at
            // compile time and return that constant; untyped → plain `returnvoid` (undefined).
            match return_type {
                None => out.push(JitOp::ReturnVoid),
                Some(rt) => {
                    let default = if rt.is_builtin_void() {
                        to_bits(Value::Undefined)
                    } else if rt.is_builtin_numeric() {
                        to_bits(Value::from(0))
                    } else if rt.is_builtin_boolean() {
                        to_bits(Value::from(false))
                    } else {
                        to_bits(Value::Null)
                    };
                    out.push(JitOp::PushBits(default));
                    out.push(JitOp::ReturnValue);
                }
            }
        }
        other => {
            record_decline(other);
            return None;
        }
    }
    Some(())
}

fn push_const<'gc>(out: &mut Vec<JitOp>, tys: &mut Vec<Ty<'gc>>, bits: u64, repr: Repr) {
    out.push(JitOp::PushBits(bits));
    tys.push(Ty::of_repr(repr));
}

/// A dynamic binary op: pops two operands, pushes one result. Unguarded fast paths when the
/// analysis proves both operands share a repr (no runtime tag guard, no throwing `binop` helper,
/// so no `BailIfError`):
/// - add/sub/mul/div of two canonical `Number`s → [`JitOp::BinOpNum`] (native `f64`).
/// - a comparison of two `Int`s (signed i32 compare) or two canonical `Number`s (f64 compare,
///   NaN-correct) → [`JitOp::BinOpCmp`] — the hot loop-condition (`i < n`) path.
/// Otherwise the guarded [`JitOp::BinOp`] + `BailIfError`, with an inferred result repr.
fn bin<'gc>(out: &mut Vec<JitOp>, tys: &mut Vec<Ty<'gc>>, code: i32) -> Option<()> {
    use binop_code as bc;
    let b = tys.pop()?;
    let a = tys.pop()?;
    // Native i32 arithmetic: both operands are proven `Int` boxes AND the op ALWAYS yields an
    // `int` box (wrapping `*_i`, bitwise, signed shifts — see `JitOp::BinOpInt`). An `int` box
    // unboxes exactly, so this needs no `ToInt32`, no tag guard and no `BailIfError`. This is the
    // dominant FlasCC/Alchemy shape (C `int` arithmetic → `add_i`/`bitand`/`lshift`/…).
    let int_op = matches!(
        code,
        bc::ADD_I
            | bc::SUBTRACT_I
            | bc::MULTIPLY_I
            | bc::BITAND
            | bc::BITOR
            | bc::BITXOR
            | bc::LSHIFT
            | bc::RSHIFT // NB: URSHIFT is conditional — see `urshift_yields_int` below.
    );
    // `urshift` computes `(a as u32) >> (b & 0x1F)` and boxes the `u32` result — as an `int` when
    // it is < 2^31, else as a `Number`. That "else" is what makes the general case `Numeric`. But
    // for a SHIFT COUNT KNOWN NON-ZERO the result is at most `u32::MAX >> 1` = 2^31 - 1, so it is
    // ALWAYS an `int` box. The count is known when it was just pushed as a literal (the same
    // just-emitted-`PushBits` trick `NullishEq` uses); we compare `& 0x1F` exactly as the
    // interpreter does, so a count of e.g. 32 (≡ a 0-shift, which passes a ≥ 2^31 `u32` straight
    // through) correctly fails this test.
    //
    // This matters far beyond `urshift` itself: FlasCC emits `>>>` for every unsigned C shift, so
    // it is the main FACTORY of `Repr::Numeric` — and `Numeric` is the shared blocker for BOTH
    // proven arithmetic AND i32-register promotion (`promotable_as` needs `Int` at every store and
    // every live block entry, and `meet(Int, Numeric) = Numeric` spreads one `Numeric` everywhere).
    let urshift_yields_int = code == bc::URSHIFT
        && matches!(out.last(), Some(&JitOp::PushBits(bits))
            if bits >> 48 == crate::emit::VALUE_INT_MARK >> 48 && (bits as u32) & 0x1F != 0);
    if (int_op || urshift_yields_int) && a.repr.is_int() && b.repr.is_int() {
        out.push(JitOp::BinOpInt(code));
        tys.push(Ty::of_repr(Repr::Int));
        return Some(());
    }
    // Native f64 arithmetic. SOUNDNESS: the interpreter's `add`/`subtract`/`multiply` have an
    // int-int arm that yields an `int` box (or a `Number` only on overflow), so an f64 op is
    // equivalent ONLY when that arm provably cannot fire — i.e. at least one operand is statically
    // a `Number`, so the pair is never both-`int` at runtime. `divide` has NO int arm (it always
    // runs `ToNumber` on both → always a `Number`), so ANY numeric pair qualifies. Either way the
    // result is a canonical `Number`. (This mirrors `binop_result_repr`'s `either_f64` reasoning.)
    // Operands are limited to `Number`/`Int` — both unbox to f64 exactly and without `valueOf`.
    // `Numeric`/`NumberBoxed` would need a runtime int-vs-double branch / a `Gc<f64>` deref, so
    // they stay on the guarded path for now.
    // Native f64 arithmetic. Operands are limited to `Number` (canonical inline — bits ARE the
    // double) and `Int` (payload IS an i32); both unbox exactly, with no branch and no `valueOf`.
    //
    // `NumberBoxed`/`Numeric` are DELIBERATELY EXCLUDED — see the `NumUnbox` doc. Generalizing to
    // them looks obviously right and is NOT: `Repr::NumberBoxed` can be an **`int` box**, because
    // `matches_type(Number)` accepts an `int`-class value (optimizer/type_aware.rs), so a
    // `Number`-typed slot can be `SetSlotNoCoerce`d with a `Value::Integer`. That breaks the unbox
    // (a blind reinterpret of an int box reads the tag as a double — an instant grey screen) AND
    // the `int_arm_impossible` reasoning below (two "NumberBoxed" operands could be int+int at
    // runtime, firing the interpreter's int-int arm → an `int` box result, not the canonical
    // `Number` this path promises). `Repr::is_f64_unboxable()` returning `true` for `NumberBoxed`
    // is therefore a LIE — a latent trap this comment exists to stop the next person re-walking.
    let f64_op = matches!(code, bc::ADD | bc::SUBTRACT | bc::MULTIPLY | bc::DIVIDE);
    let exact_unbox = |r: Repr| matches!(r, Repr::Number | Repr::Int);
    if f64_op && exact_unbox(a.repr) && exact_unbox(b.repr) {
        // SOUNDNESS: `add`/`subtract`/`multiply` have an int-int arm yielding an `int` box (a
        // `Number` only on overflow), so an f64 op is equivalent ONLY when that arm provably cannot
        // fire — i.e. some operand is a statically canonical `Number`, never an int box at runtime.
        // `divide` has NO int arm (always `ToNumber` on both → always a `Number`), so any pair of
        // exactly-unboxable operands qualifies. Either way the result is a canonical `Number`.
        let int_arm_impossible =
            code == bc::DIVIDE || a.repr.is_canonical_number() || b.repr.is_canonical_number();
        if int_arm_impossible {
            out.push(JitOp::BinOpNum { code, a: num_unbox(a.repr), b: num_unbox(b.repr) });
            tys.push(Ty::of_repr(Repr::Number));
            return Some(());
        }
    }
    // Comparisons: `==`/`===`/`<`/`<=`/`>`/`>=`. When both operands are the SAME proven numeric
    // repr the abstract compare reduces to a native same-type compare (mixed int/Number takes the
    // guarded path). Two `Int`s → signed i32; two canonical `Number`s → f64 (every NaN compare
    // false, matching AS3). Result is a `Bool` box (repr `Boxed` — it feeds `iftrue`/`truthy`).
    let is_cmp = matches!(
        code,
        bc::EQUALS
            | bc::STRICT_EQUALS
            | bc::LESS_THAN
            | bc::LESS_EQUALS
            | bc::GREATER_THAN
            | bc::GREATER_EQUALS
    );
    // Any MIX of exactly-unboxable numerics compares natively. Unlike the arithmetic ops there is
    // no int-int arm to reason about: every comparison reduces two numerics to a compare of their
    // `packed_number()` f64s (`abstract_eq`/`abstract_lt`, and `strict_eq` via `Value`'s custom
    // `PartialEq`, whose `(Number, Integer)` arms compare numerically — NOT by bits), an `i32`
    // converts to `f64` exactly, and the result is always a `Bool`. `NumberBoxed`/`Numeric` still
    // stay guarded — only because their UNBOX needs the 3-way branch (see `NumUnbox`), not for any
    // int-arm reason.
    if is_cmp && a.repr.is_numeric() && b.repr.is_numeric() {
        out.push(JitOp::BinOpCmp { code, a: num_unbox_cmp(a.repr), b: num_unbox_cmp(b.repr) });
        tys.push(Ty::of_repr(Repr::Bool));
        return Some(());
    }
    // `x ==/=== null|undefined`: the OTHER operand is a `null`/`undefined` constant just pushed.
    // Inline the nullish test (a VERY common null check) instead of the `abstract_eq` helper.
    // Sound for any `x`: only `null`/`undefined` are `== null` (no coercion of `x`), and `===`
    // is exact identity. The constant is always the just-emitted `PushBits` (`… pushnull; equals`).
    if matches!(code, bc::EQUALS | bc::STRICT_EQUALS) {
        if let Some(&JitOp::PushBits(bits)) = out.last() {
            if bits == crate::emit::NULL_BITS || bits == crate::emit::UNDEFINED_BITS {
                out.pop(); // drop the constant push; `NullishEq` consumes the other operand
                out.push(JitOp::NullishEq { against: bits, loose: code == bc::EQUALS });
                tys.push(Ty::of_repr(Repr::Bool));
                return Some(());
            }
        }
    }
    record_guarded_bin(code, a.repr, b.repr);
    out.push(JitOp::BinOp(code));
    out.push(JitOp::BailIfError);
    // Even on the GUARDED path a known-non-zero `urshift` yields an `int` box: the op coerces both
    // operands to `u32` FIRST, so whatever the operand reprs (they only decide whether the coerce
    // can throw — and if it throws we bail here), the value that reaches the box is a `u32 >> k`
    // with k ≥ 1, i.e. < 2^31. Overriding `Numeric` → `Int` here is what stops the poisoning at its
    // source for the operand shapes we can't yet prove natively.
    let repr = if urshift_yields_int {
        Repr::Int
    } else {
        binop_result_repr(code, a.repr, b.repr)
    };
    tys.push(Ty::of_repr(repr));
    Some(())
}

// TEMPORARY diagnostic: why a `BinOp` fell through to the GUARDED path — the `(code, a, b)` repr
// triple. Drained per-method by the `RUFFLE_JIT3_HISTO` dump. See `record_guarded_bin`.
thread_local! {
    static GUARD_LOG: std::cell::RefCell<std::collections::BTreeMap<String, usize>> =
        Default::default();
}

fn binop_name(code: i32) -> &'static str {
    use binop_code as bc;
    match code {
        bc::ADD => "add",
        bc::SUBTRACT => "subtract",
        bc::MULTIPLY => "multiply",
        bc::DIVIDE => "divide",
        bc::MODULO => "modulo",
        bc::BITAND => "bitand",
        bc::BITOR => "bitor",
        bc::BITXOR => "bitxor",
        bc::LSHIFT => "lshift",
        bc::RSHIFT => "rshift",
        bc::URSHIFT => "urshift",
        bc::EQUALS => "equals",
        bc::STRICT_EQUALS => "strictequals",
        bc::LESS_THAN => "lessthan",
        bc::LESS_EQUALS => "lessequals",
        bc::GREATER_THAN => "greaterthan",
        bc::GREATER_EQUALS => "greaterequals",
        bc::ADD_I => "add_i",
        bc::SUBTRACT_I => "subtract_i",
        bc::MULTIPLY_I => "multiply_i",
        _ => "?",
    }
}

/// Record one guarded-`BinOp` fallthrough (diagnostic; only when the histogram env is set).
fn record_guarded_bin(code: i32, a: Repr, b: Repr) {
    if crate::histo_min_ops() == 0 {
        return;
    }
    let key = format!("{}({a:?},{b:?})", binop_name(code));
    GUARD_LOG.with(|g| *g.borrow_mut().entry(key).or_default() += 1);
}

// TEMPORARY diagnostic: why a local failed register promotion (`promotable_as`), keyed
// `<target>:<reason>`. Only the FIRST failing reason per (local, target) is recorded — that is the
// one that would have to be fixed first.
thread_local! {
    static PROMO_LOG: std::cell::RefCell<std::collections::BTreeMap<String, usize>> =
        Default::default();
}

/// Record a store that DISQUALIFIES its target local from i32-register promotion: the repr stored
/// and the AVM2 op that stored it — i.e. exactly which producer's repr would have to improve.
fn record_store_non_int(stored: Option<Repr>, out: &[JitOp]) {
    if crate::histo_min_ops() == 0 {
        return;
    }
    // Walk back past the store op itself and any repr-FORWARDING ops (`Dup` — which `StoreLocal`
    // emits itself — and `BailIfError`) to the op that actually ORIGINATED the repr: the producer
    // whose repr would have to improve. Keep just the variant name.
    let producer = match out
        .iter()
        .rev()
        .skip(1)
        .find(|op| !matches!(op, JitOp::Dup | JitOp::BailIfError))
    {
        Some(op) => {
            let s = format!("{op:?}");
            s.split(|c: char| c == ' ' || c == '{' || c == '(').next().unwrap_or("?").to_string()
        }
        None => "<block-entry>".to_string(),
    };
    let key = format!("STORE {producer}->{stored:?}");
    PROMO_LOG.with(|g| *g.borrow_mut().entry(key).or_default() += 1);
}

fn record_promo_fail(target: Repr, reason: &str) {
    if crate::histo_min_ops() == 0 {
        return;
    }
    let key = format!("{target:?}:{reason}");
    PROMO_LOG.with(|g| *g.borrow_mut().entry(key).or_default() += 1);
}

/// Drain the promotion-failure log (diagnostic).
pub(crate) fn take_promo_log() -> Vec<(String, usize)> {
    PROMO_LOG.with(|g| {
        let mut v: Vec<(String, usize)> =
            std::mem::take(&mut *g.borrow_mut()).into_iter().collect();
        v.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
        v
    })
}

/// Drain the guarded-`BinOp` reason log (diagnostic).
pub(crate) fn take_guard_log() -> Vec<(String, usize)> {
    GUARD_LOG.with(|g| {
        let mut v: Vec<(String, usize)> = std::mem::take(&mut *g.borrow_mut()).into_iter().collect();
        v.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
        v
    })
}

/// The unbox strategy for an EXACTLY-unboxable ARITHMETIC operand. Only `Number`/`Int` qualify —
/// see the `NumberBoxed` trap documented in [`bin`]. Panics otherwise: callers must check first.
fn num_unbox(repr: Repr) -> NumUnbox {
    match repr {
        Repr::Number => NumUnbox::Canonical,
        Repr::Int => NumUnbox::Int,
        _ => unreachable!("not an exactly-unboxable repr: {repr:?}"),
    }
}

/// The unbox strategy for a COMPARISON operand — any proven-numeric repr qualifies, because a
/// comparison has no int-int arm to mis-predict (its result is always a `Bool`); only the unbox has
/// to be right, and `AnyNumeric` covers all three runtime shapes. Panics on a non-numeric repr.
fn num_unbox_cmp(repr: Repr) -> NumUnbox {
    match repr {
        Repr::Number => NumUnbox::Canonical,
        Repr::Int => NumUnbox::Int,
        // Same runtime set despite the different names: {int box, canonical f64, boxed double}.
        Repr::NumberBoxed | Repr::Numeric => NumUnbox::AnyNumeric,
        Repr::Bool | Repr::Boxed => unreachable!("not a numeric repr: {repr:?}"),
    }
}

/// The repr of a boxed `BinOp`'s result.
///
/// SOUNDNESS: `emit`'s int fast path boxes `subtract`/`multiply`/`add` of two runtime-`int`s
/// as an **Integer** (not a canonical `Number`), so a result is provably a canonical `Number`
/// ONLY when at least one operand is statically f64-unboxable — then the pair is never
/// both-int at runtime, the int path can't fire, and the f64/helper path yields a canonical
/// `Number`. Otherwise the arithmetic result is `int`-or-`Number` → `Numeric`.
///
/// `subtract`/`multiply`/`divide`/`modulo` always run `ToNumber` → a numeric result regardless
/// of input. `add` is numeric only if both operands are (else it may be string concat).
/// Bitwise/shift/`*_i` always yield an `int`/`uint` → `Numeric`. Comparisons yield a `Bool`
/// (not yet a tracked repr → `Boxed`).
fn binop_result_repr(code: i32, a: Repr, b: Repr) -> Repr {
    use binop_code as bc;
    // "The interpreter's int-int arm provably cannot fire", i.e. some operand is NEVER an `int` box
    // at runtime. ONLY `is_canonical_number()` proves that. `is_f64_unboxable()` must NOT be used
    // here even though it reads as the natural fit: it is `true` for `NumberBoxed`, which CAN be an
    // int box (`matches_type(Number)` accepts an `int`-class value, so a `Number`-typed slot can be
    // `SetSlotNoCoerce`d with a `Value::Integer` — see the trap note in `bin`). Using it claimed a
    // canonical `Number` for e.g. `multiply(NumberBoxed, Boxed)` when both are ints at runtime and
    // the result is really an `int` box — a latent lie that a downstream `BinOpNum` would turn into
    // a blind reinterpret.
    let int_arm_impossible = a.is_canonical_number() || b.is_canonical_number();
    match code {
        // `divide`/`modulo` have NO int-int arm at all: both always run `ToNumber` and the f64
        // result is boxed as a canonical `Number` (`(a / b).into()`). So the result is `Number`
        // for ANY operands — on the non-throwing path, which is the only one that continues (a
        // coercion throw takes the following `BailIfError` straight out).
        bc::DIVIDE | bc::MODULO => Repr::Number,
        bc::SUBTRACT | bc::MULTIPLY => {
            if int_arm_impossible {
                Repr::Number
            } else {
                Repr::Numeric
            }
        }
        bc::ADD => {
            if a.is_numeric() && b.is_numeric() {
                if int_arm_impossible {
                    Repr::Number
                } else {
                    Repr::Numeric
                }
            } else {
                Repr::Boxed
            }
        }
        // Bitwise / signed shifts / wrapping `*_i` always yield an `int` box → `Int`. URSHIFT is
        // the exception: a `u32` ≥ 2^31 boxes as `Number` → `Numeric`.
        bc::BITAND | bc::BITOR | bc::BITXOR | bc::LSHIFT | bc::RSHIFT | bc::ADD_I | bc::SUBTRACT_I
        | bc::MULTIPLY_I => Repr::Int,
        bc::URSHIFT => Repr::Numeric,
        // Comparisons yield a `Bool` box (regardless of operand reprs).
        bc::EQUALS | bc::STRICT_EQUALS | bc::LESS_THAN | bc::LESS_EQUALS | bc::GREATER_THAN
        | bc::GREATER_EQUALS => Repr::Bool,
        _ => Repr::Boxed,
    }
}

/// A dynamic unary op: pops one operand, pushes one result. A `coerce_d` (`convert_d`) whose
/// operand is already a `Number` (inline or heap-boxed) is a bit-identity → drop it entirely.
fn un<'gc>(out: &mut Vec<JitOp>, tys: &mut Vec<Ty<'gc>>, code: i32) -> Option<()> {
    let a = tys.pop()?;
    // Elide ONLY on a canonical inline `Number`: `convert_d` on it is a true bit-identity
    // (`Value::from(same f64)` → same bits). A `NumberBoxed` (heap colliding-NaN) must NOT be
    // elided — the interpreter's `convert_d` canonicalizes it, so eliding would diverge.
    if code == unop_code::COERCE_D && a.repr.is_canonical_number() {
        tys.push(a); // unchanged — the coerce is dead
        return Some(());
    }
    tys.push(Ty::of_repr(unop_result_repr(code, a.repr)));
    out.push(JitOp::UnOp(code));
    out.push(JitOp::BailIfError);
    Some(())
}

/// The repr of a `UnOp`'s result. `coerce_d` (`convert_d`) is `ToNumber` by definition → a
/// canonical `Number`, never an int box. `negate`/`increment`/`decrement` (generic) are
/// numeric but MAY be int-boxed by the runtime op, so conservatively `Numeric`.
/// `coerce_i`/`coerce_u`/`bitnot`/`*_i` yield an `int`/`uint` → `Numeric`. Bool/String
/// coercions are not yet tracked → `Boxed`.
fn unop_result_repr(code: i32, _a: Repr) -> Repr {
    use unop_code as uc;
    match code {
        uc::COERCE_D => Repr::Number,
        // `coerce_i` (`ToInt32`), `bitnot` (`~ToInt32`), and the wrapping `*_i` ops yield an
        // `int` box → `Int`. `coerce_u` (`ToUint32`) can be `Number` (≥ 2^31) → `Numeric`.
        // Generic `negate`/`increment`/`decrement` MAY be int-boxed by the runtime → `Numeric`.
        uc::COERCE_I | uc::BITNOT | uc::INCREMENT_I | uc::DECREMENT_I | uc::NEGATE_I => Repr::Int,
        uc::COERCE_U | uc::NEGATE | uc::INCREMENT | uc::DECREMENT => Repr::Numeric,
        // `not` and `coerce_b` (`convert_b`) yield a `Bool` box.
        uc::NOT | uc::COERCE_B => Repr::Bool,
        _ => Repr::Boxed,
    }
}

/// A MOP load / sign-extend: pops the address (or value), pushes one result. May throw. A
/// float load (`lf32`/`lf64`) uses `number_lossless` → a possibly-heap-boxed `Number`
/// ([`Repr::NumberBoxed`]); integer loads / sign-extends yield an `int` → `Numeric`.
fn mop_load_op<'gc>(out: &mut Vec<JitOp>, tys: &mut Vec<Ty<'gc>>, code: i32) -> Option<()> {
    use mop_code as mc;
    // A proven-`Int` address lets the inline path skip the per-access address tag check
    // (FlasCC/Alchemy pointer arithmetic produces `Int` addresses).
    let addr_int = tys.pop()?.repr.is_int();
    let repr = match code {
        // `lf32`/`lf64` read raw bits that may be a box-colliding NaN, which the interpreter
        // heap-boxes to stay byte-exact (`number_lossless`) — hence the box-safe repr.
        mc::LF32 | mc::LF64 => Repr::NumberBoxed,
        // Every INTEGER load (`li8`/`li16`/`li32`) and sign-extend (`sxi1`/`sxi8`/`sxi16`) yields
        // `Value::Integer` on BOTH paths — the helper builds one directly (`helpers::
        // mop_load_inner`), and the inline emit re-boxes with `VALUE_INT_MARK`. So the result is
        // ALWAYS an `int` box: `Int`, not `Numeric`.
        //
        // This one word was the main factory of `Repr::Numeric` on FlasCC: the Lua VM reads its
        // whole state out of domainMemory via `li32`, so every loaded value was `Numeric`, which
        // then poisoned all downstream arithmetic (`add_i(Numeric,Int)`, `bitand(Numeric,Int)`, …)
        // AND blocked i32-register promotion (`promotable_as` demands `Int` at every store and
        // live-block entry, and `meet(Int, Numeric) = Numeric` spreads a single `Numeric`).
        _ => Repr::Int,
    };
    tys.push(Ty::of_repr(repr));
    out.push(JitOp::MopLoad(code, addr_int));
    out.push(JitOp::BailIfError);
    Some(())
}

/// A MOP store: pops the address then the value, no result (discards the undefined). May throw.
fn mop_store_op<'gc>(out: &mut Vec<JitOp>, tys: &mut Vec<Ty<'gc>>, code: i32) -> Option<()> {
    let addr_int = tys.pop()?.repr.is_int(); // address (top)
    tys.pop()?; // value
    out.push(JitOp::MopStore(code, addr_int));
    out.push(JitOp::BailIfError);
    out.push(JitOp::Pop); // discard the (undefined) result
    Some(())
}

/// A slot write: pops `value` then `object`, no result. Guarded by a `BailIfError`.
fn set_slot<'gc>(
    out: &mut Vec<JitOp>,
    tys: &mut Vec<Ty<'gc>>,
    index: usize,
    mode: i32,
) -> Option<()> {
    tys.pop()?; // value
    tys.pop()?; // object
    out.push(JitOp::SetSlot(index as u32, mode));
    out.push(JitOp::BailIfPerr);
    Some(())
}

/// In-place `inclocal`/`declocal`: `local[i] = unop(local[i], code)`, no net stack effect. The
/// result repr follows `unop_result_repr` (a `Number` for `inc/declocal`, an `int` for the
/// `_i` variants) — sound because the emitted `UnOp` helper produces exactly that box.
fn local_inplace<'gc>(
    out: &mut Vec<JitOp>,
    locals: &mut [Ty<'gc>],
    index: u32,
    code: i32,
) -> Option<()> {
    out.push(JitOp::GetLocal(index));
    out.push(JitOp::UnOp(code));
    out.push(JitOp::BailIfError);
    out.push(JitOp::SetLocal(index));
    if let Some(slot) = locals.get_mut(index as usize) {
        *slot = Ty::of_repr(unop_result_repr(code, slot.repr));
    }
    Some(())
}

/// Records an unsupported op for the decline log — the first sighting of each op variant is
/// logged (to stderr natively / the browser console on web) so we can see, from real
/// content, which ops to implement next. Subsequent sightings just bump a counter.
use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, HashMap};

thread_local! {
    /// Interns op-name strings to `&'static str` (bounded: ~one leak per distinct op
    /// variant) so the compile-decline reason can flow into the cache + profiler.
    static INTERNED: RefCell<HashMap<String, &'static str>> = RefCell::new(HashMap::new());
    /// The reason `translate` most recently declined — an `op:<Variant>` for a genuinely
    /// unsupported op, or the default `shape/guard` for a structural decline (non-empty
    /// operand stack across an edge, a guarded op like non-null-safe `getslot`, a lazy
    /// multiname, …). Read by `try_compile` after a `None` return.
    static LAST_DECLINE: Cell<&'static str> = const { Cell::new("shape/guard") };
}

/// Interns `name` into a process-lifetime `&'static str` (leaks once per distinct string).
fn intern(name: &str) -> &'static str {
    INTERNED.with(|m| {
        let mut m = m.borrow_mut();
        if let Some(s) = m.get(name) {
            return *s;
        }
        let leaked: &'static str = Box::leak(name.to_string().into_boxed_str());
        m.insert(name.to_string(), leaked);
        leaked
    })
}

/// Resets the decline reason to the structural default; called at the top of `translate`.
pub(crate) fn reset_decline_reason() {
    LAST_DECLINE.with(|c| c.set("shape/guard"));
}

/// Labels the current structural decline (sub-buckets `shape/guard` for the profiler).
fn set_decline(reason: &'static str) {
    LAST_DECLINE.with(|c| c.set(reason));
}

/// The reason `translate` last declined (see [`LAST_DECLINE`]). Read by `try_compile`.
pub(crate) fn last_decline_reason() -> &'static str {
    LAST_DECLINE.with(|c| c.get())
}

fn record_decline(op: &Op) {
    thread_local! {
        static SEEN: RefCell<BTreeMap<String, u64>> = const { RefCell::new(BTreeMap::new()) };
    }
    // Variant name = the leading identifier of the `Debug` form (`Foo { .. }` → `Foo`).
    let dbg = format!("{op:?}");
    let name = dbg
        .split(|c: char| c == ' ' || c == '{' || c == '(')
        .next()
        .unwrap_or("?")
        .to_string();
    // Record the specific op as the current decline reason (frequency-weighted profiler).
    LAST_DECLINE.with(|c| c.set(intern(&format!("op:{name}"))));
    SEEN.with(|m| {
        let mut m = m.borrow_mut();
        let count = m.entry(name.clone()).or_insert(0);
        *count += 1;
        if *count == 1 {
            crate::runner::log_decline(&name);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binopnum_only_for_two_canonical_numbers() {
        use binop_code as bc;
        // Both canonical `Number` → unguarded `BinOpNum`, no `BailIfError`, canonical result.
        let mut out = Vec::new();
        let mut tys = vec![Ty::of_repr(Repr::Number), Ty::of_repr(Repr::Number)];
        bin(&mut out, &mut tys, bc::MULTIPLY).unwrap();
        assert_eq!(out, vec![JitOp::BinOpNum { code: bc::MULTIPLY, a: NumUnbox::Canonical, b: NumUnbox::Canonical }]);
        assert_eq!(tys.len(), 1);
        assert!(tys[0].repr.is_canonical_number());

        // A `NumberBoxed` operand (heap colliding-NaN) must NOT take the reinterpret path.
        for other in [Repr::NumberBoxed, Repr::Numeric, Repr::Boxed] {
            let mut out = Vec::new();
            let mut tys = vec![Ty::of_repr(Repr::Number), Ty::of_repr(other)];
            bin(&mut out, &mut tys, bc::ADD).unwrap();
            assert_eq!(out, vec![JitOp::BinOp(bc::ADD), JitOp::BailIfError], "other={other:?}");
        }
    }

    #[test]
    fn int_arithmetic_of_two_int_boxes_is_unguarded() {
        use binop_code as bc;
        // Ops whose result is ALWAYS an `int` box → native i32, no guard, no `BailIfError`, and
        // the result repr stays `Int` (which is what feeds the i32-register promotion).
        for code in [
            bc::ADD_I,
            bc::SUBTRACT_I,
            bc::MULTIPLY_I,
            bc::BITAND,
            bc::BITOR,
            bc::BITXOR,
            bc::LSHIFT,
            bc::RSHIFT,
        ] {
            let mut out = Vec::new();
            let mut tys = vec![Ty::of_repr(Repr::Int), Ty::of_repr(Repr::Int)];
            bin(&mut out, &mut tys, code).unwrap();
            assert_eq!(out, vec![JitOp::BinOpInt(code)], "code={code}");
            assert!(tys[0].repr.is_int(), "code={code}");
        }
        // `urshift` with a NON-constant count stays guarded and `Numeric`: a 0-shift passes a
        // `u32` ≥ 2^31 straight through, which boxes as a `Number`, not an `int`.
        let mut out = Vec::new();
        let mut tys = vec![Ty::of_repr(Repr::Int), Ty::of_repr(Repr::Int)];
        bin(&mut out, &mut tys, bc::URSHIFT).unwrap();
        assert_eq!(out, vec![JitOp::BinOp(bc::URSHIFT), JitOp::BailIfError]);
        assert_eq!(tys[0].repr, Repr::Numeric);
        // A non-`Int` operand keeps the guard (`ToInt32` of a `Numeric` needs a runtime branch).
        for other in [Repr::Numeric, Repr::Number, Repr::NumberBoxed, Repr::Boxed] {
            let mut out = Vec::new();
            let mut tys = vec![Ty::of_repr(other), Ty::of_repr(Repr::Int)];
            bin(&mut out, &mut tys, bc::ADD_I).unwrap();
            assert_eq!(out, vec![JitOp::BinOp(bc::ADD_I), JitOp::BailIfError], "other={other:?}");
        }
    }

    #[test]
    fn urshift_by_a_known_nonzero_count_yields_int() {
        use binop_code as bc;
        let push_int = |k: i32| JitOp::PushBits(to_bits(Value::from(k)));
        // A literal count whose `& 0x1F` is non-zero ⇒ the `u32` result is < 2^31 ⇒ an `int` box.
        // With both operands `Int` this also becomes a native `i32.shr_u`.
        for k in [1, 2, 31, 33, -1] {
            let mut out = vec![push_int(k)];
            let mut tys = vec![Ty::of_repr(Repr::Int), Ty::of_repr(Repr::Int)];
            bin(&mut out, &mut tys, bc::URSHIFT).unwrap();
            assert_eq!(out, vec![push_int(k), JitOp::BinOpInt(bc::URSHIFT)], "k={k}");
            assert_eq!(tys[0].repr, Repr::Int, "k={k}");
        }
        // THE POISONING FIX: even when the value operand isn't provable (so the op stays guarded),
        // the RESULT is still `Int` — `urshift` coerces to `u32` before boxing, so the operand repr
        // only decides whether the coerce can throw (and a throw bails at the `BailIfError`).
        for other in [Repr::Numeric, Repr::Boxed, Repr::NumberBoxed, Repr::Number] {
            let mut out = vec![push_int(2)];
            let mut tys = vec![Ty::of_repr(other), Ty::of_repr(Repr::Int)];
            bin(&mut out, &mut tys, bc::URSHIFT).unwrap();
            assert_eq!(
                out,
                vec![push_int(2), JitOp::BinOp(bc::URSHIFT), JitOp::BailIfError],
                "other={other:?}"
            );
            assert_eq!(tys[0].repr, Repr::Int, "other={other:?}");
        }
        // SOUNDNESS EDGE: a count whose `& 0x1F` is ZERO is a no-op shift, which passes a `u32`
        // ≥ 2^31 through unchanged → that boxes as a `Number`. Must stay `Numeric` and guarded.
        for k in [0, 32, 64] {
            let mut out = vec![push_int(k)];
            let mut tys = vec![Ty::of_repr(Repr::Int), Ty::of_repr(Repr::Int)];
            bin(&mut out, &mut tys, bc::URSHIFT).unwrap();
            assert_eq!(
                out,
                vec![push_int(k), JitOp::BinOp(bc::URSHIFT), JitOp::BailIfError],
                "k={k}"
            );
            assert_eq!(tys[0].repr, Repr::Numeric, "k={k}");
        }
    }

    #[test]
    fn mixed_number_int_arith_is_unguarded_but_only_when_the_int_arm_cannot_fire() {
        use binop_code as bc;
        // An `Int` operand converts to f64 exactly, so mixed Number/Int is guard-free — BUT only
        // where the interpreter's int-int arm provably cannot fire, else the result would be an
        // `int` box (or an overflowing `Number`), not the canonical `Number` this path produces.
        for code in [bc::ADD, bc::SUBTRACT, bc::MULTIPLY, bc::DIVIDE] {
            // One side statically `Number` ⇒ never both-int at runtime ⇒ pure f64.
            let mut out = Vec::new();
            let mut tys = vec![Ty::of_repr(Repr::Number), Ty::of_repr(Repr::Int)];
            bin(&mut out, &mut tys, code).unwrap();
            assert_eq!(
                out,
                vec![JitOp::BinOpNum { code, a: NumUnbox::Canonical, b: NumUnbox::Int }],
                "code={code}"
            );
            assert!(tys[0].repr.is_canonical_number(), "code={code}");
        }
        // SOUNDNESS EDGE: `add`/`subtract`/`multiply` of two `Int`s CAN take the interpreter's
        // int-int arm (→ an `int` box, or a `Number` only on overflow), so they must stay guarded.
        for code in [bc::ADD, bc::SUBTRACT, bc::MULTIPLY] {
            let mut out = Vec::new();
            let mut tys = vec![Ty::of_repr(Repr::Int), Ty::of_repr(Repr::Int)];
            bin(&mut out, &mut tys, code).unwrap();
            assert_eq!(out, vec![JitOp::BinOp(code), JitOp::BailIfError], "code={code}");
        }
        // `divide` has NO int arm (it always runs `ToNumber` on both) → two `Int`s are fine.
        let mut out = Vec::new();
        let mut tys = vec![Ty::of_repr(Repr::Int), Ty::of_repr(Repr::Int)];
        bin(&mut out, &mut tys, bc::DIVIDE).unwrap();
        assert_eq!(out, vec![JitOp::BinOpNum { code: bc::DIVIDE, a: NumUnbox::Int, b: NumUnbox::Int }]);
        assert!(tys[0].repr.is_canonical_number());
    }

    #[test]
    fn arith_result_is_canonical_number_only_when_an_operand_is_canonical() {
        use binop_code as bc;
        // SOUNDNESS: `subtract`/`multiply`/`add` of two runtime-`int`s take the interpreter's
        // int-int arm and yield an Integer box, NOT a canonical `Number` — so the result is only
        // `Number` when an operand is a statically CANONICAL `Number` (never an int box at run
        // time). Anything weaker is `Numeric`.
        assert_eq!(binop_result_repr(bc::SUBTRACT, Repr::Numeric, Repr::Numeric), Repr::Numeric);
        assert_eq!(binop_result_repr(bc::SUBTRACT, Repr::Boxed, Repr::Boxed), Repr::Numeric);
        assert_eq!(binop_result_repr(bc::SUBTRACT, Repr::Number, Repr::Numeric), Repr::Number);
        // REGRESSION (this line asserted `Number` and was WRONG — it cost a grey screen once a
        // consumer trusted it): `NumberBoxed` can be an **int box** (`matches_type(Number)` admits
        // an `int`-class value, so a `Number` slot can be `SetSlotNoCoerce`d with a `Value::Integer`
        // — see `bin`). So `NumberBoxed`+`Boxed` can be int+int at runtime → an Integer result.
        // `is_f64_unboxable()` must never gate this; only `is_canonical_number()`.
        assert_eq!(binop_result_repr(bc::MULTIPLY, Repr::NumberBoxed, Repr::Boxed), Repr::Numeric);
        assert_eq!(binop_result_repr(bc::SUBTRACT, Repr::NumberBoxed, Repr::Int), Repr::Numeric);
        assert_eq!(binop_result_repr(bc::ADD, Repr::NumberBoxed, Repr::NumberBoxed), Repr::Numeric);
        // `divide`/`modulo` have NO int-int arm — they always `ToNumber` both and box the f64, so
        // the result is a canonical `Number` for ANY operands (a coercion throw bails first).
        assert_eq!(binop_result_repr(bc::DIVIDE, Repr::Int, Repr::Int), Repr::Number);
        assert_eq!(binop_result_repr(bc::DIVIDE, Repr::Boxed, Repr::Boxed), Repr::Number);
        assert_eq!(binop_result_repr(bc::MODULO, Repr::Numeric, Repr::Numeric), Repr::Number);
        // add can be string concat unless both operands are numeric.
        assert_eq!(binop_result_repr(bc::ADD, Repr::Number, Repr::Boxed), Repr::Boxed);
        assert_eq!(binop_result_repr(bc::ADD, Repr::Numeric, Repr::Numeric), Repr::Numeric);
        assert_eq!(binop_result_repr(bc::ADD, Repr::Number, Repr::Numeric), Repr::Number);
        // bitwise/shift always a provable int box.
        assert_eq!(binop_result_repr(bc::BITAND, Repr::Boxed, Repr::Boxed), Repr::Int);
        assert_eq!(binop_result_repr(bc::ADD_I, Repr::Boxed, Repr::Boxed), Repr::Int);
        assert_eq!(binop_result_repr(bc::URSHIFT, Repr::Int, Repr::Int), Repr::Numeric);
    }

    #[test]
    fn coerce_d_elided_only_on_canonical_number() {
        use unop_code as uc;
        // Canonical `Number` → coerce_d is a bit-identity → dropped.
        let mut out = Vec::new();
        let mut tys = vec![Ty::of_repr(Repr::Number)];
        un(&mut out, &mut tys, uc::COERCE_D).unwrap();
        assert!(out.is_empty(), "coerce_d on a canonical Number is dead");
        assert_eq!(tys.len(), 1);

        // NumberBoxed / int / Boxed → the conversion is real (interp canonicalizes / converts).
        for r in [Repr::NumberBoxed, Repr::Numeric, Repr::Boxed] {
            let mut out = Vec::new();
            let mut tys = vec![Ty::of_repr(r)];
            un(&mut out, &mut tys, uc::COERCE_D).unwrap();
            assert_eq!(out, vec![JitOp::UnOp(uc::COERCE_D), JitOp::BailIfError], "r={r:?}");
            assert!(tys[0].repr.is_canonical_number(), "coerce_d result is a canonical Number");
        }
    }

    #[test]
    fn subtract_then_multiply_fuses() {
        use binop_code as bc;
        // (a - b) with an f64 operand is a canonical `Number`; a following `* literal` fuses.
        let mut out = Vec::new();
        let mut tys = vec![Ty::of_repr(Repr::Number), Ty::of_repr(Repr::Boxed)];
        bin(&mut out, &mut tys, bc::SUBTRACT).unwrap(); // guarded (Boxed operand present)
        assert!(tys[0].repr.is_canonical_number());
        tys.push(Ty::of_repr(Repr::Number));
        out.clear();
        bin(&mut out, &mut tys, bc::MULTIPLY).unwrap();
        assert_eq!(out, vec![JitOp::BinOpNum { code: bc::MULTIPLY, a: NumUnbox::Canonical, b: NumUnbox::Canonical }]);
    }
}
