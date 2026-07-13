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

use crate::emit::{Block, JitOp, Promotion, RegKind, Term};
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
    let mut entry_locals: Vec<Option<Vec<Ty<'gc>>>> = vec![None; num_blocks];
    entry_locals[0] = Some(param_seeds);

    // `stored_non_{number,int}[i]` = some store to local `i` stored a value that is NOT a
    // canonical `Number` / NOT a provable `Int` — so `i` is unsound to promote to an f64 / i32
    // register (a `GetLocal` could read the wrong shape). Recomputed each pass (store reprs
    // depend on the seeds); after the fixpoint holds the final (stable) verdict. `hasnext2`/
    // `kill` stores exclude their locals from both automatically.
    let mut stored_non_number = vec![false; nlocals];
    let mut stored_non_int = vec![false; nlocals];
    let mut stored_non_bool = vec![false; nlocals];

    let mut blocks = Vec::with_capacity(num_blocks);
    loop {
        let mut changed = false;
        stored_non_number.iter_mut().for_each(|x| *x = false);
        stored_non_int.iter_mut().for_each(|x| *x = false);
        stored_non_bool.iter_mut().for_each(|x| *x = false);
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
                    op @ (Op::ReturnValue { .. } | Op::ReturnVoid { .. }) => {
                        translate_op(op, i, null_safe, needs_scopes, &mut out, &mut tys, &mut locals)?;
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
                        translate_op(op, i, null_safe, needs_scopes, &mut out, &mut tys, &mut locals)?;
                        // Track store reprs for promotion: after `translate_op`, a store op's
                        // target `locals[idx]` holds the stored repr. Flag stores that aren't a
                        // canonical `Number` / a provable `Int` (each disqualifies that target).
                        for idx in store_targets(op) {
                            let r = locals.get(idx as usize).map(|t| t.repr);
                            if r != Some(Repr::Number) {
                                if let Some(s) = stored_non_number.get_mut(idx as usize) {
                                    *s = true;
                                }
                            }
                            if r != Some(Repr::Int) {
                                if let Some(s) = stored_non_int.get_mut(idx as usize) {
                                    *s = true;
                                }
                            }
                            if r != Some(Repr::Bool) {
                                if let Some(s) = stored_non_bool.get_mut(idx as usize) {
                                    *s = true;
                                }
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
    //   (1) every store to `i` is a canonical `Number` (`!stored_nonnumber[i]`), AND
    //   (2) at every block where `i` is LIVE-on-entry, `i` is a canonical `Number` there.
    // (2) uses liveness so a non-param whose `undefined` initial value is dead (def before use)
    // still qualifies — at its live blocks it is already `Number`. Combined, `i` holds a
    // canonical `Number` at every point it can be READ, so `GetLocal`↔f64 register round-trips
    // are bit-identity. `hasnext2` regs / `kill`ed / `inc·declocal`'d locals fail (1) (their
    // stores aren't canonical `Number`), so they are excluded — no frame desync. No writeback:
    // nothing reads the (now stale) frame slot (only `GetLocal`, redirected; reification helpers
    // never read caller locals), and the stale value is a non-pointer.
    let live_in = compute_liveness(ops, &leaders, &blocks, nlocals);
    // Is local `i` promotable to a `target` (`Number`→f64 / `Int`→i32) register? Yes iff every
    // store to it is that repr (`!stored_non`) and, at every block where it is LIVE-on-entry, it
    // is that repr. Returns `init_from_frame` (`i` is that exact repr at method entry — a typed
    // param, loaded from the frame; else a dead-on-entry accumulator → default register value).
    let promotable_as = |i: usize, target: Repr, stored_non: &[bool]| -> Option<bool> {
        if stored_non[i] {
            return None;
        }
        for b in 0..num_blocks {
            if live_in[b][i] && entry_locals[b].as_ref().map(|l| l[i].repr) != Some(target) {
                return None;
            }
        }
        Some(entry_locals[0].as_ref().is_some_and(|l| l[i].repr == target))
    };
    let promoted: Vec<Promotion> = (0..nlocals)
        .filter_map(|i| {
            let local = i as u32;
            if let Some(init_from_frame) = promotable_as(i, Repr::Number, &stored_non_number) {
                Some(Promotion { local, kind: RegKind::F64, init_from_frame })
            } else if let Some(init_from_frame) = promotable_as(i, Repr::Int, &stored_non_int) {
                Some(Promotion { local, kind: RegKind::IntI32, init_from_frame })
            } else if let Some(init_from_frame) = promotable_as(i, Repr::Bool, &stored_non_bool) {
                Some(Promotion { local, kind: RegKind::BoolI32, init_from_frame })
            } else {
                None
            }
        })
        .collect();

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
    needs_scopes: bool,
    out: &mut Vec<JitOp>,
    tys: &mut Vec<Ty<'gc>>,
    locals: &mut Vec<Ty<'gc>>,
) -> Option<()> {
    match op {
        Op::GetLocal { index } => {
            out.push(JitOp::GetLocal(*index));
            tys.push(locals.get(*index as usize).copied().unwrap_or_else(Ty::boxed));
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
            tys.push(Ty::boxed());
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
        Op::PushTrue => push_const(out, tys, to_bits(Value::from(true)), Repr::Boxed),
        Op::PushFalse => push_const(out, tys, to_bits(Value::from(false)), Repr::Boxed),
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
    let f64_op = matches!(code, bc::ADD | bc::SUBTRACT | bc::MULTIPLY | bc::DIVIDE);
    if f64_op && a.repr.is_canonical_number() && b.repr.is_canonical_number() {
        out.push(JitOp::BinOpNum(code));
        tys.push(Ty::of_repr(Repr::Number));
        return Some(());
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
    if is_cmp {
        if a.repr.is_int() && b.repr.is_int() {
            out.push(JitOp::BinOpCmp(code, false)); // i32 compare
            tys.push(Ty::of_repr(Repr::Bool));
            return Some(());
        }
        if a.repr.is_canonical_number() && b.repr.is_canonical_number() {
            out.push(JitOp::BinOpCmp(code, true)); // f64 compare
            tys.push(Ty::of_repr(Repr::Bool));
            return Some(());
        }
    }
    out.push(JitOp::BinOp(code));
    out.push(JitOp::BailIfError);
    tys.push(Ty::of_repr(binop_result_repr(code, a.repr, b.repr)));
    Some(())
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
    let either_f64 = a.is_f64_unboxable() || b.is_f64_unboxable();
    match code {
        bc::SUBTRACT | bc::MULTIPLY | bc::DIVIDE | bc::MODULO => {
            if either_f64 {
                Repr::Number
            } else {
                Repr::Numeric
            }
        }
        bc::ADD => {
            if a.is_numeric() && b.is_numeric() {
                if either_f64 {
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
    tys.pop()?;
    let repr = match code {
        mc::LF32 | mc::LF64 => Repr::NumberBoxed,
        _ => Repr::Numeric,
    };
    tys.push(Ty::of_repr(repr));
    out.push(JitOp::MopLoad(code));
    out.push(JitOp::BailIfError);
    Some(())
}

/// A MOP store: pops the address then the value, no result (discards the undefined). May throw.
fn mop_store_op<'gc>(out: &mut Vec<JitOp>, tys: &mut Vec<Ty<'gc>>, code: i32) -> Option<()> {
    tys.pop()?; // address
    tys.pop()?; // value
    out.push(JitOp::MopStore(code));
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
        assert_eq!(out, vec![JitOp::BinOpNum(bc::MULTIPLY)]);
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
    fn arith_result_is_canonical_number_only_when_an_operand_is_f64() {
        use binop_code as bc;
        // SOUNDNESS: subtract of two `int` boxes yields an Integer box, NOT a canonical
        // `Number` — so the result is `Numeric` unless an operand is already f64-unboxable.
        assert_eq!(binop_result_repr(bc::SUBTRACT, Repr::Numeric, Repr::Numeric), Repr::Numeric);
        assert_eq!(binop_result_repr(bc::SUBTRACT, Repr::Boxed, Repr::Boxed), Repr::Numeric);
        assert_eq!(binop_result_repr(bc::SUBTRACT, Repr::Number, Repr::Numeric), Repr::Number);
        assert_eq!(binop_result_repr(bc::MULTIPLY, Repr::NumberBoxed, Repr::Boxed), Repr::Number);
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
        assert_eq!(out, vec![JitOp::BinOpNum(bc::MULTIPLY)]);
    }
}
