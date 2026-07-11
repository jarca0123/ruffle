// Until the lowering pass consumes it, this analysis is unreferenced.
#![allow(dead_code)]

//! Value-representation inference for the boxed JIT path — the first, pure
//! (analysis-only) half of type specialization.
//!
//! The boxed path threads NaN-boxed `Value`s (i64) and, for every arithmetic op,
//! runtime-checks `is_numeric` and boxes/unboxes. When we can *prove* statically
//! that a value is numeric (an `int` box or a `Number`), the emitted arithmetic
//! can skip that runtime check and its helper fallback — and, later, keep the
//! value unboxed as `f64` across a run.
//!
//! This module computes, soundly and conservatively, which operand-stack slots
//! and locals are provably [`Repr::Numeric`] at each program point. It touches no
//! lowering and cannot cause miscompilation on its own; a lowering pass consumes
//! its output. "Unsure" always resolves to [`Repr::Boxed`], so a wrong inference
//! can only *miss* an optimization, never enable an unsound one.
//!
//! Soundness rests on the stack model staying aligned with the real lowering.
//! [`stack_arity`] is the single source of truth for each op's operand-stack
//! `(pop, push)`; the lowering (`emit_body`) cross-checks its own `depth` against
//! it under `debug_assertions`, so an arity mistake trips tests, never ships.

use crate::lower::JitOp;

/// The proven representation of a value. A two-point lattice: `Numeric` (an `int`
/// box **or** a `Number` — either way safe to feed `emit_numval`/`ToInt32`) meets
/// with `Boxed` (anything) to `Boxed`. We deliberately do NOT split `Int` vs
/// `Number` here: this first slice only *elides the runtime numeric check*, it
/// does not rebox, so the int-vs-Number distinction (which would matter for
/// `is int` if we reboxed) never arises.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Repr {
    /// Provably a `Number` box whose bits ARE a **canonical, inline** `f64` (never a
    /// heap-boxed colliding NaN). Safe to `f64.reinterpret` (`emit_numval`) and to
    /// re-box with `box_double` — a sound unbox/box round-trip. This is the stronger
    /// fact phase-2 physical unboxing requires. Produced by AS3 arithmetic /
    /// `Number`-typed slots / literals, all of which coerce (canonicalize) first.
    Number,
    /// Provably numeric, but possibly a **heap-boxed double**
    /// (`TAG_BOXED_DOUBLE`) — a colliding NaN that `number_lossless` stashed in a
    /// `Gc<f64>`, or a non-canonical inline NaN. Must be unboxed with
    /// [`crate::lower::emit_numval_boxsafe`] (plain `emit_numval` would read a
    /// pointer as an int). Produced only by `lf32`/`lf64` (`DmLoadF`), whose result
    /// is always a `Number` or a swallowed-OOB `Undefined` — never an object
    /// needing `valueOf` — so eliding the runtime `is_numeric` guard is sound (an
    /// `Undefined` unboxes to a pure `0.0`, differing from Adobe only in the already
    /// non-conforming swallowed-OOB path). NEVER an arith result (those are
    /// canonical `Number`), so a physically-unboxed `f64` is never `NumberBoxed`.
    NumberBoxed,
    /// Provably an `int` box OR a canonical `Number` — safe to feed `emit_numval`
    /// (which handles both), so safe to skip the runtime `is_numeric` check (phase
    /// 1), but NOT safe to blindly re-box as `Number` (would break `is int`).
    Numeric,
    /// Unknown — could be any `Value`.
    Boxed,
}

impl Repr {
    /// True if the value is provably numeric — an `int` box, a canonical `Number`,
    /// or a possibly-heap-boxed `Number` ([`Repr::NumberBoxed`]). Gates dropping the
    /// runtime `is_numeric` check; every numeric repr is unboxable to `f64` without
    /// a `valueOf` coercion.
    pub(crate) fn is_numeric(self) -> bool {
        matches!(self, Repr::Numeric | Repr::Number | Repr::NumberBoxed)
    }

    /// True if the value is unboxable to a raw `f64` (a genuine `Number`, inline or
    /// heap-boxed) — the eligibility for phase-2 physical unboxing. An `int`-box
    /// (`Numeric` but not `Number`/`NumberBoxed`) is excluded: it would need
    /// `convert_i32`, not the `f64` fast path.
    pub(crate) fn is_f64_unboxable(self) -> bool {
        matches!(self, Repr::Number | Repr::NumberBoxed)
    }

    /// True if unboxing requires the heap-box-safe path
    /// ([`crate::lower::emit_numval_boxsafe`]) rather than plain `emit_numval`.
    pub(crate) fn needs_boxsafe_unbox(self) -> bool {
        matches!(self, Repr::NumberBoxed)
    }

    /// Lattice meet (for merging a slot/local across control-flow predecessors).
    /// `Number` survives only if `Number` on every path; any numeric/heap-boxed mix
    /// degrades to the weakest still-sound numeric fact (`NumberBoxed` if a
    /// possibly-boxed `Number` is on any path, else `Numeric`).
    fn meet(self, other: Repr) -> Repr {
        match (self, other) {
            (Repr::Number, Repr::Number) => Repr::Number,
            // A possibly-heap-boxed `Number` on either path: the merge is still an
            // f64-unboxable `Number`, but must use the box-safe unbox.
            (a, b)
                if a.is_f64_unboxable()
                    && b.is_f64_unboxable()
                    && (a == Repr::NumberBoxed || b == Repr::NumberBoxed) =>
            {
                Repr::NumberBoxed
            }
            (a, b) if a.is_numeric() && b.is_numeric() => Repr::Numeric,
            _ => Repr::Boxed,
        }
    }
}

/// The operand-stack effect `(pop, push)` of `op` in the boxed model — the single
/// source of truth mirrored by `emit_body`'s `depth` bookkeeping. Variable-arity
/// ops read their count from their own fields (e.g. `CallMethod`'s `argc`), so a
/// call popping `argc` args plus its receiver is `(argc + 1, push as u32)`.
pub(crate) fn stack_arity(op: &JitOp) -> (u32, u32) {
    use JitOp::*;
    match *op {
        // Pushes (no pop).
        GetLocal(_) | GetLocalValue(_) | GetLocalDouble(_) | PushInt(_) | PushIntValue(_)
        | PushConst(_) | PushDouble(_) | PushString(_) | PushBits(_) | GetScopeObject(_)
        | GetOuterScope(_) | GetScriptGlobals(_) | NewCatch(_) => (0, 1),

        // Pure stack shuffles.
        Dup | DupValue => (1, 2),
        Nop => (0, 0),
        Pop => (1, 0),
        SwapValue => (2, 2),

        // Stores: pop one.
        SetLocal(_) | SetLocalValue(_) | SetLocalDouble(_) => (1, 0),
        // Peek-and-store: net 0 (value stays on the stack).
        StoreLocalValue(_) | StoreLocalDouble(_) => (1, 1),

        // In-place local inc/dec: no stack effect.
        IncLocalI(_) | DecLocalI(_) | IncDecLocalIValue(_, _) | IncDecLocalNum(_, _) => (0, 0),

        // Unary: pop one, push one.
        IncrementI | DecrementI | CoerceInt(_) | CoerceBool | CoerceString | Coerce(_)
        | IncrementIBoxed | DecrementIBoxed | IncrementD | DecrementD | NegateD => (1, 1),

        // Binary arithmetic / compares: pop two, push one.
        AddI | SubtractI | MultiplyI | LessThan | LessEquals | GreaterThan | GreaterEquals
        | Equals | CmpNum(_) | BitOpInt(_) | ArithInt(_) | ArithNum(_) | CallHelper2(_)
        | AddIBoxed | SubtractIBoxed | MultiplyIBoxed | AddD | SubtractD | MultiplyD
        | DivideD => (2, 1),

        // Arity-1 helper (h{i}).
        CallHelper(_) => (1, 1),

        // Property/slot reads: pop the receiver, push the value.
        GetProperty(_) | GetPropertyIc(_, _) | GetSlot(_, _) => (1, 1),
        // findpropstrict/findproperty: searches the scope chain by its multiname
        // immediate and pushes the found object — no operand-stack pop.
        FindProp(_, _) => (0, 1),
        // getpropertyfast: `[receiver, name]` -> value.
        GetPropertyFast(_, _) => (2, 1),
        // set3 helper: pops (receiver, value); the 3rd arg (slot-id/width) is the
        // op's immediate, not a stack operand. Net `-2`, pushes nothing.
        CallHelper3(_, _) => (2, 0),

        // domainMemory: load pops addr -> value; store pops (value, addr) -> nothing.
        DmLoad(_) | DmLoadF(_) => (1, 1),
        DmStore(_) | DmStoreF(_) => (2, 0),

        // hasnext2: reads two locals, pushes a Boolean — no operand-stack pop.
        HasNext2(_, _) => (0, 1),

        // Branches consume their condition/operands, push nothing.
        IfFalse(_) | IfTrue(_) | IfTrueBoxed(_) | IfFalseBoxed(_) => (1, 0),
        IfLt(_) | IfGe(_) => (2, 0),
        // `pushscope` pops the scope object off the operand stack (onto the scope
        // stack); `popscope` only touches the scope stack.
        PushScopeReal => (1, 0),
        Jump(_) | PopScopeReal => (0, 0),
        LookupSwitch(_) => (1, 0),

        // Returns / throw are terminators; model their pop for stack tracking.
        ReturnValue | ReturnValueBoxed | ReturnValueCoerced | ReturnValueCoerceBaked(_)
        | ReturnDouble | Throw => (1, 0),
        ReturnVoidBoxed(_) => (0, 0),

        // Calls: pop `argc` spilled args + the receiver; push the result iff `push`.
        CallMethod(_, argc, push) | CallProperty(_, argc, push) => {
            (argc + 1, push as u32)
        }
        // Direct-call (§8): same stack effect as `callmethod`.
        CallMethodDirect(_, argc, push, _) => (argc + 1, push as u32),
        // `call value`: function + receiver + argc args.
        CallValue(argc) => (argc + 2, 1),
        // constructsuper: receiver + argc args, pushes nothing.
        ConstructSuper(argc) => (argc + 1, 0),
        // vcall: `spill` operands + the `a` receiver operand IFF the kind has one;
        // pushes the result iff `push`. Mirrors the `VCall` lowering exactly.
        VCall(kind, _, spill, push) => {
            (spill + crate::lower::vc::has_receiver(kind) as u32, push as u32)
        }
    }
}

/// Per-op forward transfer for the operand-stack repr, applied over a single
/// block. `locals` is the block-entry local repr (mutated as `SetLocal*` fire).
/// Returns the outgoing local repr, or `None` if the stack under/overflows (a
/// malformed block — the caller then bails out of specialization).
///
/// Only a small, provably-numeric-producing op set yields [`Repr::Numeric`];
/// everything else produces [`Repr::Boxed`]. Conservative by construction.
pub(crate) fn simulate_block(
    block: &[JitOp],
    mut locals: Vec<Repr>,
) -> Option<(Vec<Repr>, Vec<Vec<Repr>>)> {
    let mut stack: Vec<Repr> = Vec::new();
    // The stack repr *before* each op — what a lowering pass consults to decide
    // whether an op's operands are numeric.
    let mut pre: Vec<Vec<Repr>> = Vec::with_capacity(block.len());

    for op in block {
        pre.push(stack.clone());
        let (pop, push) = stack_arity(op);
        if (stack.len() as u32) < pop {
            return None; // malformed / not modelled — bail
        }

        // Result repr for the pushed value(s), by op.
        let result = op_result_repr(op, &stack);

        // Special local-affecting ops (their arity is (…,) above; here we also
        // move repr into/out of `locals`).
        match *op {
            JitOp::GetLocal(i) | JitOp::GetLocalValue(i) | JitOp::GetLocalDouble(i) => {
                // Overwrite the just-computed push with the local's repr.
                let r = *locals.get(i as usize)?;
                for _ in 0..pop {
                    stack.pop();
                }
                for _ in 0..push {
                    stack.push(r);
                }
                continue;
            }
            JitOp::SetLocal(i) | JitOp::SetLocalValue(i) | JitOp::SetLocalDouble(i) => {
                let v = stack.pop()?;
                *locals.get_mut(i as usize)? = v;
                continue;
            }
            JitOp::StoreLocalValue(i) | JitOp::StoreLocalDouble(i) => {
                let v = *stack.last()?;
                *locals.get_mut(i as usize)? = v;
                continue;
            }
            _ => {}
        }

        for _ in 0..pop {
            stack.pop();
        }
        for _ in 0..push {
            stack.push(result);
        }
    }
    Some((locals, pre))
}

/// The repr of an op's pushed result, given the current stack (top-of-stack last).
/// Numeric only where the op provably yields an `int`/`Number`.
fn op_result_repr(op: &JitOp, stack: &[Repr]) -> Repr {
    use JitOp::*;
    let n = stack.len();
    let top = |k: usize| stack.get(n.wrapping_sub(k)).copied().unwrap_or(Repr::Boxed);
    let both_numeric = || n >= 2 && top(1).is_numeric() && top(2).is_numeric();
    // A genuine `Number` operand (inline or heap-boxed) — its presence forces an
    // arith result down the f64 path (never the int-box result), so the result is a
    // (canonical) `Number`.
    let either_numberish = || n >= 2 && (top(1).is_f64_unboxable() || top(2).is_f64_unboxable());
    match *op {
        // `lf32`/`lf64` domainMemory load: a `Number` (inline OR heap-boxed via
        // `number_lossless`) or a swallowed-OOB `Undefined`. Never int/object.
        DmLoadF(_) => Repr::NumberBoxed,
        // Provably a `Number` box.
        PushDouble(_) => Repr::Number,
        // `int`/`uint` constants + coercions: numeric, but an int box (not `Number`).
        PushInt(_) | PushIntValue(_) | CoerceInt(_) => Repr::Numeric,
        // Double-path ops (operands already `f64`) → `Number`.
        AddD | SubtractD | MultiplyD | DivideD | IncrementD | DecrementD | NegateD => {
            Repr::Number
        }
        // `divide` (boxed): always a `Number` when operands are numeric.
        ArithNum(_) => {
            if both_numeric() { Repr::Number } else { Repr::Boxed }
        }
        // `multiply`/`subtract`/`add` (boxed): AVM2 yields a `Number` when a
        // `Number` operand is present (the int-box path can't be taken); when both
        // are only-`Numeric` (possibly int) the result is int-or-Number → `Numeric`.
        ArithInt(_) | AddIBoxed | SubtractIBoxed | MultiplyIBoxed => {
            if both_numeric() {
                // A genuine `Number` operand (inline or heap-boxed) forces the f64
                // result path → a canonical `Number`. Only int-box × int-box stays
                // int-or-`Number` (`Numeric`).
                if either_numberish() {
                    Repr::Number
                } else {
                    Repr::Numeric
                }
            } else {
                Repr::Boxed
            }
        }
        IncrementIBoxed | DecrementIBoxed => {
            if top(1).is_numeric() { Repr::Numeric } else { Repr::Boxed }
        }
        // A `getslot` / `getpropertyfast` proven to return a `Number` (a `Number`
        // slot, or a `Vector.<Number>` element).
        GetSlot(_, true) | GetPropertyFast(_, true) => Repr::Number,
        // Everything else: unknown.
        _ => Repr::Boxed,
    }
}

/// The basic blocks control can reach from block `bb` (by its terminator).
fn successors(ops: &[JitOp], leaders: &[usize], bb: usize, bb_of: &impl Fn(usize) -> Option<usize>) -> Vec<usize> {
    let end = leaders.get(bb + 1).copied().unwrap_or(ops.len());
    let next = (bb + 1 < leaders.len()).then_some(bb + 1);
    match ops[end - 1] {
        JitOp::ReturnValue | JitOp::ReturnValueBoxed | JitOp::ReturnValueCoerced
        | JitOp::ReturnValueCoerceBaked(_)
        | JitOp::ReturnDouble | JitOp::ReturnVoidBoxed(_) => vec![],
        JitOp::Jump(t) => bb_of(t).into_iter().collect(),
        JitOp::IfTrue(t) | JitOp::IfFalse(t) | JitOp::IfLt(t) | JitOp::IfGe(t)
        | JitOp::IfTrueBoxed(t) | JitOp::IfFalseBoxed(t) => bb_of(t).into_iter().chain(next).collect(),
        // Throw / lookupswitch have irregular edges; be conservative and treat as
        // having no modelled successors (their targets start Boxed anyway).
        JitOp::Throw | JitOp::LookupSwitch(_) => vec![],
        _ => next.into_iter().collect(),
    }
}

/// Fixpoint over the CFG: the block-entry local reprs, meeting (numeric-only-if-
/// numeric-on-every-path) at merges. `entry_locals` seeds block 0 from the
/// method's parameter types. Returns per-block entry local reprs, or `None` if a
/// block is malformed / unmodelled (caller then skips specialization).
pub(crate) fn infer_local_reprs(ops: &[JitOp], entry_locals: &[Repr]) -> Option<Vec<Vec<Repr>>> {
    let leaders = crate::lower::basic_block_leaders(ops)?;
    let n = leaders.len();
    let bb_of = |op_idx: usize| leaders.iter().position(|&l| l == op_idx);

    let mut entry: Vec<Option<Vec<Repr>>> = vec![None; n];
    entry[0] = Some(entry_locals.to_vec());
    let mut work = vec![0usize];

    while let Some(bb) = work.pop() {
        let locals_in = entry[bb].clone().expect("queued blocks are initialised");
        let start = leaders[bb];
        let end = leaders.get(bb + 1).copied().unwrap_or(ops.len());
        let (locals_out, _pre) = simulate_block(&ops[start..end], locals_in)?;

        for succ in successors(ops, &leaders, bb, &bb_of) {
            let merged = match &entry[succ] {
                None => locals_out.clone(),
                Some(existing) => existing
                    .iter()
                    .zip(&locals_out)
                    .map(|(&a, &b)| a.meet(b))
                    .collect(),
            };
            if entry[succ].as_deref() != Some(merged.as_slice()) {
                entry[succ] = Some(merged);
                work.push(succ);
            }
        }
    }

    // Unreached blocks (dead code) start Boxed.
    Some(entry.into_iter().map(|e| e.unwrap_or_else(|| entry_locals.iter().map(|_| Repr::Boxed).collect())).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lower::JitOp;

    #[test]
    fn number_param_arith_is_numeric() {
        // f(x:Number): return x * x  — with local[1]=x seeded Numeric.
        let ops = vec![
            JitOp::GetLocalValue(1),
            JitOp::GetLocalValue(1),
            JitOp::ArithNum(11), // multiply
            JitOp::ReturnValueBoxed,
        ];
        let entry = vec![Repr::Boxed, Repr::Numeric]; // this, x
        let (_locals, pre) = simulate_block(&ops, entry).unwrap();
        // Before ArithNum, both operands must be Numeric.
        assert_eq!(pre[2], vec![Repr::Numeric, Repr::Numeric]);
    }

    #[test]
    fn boxed_operand_taints_result() {
        // getslot (unknown) * number  → operands not both numeric.
        let ops = vec![
            JitOp::GetLocalValue(0), // this (Boxed)
            JitOp::GetSlot(0, false),       // Boxed
            JitOp::PushDouble(0),    // Numeric
            JitOp::ArithNum(11),
            JitOp::ReturnValueBoxed,
        ];
        let (_l, pre) = simulate_block(&ops, vec![Repr::Boxed]).unwrap();
        assert_eq!(pre[3], vec![Repr::Boxed, Repr::Number]); // getslot Boxed, pushdouble Number
    }

    #[test]
    fn meet_across_branch_demotes() {
        // Two predecessors set local 1 to Numeric and Boxed respectively; the
        // merge must see Boxed.
        let ops = vec![
            JitOp::GetLocalValue(0), // 0
            JitOp::IfFalse(4),       // 1 -> block at 4 or fallthrough
            JitOp::PushDouble(0),    // 2
            JitOp::SetLocalValue(1), // 3: local1 = Numeric ; falls to 4
            JitOp::GetLocalValue(1), // 4 (merge): local1 read
            JitOp::ReturnValueBoxed, // 5
        ];
        // Seed local1 Boxed. Path through fallthrough sets Numeric; the taken
        // edge (2->4 skipped) keeps Boxed → merge is Boxed.
        let entry = infer_local_reprs(&ops, &[Repr::Boxed, Repr::Boxed]).unwrap();
        // The merge block starting at op 4: local 1 must be Boxed (not numeric on
        // the taken path).
        let leaders = crate::lower::basic_block_leaders(&ops).unwrap();
        let merge_bb = leaders.iter().position(|&l| l == 4).unwrap();
        assert_eq!(entry[merge_bb][1], Repr::Boxed);
    }
}
