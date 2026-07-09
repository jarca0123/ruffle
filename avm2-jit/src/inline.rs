//! Method inlining for the boxed path.
//!
//! Each JIT'd method is its own WASM module; a call to another JIT'd method pays the
//! full per-invocation overhead (`try_run`: reg snapshot, thread-local setup, a fresh
//! entry). For small, statically-resolvable callees that overhead dwarfs the callee's
//! actual work. Inlining splices the callee's [`JitOp`]s into the caller so the call
//! becomes wasm-internal — no re-entry.
//!
//! ## Constraints (this module)
//! Inlining at the [`JitOp`] level with op-index branch targets makes a general
//! splice error-prone, so an inlinable callee must be:
//! - **single trailing return** (`ReturnValueBoxed`/`ReturnVoidBoxed` as the last op),
//!   with no branch targeting it (all paths fall through to it) — branches *within* the
//!   body are fine, the splice remaps their targets to absolute output indices;
//! - **scope-free, call-free, and free of ops with a caller-keyed side-table**
//!   (`getscriptglobals`, `pushstring`, `coerce` — their tables are built from the
//!   *caller's* `parsed_code` and indexed by a `k` the splice does not remap) — but
//!   `GetProperty`/`GetPropertyFast` (multiname *reads*) ARE allowed: the caller builds
//!   a **combined** multiname list (caller's + each callee's) and the splice remaps the
//!   callee's `k`s into it (`mn_base`);
//! - **small** (see [`MAX_INLINE_OPS`]).
//!
//! Trivial super constructors and simple leaf getters/computations satisfy this. The
//! splice remaps the callee's locals by a base offset, sets up the receiver+args from
//! the caller's stack, and drops the callee's return; the caller's branch targets past
//! the splice point are shifted. Resolution + wiring live in [`crate::lib`].

use crate::lower::JitOp;

/// Max callee op count to inline (keeps the frame + code size bounded).
pub(crate) const MAX_INLINE_OPS: usize = 24;

/// How the caller consumes the inlined callee's result.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum ResultMode {
    /// The caller wants the callee's value on the stack (`callmethod`/`call` push).
    /// Unused by phase-1 (super ctors are void); reserved for `this`-call inlining.
    #[allow(dead_code)]
    Push,
    /// The caller discards it (`callpropvoid`/`constructsuper`).
    Discard,
}

/// Whether `callee` (already translated to boxed [`JitOp`]s) may be inlined under the
/// phase-1 constraints. Returns the body length (ops before the final return) and
/// whether that return carries a value, or `None` if not inlinable.
pub(crate) fn callee_inlinable(callee: &[JitOp]) -> Option<(usize, bool)> {
    if callee.is_empty() || callee.len() > MAX_INLINE_OPS {
        return None;
    }
    // Exactly one return, as the last op.
    let (last, body) = callee.split_last()?;
    let ends_in_value = match last {
        JitOp::ReturnValueBoxed => true,
        JitOp::ReturnVoidBoxed(_) => false,
        _ => return None, // ReturnValueCoerced / non-return tail → not inlinable
    };
    // Each body op must be allowed, and any branch must stay *within* the body — a
    // branch to the trailing return (index `body_len`) would create a merge point with
    // the result already on the stack, violating the lowering's empty-stack-at-block-
    // boundary invariant. (In practice a single-return method's branches converge by
    // falling through to the return, so this holds; the check just makes it sound.)
    let body_len = body.len();
    for op in body {
        if !body_op_ok(*op) {
            return None;
        }
        if op.target().is_some_and(|t| t >= body_len) {
            return None;
        }
    }
    Some((body_len, ends_in_value))
}

/// Whether a callee *body* op (everything before the trailing return) is allowed in an
/// inline. Branches (`Jump`/`If*`) ARE allowed — the splice remaps their targets — but
/// returns, scope ops, `getscriptglobals`, and nested calls are not.
fn body_op_ok(op: JitOp) -> bool {
    match op {
        // Returns — excluded (a body has exactly one, the trailing op we split off).
        JitOp::ReturnValue
        | JitOp::ReturnValueBoxed
        | JitOp::ReturnValueCoerced
        | JitOp::ReturnVoidBoxed(_)
        | JitOp::ReturnDouble => false,
        // Scope ops (their scope stack is the caller's), and ops whose per-run
        // side-table is built from the *caller's* `parsed_code` and indexed by a
        // caller-relative `k` the splice does NOT remap — `getscriptglobals`
        // (resolved-bits table), `pushstring` (resolved-string table), and `coerce`
        // (class table). An inlined callee's op carries a *callee*-relative `k`, which
        // would index the caller's table at the wrong (or out-of-range) slot — so
        // exclude them. Nested calls are excluded too.
        JitOp::PushScopeReal
        | JitOp::GetScopeObject(_)
        | JitOp::GetScriptGlobals(_)
        | JitOp::PushString(_)
        | JitOp::Coerce(_)
        | JitOp::CallMethod(..)
        | JitOp::CallProperty(..)
        | JitOp::ConstructSuper(_)
        | JitOp::CallValue(_)
        // `VCall` imms index caller-keyed per-run tables too (multinames the splice
        // COULD remap, but also coerce classes/natives/namespaces it can't) — exclude.
        | JitOp::VCall(..)
        // `lookupswitch`'s targets live in a side-table that splicing doesn't remap.
        | JitOp::LookupSwitch(_) => false,
        // `GetProperty`/`GetPropertyFast` (multiname reads) ARE allowed: the splice
        // remaps their `k` into the caller's combined multiname list (see `mn_base`).
        // Everything else (locals, arithmetic, dm, getslot-by-index, setslot, consts,
        // compares, pop/dup/swap, `coerces`/`coerceb`/`coercei`, …) is a pure
        // straight-line op with no caller-keyed table — inlinable.
        _ => true,
    }
}

/// Whether the receiver of the call at `call_idx` (taking `argc` args) is provably
/// `this` (local 0). Conservative pattern: `GetLocalValue(0); <argc single-push ops>;
/// call` — the receiver push is a `getlocal0` and every op between it and the call is
/// a fresh single push (so nothing below the receiver is disturbed). Sound but not
/// complete (misses compound arg expressions like `this.m(a + b)`), which is fine —
/// a `false` just leaves the call un-inlined.
pub(crate) fn receiver_is_this(caller: &[JitOp], call_idx: usize, argc: u32) -> bool {
    let argc = argc as usize;
    if call_idx < argc + 1 {
        return false;
    }
    let recv_idx = call_idx - argc - 1;
    matches!(caller[recv_idx], JitOp::GetLocalValue(0))
        && caller[recv_idx + 1..call_idx].iter().all(|op| is_single_push(*op))
}

/// Whether `op` pushes exactly one fresh value without consuming anything below it
/// (so it can sit between a `this` receiver and its call without disturbing it).
fn is_single_push(op: JitOp) -> bool {
    matches!(
        op,
        JitOp::GetLocalValue(_)
            | JitOp::PushConst(_)
            | JitOp::PushIntValue(_)
            | JitOp::GetScopeObject(_)
            | JitOp::GetScriptGlobals(_)
    )
}

/// Splices `callee`'s inlinable body into `caller`, replacing the call op at
/// `call_idx`. `local_base` is where the callee's locals are placed in the caller's
/// frame (callee local `i` → caller local `local_base + i`); `argc` is the call's
/// argument count (callee locals `1..=argc` are the params, `0` is the receiver, all
/// popped from the caller's stack top-first); `result` says whether the callee's
/// value stays on the stack or is discarded. Returns the rebuilt op vector, or `None`
/// if the callee isn't inlinable.
///
/// The callee is branchless (see [`callee_inlinable`]), so its ops carry no targets to
/// remap; only the caller's targets past `call_idx` shift by the net op-count delta.
pub(crate) fn splice(
    caller: &[JitOp],
    call_idx: usize,
    callee: &[JitOp],
    local_base: u32,
    mn_base: u32,
    argc: u32,
    result: ResultMode,
) -> Option<Vec<JitOp>> {
    let (body_len, ends_in_value) = callee_inlinable(callee)?;
    let body = &callee[..body_len];

    // The replacement for the single call op: arg-setup + remapped body + result glue.
    let mut repl: Vec<JitOp> = Vec::new();
    // Store receiver+args into the callee's (remapped) locals. Stack top is the last
    // arg; pop into `base+argc`, …, args into `base+1`, receiver into `base+0`.
    for i in (0..=argc).rev() {
        repl.push(JitOp::SetLocalValue(local_base + i));
    }
    // Where the callee body lands in the output: after the caller prefix (`call_idx`)
    // and the arg-setup (`argc + 1` `SetLocalValue`s). A callee branch target `t`
    // (0-based in the callee) maps to `body_base + t`.
    let body_base = call_idx + (argc as usize + 1);
    // The callee body with locals remapped by `local_base`, multiname `k`s by
    // `mn_base`, and branch targets to absolute output indices (`+ body_base`).
    for &op in body {
        repl.push(remap_op(op, local_base, mn_base, body_base));
    }
    // Result glue: reconcile what the callee produced with what the caller wants.
    match (ends_in_value, result) {
        (true, ResultMode::Push) => {}                          // value already on stack
        (true, ResultMode::Discard) => repl.push(JitOp::Pop),   // drop the value
        (false, ResultMode::Discard) => {}                      // nothing on either side
        (false, ResultMode::Push) => {
            // Void callee but the caller wants a value: push what the void return
            // coerced to (the trailing `ReturnVoidBoxed(bits)`).
            let JitOp::ReturnVoidBoxed(bits) = callee[body_len] else {
                return None;
            };
            repl.push(JitOp::PushConst(bits));
        }
    }

    // Rebuild: caller[..call_idx] + repl + caller[call_idx+1..], shifting every branch
    // target that points *after* the removed call op by the net delta.
    let delta = repl.len() as isize - 1; // we removed 1 op, inserted repl.len()
    let mut out: Vec<JitOp> = Vec::with_capacity(caller.len() + repl.len());
    out.extend(caller[..call_idx].iter().map(|&op| shift_target(op, call_idx, delta)));
    out.extend(repl);
    out.extend(caller[call_idx + 1..].iter().map(|&op| shift_target(op, call_idx, delta)));
    Some(out)
}

/// Returns `op` with any local index offset by `local_base`, any multiname index by
/// `mn_base`, and any branch target by `target_base` (callee target `t` → absolute
/// output index `target_base + t`) — so the spliced callee body reads the caller's
/// frame slots + combined multiname list and its branches point into the output.
fn remap_op(op: JitOp, local_base: u32, mn_base: u32, target_base: usize) -> JitOp {
    match op {
        JitOp::GetLocalValue(i) => JitOp::GetLocalValue(i + local_base),
        JitOp::SetLocalValue(i) => JitOp::SetLocalValue(i + local_base),
        JitOp::StoreLocalValue(i) => JitOp::StoreLocalValue(i + local_base),
        JitOp::GetLocal(i) => JitOp::GetLocal(i + local_base),
        JitOp::SetLocal(i) => JitOp::SetLocal(i + local_base),
        JitOp::IncLocalI(i) => JitOp::IncLocalI(i + local_base),
        JitOp::DecLocalI(i) => JitOp::DecLocalI(i + local_base),
        JitOp::GetLocalDouble(i) => JitOp::GetLocalDouble(i + local_base),
        JitOp::SetLocalDouble(i) => JitOp::SetLocalDouble(i + local_base),
        JitOp::StoreLocalDouble(i) => JitOp::StoreLocalDouble(i + local_base),
        JitOp::GetProperty(k) => JitOp::GetProperty(k + mn_base),
        JitOp::GetPropertyFast(k, num) => JitOp::GetPropertyFast(k + mn_base, num),
        JitOp::Jump(t) => JitOp::Jump(target_base + t),
        JitOp::IfTrue(t) => JitOp::IfTrue(target_base + t),
        JitOp::IfFalse(t) => JitOp::IfFalse(target_base + t),
        JitOp::IfLt(t) => JitOp::IfLt(target_base + t),
        JitOp::IfGe(t) => JitOp::IfGe(target_base + t),
        JitOp::IfTrueBoxed(t) => JitOp::IfTrueBoxed(target_base + t),
        JitOp::IfFalseBoxed(t) => JitOp::IfFalseBoxed(target_base + t),
        other => other,
    }
}

/// Returns `op` with its branch target shifted by `delta` if it points *after*
/// `removed_idx` (the spliced-out call op's index). Targets at/before are unchanged.
fn shift_target(op: JitOp, removed_idx: usize, delta: isize) -> JitOp {
    let adjust = |t: usize| -> usize {
        if t > removed_idx {
            (t as isize + delta) as usize
        } else {
            t
        }
    };
    match op {
        JitOp::Jump(t) => JitOp::Jump(adjust(t)),
        JitOp::IfTrue(t) => JitOp::IfTrue(adjust(t)),
        JitOp::IfFalse(t) => JitOp::IfFalse(adjust(t)),
        JitOp::IfLt(t) => JitOp::IfLt(adjust(t)),
        JitOp::IfGe(t) => JitOp::IfGe(adjust(t)),
        JitOp::IfTrueBoxed(t) => JitOp::IfTrueBoxed(adjust(t)),
        JitOp::IfFalseBoxed(t) => JitOp::IfFalseBoxed(adjust(t)),
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const UNDEF: u64 = 0xFFF8_0000_0000_0000;

    #[test]
    fn rejects_non_inlinable() {
        // A branch that targets the trailing return (index 2 = body_len) — rejected
        // (would merge with the result on the stack).
        assert!(callee_inlinable(&[
            JitOp::IfFalseBoxed(2),
            JitOp::Nop,
            JitOp::ReturnVoidBoxed(UNDEF)
        ])
        .is_none());
        // A branch that stays within the body IS allowed.
        assert!(callee_inlinable(&[
            JitOp::IfFalseBoxed(1),
            JitOp::Nop,
            JitOp::ReturnVoidBoxed(UNDEF)
        ])
        .is_some());
        // No trailing return.
        assert!(callee_inlinable(&[JitOp::GetLocalValue(0)]).is_none());
        // A nested call (still excluded).
        assert!(callee_inlinable(&[JitOp::CallMethod(0, 0, true), JitOp::ReturnValueBoxed]).is_none());
        // A multiname *read* IS now allowed (its `k` is remapped by the splice).
        assert!(callee_inlinable(&[JitOp::GetProperty(0), JitOp::ReturnValueBoxed]).is_some());
        // Ops with a caller-keyed per-run table are excluded: the splice does NOT
        // remap their `k`, so a callee-relative index would read the caller's table at
        // the wrong slot. `coerce` (class table) and `pushstring` (string table) —
        // like `getscriptglobals` (globals table).
        assert!(callee_inlinable(&[JitOp::Coerce(0), JitOp::ReturnValueBoxed]).is_none());
        assert!(callee_inlinable(&[JitOp::PushString(0), JitOp::ReturnValueBoxed]).is_none());
        assert!(callee_inlinable(&[JitOp::GetScriptGlobals(0), JitOp::ReturnValueBoxed]).is_none());
        // `coerces`/`coercei` (no side-table — stateless) stay inlinable.
        assert!(callee_inlinable(&[JitOp::CoerceString, JitOp::ReturnValueBoxed]).is_some());
        // Too big.
        let big: Vec<JitOp> = std::iter::repeat(JitOp::Nop)
            .take(MAX_INLINE_OPS + 1)
            .chain([JitOp::ReturnVoidBoxed(UNDEF)])
            .collect();
        assert!(callee_inlinable(&big).is_none());
    }

    #[test]
    fn accepts_simple_callee() {
        // return local1 + local2  (boxed int add), single value return.
        let callee = [
            JitOp::GetLocalValue(1),
            JitOp::GetLocalValue(2),
            JitOp::AddIBoxed,
            JitOp::ReturnValueBoxed,
        ];
        assert_eq!(callee_inlinable(&callee), Some((3, true)));
    }

    #[test]
    fn splice_void_callee_discards_and_shifts_targets() {
        // Caller: [GetLocalValue0, ConstructSuper(0)@1, Jump(4)@2, Nop@3, ReturnVoidBoxed@4]
        // Callee (super ctor): [GetLocalValue0, Pop, ReturnVoidBoxed]  (trivial).
        let caller = [
            JitOp::GetLocalValue(0),   // 0
            JitOp::ConstructSuper(0),  // 1  (the call, argc=0)
            JitOp::Jump(4),            // 2  → target 4 (after the call)
            JitOp::Nop,                // 3
            JitOp::ReturnVoidBoxed(UNDEF), // 4
        ];
        let callee = [JitOp::GetLocalValue(0), JitOp::Pop, JitOp::ReturnVoidBoxed(UNDEF)];
        // argc=0 → 1 SetLocalValue (receiver into base+0), then body (2 ops), no result.
        let out = splice(&caller, 1, &callee, 5, 0, 0, ResultMode::Discard).expect("inlinable");
        // repl = [SetLocalValue(5), GetLocalValue(5), Pop]  (3 ops replacing 1 → delta +2)
        assert_eq!(
            out,
            vec![
                JitOp::GetLocalValue(0),   // 0
                JitOp::SetLocalValue(5),   // 1: receiver → base+0 (=5)
                JitOp::GetLocalValue(5),   // 2: callee getlocal0 remapped
                JitOp::Pop,                // 3: callee pushscope→pop
                JitOp::Jump(6),            // 4: was Jump(4); target 4>1 → 4+2=6
                JitOp::Nop,                // 5
                JitOp::ReturnVoidBoxed(UNDEF), // 6
            ]
        );
    }

    #[test]
    fn splice_remaps_callee_multinames() {
        // Callee: `return this.prop`  → getlocal0; getproperty(0); returnvalue.
        let caller = [JitOp::CallMethod(0, 0, true), JitOp::ReturnValueBoxed];
        let callee = [
            JitOp::GetLocalValue(0),
            JitOp::GetProperty(0),
            JitOp::ReturnValueBoxed,
        ];
        // local_base=5, mn_base=7 → the callee's GetProperty(0) becomes GetProperty(7).
        let out = splice(&caller, 0, &callee, 5, 7, 0, ResultMode::Push).expect("inlinable");
        assert_eq!(
            out,
            vec![
                JitOp::SetLocalValue(5), // receiver → base+0
                JitOp::GetLocalValue(5), // callee getlocal0 remapped
                JitOp::GetProperty(7),   // callee getproperty(0) → mn_base+0 = 7
                JitOp::ReturnValueBoxed, // caller's own return (value kept on stack)
            ]
        );
    }

    #[test]
    fn splice_remaps_callee_branch_targets() {
        // Callee with an internal branch: body = [IfFalseBoxed(2), Nop, PushConst(9)],
        // return @3. The `IfFalseBoxed(2)` targets the callee's `PushConst` (index 2).
        let caller = [JitOp::CallMethod(0, 0, true), JitOp::ReturnValueBoxed];
        let callee = [
            JitOp::IfFalseBoxed(2),
            JitOp::Nop,
            JitOp::PushConst(9),
            JitOp::ReturnValueBoxed,
        ];
        // argc=0 → 1 arg-setup op → body_base = 0 + 1 = 1, so target 2 → 1 + 2 = 3.
        let out = splice(&caller, 0, &callee, 5, 0, 0, ResultMode::Push).expect("inlinable");
        assert_eq!(
            out,
            vec![
                JitOp::SetLocalValue(5),
                JitOp::IfFalseBoxed(3), // callee target 2 → body_base(1)+2 = 3 (the PushConst)
                JitOp::Nop,
                JitOp::PushConst(9),
                JitOp::ReturnValueBoxed, // caller's own return
            ]
        );
    }

    #[test]
    fn receiver_is_this_pattern() {
        // this.m()  → getlocal0; callmethod(argc=0)
        let ops = [JitOp::GetLocalValue(0), JitOp::CallMethod(1, 0, true)];
        assert!(receiver_is_this(&ops, 1, 0));
        // this.m(local1, const)  → getlocal0; getlocal1; pushconst; call(argc=2)
        let ops = [
            JitOp::GetLocalValue(0),
            JitOp::GetLocalValue(1),
            JitOp::PushConst(0),
            JitOp::CallMethod(1, 2, true),
        ];
        assert!(receiver_is_this(&ops, 3, 2));
        // Receiver is NOT this (getlocal1 receiver).
        let ops = [JitOp::GetLocalValue(1), JitOp::CallMethod(1, 0, true)];
        assert!(!receiver_is_this(&ops, 1, 0));
        // A compound arg (AddIBoxed pops 2) → not the simple pattern.
        let ops = [
            JitOp::GetLocalValue(0),
            JitOp::GetLocalValue(1),
            JitOp::GetLocalValue(2),
            JitOp::AddIBoxed,
            JitOp::CallMethod(1, 1, true),
        ];
        assert!(!receiver_is_this(&ops, 4, 1));
    }

    #[test]
    fn splice_value_callee_pushes_result_and_remaps_locals() {
        // Callee: return arg0 + arg1 → locals 1,2. argc=2 → receiver+2 args.
        let caller = [JitOp::CallMethod(0, 2, true), JitOp::ReturnValueBoxed];
        let callee = [
            JitOp::GetLocalValue(1),
            JitOp::GetLocalValue(2),
            JitOp::AddIBoxed,
            JitOp::ReturnValueBoxed,
        ];
        let out = splice(&caller, 0, &callee, 10, 0, 2, ResultMode::Push).expect("inlinable");
        // Arg-setup: SetLocalValue(12), SetLocalValue(11), SetLocalValue(10)
        // (arg1→base+2=12, arg0→base+1=11, receiver→base+0=10), then body remapped +10.
        assert_eq!(
            out,
            vec![
                JitOp::SetLocalValue(12),
                JitOp::SetLocalValue(11),
                JitOp::SetLocalValue(10),
                JitOp::GetLocalValue(11), // local1 remapped
                JitOp::GetLocalValue(12), // local2 remapped
                JitOp::AddIBoxed,
                // value stays on stack (Push), return dropped
                JitOp::ReturnValueBoxed, // caller's own return
            ]
        );
    }
}
