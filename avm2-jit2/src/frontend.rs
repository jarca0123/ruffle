//! Frontend: decide whether a method is in the JIT's supported subset and, if so,
//! compute its CFG, the machine type of every AS3 local, and which locals are
//! parameters.
//!
//! Step 1 supports numeric methods with **branches**: `int`/`Number` arithmetic,
//! comparisons, locals, `Jump`/`IfTrue`/`IfFalse`, and returns. Two structural
//! restrictions keep it sound and simple; anything violating them declines to the
//! interpreter (never a miscompile):
//!
//! 1. **Empty operand stack at every basic-block boundary.** Each block is
//!    simulated from an empty stack and must end empty, so no unboxed operand
//!    value has to cross a CFG edge (only locals do, in their WASM locals). This
//!    covers ordinary loops/conditionals; ternary/short-circuit patterns that
//!    leave a value live across an edge decline. (Cross-block operand SSA is a
//!    later step.)
//! 2. **Local types are pinned by the entry block.** A local's machine type is
//!    fixed by its type as a parameter or its first assignment in block 0 (the
//!    entry, which dominates every other block — so the type is valid at every
//!    use). A local first assigned only in a non-entry block declines. This needs
//!    no dataflow fixpoint or dominator tree.

use std::collections::BTreeSet;

use ruffle_core::avm2::Op;

use crate::ir::{arith_lower, BinArith, ReturnCoerce, Ty};

/// The result of a successful analysis.
pub struct Analysis {
    /// Machine type of each AS3 local (indexed by local id). Meaningful only for
    /// referenced locals; unused slots are filler.
    pub local_ty: Vec<Ty>,
    /// Whether each AS3 local is referenced by any op.
    pub used: Vec<bool>,
    /// AS3 local indices passed to the compiled function as WASM parameters
    /// (a used, numerically-typed method parameter), ascending.
    pub param_locals: Vec<u32>,
    /// How a `ReturnValue` coerces its operand (from the declared return type);
    /// `None` = a type step 1 can't coerce to (a `ReturnValue` then declines, a
    /// `ReturnVoid` is still fine).
    pub ret_coerce: Option<ReturnCoerce>,
    /// Basic-block start op-indices, ascending (block `bi` spans
    /// `[leaders[bi], block_end(bi))`). `leaders[0] == 0`.
    pub leaders: Vec<usize>,
    /// Whether the method has any control flow (⇒ needs the dispatch loop). When
    /// `false` there is exactly one straight-line block ending in a return.
    pub has_control_flow: bool,
    /// Operand-stack types on entry to each block (depth 0 = bottom). Empty for
    /// the common case; a non-empty entry means operand values cross a CFG edge
    /// into this block and are carried in the boundary WASM locals (step 2).
    pub entry_stack: Vec<Vec<Ty>>,
    /// Type of each operand-boundary slot by depth — the type of the global WASM
    /// local that carries an operand value across CFG edges at that depth.
    pub boundary_ty: Vec<Ty>,
    /// Whether the method uses any domainMemory op (`li*`/`si*`) — the emitted
    /// module then needs a linear memory and the `dm_len` parameter (step 3).
    pub uses_dm: bool,
    /// Whether the method uses `GetSlot` — the emitted module then imports the
    /// `get_slot` runtime helper (step 3b).
    pub uses_get_slot: bool,
    /// Whether the method writes a slot (`SetSlotNoCoerce`/`SetSlotCoerceI`) — the
    /// module then imports the `set_slot` helper, which needs the GC `Mutation`.
    pub uses_set_slot: bool,
    /// Whether the method needs the `to_int32` helper (a `Number`→int coercion).
    pub uses_to_int32: bool,
    /// Whether the method makes a JIT→JIT call — imports the call helpers
    /// (`push_call_arg`, `call_method`, `pending_error`) and needs the live
    /// `Activation` + error propagation (step 4).
    pub uses_call: bool,
    num_ops: usize,
}

impl Analysis {
    /// Block index of a leader op-index (a branch target is always a leader).
    pub fn block_of(&self, op: usize) -> usize {
        self.leaders
            .binary_search(&op)
            .expect("branch target must be a block leader")
    }
    /// Exclusive end op-index of block `bi`.
    pub fn block_end(&self, bi: usize) -> usize {
        self.leaders.get(bi + 1).copied().unwrap_or(self.num_ops)
    }
    pub fn num_blocks(&self) -> usize {
        self.leaders.len()
    }
}

/// Whether an op is a basic-block terminator (control leaves it explicitly).
fn is_terminator(op: &Op<'_>) -> bool {
    matches!(
        op,
        Op::Jump { .. }
            | Op::IfTrue { .. }
            | Op::IfFalse { .. }
            | Op::PopJump { .. }
            | Op::ReturnValue { .. }
            | Op::ReturnVoid { .. }
    )
}

/// Branch target op-index, if `op` is a (conditional or unconditional) branch.
fn branch_target(op: &Op<'_>) -> Option<usize> {
    match op {
        Op::Jump { offset } | Op::IfTrue { offset } | Op::IfFalse { offset } | Op::PopJump { offset } => {
            Some(*offset)
        }
        _ => None,
    }
}

/// Compute basic-block leaders (block starts). `None` if a branch target is out
/// of range. Targets are op-indices into `ops` (the verifier resolves them).
fn compute_leaders(ops: &[Op<'_>]) -> Option<Vec<usize>> {
    let len = ops.len();
    let mut leaders = BTreeSet::new();
    leaders.insert(0usize);
    for (i, op) in ops.iter().enumerate() {
        if let Some(t) = branch_target(op) {
            if t >= len {
                return None;
            }
            leaders.insert(t);
        }
        if is_terminator(op) && i + 1 < len {
            leaders.insert(i + 1);
        }
    }
    Some(leaders.into_iter().collect())
}

/// Whether stores in this block pin a local's type (entry block) or must match an
/// already-pinned type (every other block).
#[derive(Copy, Clone, PartialEq, Eq)]
enum StoreMode {
    Pin,
    Strict,
}

pub fn analyze(
    ops: &[Op<'_>],
    num_locals: u32,
    entry_local_tys: &[Option<Ty>],
    ret_coerce: Option<ReturnCoerce>,
) -> Option<Analysis> {
    debug_assert_eq!(entry_local_tys.len(), num_locals as usize);
    if ops.is_empty() {
        return None;
    }

    let leaders = compute_leaders(ops)?;
    let num_blocks = leaders.len();

    // Every block must end by transferring control: a terminator, or a
    // fall-through into the (existing) next block. The last block therefore must
    // end in a terminator (AVM2 methods always do).
    for bi in 0..num_blocks {
        let end = leaders.get(bi + 1).copied().unwrap_or(ops.len());
        let last_is_term = is_terminator(&ops[end - 1]);
        if !last_is_term && bi + 1 >= num_blocks {
            return None; // falls off the end of the method
        }
    }

    let block_of = |op: usize| leaders.binary_search(&op).ok();

    let mut local_ty: Vec<Option<Ty>> = entry_local_tys.to_vec();
    let mut used = vec![false; num_locals as usize];
    let mut uses_to_int32 = false;

    // Cross-block fixpoint over operand-stack types (step 2). Each reachable block
    // is simulated from its entry stack; the exit stack propagates to successors
    // and must be consistent across every CFG edge into a block. Block 0 pins
    // local types; the rest must match. A non-empty boundary means operand values
    // live across the edge — carried in typed WASM locals, never boxed memory.
    let mut entry_stack: Vec<Option<Vec<Ty>>> = vec![None; num_blocks];
    entry_stack[0] = Some(Vec::new());
    let mut worklist = vec![0usize];

    while let Some(bi) = worklist.pop() {
        let start = leaders[bi];
        let end = leaders.get(bi + 1).copied().unwrap_or(ops.len());
        let mode = if bi == 0 { StoreMode::Pin } else { StoreMode::Strict };
        let mut stack = entry_stack[bi].clone().expect("worklist ⇒ entry known");
        for op in &ops[start..end] {
            step(
                op,
                &mut stack,
                &mut local_ty,
                &mut used,
                ret_coerce,
                mode,
                &mut uses_to_int32,
            )?;
        }
        // `stack` is the exit stack (terminators pop their condition/value in
        // `step`); propagate it to each successor.
        let succs: Vec<usize> = match &ops[end - 1] {
            Op::Jump { offset } | Op::PopJump { offset } => vec![block_of(*offset)?],
            Op::IfTrue { offset } | Op::IfFalse { offset } => vec![block_of(*offset)?, bi + 1],
            Op::ReturnValue { .. } | Op::ReturnVoid { .. } => vec![],
            _ => vec![bi + 1], // fall-through
        };
        for succ in succs {
            if succ >= num_blocks {
                return None;
            }
            match &entry_stack[succ] {
                None => {
                    entry_stack[succ] = Some(stack.clone());
                    worklist.push(succ);
                }
                Some(existing) if *existing != stack => return None, // inconsistent edge
                Some(_) => {}
            }
        }
    }

    // One global WASM local per operand-boundary depth; a depth that is different
    // types at different boundaries can't share a local ⇒ decline (boxing such a
    // slot is a later step). All reachable blocks have `Some` entry stacks.
    let max_depth = entry_stack.iter().flatten().map(Vec::len).max().unwrap_or(0);
    let mut boundary_ty: Vec<Option<Ty>> = vec![None; max_depth];
    for es in entry_stack.iter().flatten() {
        for (d, &t) in es.iter().enumerate() {
            match boundary_ty[d] {
                None => boundary_ty[d] = Some(t),
                Some(u) if u != t => return None,
                Some(_) => {}
            }
        }
    }
    let boundary_ty: Vec<Ty> = boundary_ty.into_iter().map(|t| t.unwrap_or(Ty::Boxed)).collect();
    let entry_stack: Vec<Vec<Ty>> = entry_stack.into_iter().map(Option::unwrap_or_default).collect();

    // Unpinned locals (never a param, never stored) are Opaque — only `this`-style
    // scope-prologue reads reach them.
    let local_ty: Vec<Ty> = local_ty.into_iter().map(|t| t.unwrap_or(Ty::Opaque)).collect();
    let param_locals: Vec<u32> = (0..num_locals)
        .filter(|&i| used[i as usize] && entry_local_tys[i as usize].is_some())
        .collect();
    let has_control_flow = ops.iter().any(|op| branch_target(op).is_some());
    let uses_dm = ops.iter().any(|op| {
        matches!(
            op,
            Op::Li8
                | Op::Li16
                | Op::Li32
                | Op::Si8
                | Op::Si16
                | Op::Si32
                | Op::Lf32
                | Op::Lf64
                | Op::Sf32
                | Op::Sf64
        )
    });
    let uses_get_slot = ops.iter().any(|op| matches!(op, Op::GetSlot { .. }));
    let uses_set_slot = ops
        .iter()
        .any(|op| matches!(op, Op::SetSlotNoCoerce { .. } | Op::SetSlotCoerceI { .. }));
    let uses_call = ops.iter().any(|op| matches!(op, Op::CallMethod { .. }));

    Some(Analysis {
        local_ty,
        used,
        param_locals,
        ret_coerce,
        leaders,
        has_control_flow,
        entry_stack,
        boundary_ty,
        uses_dm,
        uses_get_slot,
        uses_set_slot,
        uses_to_int32,
        uses_call,
        num_ops: ops.len(),
    })
}

/// Advance the abstract state by one op, or `None` to decline the method.
fn step(
    op: &Op<'_>,
    stack: &mut Vec<Ty>,
    local_ty: &mut [Option<Ty>],
    used: &mut [bool],
    ret_coerce: Option<ReturnCoerce>,
    mode: StoreMode,
    to_int32: &mut bool,
) -> Option<()> {
    // Store into a local: pin (entry block) or require a matching pin.
    fn store_local(local_ty: &mut [Option<Ty>], i: u32, t: Ty, mode: StoreMode) -> Option<()> {
        let slot = local_ty.get_mut(i as usize)?;
        match (mode, *slot) {
            (_, Some(u)) if u == t => Some(()),
            (StoreMode::Pin, None) => {
                *slot = Some(t);
                Some(())
            }
            // Strict: unpinned or mismatched ⇒ decline.
            _ => None,
        }
    }

    match op {
        // ---- constant pushes ----
        Op::PushInt { .. } => stack.push(Ty::Int),
        Op::PushDouble { .. } => stack.push(Ty::Num),
        Op::PushTrue | Op::PushFalse => stack.push(Ty::Bool),

        // ---- locals ----
        Op::GetLocal { index } => {
            *used.get_mut(*index as usize)? = true;
            // An untyped local (`this`, or an object/`*` param) reads as Opaque —
            // usable only for the scope prologue.
            let ty = local_ty.get(*index as usize)?.unwrap_or(Ty::Opaque);
            stack.push(ty);
        }
        Op::SetLocal { index } => {
            *used.get_mut(*index as usize)? = true;
            let t = stack.pop()?;
            if t == Ty::Opaque {
                return None;
            }
            store_local(local_ty, *index, t, mode)?;
        }
        Op::StoreLocal { index } => {
            *used.get_mut(*index as usize)? = true;
            let t = *stack.last()?;
            if t == Ty::Opaque {
                return None;
            }
            store_local(local_ty, *index, t, mode)?;
        }
        Op::IncLocalI { index } | Op::DecLocalI { index } => {
            *used.get_mut(*index as usize)? = true;
            store_local(local_ty, *index, Ty::Int, mode)?;
            if local_ty[*index as usize] != Some(Ty::Int) {
                return None;
            }
        }

        // ---- generic arithmetic ----
        Op::Add { .. } => bin_arith(stack, BinArith::Add)?,
        Op::Subtract { .. } => bin_arith(stack, BinArith::Sub)?,
        Op::Multiply => bin_arith(stack, BinArith::Mul)?,
        Op::Divide => bin_arith(stack, BinArith::Div)?,
        Op::Negate | Op::Increment | Op::Decrement => {
            let a = stack.pop()?;
            if !matches!(a, Ty::Int | Ty::Num | Ty::NumRaw) {
                return None;
            }
            stack.push(Ty::Num);
        }

        // ---- integer (`*I`) arithmetic ----
        Op::AddI | Op::SubtractI | Op::MultiplyI => {
            let b = stack.pop()?;
            let a = stack.pop()?;
            if a != Ty::Int || b != Ty::Int {
                return None;
            }
            stack.push(Ty::Int);
        }
        Op::NegateI | Op::IncrementI | Op::DecrementI => {
            let a = stack.pop()?;
            if a != Ty::Int {
                return None;
            }
            stack.push(Ty::Int);
        }

        // ---- JIT→JIT method call (§step 4). Stack is [receiver, arg0..argN];
        // pop the args and receiver, push the result if kept. The call may throw —
        // handled by the emitted error check + `try_run` propagation. ----
        Op::CallMethod {
            num_args,
            push_return_value,
            ..
        } => {
            for _ in 0..*num_args {
                // Args are boxed for the call; a raw float would canonicalize, and
                // Opaque isn't a value.
                if matches!(stack.pop()?, Ty::Opaque | Ty::NumRaw) {
                    return None;
                }
            }
            if stack.pop()? != Ty::Boxed {
                return None; // receiver must be an object
            }
            if *push_return_value {
                stack.push(Ty::Boxed);
            }
        }

        // ---- comparisons (produce Bool). StrictEquals declined: strict_eq is a
        // bit-compare, so Integer(5) !== Number(5.0). ----
        Op::LessThan | Op::LessEquals | Op::GreaterThan | Op::GreaterEquals | Op::Equals => {
            let numeric = |t: Ty| matches!(t, Ty::Int | Ty::Num | Ty::NumRaw);
            let b = stack.pop()?;
            let a = stack.pop()?;
            if !numeric(a) || !numeric(b) {
                return None;
            }
            stack.push(Ty::Bool);
        }

        // ---- domainMemory (FlasCC hot path): integer loads/stores against the
        // linear memory. Loads pop an int address, push an int; stores pop the
        // address then the value (both int). §step 3. ----
        Op::Li8 | Op::Li16 | Op::Li32 => {
            if stack.pop()? != Ty::Int {
                return None;
            }
            stack.push(Ty::Int);
        }
        Op::Si8 | Op::Si16 | Op::Si32 => {
            let addr = stack.pop()?;
            let val = stack.pop()?;
            if addr != Ty::Int || val != Ty::Int {
                return None;
            }
        }
        // Float loads produce a raw (lossless) Number; float stores coerce any
        // numeric value to f64 and write its bytes.
        Op::Lf32 | Op::Lf64 => {
            if stack.pop()? != Ty::Int {
                return None;
            }
            stack.push(Ty::NumRaw);
        }
        Op::Sf32 | Op::Sf64 => {
            let addr = stack.pop()?;
            let val = stack.pop()?;
            if addr != Ty::Int || !matches!(val, Ty::Int | Ty::Num | Ty::NumRaw) {
                return None;
            }
        }

        // ---- coercions ----
        Op::CoerceA => {}
        Op::CoerceD => {
            let a = stack.pop()?;
            if !matches!(a, Ty::Int | Ty::Num | Ty::Bool) {
                return None;
            }
            stack.push(Ty::Num);
        }
        Op::CoerceI => {
            let a = stack.pop()?;
            match a {
                Ty::Int | Ty::Bool => {} // already the right i32
                Ty::Num => *to_int32 = true, // ToInt32 via helper
                _ => return None,
            }
            stack.push(Ty::Int);
        }

        // ---- scope prologue: unobservable to the numeric subset (no accepted op
        // reads the scope), so push/pop scope are no-ops. `pushscope` consumes the
        // Opaque `this`. ----
        Op::PushScope => {
            // `this` (Boxed object) or an Opaque untyped local — either is fine to
            // push as scope; the numeric subset never reads the scope back.
            if !matches!(stack.pop()?, Ty::Opaque | Ty::Boxed) {
                return None;
            }
        }
        Op::PopScope => {}

        // ---- early-bound slot read (§step 3b): the verifier resolves a typed
        // `getproperty` to `GetSlot { index }` on a proven object; we call the
        // pure `get_slot` helper. Result is a Boxed value. ----
        Op::GetSlot { .. } => {
            if stack.pop()? != Ty::Boxed {
                return None;
            }
            stack.push(Ty::Boxed);
        }
        // Slot write (verifier-resolved from a typed `setproperty`): stack is
        // [object, value]. NoCoerce stores the value's natural boxed form;
        // CoerceI coerces to int first (so the value must already be int/bool).
        Op::SetSlotNoCoerce { .. } => {
            let value = stack.pop()?;
            let object = stack.pop()?;
            // NumRaw would need a lossless box (deferred); Opaque isn't a value.
            if object != Ty::Boxed || matches!(value, Ty::Opaque | Ty::NumRaw) {
                return None;
            }
        }
        Op::SetSlotCoerceI { .. } => {
            let value = stack.pop()?;
            let object = stack.pop()?;
            if object != Ty::Boxed {
                return None;
            }
            match value {
                Ty::Int | Ty::Bool => {}
                // ToInt32 launders a raw Number (NaN → 0), so it's safe here.
                Ty::Num | Ty::NumRaw => *to_int32 = true,
                _ => return None,
            }
        }

        // ---- stack shuffles ----
        Op::Dup => {
            let t = *stack.last()?;
            if t == Ty::Opaque {
                return None;
            }
            stack.push(t);
        }
        Op::Pop => {
            stack.pop()?;
        }
        Op::Swap => {
            let len = stack.len();
            if len < 2 || stack[len - 1] == Ty::Opaque || stack[len - 2] == Ty::Opaque {
                return None;
            }
            stack.swap(len - 1, len - 2);
        }
        Op::Nop => {}

        // ---- control flow (terminators) ----
        Op::Jump { .. } => {}
        Op::PopJump { .. } => {
            stack.pop()?;
        }
        Op::IfTrue { .. } | Op::IfFalse { .. } => {
            // coerce_to_boolean: Bool/Int are already the WASM condition
            // (non-zero = true); a Num converts inline via `(f==f) & (f!=0)`.
            let c = stack.pop()?;
            if !matches!(c, Ty::Bool | Ty::Int | Ty::Num | Ty::NumRaw) {
                return None;
            }
        }
        Op::ReturnValue { .. } => {
            let a = stack.pop()?;
            // A raw-loaded Number can't be boxed here (would canonicalize a
            // colliding NaN); it must reach a store/arithmetic instead.
            if a == Ty::NumRaw {
                return None;
            }
            let ok = match ret_coerce? {
                ReturnCoerce::Any => a != Ty::Opaque,
                ReturnCoerce::Number => matches!(a, Ty::Int | Ty::Num | Ty::Bool),
                ReturnCoerce::Int => {
                    if a == Ty::Num {
                        *to_int32 = true; // ToInt32 on the returned Number
                    }
                    matches!(a, Ty::Int | Ty::Bool | Ty::Num)
                }
                ReturnCoerce::Boolean => a == Ty::Bool,
            };
            if !ok {
                return None;
            }
        }
        Op::ReturnVoid { .. } => {}

        // Everything else (LookupSwitch, StrictEquals, property/method access,
        // string/object ops, scope ops, ...) is outside the step-1 subset.
        _ => return None,
    }
    Some(())
}

/// Type transition for a generic binary arithmetic op, shared with the emitter
/// via [`arith_lower`].
fn bin_arith(stack: &mut Vec<Ty>, op: BinArith) -> Option<()> {
    let b = stack.pop()?;
    let a = stack.pop()?;
    let (_lower, result) = arith_lower(op, a, b)?;
    stack.push(result);
    Some(())
}
