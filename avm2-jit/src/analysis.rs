//! "Everything is `int`" soundness check for the raw-`i32` JIT model.
//!
//! The lowering treats every operand as a 32-bit int (it `i32.wrap`s each slot
//! it loads and boxes results as `int`). That matches AVM2 semantics *only* when
//! every value it consumes as an int really is an int. This pass proves that by
//! abstract interpretation before we agree to compile a method; if it can't, the
//! method falls back to the interpreter.
//!
//! It tracks a single bit per local — "provably int" — across the CFG to a
//! fixpoint (a value is int only if int on *all* paths that reach a use). The
//! operand stack needs no cross-block tracking: [`crate::lower`] already requires
//! the stack to be empty at every basic-block boundary, so each block is
//! analysed with a fresh empty stack. Only ops that *use* a value as an int
//! (arithmetic, comparisons, conditional branches, `inclocali`, `returnvalue`)
//! impose a requirement; loading a non-int and then dropping it (the
//! `getlocal0; pushscope` prologue) is fine.

use crate::lower::{basic_block_leaders, JitOp};

/// Returns `true` if the raw-`i32` lowering of `ops` is faithful to AVM2 given
/// `entry_locals_int` (per-local "is int" at method entry: `this`/object params
/// `false`, `int`/`uint` params `true`, fresh locals `false`). Conservative:
/// anything not provably int is treated as non-int.
pub fn int_sound(ops: &[JitOp], entry_locals_int: &[bool]) -> bool {
    let Some(leaders) = basic_block_leaders(ops) else {
        return false;
    };
    let n = leaders.len();
    let bb_of = |op_idx: usize| leaders.iter().position(|&l| l == op_idx);

    // Per-block entry state; `None` = not yet reached.
    let mut entry: Vec<Option<Vec<bool>>> = vec![None; n];
    entry[0] = Some(entry_locals_int.to_vec());
    let mut work = vec![0usize];

    while let Some(bb) = work.pop() {
        let locals_in = entry[bb].clone().expect("queued blocks are initialised");
        let start = leaders[bb];
        let end = leaders.get(bb + 1).copied().unwrap_or(ops.len());

        let Some(locals_out) = simulate_block(&ops[start..end], locals_in) else {
            return false; // a use consumed a non-int: unsound
        };

        // Propagate to successors, meeting (AND) with any existing state.
        for succ in successors(ops, &leaders, bb, bb_of) {
            let merged = match &entry[succ] {
                None => locals_out.clone(),
                Some(existing) => existing
                    .iter()
                    .zip(&locals_out)
                    .map(|(&a, &b)| a && b)
                    .collect(),
            };
            if entry[succ].as_deref() != Some(merged.as_slice()) {
                entry[succ] = Some(merged);
                work.push(succ);
            }
        }
    }
    true
}

/// Returns `true` if the unboxed-`f64` double-path lowering of `ops` is faithful
/// to AVM2 given `entry_locals_double` (per-local "is Number" at entry). Every
/// value consumed as an `f64` (arithmetic, `setlocaldouble`, `returndouble`) must
/// be a `Number`, else the `f64.reinterpret` would read a boxed value's bits as
/// garbage (or `add` would be string concat, not `f64.add`). Straight-line only —
/// the double path emits no branches yet — so a single linear pass suffices.
pub fn double_sound(ops: &[JitOp], entry_locals_double: &[bool]) -> bool {
    let mut locals = entry_locals_double.to_vec();
    let mut stack: Vec<bool> = Vec::new();
    for &op in ops {
        match op {
            JitOp::GetLocalDouble(i) => match locals.get(i as usize) {
                Some(&d) => stack.push(d),
                None => return false,
            },
            JitOp::PushDouble(_) => stack.push(true),
            JitOp::SetLocalDouble(i) => {
                let Some(v) = stack.pop() else { return false };
                if !v {
                    return false; // storing a non-Number as a double local
                }
                match locals.get_mut(i as usize) {
                    Some(l) => *l = true,
                    None => return false,
                }
            }
            JitOp::StoreLocalDouble(i) => {
                // Peek-and-store: top must be a Number, and it stays on the stack.
                match stack.last() {
                    Some(true) => {}
                    _ => return false,
                }
                match locals.get_mut(i as usize) {
                    Some(l) => *l = true,
                    None => return false,
                }
            }
            JitOp::AddD | JitOp::SubtractD | JitOp::MultiplyD | JitOp::DivideD => {
                let (Some(b), Some(a)) = (stack.pop(), stack.pop()) else {
                    return false;
                };
                if !(a && b) {
                    return false;
                }
                stack.push(true);
            }
            JitOp::IncrementD | JitOp::DecrementD | JitOp::NegateD => {
                let Some(a) = stack.pop() else { return false };
                if !a {
                    return false;
                }
                stack.push(true);
            }
            JitOp::Pop => {
                if stack.pop().is_none() {
                    return false;
                }
            }
            JitOp::Nop => {}
            JitOp::ReturnDouble => {
                let Some(v) = stack.pop() else { return false };
                if !v {
                    return false; // returned value must be a Number (sound to box)
                }
            }
            // Any int/boxed/branch op means this isn't a pure straight-line
            // double method.
            _ => return false,
        }
    }
    true
}

/// Simulates one basic block over a fresh empty stack, returning the outgoing
/// local int-ness, or `None` if a use consumes a non-int (or the stack under- or
/// over-flows relative to what the block should look like).
fn simulate_block(block: &[JitOp], mut locals: Vec<bool>) -> Option<Vec<bool>> {
    let mut stack: Vec<bool> = Vec::new();

    for &op in block {
        match op {
            JitOp::GetLocal(i) => stack.push(*locals.get(i as usize)?),
            JitOp::SetLocal(i) => {
                let v = stack.pop()?;
                *locals.get_mut(i as usize)? = v;
            }
            JitOp::PushInt(_) => stack.push(true),
            // Int-consuming, int-producing binary ops.
            JitOp::AddI
            | JitOp::SubtractI
            | JitOp::MultiplyI
            | JitOp::LessThan
            | JitOp::LessEquals
            | JitOp::GreaterThan
            | JitOp::GreaterEquals
            | JitOp::Equals => {
                let b = stack.pop()?;
                let a = stack.pop()?;
                if !(a && b) {
                    return None;
                }
                stack.push(true);
            }
            JitOp::IncrementI | JitOp::DecrementI => {
                let a = stack.pop()?;
                if !a {
                    return None;
                }
                stack.push(true);
            }
            JitOp::IncLocalI(i) | JitOp::DecLocalI(i) => {
                if !*locals.get(i as usize)? {
                    return None;
                }
                *locals.get_mut(i as usize)? = true;
            }
            JitOp::Pop => {
                stack.pop()?; // value discarded — no int requirement
            }
            JitOp::Dup => {
                let v = *stack.last()?;
                stack.push(v);
            }
            JitOp::Nop => {}
            JitOp::IfTrue(_) | JitOp::IfFalse(_) => {
                let c = stack.pop()?;
                if !c {
                    return None; // condition must be a genuine int (from a compare)
                }
            }
            JitOp::IfLt(_) | JitOp::IfGe(_) => {
                let b = stack.pop()?;
                let a = stack.pop()?;
                if !(a && b) {
                    return None;
                }
            }
            JitOp::Jump(_) => {}
            JitOp::ReturnValue => {
                let v = stack.pop()?;
                if !v {
                    return None; // we box the result as int
                }
            }
            // Boxed-`Value` and double-path ops aren't part of the int model; an
            // int-path method never contains them, so treat as unsound here.
            JitOp::GetLocalValue(_)
            | JitOp::SetLocalValue(_)
            | JitOp::CallHelper(_)
            | JitOp::CallHelper2(_)
            | JitOp::CmpNum(_)
            | JitOp::BitOpInt(_)
            | JitOp::ArithInt(_)
            | JitOp::ArithNum(_)
            | JitOp::CoerceInt(_)
            | JitOp::CoerceBool
            | JitOp::SwapValue
            | JitOp::ReturnValueBoxed
            | JitOp::ReturnValueCoerced
            | JitOp::ReturnVoidBoxed(_)
            | JitOp::DupValue
            | JitOp::StoreLocalValue(_)
            | JitOp::AddIBoxed
            | JitOp::SubtractIBoxed
            | JitOp::MultiplyIBoxed
            | JitOp::IncrementIBoxed
            | JitOp::DecrementIBoxed
            | JitOp::GetProperty(_)
            | JitOp::GetPropertyIc(_, _)
            | JitOp::GetPropertyFast(_, _)
            | JitOp::GetSlot(_, _)
            | JitOp::FindProp(_, _)
            | JitOp::PushIntValue(_)
            | JitOp::PushConst(_)
            | JitOp::CallHelper3(_, _)
            | JitOp::IfTrueBoxed(_)
            | JitOp::IfFalseBoxed(_)
            | JitOp::PushScopeReal
            | JitOp::GetScopeObject(_)
            | JitOp::GetOuterScope(_)
            | JitOp::IncDecLocalIValue(_, _)
            | JitOp::IncDecLocalNum(_, _)
            | JitOp::HasNext2(_, _)
            | JitOp::CoerceString
            | JitOp::GetScriptGlobals(_)
            | JitOp::GetLocalDouble(_)
            | JitOp::SetLocalDouble(_)
            | JitOp::StoreLocalDouble(_)
            | JitOp::PushDouble(_)
            | JitOp::AddD
            | JitOp::SubtractD
            | JitOp::MultiplyD
            | JitOp::DivideD
            | JitOp::IncrementD
            | JitOp::DecrementD
            | JitOp::NegateD
            | JitOp::ReturnDouble
            | JitOp::CallMethod(_, _, _)
            | JitOp::CallMethodDirect(_, _, _, _)
            | JitOp::CallProperty(_, _, _)
            | JitOp::ConstructSuper(_)
            | JitOp::CallValue(_)
            | JitOp::DmLoad(_)
            | JitOp::DmStore(_)
            | JitOp::DmLoadF(_)
            | JitOp::DmStoreF(_)
            // These are boxed-path only; never in the int analysis.
            | JitOp::LookupSwitch(_)
            | JitOp::PushString(_)
            | JitOp::Throw
            | JitOp::NewCatch(_)
            | JitOp::Coerce(_)
            | JitOp::VCall(..)
            | JitOp::PopScopeReal => return None,
        }
    }
    Some(locals)
}

/// The basic blocks that control can flow to from block `bb`.
fn successors(
    ops: &[JitOp],
    leaders: &[usize],
    bb: usize,
    bb_of: impl Fn(usize) -> Option<usize>,
) -> Vec<usize> {
    let end = leaders.get(bb + 1).copied().unwrap_or(ops.len());
    let last = ops[end - 1];
    let next = (bb + 1 < leaders.len()).then_some(bb + 1);
    match last {
        JitOp::ReturnValue => vec![],
        JitOp::Jump(t) => bb_of(t).into_iter().collect(),
        JitOp::IfTrue(t) | JitOp::IfFalse(t) | JitOp::IfLt(t) | JitOp::IfGe(t) => {
            bb_of(t).into_iter().chain(next).collect()
        }
        // Fell off the end without a terminator: straight-line to the next block.
        _ => next.into_iter().collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // sum(n:int): pure-int loop reading an int param — sound.
    fn sum_ops() -> Vec<JitOp> {
        vec![
            JitOp::GetLocal(0),  // 0: this
            JitOp::Pop,          // 1: pushscope
            JitOp::PushInt(0),   // 2
            JitOp::SetLocal(2),  // 3: s=0
            JitOp::PushInt(0),   // 4
            JitOp::SetLocal(3),  // 5: i=0
            JitOp::Jump(12),     // 6
            JitOp::GetLocal(2),  // 7
            JitOp::GetLocal(3),  // 8
            JitOp::AddI,         // 9
            JitOp::SetLocal(2),  // 10: s+=i
            JitOp::IncLocalI(3), // 11: i++
            JitOp::GetLocal(3),  // 12
            JitOp::GetLocal(1),  // 13: n
            JitOp::LessThan,     // 14
            JitOp::IfTrue(7),    // 15
            JitOp::GetLocal(2),  // 16
            JitOp::ReturnValue,  // 17
        ]
    }

    #[test]
    fn sound_when_param_is_int() {
        // locals: 0=this(false), 1=n(int:true), 2=s, 3=i.
        let seed = [false, true, false, false];
        assert!(int_sound(&sum_ops(), &seed));
    }

    #[test]
    fn unsound_when_param_is_not_int() {
        // n is a Number/Object (false): `getlocal1; lessthan` uses a non-int.
        let seed = [false, false, false, false];
        assert!(!int_sound(&sum_ops(), &seed));
    }

    #[test]
    fn dropping_a_non_int_is_fine() {
        // getlocal0 (this, non-int); pushscope (pop); pushint 1; returnvalue.
        let ops = [JitOp::GetLocal(0), JitOp::Pop, JitOp::PushInt(1), JitOp::ReturnValue];
        assert!(int_sound(&ops, &[false]));
    }

    #[test]
    fn returning_a_non_int_local_is_unsound() {
        let ops = [JitOp::GetLocal(0), JitOp::ReturnValue];
        assert!(!int_sound(&ops, &[false]));
        assert!(int_sound(&ops, &[true]));
    }
}
