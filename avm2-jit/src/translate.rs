//! Core `Op` → [`JitOp`] translation.
//!
//! Maps the supported subset of `ruffle_core`'s AVM2 `Op` enum onto the JIT's
//! self-contained [`JitOp`] IR, and bails (`None`) on anything outside that
//! subset so the backend can fall back to the interpreter.
//!
//! A key simplification: core `Op` branch offsets are **op indices** (the
//! verifier resolves byte offsets to indices; the interpreter does `ip =
//! offset`). [`JitOp`] branch targets are also op indices, so branches map
//! straight across with no offset arithmetic.
//!
//! ## Soundness note
//! This translation is *structural* — it assumes every operand it touches is an
//! `int`-typed [`Value`](ruffle_core::avm2::Value) (the raw-`i32` execution
//! model in [`crate::lower`]). It carries no type check, so the backend must
//! only enable it for methods whose relevant locals/stack are provably `int`.
//! The differential tests here pin the *semantics* of the supported ops; gating
//! by verifier type info is future work.

use ruffle_core::avm2::Op;

use crate::lower::JitOp;

/// Translates a verified core op slice to [`JitOp`], or returns `None` if it
/// contains any op outside the supported numeric+control-flow subset.
pub fn translate(ops: &[Op]) -> Option<Vec<JitOp>> {
    let mut out = Vec::with_capacity(ops.len());
    for op in ops {
        out.push(match *op {
            Op::GetLocal { index } => JitOp::GetLocal(index),
            Op::SetLocal { index } => JitOp::SetLocal(index),
            Op::PushInt { value } => JitOp::PushInt(value),
            Op::AddI => JitOp::AddI,
            Op::SubtractI => JitOp::SubtractI,
            Op::MultiplyI => JitOp::MultiplyI,
            Op::IncrementI => JitOp::IncrementI,
            Op::DecrementI => JitOp::DecrementI,
            Op::IncLocalI { index } => JitOp::IncLocalI(index),
            Op::DecLocalI { index } => JitOp::DecLocalI(index),
            Op::LessThan => JitOp::LessThan,
            Op::LessEquals => JitOp::LessEquals,
            Op::GreaterThan => JitOp::GreaterThan,
            Op::GreaterEquals => JitOp::GreaterEquals,
            Op::Equals => JitOp::Equals,
            Op::Jump { offset } => JitOp::Jump(offset),
            Op::IfFalse { offset } => JitOp::IfFalse(offset),
            Op::IfTrue { offset } => JitOp::IfTrue(offset),
            Op::ReturnValue { .. } => JitOp::ReturnValue,
            // Structurally-neutral ops (in the int model): keep them 1:1 so that
            // branch targets (op indices) stay aligned.
            Op::Pop | Op::PushScope => JitOp::Pop,
            Op::Dup => JitOp::Dup,
            Op::Nop | Op::CoerceI | Op::Kill { .. } => JitOp::Nop,
            _ => return None,
        });
    }
    Some(out)
}

/// A tiny reference interpreter over [`JitOp`], mirroring the raw-`i32` model the
/// WASM lowering compiles to. It is the "interpreter" side of the differential
/// tests: `compile(ops)` executed under a WASM runtime must agree with this for
/// every input. Locals are `Value` bit patterns; the operand stack is raw `i32`.
#[cfg(test)]
pub(crate) fn reference_run(ops: &[JitOp], regs: &[u64]) -> u64 {
    // MUST match `crate::lower::VALUE_INT_MARK`.
    const INT_MARK: u64 = 0xFFFB_0000_0000_0000;
    let box_int = |v: i32| INT_MARK | (v as u32 as u64);

    let mut locals: Vec<u64> = regs.to_vec();
    let mut stack: Vec<i32> = Vec::new();
    let mut pc: usize = 0;
    loop {
        match ops[pc] {
            JitOp::GetLocal(i) => {
                // Mirrors I64Load + I32WrapI64: low 32 bits of the slot.
                stack.push(locals[i as usize] as u32 as i32);
                pc += 1;
            }
            JitOp::SetLocal(i) => {
                let v = stack.pop().unwrap();
                locals[i as usize] = box_int(v);
                pc += 1;
            }
            JitOp::PushInt(v) => {
                stack.push(v);
                pc += 1;
            }
            JitOp::AddI => {
                let b = stack.pop().unwrap();
                let a = stack.pop().unwrap();
                stack.push(a.wrapping_add(b));
                pc += 1;
            }
            JitOp::SubtractI => {
                let b = stack.pop().unwrap();
                let a = stack.pop().unwrap();
                stack.push(a.wrapping_sub(b));
                pc += 1;
            }
            JitOp::MultiplyI => {
                let b = stack.pop().unwrap();
                let a = stack.pop().unwrap();
                stack.push(a.wrapping_mul(b));
                pc += 1;
            }
            JitOp::IncrementI => {
                let a = stack.pop().unwrap();
                stack.push(a.wrapping_add(1));
                pc += 1;
            }
            JitOp::DecrementI => {
                let a = stack.pop().unwrap();
                stack.push(a.wrapping_sub(1));
                pc += 1;
            }
            JitOp::IncLocalI(i) => {
                let v = (locals[i as usize] as u32 as i32).wrapping_add(1);
                locals[i as usize] = box_int(v);
                pc += 1;
            }
            JitOp::DecLocalI(i) => {
                let v = (locals[i as usize] as u32 as i32).wrapping_sub(1);
                locals[i as usize] = box_int(v);
                pc += 1;
            }
            JitOp::LessThan
            | JitOp::LessEquals
            | JitOp::GreaterThan
            | JitOp::GreaterEquals
            | JitOp::Equals => {
                let b = stack.pop().unwrap();
                let a = stack.pop().unwrap();
                let r = match ops[pc] {
                    JitOp::LessThan => a < b,
                    JitOp::LessEquals => a <= b,
                    JitOp::GreaterThan => a > b,
                    JitOp::GreaterEquals => a >= b,
                    _ => a == b,
                };
                stack.push(r as i32);
                pc += 1;
            }
            JitOp::Pop => {
                stack.pop().unwrap();
                pc += 1;
            }
            JitOp::Dup => {
                let v = *stack.last().unwrap();
                stack.push(v);
                pc += 1;
            }
            JitOp::Nop => pc += 1,
            JitOp::Jump(t) => pc = t,
            JitOp::IfLt(t) => {
                let b = stack.pop().unwrap();
                let a = stack.pop().unwrap();
                pc = if a < b { t } else { pc + 1 };
            }
            JitOp::IfGe(t) => {
                let b = stack.pop().unwrap();
                let a = stack.pop().unwrap();
                pc = if a >= b { t } else { pc + 1 };
            }
            JitOp::IfFalse(t) => {
                let c = stack.pop().unwrap();
                pc = if c == 0 { t } else { pc + 1 };
            }
            JitOp::IfTrue(t) => {
                let c = stack.pop().unwrap();
                pc = if c != 0 { t } else { pc + 1 };
            }
            JitOp::ReturnValue => return box_int(stack.pop().unwrap()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn translates_numeric_subset() {
        let ops = [
            Op::GetLocal { index: 1 },
            Op::GetLocal { index: 2 },
            Op::AddI,
            Op::ReturnValue { return_type: None },
        ];
        assert_eq!(
            translate(&ops),
            Some(vec![
                JitOp::GetLocal(1),
                JitOp::GetLocal(2),
                JitOp::AddI,
                JitOp::ReturnValue,
            ])
        );
    }

    #[test]
    fn branch_offsets_pass_through_as_indices() {
        let ops = [
            Op::GetLocal { index: 0 },
            Op::IfFalse { offset: 4 },
            Op::PushInt { value: 1 },
            Op::Jump { offset: 5 },
            Op::PushInt { value: 0 },
            Op::ReturnValue { return_type: None },
        ];
        let out = translate(&ops).expect("supported");
        assert_eq!(out[1], JitOp::IfFalse(4));
        assert_eq!(out[3], JitOp::Jump(5));
    }

    #[test]
    fn bails_on_unsupported_op() {
        // `GetScopeObject` (scope access) is outside the supported subset.
        assert_eq!(translate(&[Op::GetScopeObject { index: 0 }]), None);
    }

    #[test]
    fn translates_compare_and_prologue() {
        let ops = [
            Op::GetLocal { index: 0 },
            Op::PushScope,
            Op::GetLocal { index: 1 },
            Op::GetLocal { index: 2 },
            Op::LessThan,
            Op::IfTrue { offset: 6 },
            Op::ReturnValue { return_type: None },
        ];
        let out = translate(&ops).expect("supported");
        assert_eq!(out[1], JitOp::Pop); // pushscope
        assert_eq!(out[4], JitOp::LessThan);
        assert_eq!(out[5], JitOp::IfTrue(6));
    }
}

/// Differential tests: the WASM lowering (run under wasmi) must agree with the
/// reference interpreter for every input. Native-only (wasmi).
#[cfg(all(test, not(target_arch = "wasm32")))]
mod diff_tests {
    use super::*;
    use crate::lower::compile;
    use crate::runner::run;

    fn int_bits(n: i32) -> u64 {
        0xFFFB_0000_0000_0000 | (n as u32 as u64)
    }

    /// Runs `ops` both ways and asserts equality.
    fn assert_equiv(ops: &[JitOp], regs: &[u64]) {
        let bytes = compile(ops).expect("compiles");
        let jit = run(&bytes, regs).expect("native runner");
        let interp = reference_run(ops, regs);
        assert_eq!(jit, interp, "JIT != interpreter for regs {regs:?}");
    }

    #[test]
    fn straight_line_matches_interpreter() {
        // (l1 + l2) * l3
        let ops = [
            JitOp::GetLocal(1),
            JitOp::GetLocal(2),
            JitOp::AddI,
            JitOp::GetLocal(3),
            JitOp::MultiplyI,
            JitOp::ReturnValue,
        ];
        for &(a, b, c) in &[(3, 4, 5), (-2, 7, 3), (100, -50, -1), (0, 0, 9)] {
            assert_equiv(&ops, &[int_bits(0), int_bits(a), int_bits(b), int_bits(c)]);
        }
    }

    #[test]
    fn counted_loop_matches_interpreter() {
        // sum(n): s=0; i=0; while (i < n) { s += i; i += 1 } return s
        let ops = [
            JitOp::PushInt(0),
            JitOp::SetLocal(2),
            JitOp::PushInt(0),
            JitOp::SetLocal(3),
            JitOp::GetLocal(3),
            JitOp::GetLocal(1),
            JitOp::IfGe(16),
            JitOp::GetLocal(2),
            JitOp::GetLocal(3),
            JitOp::AddI,
            JitOp::SetLocal(2),
            JitOp::GetLocal(3),
            JitOp::PushInt(1),
            JitOp::AddI,
            JitOp::SetLocal(3),
            JitOp::Jump(4),
            JitOp::GetLocal(2),
            JitOp::ReturnValue,
        ];
        for n in [0, 1, 2, 5, 10, 100] {
            let regs = [int_bits(0), int_bits(n), int_bits(0), int_bits(0)];
            assert_equiv(&ops, &regs);
        }
    }

    #[test]
    fn real_shaped_loop_through_translate_matches_interpreter() {
        // Mirrors what the verifier emits for a real `sum(n:int):int`: a
        // `getlocal0/pushscope` prologue, a jump-to-condition `for` loop, an
        // `inclocali` step, and a `lessthan`/`iftrue` back-edge.
        use ruffle_core::avm2::Op;
        let core_ops = [
            Op::GetLocal { index: 0 }, // 0: this
            Op::PushScope,             // 1
            Op::PushInt { value: 0 },  // 2
            Op::SetLocal { index: 2 }, // 3: s = 0
            Op::PushInt { value: 0 },  // 4
            Op::SetLocal { index: 3 }, // 5: i = 0
            Op::Jump { offset: 12 },   // 6: -> condition
            Op::GetLocal { index: 2 }, // 7: body: s
            Op::GetLocal { index: 3 }, // 8: i
            Op::AddI,                  // 9
            Op::SetLocal { index: 2 }, // 10: s += i
            Op::IncLocalI { index: 3 }, // 11: i++
            Op::GetLocal { index: 3 }, // 12: condition: i
            Op::GetLocal { index: 1 }, // 13: n
            Op::LessThan,              // 14: i < n
            Op::IfTrue { offset: 7 },  // 15: -> body
            Op::GetLocal { index: 2 }, // 16: s
            Op::ReturnValue { return_type: None }, // 17
        ];
        let ops = translate(&core_ops).expect("supported");
        for n in [0, 1, 3, 7, 25, 100] {
            // regs: 0=this (dummy), 1=n, 2=s, 3=i.
            let regs = [int_bits(0), int_bits(n), int_bits(0), int_bits(0)];
            assert_equiv(&ops, &regs);
            // And it computes the actual triangular number.
            let bytes = compile(&ops).unwrap();
            let got = run(&bytes, &regs).unwrap();
            let expected: i32 = (0..n).sum();
            assert_eq!(got, int_bits(expected), "sum({n})");
        }
    }

    #[test]
    fn iftrue_branch_matches_interpreter() {
        // if (l1) return 111 else return 222
        let ops = [
            JitOp::GetLocal(1),
            JitOp::IfTrue(4),
            JitOp::PushInt(222),
            JitOp::ReturnValue,
            JitOp::PushInt(111),
            JitOp::ReturnValue,
        ];
        for v in [0, 1, -5, 42] {
            assert_equiv(&ops, &[int_bits(0), int_bits(v)]);
        }
    }
}
