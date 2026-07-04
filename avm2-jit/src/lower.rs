//! ABC → WASM lowering (numeric subset, with control flow).
//!
//! The emitter works on a small, self-contained [`JitOp`] IR rather than
//! `ruffle_core`'s evolving `Op` enum; a thin translation (in the backend) maps
//! supported core ops to `JitOp` and bails to the interpreter for the rest.
//!
//! ## Model
//! The compiled function is `run(state_ptr: i32) -> i64` over an **imported
//! linear memory** shared with Ruffle. `state_ptr` is the base of the method's
//! frame; register/stack slot `i` lives at `state_ptr + i*8` as an 8-byte
//! NaN-boxed `Value`. Within a basic block the ABC operand stack maps onto the
//! WASM operand stack holding **raw `i32`** integers.
//!
//! ## Control flow
//! WASM only has structured control flow, so ABC's arbitrary jumps are lowered
//! via a **dispatch loop**: a `loop` wrapping nested `block`s, entered by a
//! `br_table` on a `$block` local that holds the current basic-block index. Each
//! block ends by setting `$block` to its successor and branching back to the
//! loop (or `return`). This handles any (even irreducible) CFG mechanically.
//! Requirement: the operand stack is empty at basic-block boundaries (true for
//! typical compiled loops, whose live values are in locals) — otherwise
//! [`compile`] bails.

use std::collections::BTreeSet;

use wasm_encoder::{
    BlockType, CodeSection, EntityType, ExportKind, ExportSection, Function, FunctionSection,
    ImportSection, Instruction, MemArg, MemoryType, Module, TypeSection, ValType,
};

/// Bit pattern OR-ed onto a 32-bit integer to form an AVM2 int [`Value`].
/// MUST match `ruffle_core::avm2::value`'s `BOX_MARK | (TAG_INT << 48)`.
const VALUE_INT_MARK: u64 = 0xFFFB_0000_0000_0000;
const VALUE_ALIGN: u32 = 3;
const STATE_PTR: u32 = 0;
const SCRATCH: u32 = 1;
const BLOCK: u32 = 2;

/// The op set the lowering supports; anything else makes [`compile`] return
/// `None` (→ interpret). Branch targets are indices into the op slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JitOp {
    GetLocal(u32),
    SetLocal(u32),
    PushInt(i32),
    AddI,
    SubtractI,
    MultiplyI,
    /// Pop the top int, push `int + 1` (`incrementi`).
    IncrementI,
    /// Pop the top int, push `int - 1` (`decrementi`).
    DecrementI,
    /// `local[i] += 1` in place, no operand-stack effect (`inclocali`).
    IncLocalI(u32),
    /// `local[i] -= 1` in place, no operand-stack effect (`declocali`).
    DecLocalI(u32),
    /// Pop two ints (a, b); push `a < b` as `0`/`1`.
    LessThan,
    /// Pop two ints (a, b); push `a <= b` as `0`/`1`.
    LessEquals,
    /// Pop two ints (a, b); push `a > b` as `0`/`1`.
    GreaterThan,
    /// Pop two ints (a, b); push `a >= b` as `0`/`1`.
    GreaterEquals,
    /// Pop two ints (a, b); push `a == b` as `0`/`1`.
    Equals,
    /// Pop the top int and discard it (`pop`, and — in the int model — `pushscope`).
    Pop,
    /// Duplicate the top int (`dup`).
    Dup,
    /// No operand-stack, no memory effect (`nop`, `coerce_i` on an int, `kill`).
    Nop,
    /// Unconditional jump to op `target`.
    Jump(usize),
    /// Pop two ints (a, b); branch to `target` if a < b.
    IfLt(usize),
    /// Pop two ints (a, b); branch to `target` if a >= b.
    IfGe(usize),
    /// Pop one int; branch to `target` if it is zero (false).
    IfFalse(usize),
    /// Pop one int; branch to `target` if it is non-zero (true).
    IfTrue(usize),
    /// Box the top int as a `Value` and return it.
    ReturnValue,
}

impl JitOp {
    /// The branch target if this op is a (conditional or unconditional) branch.
    fn target(self) -> Option<usize> {
        match self {
            JitOp::Jump(t)
            | JitOp::IfLt(t)
            | JitOp::IfGe(t)
            | JitOp::IfFalse(t)
            | JitOp::IfTrue(t) => Some(t),
            _ => None,
        }
    }

    /// Whether this op ends a basic block (branch or return).
    fn is_terminator(self) -> bool {
        matches!(
            self,
            JitOp::Jump(_)
                | JitOp::IfLt(_)
                | JitOp::IfGe(_)
                | JitOp::IfFalse(_)
                | JitOp::IfTrue(_)
                | JitOp::ReturnValue
        )
    }
}

fn slot(i: u32) -> MemArg {
    MemArg {
        offset: i as u64 * 8,
        align: VALUE_ALIGN,
        memory_index: 0,
    }
}

/// Emits a non-branch op onto `body`, updating the compile-time operand-stack
/// `depth`. Returns `None` if the op isn't a linear op or underflows.
fn emit_linear(body: &mut Function, op: JitOp, depth: &mut i32) -> Option<()> {
    match op {
        JitOp::GetLocal(i) => {
            body.instruction(&Instruction::LocalGet(STATE_PTR));
            body.instruction(&Instruction::I64Load(slot(i)));
            body.instruction(&Instruction::I32WrapI64);
            *depth += 1;
        }
        JitOp::SetLocal(i) => {
            if *depth < 1 {
                return None;
            }
            body.instruction(&Instruction::LocalSet(SCRATCH));
            body.instruction(&Instruction::LocalGet(STATE_PTR));
            body.instruction(&Instruction::LocalGet(SCRATCH));
            body.instruction(&Instruction::I64ExtendI32U);
            body.instruction(&Instruction::I64Const(VALUE_INT_MARK as i64));
            body.instruction(&Instruction::I64Or);
            body.instruction(&Instruction::I64Store(slot(i)));
            *depth -= 1;
        }
        JitOp::PushInt(v) => {
            body.instruction(&Instruction::I32Const(v));
            *depth += 1;
        }
        JitOp::AddI | JitOp::SubtractI | JitOp::MultiplyI => {
            if *depth < 2 {
                return None;
            }
            body.instruction(&match op {
                JitOp::AddI => Instruction::I32Add,
                JitOp::SubtractI => Instruction::I32Sub,
                _ => Instruction::I32Mul,
            });
            *depth -= 1;
        }
        // Binary comparisons: pop two ints, push a signed `0`/`1` result.
        JitOp::LessThan
        | JitOp::LessEquals
        | JitOp::GreaterThan
        | JitOp::GreaterEquals
        | JitOp::Equals => {
            if *depth < 2 {
                return None;
            }
            body.instruction(&match op {
                JitOp::LessThan => Instruction::I32LtS,
                JitOp::LessEquals => Instruction::I32LeS,
                JitOp::GreaterThan => Instruction::I32GtS,
                JitOp::GreaterEquals => Instruction::I32GeS,
                _ => Instruction::I32Eq,
            });
            *depth -= 1;
        }
        JitOp::IncrementI | JitOp::DecrementI => {
            if *depth < 1 {
                return None;
            }
            body.instruction(&Instruction::I32Const(1));
            body.instruction(&match op {
                JitOp::IncrementI => Instruction::I32Add,
                _ => Instruction::I32Sub,
            });
        }
        JitOp::IncLocalI(i) | JitOp::DecLocalI(i) => {
            // load-modify-store a local; balanced, so no operand-stack effect.
            body.instruction(&Instruction::LocalGet(STATE_PTR)); // store address
            body.instruction(&Instruction::LocalGet(STATE_PTR));
            body.instruction(&Instruction::I64Load(slot(i)));
            body.instruction(&Instruction::I32WrapI64);
            body.instruction(&Instruction::I32Const(1));
            body.instruction(&match op {
                JitOp::IncLocalI(_) => Instruction::I32Add,
                _ => Instruction::I32Sub,
            });
            body.instruction(&Instruction::I64ExtendI32U);
            body.instruction(&Instruction::I64Const(VALUE_INT_MARK as i64));
            body.instruction(&Instruction::I64Or);
            body.instruction(&Instruction::I64Store(slot(i)));
        }
        JitOp::Pop => {
            if *depth < 1 {
                return None;
            }
            body.instruction(&Instruction::Drop);
            *depth -= 1;
        }
        JitOp::Dup => {
            if *depth < 1 {
                return None;
            }
            body.instruction(&Instruction::LocalTee(SCRATCH));
            body.instruction(&Instruction::LocalGet(SCRATCH));
            *depth += 1;
        }
        JitOp::Nop => {}
        _ => return None,
    }
    Some(())
}

/// Boxes the top-of-stack int as an AVM2 int `Value`.
fn emit_box_int(body: &mut Function) {
    body.instruction(&Instruction::I64ExtendI32U);
    body.instruction(&Instruction::I64Const(VALUE_INT_MARK as i64));
    body.instruction(&Instruction::I64Or);
}

/// Lowers `ops` to a WASM module exporting `run(state_ptr: i32) -> i64` (the
/// returned `Value`'s bits), importing the shared memory as `("env", "memory")`.
/// Returns `None` for anything unsupported.
pub fn compile(ops: &[JitOp]) -> Option<Vec<u8>> {
    if ops.is_empty() {
        return None;
    }

    // Basic-block leaders: op 0, every branch target, and every op after a
    // terminator.
    let mut leaders = BTreeSet::new();
    leaders.insert(0usize);
    for (i, op) in ops.iter().enumerate() {
        if let Some(t) = op.target() {
            if t >= ops.len() {
                return None;
            }
            leaders.insert(t);
        }
        if op.is_terminator() && i + 1 < ops.len() {
            leaders.insert(i + 1);
        }
    }
    let leaders: Vec<usize> = leaders.into_iter().collect();
    // Map an op index (that is a leader) to its basic-block index.
    let bb_of = |op_idx: usize| -> Option<usize> { leaders.iter().position(|&l| l == op_idx) };
    let num_bbs = leaders.len();

    let mut body = Function::new([(2, ValType::I32)]); // SCRATCH, BLOCK

    // loop { block(K-1) { ... block(0) { br_table } } }
    body.instruction(&Instruction::Loop(BlockType::Empty));
    for _ in 0..num_bbs {
        body.instruction(&Instruction::Block(BlockType::Empty));
    }
    let targets: Vec<u32> = (0..num_bbs as u32).collect();
    body.instruction(&Instruction::LocalGet(BLOCK));
    body.instruction(&Instruction::BrTable(targets.into(), 0));

    for bb in 0..num_bbs {
        // Close block `bb`; its code follows. Depth from here to the loop:
        let loop_depth = (num_bbs - 1 - bb) as u32;
        body.instruction(&Instruction::End);

        let start = leaders[bb];
        let end = leaders.get(bb + 1).copied().unwrap_or(ops.len());
        let mut depth: i32 = 0;

        for &op in &ops[start..end] {
            match op {
                JitOp::ReturnValue => {
                    if depth < 1 {
                        return None;
                    }
                    emit_box_int(&mut body);
                    body.instruction(&Instruction::Return);
                    depth = 0;
                }
                JitOp::Jump(t) => {
                    if depth != 0 {
                        return None; // non-empty stack across a branch: bail
                    }
                    body.instruction(&Instruction::I32Const(bb_of(t)? as i32));
                    body.instruction(&Instruction::LocalSet(BLOCK));
                    body.instruction(&Instruction::Br(loop_depth));
                }
                JitOp::IfLt(t) | JitOp::IfGe(t) | JitOp::IfFalse(t) | JitOp::IfTrue(t) => {
                    let unary = matches!(op, JitOp::IfFalse(_) | JitOp::IfTrue(_));
                    let needed = if unary { 1 } else { 2 };
                    if depth != needed {
                        return None;
                    }
                    // Condition -> i32 (non-zero = take branch). `IfTrue` uses the
                    // popped int directly as the condition, so it emits nothing.
                    match op {
                        JitOp::IfLt(_) => {
                            body.instruction(&Instruction::I32LtS);
                        }
                        JitOp::IfGe(_) => {
                            body.instruction(&Instruction::I32GeS);
                        }
                        JitOp::IfFalse(_) => {
                            body.instruction(&Instruction::I32Eqz);
                        }
                        _ => {} // IfTrue: value is already the condition
                    }
                    let taken = bb_of(t)? as i32;
                    let fallthrough = (bb + 1) as i32;
                    body.instruction(&Instruction::If(BlockType::Empty));
                    body.instruction(&Instruction::I32Const(taken));
                    body.instruction(&Instruction::LocalSet(BLOCK));
                    body.instruction(&Instruction::Else);
                    body.instruction(&Instruction::I32Const(fallthrough));
                    body.instruction(&Instruction::LocalSet(BLOCK));
                    body.instruction(&Instruction::End);
                    body.instruction(&Instruction::Br(loop_depth));
                    depth = 0;
                }
                other => emit_linear(&mut body, other, &mut depth)?,
            }
        }

        // Fell off the block end without a terminator: continue to the next BB.
        if end == ops.len() || !ops[end - 1].is_terminator() {
            if depth != 0 {
                return None;
            }
            body.instruction(&Instruction::I32Const((bb + 1) as i32));
            body.instruction(&Instruction::LocalSet(BLOCK));
            body.instruction(&Instruction::Br(loop_depth));
        }
    }

    body.instruction(&Instruction::End); // loop
    body.instruction(&Instruction::Unreachable);
    body.instruction(&Instruction::End); // function

    // Assemble module.
    let mut module = Module::new();
    let mut types = TypeSection::new();
    types.ty().function([ValType::I32], [ValType::I64]);
    module.section(&types);
    let mut imports = ImportSection::new();
    imports.import(
        "env",
        "memory",
        EntityType::Memory(MemoryType {
            minimum: 1,
            maximum: None,
            memory64: false,
            shared: false,
            page_size_log2: None,
        }),
    );
    module.section(&imports);
    let mut functions = FunctionSection::new();
    functions.function(0);
    module.section(&functions);
    let mut exports = ExportSection::new();
    exports.export("run", ExportKind::Func, 0);
    module.section(&exports);
    let mut code = CodeSection::new();
    code.function(&body);
    module.section(&code);
    Some(module.finish())
}

// Native-only: these execute the emitted module through wasmi.
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use wasmi::{Engine, Instance, Memory, MemoryType as WMemoryType, Module, Store};

    #[test]
    fn int_mark_matches_core() {
        let bits: u64 = unsafe { std::mem::transmute(ruffle_core::avm2::Value::from(0i32)) };
        assert_eq!(bits, VALUE_INT_MARK, "int box mark drifted from core");
    }

    fn int_value_bits(n: i32) -> u64 {
        unsafe { std::mem::transmute(ruffle_core::avm2::Value::from(n)) }
    }

    fn run(bytes: &[u8], slots: &[u64]) -> u64 {
        let engine = Engine::default();
        let module = Module::new(&engine, bytes).expect("valid wasm");
        let mut store = Store::new(&engine, ());
        let memory = Memory::new(&mut store, WMemoryType::new(1, None).unwrap()).unwrap();
        let mut buf = Vec::new();
        for s in slots {
            buf.extend_from_slice(&s.to_le_bytes());
        }
        memory.write(&mut store, 0, &buf).unwrap();
        let instance = Instance::new(&mut store, &module, &[memory.into()]).expect("instantiates");
        let run = instance
            .get_typed_func::<i32, i64>(&store, "run")
            .expect("run export");
        run.call(&mut store, 0).expect("runs") as u64
    }

    #[test]
    fn lowers_local_add() {
        let ops = [
            JitOp::GetLocal(1),
            JitOp::GetLocal(2),
            JitOp::AddI,
            JitOp::ReturnValue,
        ];
        let bytes = compile(&ops).expect("compiles");
        let slots = [int_value_bits(0), int_value_bits(10), int_value_bits(20)];
        assert_eq!(run(&bytes, &slots), int_value_bits(30));
    }

    #[test]
    fn lowers_counted_loop() {
        // sum(n): s=0; i=0; while (i < n) { s += i; i += 1 } return s
        // locals: 1 = n, 2 = s, 3 = i.
        let ops = [
            JitOp::PushInt(0),
            JitOp::SetLocal(2), // 0,1
            JitOp::PushInt(0),
            JitOp::SetLocal(3), // 2,3
            // 4: loop head
            JitOp::GetLocal(3),
            JitOp::GetLocal(1),
            JitOp::IfGe(16), // 4,5,6: if i>=n exit
            // 7: body
            JitOp::GetLocal(2),
            JitOp::GetLocal(3),
            JitOp::AddI,
            JitOp::SetLocal(2), // 7-10: s+=i
            JitOp::GetLocal(3),
            JitOp::PushInt(1),
            JitOp::AddI,
            JitOp::SetLocal(3), // 11-14: i+=1
            JitOp::Jump(4),     // 15: back-edge
            // 16: exit
            JitOp::GetLocal(2),
            JitOp::ReturnValue, // 16,17
        ];
        let bytes = compile(&ops).expect("compiles");
        // sum(5) = 0+1+2+3+4 = 10
        let slots = [int_value_bits(0), int_value_bits(5), int_value_bits(0), int_value_bits(0)];
        assert_eq!(run(&bytes, &slots), int_value_bits(10));
        // sum(0) = 0 (loop body never runs)
        let slots0 = [int_value_bits(0), int_value_bits(0), int_value_bits(0), int_value_bits(0)];
        assert_eq!(run(&bytes, &slots0), int_value_bits(0));
    }
}
