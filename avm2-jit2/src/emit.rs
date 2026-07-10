//! Backend: lower an analyzed single-block method to a WebAssembly module.
//!
//! The emitted module exports one function, `run`, whose parameters are the
//! method's numeric locals **already unboxed** (the caller reads them from the
//! frame and passes `i32`/`f64`), and whose single `i64` result is the return
//! value's NaN-box bits. Every AS3 local and every operand-stack value is a typed
//! WASM local (`i32`/`f64`/`i64`), so the host engine register-allocates them —
//! there is no boxed, memory-resident frame (AVM2_JIT_DESIGN.md §1/§6).
//!
//! Values are boxed only at the return boundary; incoming args arrive unboxed.
//! The only place a NaN-box is *constructed* in WASM is a return, and the only
//! `Boxed`-typed value produced is an overflow-capable `int` add/sub/mul, whose
//! `Integer`-vs-`Number` tag is selected inline exactly as the interpreter does.

use wasm_encoder::{
    BlockType, CodeSection, EntityType, ExportKind, ExportSection, Function, FunctionSection,
    ImportSection, Instruction, MemArg, MemorySection, MemoryType, Module, TypeSection, ValType,
};

use ruffle_core::avm2::Op;

use crate::frontend::Analysis;
use crate::ir::{arith_lower, ArithLower, BinArith, ReturnCoerce, Ty};
use crate::value_bits::{BOOL_BOX_HI, CANON_NAN, INT_BOX_HI, UNDEFINED_BITS};

/// The name the runner looks up in the instantiated module.
pub const ENTRY_NAME: &str = "run";
/// The exported linear memory holding domainMemory bytes (present iff `uses_dm`).
pub const MEMORY_NAME: &str = "mem";

/// A direct JIT→JIT call target: a `CallMethod` disp_id that resolves (against the
/// shared receiver `this` vtable) to a method co-compiled into this same module
/// (design §8, the per-caller call-tree amalgam). The call becomes a direct WASM
/// `call` to that function — native-typed args, no interpreter trampoline —
/// instead of the `ca`/`cm` helper. Computed in `lib.rs` (needs the live vtable).
#[derive(Clone)]
pub struct DirectTarget {
    /// The `CallMethod.index` (disp_id) that resolves to the callee.
    pub disp_id: usize,
    /// Index of the callee unit within the amalgam (0 = the top/exported method).
    pub ordinal: usize,
    /// The callee's parameters in WASM-signature (ascending-local) order, as
    /// `(as3 local, machine type)`. Local 0 is `this`; a callee that doesn't use
    /// `this` simply omits it (the optimizer strips an unused scope prologue). The
    /// caller maps local 0 → the receiver and local `k` → the call's arg `k-1`.
    pub params: Vec<(u32, Ty)>,
}

/// One method compiled into the amalgam module. The operand lifetime `'gc` is
/// decoupled from the borrow lifetime `'a` because `Op<'gc>` is invariant.
pub struct Unit<'a, 'gc> {
    pub ana: &'a Analysis,
    pub ops: &'a [Op<'gc>],
    /// The disp_ids in this unit's body that dispatch directly to a co-compiled
    /// callee (only applied to calls whose receiver is provably `this`).
    pub direct_targets: Vec<DirectTarget>,
    /// Whether this unit's `this` (local 0) is never reassigned, so a self/sibling
    /// call's receiver read from it is provably the original receiver.
    pub this_immutable: bool,
}

/// A [`DirectTarget`] with its `ordinal` resolved to an absolute WASM function
/// index (imports occupy the low indices, so `func_index = import_count + ordinal`).
struct ResolvedTarget {
    disp_id: usize,
    func_index: u32,
    params: Vec<(u32, Ty)>,
}

/// The module-shared runtime-helper imports and their function indices (imports
/// occupy the low function-index space, before the defined method functions). A
/// helper is imported iff *any* unit uses it (the union across the amalgam).
#[derive(Copy, Clone, Default)]
struct Imports {
    get_slot: Option<u32>,
    set_slot: Option<u32>,
    to_int32: Option<u32>,
    call_arg: Option<u32>,
    call_method: Option<u32>,
    pending_error: Option<u32>,
    count: u32,
}

impl Imports {
    fn compute(units: &[Unit<'_, '_>]) -> Self {
        let any = |f: fn(&Analysis) -> bool| units.iter().any(|u| f(u.ana));
        let uses_gs = any(|a| a.uses_get_slot);
        let uses_ss = any(|a| a.uses_set_slot);
        let uses_ti = any(|a| a.uses_to_int32);
        let uses_call = any(|a| a.uses_call);
        let mut n = 0u32;
        let mut next = |used: bool| {
            if used {
                let i = n;
                n += 1;
                Some(i)
            } else {
                None
            }
        };
        // Canonical order: [get_slot, set_slot, to_int32, call_arg, call_method,
        // pending_error] — must match `runner::instantiate`.
        let get_slot = next(uses_gs);
        let set_slot = next(uses_ss);
        let to_int32 = next(uses_ti);
        let call_arg = next(uses_call);
        let call_method = next(uses_call);
        let pending_error = next(uses_call);
        Imports {
            get_slot,
            set_slot,
            to_int32,
            call_arg,
            call_method,
            pending_error,
            count: n,
        }
    }
}

/// Lower a single method with no co-compiled callees. Used by the codegen tests
/// (native only); production always goes through [`emit`] with an amalgam.
#[cfg(not(target_arch = "wasm32"))]
pub fn emit_single<'a, 'gc>(ops: &'a [Op<'gc>], ana: &'a Analysis) -> Vec<u8> {
    let units = [Unit {
        ana,
        ops,
        direct_targets: Vec::new(),
        this_immutable: false,
    }];
    emit(&units).0
}

/// Lower an amalgam (a top method plus the call-tree of methods it directly
/// calls) to one WASM module: imports shared across the units, one function per
/// unit, `units[0]` exported as `run`. A direct call from one unit to another is a
/// WASM `call` to the callee's function index. Returns the module bytes and
/// whether any direct call was emitted (for diagnostics).
///
/// `units[0]` is the top (exported) method; only it may use domainMemory (in
/// which case it is the sole unit — direct-call units never use dm).
pub fn emit(units: &[Unit<'_, '_>]) -> (Vec<u8>, bool) {
    let imports = Imports::compute(units);
    let has_memory = units.iter().any(|u| u.ana.uses_dm);
    debug_assert!(
        !has_memory || units.len() == 1,
        "domainMemory only supported for a lone unit (no co-compiled callees)"
    );

    let mut bodies: Vec<FunctionBody> = Vec::with_capacity(units.len());
    let mut emitted_direct = false;
    for u in units {
        // Resolve each target's ordinal to an absolute function index.
        let targets: Vec<ResolvedTarget> = u
            .direct_targets
            .iter()
            .map(|t| ResolvedTarget {
                disp_id: t.disp_id,
                func_index: imports.count + t.ordinal as u32,
                params: t.params.clone(),
            })
            .collect();
        let mut e = Emitter::new(u.ana, imports, targets, u.this_immutable);
        if u.ana.has_control_flow {
            e.emit_dispatch(u.ops);
        } else {
            for op in u.ops {
                e.emit_op(op);
            }
        }
        emitted_direct |= e.emitted_direct;
        bodies.push(e.into_body());
    }
    (build_module(&bodies, imports, has_memory), emitted_direct)
}

struct Emitter<'a> {
    ana: &'a Analysis,
    /// AS3 local id → WASM local index (`u32::MAX` for unreferenced locals).
    as3_to_wasm: Vec<u32>,
    /// First WASM local index after all parameters (arg locals + the optional
    /// `dm_len`); declared locals start here.
    local_base: u32,
    /// The `dm_len` parameter (domainMemory length), present iff `uses_dm`.
    dm_len_param: Option<u32>,
    /// WASM types of the *declared* locals (those after the params), in index
    /// order — non-param locals first, then scratch/operand temps.
    declared: Vec<ValType>,
    insns: Vec<Instruction<'static>>,
    /// Compile-time operand stack: (WASM local holding the value, its type).
    stack: Vec<(u32, Ty)>,
    /// The `i32` WASM local holding the current block index (dispatch mode only).
    block_local: Option<u32>,
    /// One WASM local per operand-boundary depth (`ana.boundary_ty`), carrying
    /// operand values across CFG edges.
    boundary_locals: Vec<u32>,
    /// The module-shared helper import indices (union across the amalgam).
    imports: Imports,
    /// Direct-call targets for this unit (disp_id → callee function index +
    /// signature), applied only to calls whose receiver is provably `this`.
    targets: Vec<ResolvedTarget>,
    /// Whether this unit's `this` (local 0) is never reassigned.
    this_immutable: bool,
    /// WASM locals (temps) that hold a copy of the immutable `this` (materialized
    /// from `GetLocal { index: 0 }`) — a receiver operand in this set is provably
    /// the original receiver, so a direct call to it is sound.
    this_temps: Vec<u32>,
    /// Whether at least one direct call was emitted (diagnostics).
    emitted_direct: bool,
}

/// The lowered body of one method: its parameter/local WASM types and instruction
/// stream, ready to be placed in the module's code section.
struct FunctionBody {
    params: Vec<ValType>,
    declared: Vec<ValType>,
    insns: Vec<Instruction<'static>>,
}

impl<'a> Emitter<'a> {
    fn new(
        ana: &'a Analysis,
        imports: Imports,
        targets: Vec<ResolvedTarget>,
        this_immutable: bool,
    ) -> Self {
        let n = ana.local_ty.len();
        let mut as3_to_wasm = vec![u32::MAX; n];

        // Params first (WASM indices 0..num_params), in param_locals order.
        for (j, &i) in ana.param_locals.iter().enumerate() {
            as3_to_wasm[i as usize] = j as u32;
        }
        let num_params = ana.param_locals.len() as u32;
        // domainMemory adds one trailing parameter, `dm_len`.
        let dm_len_param = if ana.uses_dm { Some(num_params) } else { None };
        let local_base = num_params + if ana.uses_dm { 1 } else { 0 };

        // Then a declared WASM local for every other referenced AS3 local.
        let mut declared = Vec::new();
        for i in 0..n {
            // Opaque locals (`this`/untyped) are never materialized, so they need
            // no WASM local.
            if ana.used[i] && as3_to_wasm[i] == u32::MAX && ana.local_ty[i] != Ty::Opaque {
                as3_to_wasm[i] = local_base + declared.len() as u32;
                declared.push(ana.local_ty[i].val_type());
            }
        }

        // The dispatch loop needs a block-index local; allocate it after the AS3
        // locals (before any operand temp) so its index is stable.
        let block_local = if ana.has_control_flow {
            let idx = local_base + declared.len() as u32;
            declared.push(ValType::I32);
            Some(idx)
        } else {
            None
        };

        // One WASM local per operand-boundary depth (carries operand values across
        // CFG edges).
        let boundary_locals: Vec<u32> = ana
            .boundary_ty
            .iter()
            .map(|&ty| {
                let idx = local_base + declared.len() as u32;
                declared.push(ty.val_type());
                idx
            })
            .collect();

        Emitter {
            ana,
            as3_to_wasm,
            local_base,
            dm_len_param,
            declared,
            insns: Vec::new(),
            stack: Vec::new(),
            block_local,
            boundary_locals,
            imports,
            targets,
            this_immutable,
            this_temps: Vec::new(),
            emitted_direct: false,
        }
    }

    /// Allocate a fresh declared WASM local of the given type; returns its index.
    fn new_temp(&mut self, ty: Ty) -> u32 {
        let idx = self.local_base + self.declared.len() as u32;
        self.declared.push(ty.val_type());
        idx
    }

    fn push_ins(&mut self, ins: Instruction<'static>) {
        self.insns.push(ins);
    }

    /// Emit a load of an operand-stack entry onto the WASM stack.
    fn load(&mut self, entry: (u32, Ty)) {
        self.push_ins(Instruction::LocalGet(entry.0));
    }

    /// Load an entry and promote it to `f64` on the WASM stack.
    fn load_as_f64(&mut self, entry: (u32, Ty)) {
        self.load(entry);
        match entry.1 {
            Ty::Num | Ty::NumRaw => {} // already f64
            // Int/Bool are i32; Bool's 0/1 converts to 0.0/1.0.
            Ty::Int | Ty::Bool => self.push_ins(Instruction::F64ConvertI32S),
            Ty::Boxed | Ty::Opaque => {
                unreachable!("Boxed/Opaque never promoted (frontend declines)")
            }
        }
    }

    /// Store the value currently on top of the WASM stack into a fresh temp of
    /// `ty` and push it onto the operand stack.
    fn materialize(&mut self, ty: Ty) {
        let t = self.new_temp(ty);
        self.push_ins(Instruction::LocalSet(t));
        self.stack.push((t, ty));
    }

    /// Box the `f64` held in WASM local `lf` into `Value` bits (on the WASM
    /// stack), canonicalizing NaN exactly as `Value::pack`'s `Number` arm.
    fn box_num_local(&mut self, lf: u32) {
        self.push_ins(Instruction::LocalGet(lf));
        self.push_ins(Instruction::LocalGet(lf));
        self.push_ins(Instruction::F64Ne); // v != v → 1 if NaN
        self.push_ins(Instruction::If(BlockType::Result(ValType::I64)));
        self.push_ins(Instruction::I64Const(CANON_NAN as i64));
        self.push_ins(Instruction::Else);
        self.push_ins(Instruction::LocalGet(lf));
        self.push_ins(Instruction::I64ReinterpretF64);
        self.push_ins(Instruction::End);
    }

    fn emit_op(&mut self, op: &Op<'_>) {
        match op {
            // ---- constant pushes ----
            Op::PushInt { value } => {
                self.push_ins(Instruction::I32Const(*value));
                self.materialize(Ty::Int);
            }
            Op::PushDouble { value } => {
                self.push_ins(Instruction::F64Const(*value));
                self.materialize(Ty::Num);
            }
            Op::PushTrue => {
                self.push_ins(Instruction::I32Const(1));
                self.materialize(Ty::Bool);
            }
            Op::PushFalse => {
                self.push_ins(Instruction::I32Const(0));
                self.materialize(Ty::Bool);
            }

            // ---- locals (snapshot GetLocal into a temp so a later SetLocal to
            // the same local can't corrupt an outstanding stack value) ----
            Op::GetLocal { index } => {
                let ty = self.ana.local_ty[*index as usize];
                if ty == Ty::Opaque {
                    // `this`/untyped read for the scope prologue: no value, no code.
                    self.stack.push((u32::MAX, Ty::Opaque));
                } else {
                    let w = self.as3_to_wasm[*index as usize];
                    self.push_ins(Instruction::LocalGet(w));
                    self.materialize(ty);
                    // A `this` read (local 0, an object) is tracked so a later
                    // self-call can prove its receiver is the original `this`.
                    if *index == 0 && ty == Ty::Boxed {
                        self.this_temps.push(self.stack.last().unwrap().0);
                    }
                }
            }
            Op::SetLocal { index } => {
                let v = self.stack.pop().unwrap();
                let w = self.as3_to_wasm[*index as usize];
                self.load(v);
                self.push_ins(Instruction::LocalSet(w));
            }
            Op::StoreLocal { index } => {
                let v = *self.stack.last().unwrap();
                let w = self.as3_to_wasm[*index as usize];
                self.load(v);
                self.push_ins(Instruction::LocalSet(w));
            }
            Op::IncLocalI { index } => {
                let w = self.as3_to_wasm[*index as usize];
                self.push_ins(Instruction::LocalGet(w));
                self.push_ins(Instruction::I32Const(1));
                self.push_ins(Instruction::I32Add);
                self.push_ins(Instruction::LocalSet(w));
            }
            Op::DecLocalI { index } => {
                let w = self.as3_to_wasm[*index as usize];
                self.push_ins(Instruction::LocalGet(w));
                self.push_ins(Instruction::I32Const(1));
                self.push_ins(Instruction::I32Sub);
                self.push_ins(Instruction::LocalSet(w));
            }

            // ---- generic arithmetic ----
            Op::Add { .. } => self.bin_arith(BinArith::Add),
            Op::Subtract { .. } => self.bin_arith(BinArith::Sub),
            Op::Multiply => self.bin_arith(BinArith::Mul),
            Op::Divide => self.bin_arith(BinArith::Div),
            Op::Negate => {
                let a = self.stack.pop().unwrap();
                self.load_as_f64(a);
                self.push_ins(Instruction::F64Neg);
                self.materialize(Ty::Num);
            }
            Op::Increment => self.unary_num_add(1.0),
            Op::Decrement => self.unary_num_add(-1.0),

            // ---- integer (`*I`) arithmetic ----
            Op::AddI => self.bin_int(Instruction::I32Add),
            Op::SubtractI => self.bin_int(Instruction::I32Sub),
            Op::MultiplyI => self.bin_int(Instruction::I32Mul),
            Op::NegateI => {
                let a = self.stack.pop().unwrap();
                self.push_ins(Instruction::I32Const(0));
                self.load(a);
                self.push_ins(Instruction::I32Sub); // 0 - a, wrapping
                self.materialize(Ty::Int);
            }
            Op::IncrementI => self.unary_int_add(1),
            Op::DecrementI => self.unary_int_add(-1),

            // ---- coercions ----
            Op::CoerceA => {}
            Op::CoerceD => {
                let a = self.stack.pop().unwrap();
                self.load_as_f64(a);
                self.materialize(Ty::Num);
            }
            Op::CoerceI => {
                let a = self.stack.pop().unwrap();
                match a.1 {
                    // Int/Bool are already the right i32 payload.
                    Ty::Int | Ty::Bool => self.load(a),
                    Ty::Num => self.coerce_num_to_i32(a), // ToInt32
                    _ => unreachable!("frontend rejects CoerceI on this type"),
                }
                self.materialize(Ty::Int);
            }

            // ---- early-bound slot read: call the pure get_slot helper with the
            // object's bits + the (constant) slot index; result is Boxed. ----
            Op::GetSlot { index } => {
                let obj = self.stack.pop().unwrap();
                let call = self.imports.get_slot.expect("GetSlot needs the helper import");
                self.load(obj);
                self.push_ins(Instruction::I32Const(*index as i32));
                self.push_ins(Instruction::Call(call));
                self.materialize(Ty::Boxed);
            }
            // Slot write: call set_slot(obj_bits, value_bits, index) -> (). Stack
            // is [object, value]; NoCoerce boxes the value naturally, CoerceI
            // boxes it as Integer (the frontend proved it int/bool).
            Op::SetSlotNoCoerce { index } => {
                let value = self.stack.pop().unwrap();
                let object = self.stack.pop().unwrap();
                let call = self.imports.set_slot.expect("SetSlot needs the helper import");
                self.load(object);
                self.box_natural(value);
                self.push_ins(Instruction::I32Const(*index as i32));
                self.push_ins(Instruction::Call(call));
            }
            Op::SetSlotCoerceI { index } => {
                let value = self.stack.pop().unwrap();
                let object = self.stack.pop().unwrap();
                let call = self.imports.set_slot.expect("SetSlot needs the helper import");
                self.load(object);
                match value.1 {
                    Ty::Int | Ty::Bool => self.box_int(value),
                    Ty::Num => {
                        self.coerce_num_to_i32(value); // ToInt32
                        self.box_i32_stack();
                    }
                    _ => unreachable!("frontend rejects SetSlotCoerceI on this type"),
                }
                self.push_ins(Instruction::I32Const(*index as i32));
                self.push_ins(Instruction::Call(call));
            }

            // ---- domainMemory loads/stores against the linear memory ----
            Op::Li8 => self.dm_load(1),
            Op::Li16 => self.dm_load(2),
            Op::Li32 => self.dm_load(4),
            Op::Si8 => self.dm_store(1),
            Op::Si16 => self.dm_store(2),
            Op::Si32 => self.dm_store(4),
            Op::Lf32 => self.dm_load_f(4),
            Op::Lf64 => self.dm_load_f(8),
            Op::Sf32 => self.dm_store_f(4),
            Op::Sf64 => self.dm_store_f(8),

            // ---- JIT→JIT call: a direct WASM `call` when the callee is provably
            // this method itself (self-recursion), otherwise the interpreter
            // trampoline. Both check for a thrown error afterward and bail. ----
            Op::CallMethod {
                index,
                num_args,
                push_return_value,
            } => {
                if !self.try_emit_direct_call(*index, *num_args, *push_return_value) {
                    self.emit_trampoline_call(*index, *num_args, *push_return_value);
                }
            }

            // ---- comparisons (numeric operands ⇒ f64 compare ⇒ Bool). These map
            // exactly to the interpreter's `abstract_lt`/`abstract_eq` with the
            // NaN/unordered case folding to false (activation.rs). ----
            Op::LessThan => self.cmp(Instruction::F64Lt),
            Op::LessEquals => self.cmp(Instruction::F64Le),
            Op::GreaterThan => self.cmp(Instruction::F64Gt),
            Op::GreaterEquals => self.cmp(Instruction::F64Ge),
            Op::Equals => self.cmp(Instruction::F64Eq),

            // ---- scope prologue (no-ops; the Opaque `this` is just dropped) ----
            Op::PushScope => {
                self.stack.pop().unwrap();
            }
            Op::PopScope => {}

            // ---- stack shuffles ----
            Op::Dup => {
                let t = *self.stack.last().unwrap();
                self.stack.push(t); // temps are write-once ⇒ aliasing is safe
            }
            Op::Pop => {
                self.stack.pop().unwrap();
            }
            Op::Swap => {
                let len = self.stack.len();
                self.stack.swap(len - 1, len - 2);
            }
            Op::Nop => {}

            // ---- returns ----
            Op::ReturnValue { .. } => {
                let a = self.stack.pop().unwrap();
                self.box_return(a);
                self.push_ins(Instruction::Return);
            }
            Op::ReturnVoid { .. } => {
                self.push_ins(Instruction::I64Const(UNDEFINED_BITS as i64));
                self.push_ins(Instruction::Return);
            }

            other => unreachable!("emit reached unsupported op {other:?} (frontend bug)"),
        }
    }

    /// `Increment`/`Decrement`: coerce to Number, add ±1.0, result Number.
    fn unary_num_add(&mut self, k: f64) {
        let a = self.stack.pop().unwrap();
        self.load_as_f64(a);
        self.push_ins(Instruction::F64Const(k));
        self.push_ins(Instruction::F64Add);
        self.materialize(Ty::Num);
    }

    /// `IncrementI`/`DecrementI`: wrapping i32 ±1, result Integer.
    fn unary_int_add(&mut self, k: i32) {
        let a = self.stack.pop().unwrap();
        self.load(a);
        self.push_ins(Instruction::I32Const(k));
        self.push_ins(Instruction::I32Add); // wrapping
        self.materialize(Ty::Int);
    }

    /// `AddI`/`SubtractI`/`MultiplyI`: both operands Int, wrapping i32.
    fn bin_int(&mut self, op: Instruction<'static>) {
        let b = self.stack.pop().unwrap();
        let a = self.stack.pop().unwrap();
        self.load(a);
        self.load(b);
        self.push_ins(op);
        self.materialize(Ty::Int);
    }

    /// The interpreter-trampoline call path: marshal args into the pending-call
    /// buffer, invoke the `call_method` helper (vtable dispatch through the live
    /// `Activation`), then check for a thrown error.
    fn emit_trampoline_call(&mut self, index: usize, num_args: u32, push_return_value: bool) {
        let n = num_args as usize;
        let mut args: Vec<(u32, Ty)> = (0..n).map(|_| self.stack.pop().unwrap()).collect();
        args.reverse(); // now arg0..argN-1
        let receiver = self.stack.pop().unwrap();

        let ca = self.imports.call_arg.unwrap();
        for arg in args {
            self.box_natural(arg); // value bits on the WASM stack
            self.push_ins(Instruction::Call(ca));
        }
        self.load(receiver); // Boxed receiver bits
        self.push_ins(Instruction::I32Const(index as i32));
        self.push_ins(Instruction::I32Const(num_args as i32));
        self.push_ins(Instruction::Call(self.imports.call_method.unwrap()));
        let rt = self.new_temp(Ty::Boxed);
        self.push_ins(Instruction::LocalSet(rt)); // stash result before the check
        self.emit_error_check();
        if push_return_value {
            self.stack.push((rt, Ty::Boxed));
        }
    }

    /// Try to emit a `CallMethod` as a direct WASM `call` to a co-compiled callee
    /// (design §8's direct JIT→JIT call: self-recursion or a sibling `this.foo()`).
    ///
    /// Sound only when: the disp_id has a resolved [`DirectTarget`] (its callee is
    /// in this amalgam, resolved against the shared `this` vtable in `lib.rs`);
    /// this unit's `this` is never reassigned and the receiver operand is provably
    /// that `this` (so the receiver we pass equals the vtable we resolved against);
    /// the call shape matches the callee's signature exactly (`this` + one native
    /// arg per declared param); and each arg's machine type marshals to its param's
    /// without a helper. Any mismatch returns `false` (leaving the operand stack
    /// untouched) so the caller falls back to the trampoline.
    fn try_emit_direct_call(&mut self, index: usize, num_args: u32, push_return_value: bool) -> bool {
        if !self.this_immutable {
            return false;
        }
        // Locate the callee for this disp_id (resolved against the `this` vtable).
        let Some(t) = self.targets.iter().position(|t| t.disp_id == index) else {
            return false;
        };
        let n = num_args as usize;
        if self.stack.len() < n + 1 {
            return false;
        }
        let top = self.stack.len();
        // The receiver (just below the n args) must be the immutable `this` — the
        // object whose vtable the callee was resolved against. This holds even when
        // the callee doesn't take `this`, since the disp_id → callee resolution
        // itself depends on the receiver's class.
        let receiver = self.stack[top - n - 1];
        if receiver.1 != Ty::Boxed || !self.this_temps.contains(&receiver.0) {
            return false;
        }
        // Each callee param maps to `this` (local 0) or a call arg (local k → arg
        // k-1) and must marshal to its native type with no helper.
        for &(local, ty) in &self.targets[t].params {
            if local == 0 {
                if ty != Ty::Boxed {
                    return false; // `this` is always a boxed object
                }
            } else {
                let ai = (local - 1) as usize;
                if ai >= n || !marshal_ok(self.stack[top - n + ai].1, ty) {
                    return false;
                }
            }
        }

        // Commit: consume the receiver + all n args, then push each callee param in
        // signature order (the callee ignores any args it doesn't name).
        let func_index = self.targets[t].func_index;
        let params = self.targets[t].params.clone();
        let mut args: Vec<(u32, Ty)> = (0..n).map(|_| self.stack.pop().unwrap()).collect();
        args.reverse(); // arg0..argN-1
        let receiver = self.stack.pop().unwrap();
        for (local, ty) in params {
            let operand = if local == 0 {
                receiver
            } else {
                args[(local - 1) as usize]
            };
            self.marshal_arg(operand, ty);
        }
        self.push_ins(Instruction::Call(func_index));
        let rt = self.new_temp(Ty::Boxed);
        self.push_ins(Instruction::LocalSet(rt));
        self.emit_error_check();
        if push_return_value {
            self.stack.push((rt, Ty::Boxed));
        }
        self.emitted_direct = true;
        true
    }

    /// Load `arg` onto the WASM stack coerced to the native type of a `param`
    /// (mirrors `build_args`' frame unboxing, but the operand is already an
    /// unboxed numeric temp). Only the helper-free coercions `marshal_ok` permits.
    fn marshal_arg(&mut self, arg: (u32, Ty), param: Ty) {
        match param {
            // Int/Bool params take the i32 payload directly (Bool's 0/1 is a valid
            // int, matching `coerce_to_i32`/`coerce_to_boolean`).
            Ty::Int | Ty::Bool => self.load(arg),
            // Number param: promote Int/Bool to f64, pass Num as-is.
            Ty::Num => self.load_as_f64(arg),
            // Object param (`this`): pass the NaN-box bits.
            Ty::Boxed => self.load(arg),
            Ty::Opaque | Ty::NumRaw => unreachable!("marshal_ok rejects these param types"),
        }
    }

    /// After a call, bail out of the function if the callee threw (the error is
    /// stashed for `try_run` to propagate). The returned bits are ignored on the
    /// error path.
    fn emit_error_check(&mut self) {
        let pe = self.imports.pending_error.unwrap();
        self.push_ins(Instruction::Call(pe));
        self.push_ins(Instruction::If(BlockType::Empty));
        self.push_ins(Instruction::I64Const(UNDEFINED_BITS as i64));
        self.push_ins(Instruction::Return);
        self.push_ins(Instruction::End);
    }

    /// A numeric comparison: promote both operands to `f64`, apply the WASM
    /// compare (which yields `i32` 0/1 and folds NaN to 0), result `Bool`.
    fn cmp(&mut self, op: Instruction<'static>) {
        let b = self.stack.pop().unwrap();
        let a = self.stack.pop().unwrap();
        self.load_as_f64(a);
        self.load_as_f64(b);
        self.push_ins(op);
        self.materialize(Ty::Bool);
    }

    // ---- domainMemory (FlasCC hot path, design §10) ----
    //
    // Direct guarded loads/stores against the module's linear memory (the domain
    // memory bytes, base 0), with the length passed as `dm_len`. Out-of-bounds is
    // swallowed on this inline path (load ⇒ 0, store ⇒ skipped) rather than
    // throwing #1506 — a deliberate, design-sanctioned relaxation; the interpreter
    // agrees on every in-bounds access.

    /// Leave `addr < dm_len` (size 1) / `addr <= dm_len - size` (size 2/4) — the
    /// in-bounds test, matching the interpreter — as an i32 bool on the WASM stack.
    fn dm_in_bounds(&mut self, addr: (u32, Ty), size: u32) {
        let dm_len = self.dm_len_param.expect("dm op needs the dm_len param");
        self.load(addr);
        self.push_ins(Instruction::LocalGet(dm_len));
        if size == 1 {
            self.push_ins(Instruction::I32LtU);
        } else {
            self.push_ins(Instruction::I32Const(size as i32));
            self.push_ins(Instruction::I32Sub); // dm_len - size
            self.push_ins(Instruction::I32LeU);
        }
    }

    fn dm_load(&mut self, size: u32) {
        let addr = self.stack.pop().unwrap();
        self.dm_in_bounds(addr, size);
        self.push_ins(Instruction::If(BlockType::Result(ValType::I32)));
        self.load(addr);
        self.push_ins(dm_load_instr(size));
        self.push_ins(Instruction::Else);
        self.push_ins(Instruction::I32Const(0));
        self.push_ins(Instruction::End);
        self.materialize(Ty::Int);
    }

    fn dm_store(&mut self, size: u32) {
        let addr = self.stack.pop().unwrap();
        let val = self.stack.pop().unwrap();
        self.dm_in_bounds(addr, size);
        self.push_ins(Instruction::If(BlockType::Empty));
        self.load(addr);
        self.load(val);
        self.push_ins(dm_store_instr(size));
        self.push_ins(Instruction::End);
    }

    /// `lf32`/`lf64`: guarded float load; result is a raw (lossless) Number. `lf32`
    /// promotes to f64. OOB is swallowed to 0.0 (inline fast path).
    fn dm_load_f(&mut self, size: u32) {
        let addr = self.stack.pop().unwrap();
        self.dm_in_bounds(addr, size);
        self.push_ins(Instruction::If(BlockType::Result(ValType::F64)));
        self.load(addr);
        if size == 4 {
            self.push_ins(Instruction::F32Load(memarg()));
            self.push_ins(Instruction::F64PromoteF32);
        } else {
            self.push_ins(Instruction::F64Load(memarg()));
        }
        self.push_ins(Instruction::Else);
        self.push_ins(Instruction::F64Const(0.0));
        self.push_ins(Instruction::End);
        self.materialize(Ty::NumRaw);
    }

    /// `sf32`/`sf64`: guarded float store; the value is coerced to f64 (`sf32`
    /// demotes to f32). OOB is swallowed (store skipped).
    fn dm_store_f(&mut self, size: u32) {
        let addr = self.stack.pop().unwrap();
        let val = self.stack.pop().unwrap();
        self.dm_in_bounds(addr, size);
        self.push_ins(Instruction::If(BlockType::Empty));
        self.load(addr);
        self.load_as_f64(val);
        if size == 4 {
            self.push_ins(Instruction::F32DemoteF64);
            self.push_ins(Instruction::F32Store(memarg()));
        } else {
            self.push_ins(Instruction::F64Store(memarg()));
        }
        self.push_ins(Instruction::End);
    }

    // ---- dispatch loop (methods with control flow) ----
    //
    // WASM has only structured control flow, so an arbitrary AVM2 CFG is run as a
    // `br_table` over a block-index local: `loop { br_table block { b0 } b1 ... }`,
    // where each block ends by setting the next index and branching back to the
    // loop (or returning). The typed locals still cross blocks as WASM locals —
    // the engine register-allocates them; the dispatch is a later relooper's job.

    fn emit_dispatch(&mut self, ops: &[Op<'_>]) {
        let b = self.ana.num_blocks();
        let block_local = self.block_local.expect("dispatch needs a block local");

        self.push_ins(Instruction::Loop(BlockType::Empty));
        for _ in 0..b {
            self.push_ins(Instruction::Block(BlockType::Empty));
        }
        // Jump table: block `k`'s label sits at depth `k` here, so target = index.
        self.push_ins(Instruction::LocalGet(block_local));
        let targets: Vec<u32> = (0..b as u32).collect();
        self.push_ins(Instruction::BrTable(targets.into(), 0));

        for bi in 0..b {
            self.push_ins(Instruction::End); // close block `bi`'s label
            self.emit_block_body(ops, bi);
        }
        self.push_ins(Instruction::End); // close the loop
        self.push_ins(Instruction::Unreachable);
    }

    fn emit_block_body(&mut self, ops: &[Op<'_>], bi: usize) {
        self.stack.clear();
        // Load this block's entry operand values from the boundary locals into
        // fresh temps (snapshot, so the exit spill can't alias a boundary local).
        let entry: Vec<Ty> = self.ana.entry_stack[bi].clone();
        for (d, ty) in entry.into_iter().enumerate() {
            self.push_ins(Instruction::LocalGet(self.boundary_locals[d]));
            self.materialize(ty);
        }
        let start = self.ana.leaders[bi];
        let end = self.ana.block_end(bi);
        let last = &ops[end - 1];
        let is_term = matches!(
            last,
            Op::Jump { .. }
                | Op::IfTrue { .. }
                | Op::IfFalse { .. }
                | Op::PopJump { .. }
                | Op::ReturnValue { .. }
                | Op::ReturnVoid { .. }
        );
        let (body_end, term) = if is_term { (end - 1, Some(last)) } else { (end, None) };
        for op in &ops[start..body_end] {
            self.emit_op(op);
        }
        self.emit_terminator(bi, term);
    }

    /// Emit a block's exit: set the next block index and branch to the loop, or
    /// return. `br_depth` is the loop label's relative depth from this block body.
    fn emit_terminator(&mut self, bi: usize, term: Option<&Op<'_>>) {
        let br_depth = (self.ana.num_blocks() - 1 - bi) as u32;
        let fall = (bi + 1) as u32;
        match term {
            None => {
                self.spill_boundary();
                self.set_block(fall);
                self.push_ins(Instruction::Br(br_depth));
            }
            Some(Op::Jump { offset }) => {
                self.spill_boundary();
                let t = self.ana.block_of(*offset) as u32;
                self.set_block(t);
                self.push_ins(Instruction::Br(br_depth));
            }
            Some(Op::PopJump { offset }) => {
                self.stack.pop();
                self.spill_boundary();
                let t = self.ana.block_of(*offset) as u32;
                self.set_block(t);
                self.push_ins(Instruction::Br(br_depth));
            }
            Some(Op::IfTrue { offset }) => {
                // true ⇒ target : select(target, fall, cond). The condition is
                // popped first; the values below it flow to both successors.
                let cond = self.stack.pop().unwrap();
                let target = self.ana.block_of(*offset) as u32;
                self.spill_boundary();
                self.select_block(target, fall, cond);
                self.push_ins(Instruction::Br(br_depth));
            }
            Some(Op::IfFalse { offset }) => {
                // false ⇒ target : select(fall, target, cond)
                let cond = self.stack.pop().unwrap();
                let target = self.ana.block_of(*offset) as u32;
                self.spill_boundary();
                self.select_block(fall, target, cond);
                self.push_ins(Instruction::Br(br_depth));
            }
            Some(Op::ReturnValue { .. }) => {
                let a = self.stack.pop().unwrap();
                self.box_return(a);
                self.push_ins(Instruction::Return);
            }
            Some(Op::ReturnVoid { .. }) => {
                self.push_ins(Instruction::I64Const(UNDEFINED_BITS as i64));
                self.push_ins(Instruction::Return);
            }
            Some(other) => unreachable!("non-terminator {other:?} routed as terminator"),
        }
    }

    /// `block = cond ? if_true : if_false` (cond is the value's i32; non-zero is
    /// true, matching `coerce_to_boolean` for Bool/Int).
    fn select_block(&mut self, if_true: u32, if_false: u32, cond: (u32, Ty)) {
        self.push_ins(Instruction::I32Const(if_true as i32));
        self.push_ins(Instruction::I32Const(if_false as i32));
        self.load_cond(cond);
        self.push_ins(Instruction::Select);
        self.push_ins(Instruction::LocalSet(self.block_local.unwrap()));
    }

    /// Leave an i32 truthiness (`coerce_to_boolean`) of `cond` on the WASM stack.
    /// Bool/Int are their own condition (non-zero = true); a Num converts via
    /// `(f == f) & (f != 0)` — NaN and 0.0 are false, matching the interpreter.
    fn load_cond(&mut self, cond: (u32, Ty)) {
        match cond.1 {
            Ty::Bool | Ty::Int => self.load(cond),
            Ty::Num | Ty::NumRaw => {
                self.load(cond);
                self.load(cond);
                self.push_ins(Instruction::F64Eq); // not NaN
                self.load(cond);
                self.push_ins(Instruction::F64Const(0.0));
                self.push_ins(Instruction::F64Ne); // non-zero
                self.push_ins(Instruction::I32And);
            }
            Ty::Boxed | Ty::Opaque => unreachable!("frontend rejects non-numeric IfTrue cond"),
        }
    }

    fn set_block(&mut self, bi: u32) {
        self.push_ins(Instruction::I32Const(bi as i32));
        self.push_ins(Instruction::LocalSet(self.block_local.unwrap()));
    }

    /// Spill the live operand stack (depth 0 = bottom) into the boundary locals so
    /// the successor block can load it. Temps and boundary locals are disjoint, so
    /// the writes don't alias regardless of order.
    fn spill_boundary(&mut self) {
        for d in 0..self.stack.len() {
            let (idx, _) = self.stack[d];
            self.push_ins(Instruction::LocalGet(idx));
            self.push_ins(Instruction::LocalSet(self.boundary_locals[d]));
        }
    }

    /// Generic `Add`/`Subtract`/`Multiply`/`Divide`, routed through the shared
    /// [`arith_lower`] decision so codegen matches the frontend's typing.
    fn bin_arith(&mut self, kind: BinArith) {
        let b = self.stack.pop().unwrap();
        let a = self.stack.pop().unwrap();
        let (lower, result) = arith_lower(kind, a.1, b.1).expect("frontend accepted this op");
        match lower {
            ArithLower::F64 => {
                self.load_as_f64(a);
                self.load_as_f64(b);
                self.push_ins(match kind {
                    BinArith::Add => Instruction::F64Add,
                    BinArith::Sub => Instruction::F64Sub,
                    BinArith::Mul => Instruction::F64Mul,
                    BinArith::Div => Instruction::F64Div,
                });
                self.materialize(result); // Num
            }
            ArithLower::IntOverflow => {
                // Full-width i64 result of two i32s (exact for +,-,*).
                self.load(a);
                self.push_ins(Instruction::I64ExtendI32S);
                self.load(b);
                self.push_ins(Instruction::I64ExtendI32S);
                self.push_ins(match kind {
                    BinArith::Add => Instruction::I64Add,
                    BinArith::Sub => Instruction::I64Sub,
                    BinArith::Mul => Instruction::I64Mul,
                    BinArith::Div => unreachable!("Div is never IntOverflow"),
                });
                let tmp = self.new_temp(Ty::Boxed); // i64 scratch
                self.push_ins(Instruction::LocalSet(tmp));
                // fits_i32 = tmp <= i32::MAX && tmp >= i32::MIN
                self.push_ins(Instruction::LocalGet(tmp));
                self.push_ins(Instruction::I64Const(i32::MAX as i64));
                self.push_ins(Instruction::I64LeS);
                self.push_ins(Instruction::LocalGet(tmp));
                self.push_ins(Instruction::I64Const(i32::MIN as i64));
                self.push_ins(Instruction::I64GeS);
                self.push_ins(Instruction::I32And);
                self.push_ins(Instruction::If(BlockType::Result(ValType::I64)));
                // fits: box Integer(tmp as i32) — low 32 bits, zero-extended.
                self.push_ins(Instruction::LocalGet(tmp));
                self.push_ins(Instruction::I32WrapI64);
                self.push_ins(Instruction::I64ExtendI32U);
                self.push_ins(Instruction::I64Const(INT_BOX_HI as i64));
                self.push_ins(Instruction::I64Or);
                self.push_ins(Instruction::Else);
                // overflow: box Number(tmp as f64). An integral f64 from an i64 is
                // never NaN, so no canonicalization is needed.
                self.push_ins(Instruction::LocalGet(tmp));
                self.push_ins(Instruction::F64ConvertI64S);
                self.push_ins(Instruction::I64ReinterpretF64);
                self.push_ins(Instruction::End);
                self.materialize(result); // Boxed
            }
        }
    }

    /// Box the popped operand into the method's return `i64`, applying the
    /// declared-return-type coercion. Leaves the boxed `i64` on the WASM stack.
    fn box_return(&mut self, a: (u32, Ty)) {
        let coerce = self
            .ana
            .ret_coerce
            .expect("ReturnValue implies a supported return coercion");
        match coerce {
            ReturnCoerce::Any => self.box_natural(a),
            ReturnCoerce::Number => {
                // a ∈ {Int, Num, Bool}: promote to f64, then box as Number.
                self.load_as_f64(a);
                let lf = self.new_temp(Ty::Num);
                self.push_ins(Instruction::LocalSet(lf));
                self.box_num_local(lf);
            }
            ReturnCoerce::Int => match a.1 {
                Ty::Int | Ty::Bool => self.box_int(a),
                Ty::Num => {
                    self.coerce_num_to_i32(a); // ToInt32
                    self.box_i32_stack();
                }
                _ => unreachable!("frontend rejects int-return of this type"),
            },
            ReturnCoerce::Boolean => self.box_bool(a),
        }
    }

    /// Box a value into its natural `Value` bits on the WASM stack (by type).
    fn box_natural(&mut self, a: (u32, Ty)) {
        match a.1 {
            Ty::Int => self.box_int(a),
            Ty::Bool => self.box_bool(a),
            Ty::Num => self.box_num_local(a.0),
            Ty::Boxed => self.load(a), // already boxed bits
            Ty::Opaque | Ty::NumRaw => {
                unreachable!("Opaque/NumRaw never boxed (frontend declines)")
            }
        }
    }

    /// Box the i32 currently on the WASM stack as an `Integer`.
    fn box_i32_stack(&mut self) {
        self.push_ins(Instruction::I64ExtendI32U);
        self.push_ins(Instruction::I64Const(INT_BOX_HI as i64));
        self.push_ins(Instruction::I64Or);
    }

    /// Box an i32-payload value (Int or Bool bits) as an `Integer`.
    fn box_int(&mut self, a: (u32, Ty)) {
        self.load(a);
        self.box_i32_stack();
    }

    /// Load a `Num` operand and coerce it to `i32` via the `to_int32` helper,
    /// leaving the i32 on the WASM stack (ECMAScript ToInt32).
    fn coerce_num_to_i32(&mut self, a: (u32, Ty)) {
        debug_assert_eq!(a.1, Ty::Num);
        let call = self.imports.to_int32.expect("ToInt32 needs the helper import");
        self.load(a);
        self.push_ins(Instruction::Call(call));
    }

    /// Box a Bool (0/1) as a `Bool`.
    fn box_bool(&mut self, a: (u32, Ty)) {
        self.load(a);
        self.push_ins(Instruction::I64ExtendI32U);
        self.push_ins(Instruction::I64Const(BOOL_BOX_HI as i64));
        self.push_ins(Instruction::I64Or);
    }

    /// Finalize this unit into its lowered [`FunctionBody`] (params, declared
    /// locals, instructions). The module is assembled from all units by
    /// [`build_module`].
    fn into_body(mut self) -> FunctionBody {
        // Guard against a frontend/emit divergence shipping stack corruption.
        debug_assert!(
            self.stack.is_empty(),
            "operand stack not empty at method end: {:?}",
            self.stack
        );

        let mut params: Vec<ValType> = self
            .ana
            .param_locals
            .iter()
            .map(|&i| self.ana.local_ty[i as usize].val_type())
            .collect();
        if self.ana.uses_dm {
            params.push(ValType::I32); // dm_len
        }
        FunctionBody {
            params,
            declared: std::mem::take(&mut self.declared),
            insns: std::mem::take(&mut self.insns),
        }
    }
}

/// Assemble the amalgam module from the lowered function bodies. Imports occupy
/// the low function indices (in canonical order), then one defined function per
/// body; `bodies[0]` is exported as `run`. A single-body module may carry the
/// domainMemory linear memory (`has_memory`).
fn build_module(bodies: &[FunctionBody], imports: Imports, has_memory: bool) -> Vec<u8> {
    let mut types = TypeSection::new();
    // One function type per body first (type index == body index), so the
    // function section can reference type `i` for body `i`.
    for b in bodies {
        types
            .ty()
            .function(b.params.iter().copied(), [ValType::I64]);
    }
    // Import types follow, in canonical order; each import's function index is its
    // slot in `Imports` (0..count), before the defined functions.
    let mut import_sec = ImportSection::new();
    let mut next_type = bodies.len() as u32;
    if imports.get_slot.is_some() {
        types.ty().function([ValType::I64, ValType::I32], [ValType::I64]);
        import_sec.import("env", "gs", EntityType::Function(next_type));
        next_type += 1;
    }
    if imports.set_slot.is_some() {
        types
            .ty()
            .function([ValType::I64, ValType::I64, ValType::I32], []);
        import_sec.import("env", "ss", EntityType::Function(next_type));
        next_type += 1;
    }
    if imports.to_int32.is_some() {
        types.ty().function([ValType::F64], [ValType::I32]);
        import_sec.import("env", "ti", EntityType::Function(next_type));
        next_type += 1;
    }
    if imports.call_arg.is_some() {
        // ca: push_call_arg(bits: i64) -> ()
        types.ty().function([ValType::I64], []);
        import_sec.import("env", "ca", EntityType::Function(next_type));
        // cm: call_method(recv: i64, disp_id: i32, argc: i32) -> i64
        types
            .ty()
            .function([ValType::I64, ValType::I32, ValType::I32], [ValType::I64]);
        import_sec.import("env", "cm", EntityType::Function(next_type + 1));
        // pe: pending_error() -> i32
        types.ty().function([], [ValType::I32]);
        import_sec.import("env", "pe", EntityType::Function(next_type + 2));
    }

    // Defined functions follow the imports; body `i` is function `count + i`.
    let mut functions = FunctionSection::new();
    for i in 0..bodies.len() {
        functions.function(i as u32); // body `i` has type `i`
    }

    let mut exports = ExportSection::new();
    exports.export(ENTRY_NAME, ExportKind::Func, imports.count); // bodies[0]

    // A domainMemory method (only ever a lone body) carries the linear memory the
    // runner fills/reads around the call.
    let memory = if has_memory {
        let mut mem = MemorySection::new();
        mem.memory(MemoryType {
            minimum: 1,
            maximum: None,
            memory64: false,
            shared: false,
            page_size_log2: None,
        });
        exports.export(MEMORY_NAME, ExportKind::Memory, 0);
        Some(mem)
    } else {
        None
    };

    let mut code = CodeSection::new();
    for b in bodies {
        let mut func = Function::new(b.declared.iter().map(|&vt| (1, vt)));
        for ins in &b.insns {
            func.instruction(ins);
        }
        func.instruction(&Instruction::End);
        code.function(&func);
    }

    let mut module = Module::new();
    module.section(&types);
    // Section order: Type, Import, Function, Memory, Export, Code.
    if imports.count > 0 {
        module.section(&import_sec);
    }
    module.section(&functions);
    if let Some(mem) = &memory {
        module.section(mem);
    }
    module.section(&exports);
    module.section(&code);
    module.finish()
}

/// Whether an arg of machine type `arg` can be marshalled to a `param`'s native
/// type without a runtime helper (so a direct self-call is sound). Mirrors the
/// coercions `build_args` applies when the interpreter enters a JIT'd method.
fn marshal_ok(arg: Ty, param: Ty) -> bool {
    match param {
        Ty::Int => matches!(arg, Ty::Int | Ty::Bool),
        Ty::Num => matches!(arg, Ty::Num | Ty::Int | Ty::Bool),
        Ty::Bool => arg == Ty::Bool,
        Ty::Boxed => arg == Ty::Boxed,
        // A raw-loaded Number would canonicalize its colliding NaN if coerced, and
        // Opaque has no value — never a self-call arg.
        Ty::Opaque | Ty::NumRaw => false,
    }
}

/// Byte-aligned (unaligned-allowed) access to domain memory 0 at the given offset.
fn memarg() -> MemArg {
    MemArg {
        offset: 0,
        align: 0,
        memory_index: 0,
    }
}

fn dm_load_instr(size: u32) -> Instruction<'static> {
    let m = memarg();
    match size {
        1 => Instruction::I32Load8U(m),
        2 => Instruction::I32Load16U(m),
        4 => Instruction::I32Load(m),
        _ => unreachable!("dm load size {size}"),
    }
}

fn dm_store_instr(size: u32) -> Instruction<'static> {
    let m = memarg();
    match size {
        1 => Instruction::I32Store8(m),
        2 => Instruction::I32Store16(m),
        4 => Instruction::I32Store(m),
        _ => unreachable!("dm store size {size}"),
    }
}
