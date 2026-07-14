//! Type-0 WASM codegen for avm2-jit3.
//!
//! Every compiled method is ONE exported function of the universal type-0
//! signature `run(env: i32, argc: i32, args: i32) -> i64` (see
//! `AVM2_JIT_REDESIGN.md` §1). `args` is a byte offset into the module's frame
//! memory where the caller has written `[this, a0, a1, …]` as 8-byte NaN-boxed
//! [`Value`](ruffle_core::avm2::Value) slots (locals beyond `argc` pre-filled with
//! `undefined`); the function reads/writes those slots and returns the result
//! `Value`'s raw bits.
//!
//! ## Heap access via imported helpers
//!
//! On native (wasmtime) the module is sandboxed and cannot touch Ruffle's heap, so
//! any op that reads a GC object (`getslot`, later `getproperty`, calls) goes through
//! a Rust **helper import** — the helper runs in Ruffle's process with full heap
//! access. Every module imports the helpers it may need (function index 0…); the
//! defined `run` follows them. Phase 2 wires exactly one: `gs` (getslot), which is a
//! pure, null-safe slot read (no `Activation`, no GC mutation).
//!
//! ## Phase 1/2 op scope
//!
//! Straight-line bodies over primitive `Value`s + null-safe slot reads: local moves,
//! primitive-constant pushes, `pop`/`dup`, `getslot`, and `returnvalue`/`returnvoid`
//! with no return-type coercion. Everything else makes translation decline.

use wasm_encoder::{
    BlockType, CodeSection, EntityType, ExportKind, ExportSection, Function, FunctionSection,
    GlobalType, ImportSection, Instruction, MemArg, MemoryType, Module, RefType, TableType,
    TypeSection, ValType,
};

/// Bytes per frame nesting level: a call places the callee frame `STRIDE` above the
/// caller's, so re-entrant runs (a helper's `valueOf` calling another JIT method) never
/// alias. 512 `Value` slots — ≫ any real `num_locals`.
pub const FRAME_STRIDE: u32 = 4096;
/// Shared frame memory size, in 64 KiB pages (4 MiB = 1024 nesting levels). ONE such
/// memory is imported by every module (see the runner) — creating a memory per module
/// exhausts the browser's Wasm address space (OOM).
pub const FRAME_PAGES: u32 = 64;

// --- NaN-box Value bit patterns. MUST stay bit-identical to `ruffle_core::avm2::value`.

/// `BOX_MARK | (TAG_INT=3 << 48)` — an int `Value` is this OR the `u32` payload.
pub const VALUE_INT_MARK: u64 = 0xFFFB_0000_0000_0000;
/// `BOX_MARK | (TAG_BOOL=2 << 48)` — OR `0`/`1`.
pub const VALUE_BOOL_MARK: u64 = 0xFFFA_0000_0000_0000;
/// `Value::Undefined` (`BOX_MARK | TAG_UNDEFINED=0`). What `returnvoid` yields.
pub const UNDEFINED_BITS: u64 = 0xFFF8_0000_0000_0000;
/// The error SENTINEL: `BOX_MARK | (TAG=7 << 48)`. Tag 7 is never produced by `Value::pack`
/// (the valid tags are 0..=6), so this bit pattern is uniquely "an error occurred". A
/// throwing i64-returning helper returns it (after stashing the error); the compiled body's
/// `BailIfError` compares the result against it (`i64.eq`) — no `call_indirect` per op, unlike
/// the old `perr` flag call. `try_enter`'s `take_error` surfaces the real stashed error.
pub const SENTINEL_BITS: u64 = 0xFFFF_0000_0000_0000;
/// `Value::Null` (`BOX_MARK | TAG_NULL=1 << 48`).
pub const NULL_BITS: u64 = 0xFFF9_0000_0000_0000;

/// Frame slots are 8-byte `Value`s → `align = 3` (2^3).
const VALUE_ALIGN: u32 = 3;

// Helpers are reached via `call_indirect` on an IMPORTED function table (`T_HELPERS`), so
// a helper call stays wasm→wasm — no JS boundary, so the i64 `Value`s never marshal through
// BigInt (the trampoline cost that made the JIT net-negative on web). Each helper's table
// index arrives as an imported i32 global (its value differs per platform: on web it's the
// helper's real `__indirect_function_table` index, on native its slot in a purpose-built
// table). There are NO imported functions, so the defined `run` is function index 0.
/// Imported helper function table.
const T_HELPERS: u32 = 0;
/// `gs(obj_bits, slot_id) -> value_bits` — null-safe getslot.
const G_GS: u32 = 0;
/// `cr(value_bits, class_ptr) -> value_bits` — return coercion via reification.
const G_CR: u32 = 1;
/// `gp(receiver_bits, mn_ptr) -> value_bits` — polymorphic getproperty via reification.
const G_GP: u32 = 2;
/// `cp(receiver_bits, mn_ptr, args_off, argc) -> value_bits` — callproperty (args read from
/// the frame memory at `args_off`, one crossing).
const G_CP: u32 = 3;
/// `perr() -> i32` — `1` if a mid-body throwing op stashed an error (bail sentinel).
const G_PERR: u32 = 4;
/// `binop(a_bits, b_bits, op) -> value_bits` — dynamic binary operator by op-code.
const G_BINOP: u32 = 5;
/// `unop(a_bits, op) -> value_bits` — dynamic unary operator by op-code.
const G_UNOP: u32 = 6;
/// `truthy(bits) -> i32` — `coerce_to_boolean` for `iftrue`/`iffalse` branch dispatch.
const G_TRUTHY: u32 = 7;
/// `set_property(receiver, mn_ptr, value)` — property write via reification.
const G_SETPROP: u32 = 8;
/// `set_slot(receiver, index, value, mode)` — slot write (+ optional coercion / int-coerce).
const G_SETSLOT: u32 = 9;
/// `call_method(receiver, disp_id, args_off, argc)` — call by disp-id via reification.
const G_CALLMETHOD: u32 = 10;
/// `new_array(args_off, argc)` — build an Array from the scratch elements.
const G_NEWARRAY: u32 = 11;
/// `outer_scope(index) -> value` — the index-th captured scope's object.
const G_OUTERSCOPE: u32 = 12;
/// `script_globals(script_ptr) -> value` — a script's globals (may throw).
const G_SCRIPTGLOBALS: u32 = 13;
/// `new_activation(class_ptr) -> value` — a fresh activation object.
const G_NEWACTIVATION: u32 = 14;
/// `push_scope(value)` — null-check + push onto the local scope stack (`pushscope`).
const G_PUSHSCOPE: u32 = 15;
/// `pop_scope()` — pop the local scope stack (`popscope`).
const G_POPSCOPE: u32 = 16;
/// `get_scope(index) -> value` — read the local scope stack (`getscopeobject`).
const G_GETSCOPE: u32 = 17;
/// `construct(ctor, args_off, argc) -> value` — `new ctor(args)`. Uses the `TY_BINOP` shape.
const G_CONSTRUCT: u32 = 18;
/// `delete_property(receiver, mn_ptr) -> bool` — `delete obj.x`. Uses the `TY_HELPER2` shape.
const G_DELPROP: u32 = 19;
/// `is_type_late(value, type) -> bool` — `value is Type`. Uses the `TY_HELPER2` shape.
const G_ISTYPE: u32 = 20;
/// `as_type_late(value, class) -> value|null` — `value as Type`. Uses the `TY_HELPER2` shape.
const G_ASTYPE: u32 = 21;
/// `get_super(receiver, mn_ptr) -> value` — `super.x`. Uses the `TY_HELPER2` shape.
const G_GETSUPER: u32 = 22;
/// `set_super(receiver, mn_ptr, value)` — `super.x = v`. Uses the `TY_SETPROP` shape.
const G_SETSUPER: u32 = 23;
/// `call_super(receiver, mn_ptr, args_off, argc) -> value` — `super.f(a)`. `TY_CP` shape.
const G_CALLSUPER: u32 = 24;
/// `construct_super(receiver, args_off, argc) -> undefined` — `super(a)`. `TY_BINOP` shape.
const G_CONSTRUCTSUPER: u32 = 25;
/// `call_native(receiver, method_ptr, args_off, argc) -> value` — direct native call. `TY_CP`.
const G_CALLNATIVE: u32 = 26;
/// `get_prop_index(object, name, mn_ptr) -> value` — `obj[name]` (dynamic index). `TY_HELPER3`.
const G_GETPROPFAST: u32 = 27;
/// `set_prop_index(object, name, value, mn_ptr) -> undefined` — `obj[name]=v`. `TY_CP` shape.
const G_SETPROPFAST: u32 = 28;
/// `op_in(name, value) -> bool`. `TY_HELPER2` shape.
const G_IN: u32 = 29;
/// `next_value(value, index) -> value`. `TY_HELPER2` shape.
const G_NEXTVALUE: u32 = 30;
/// `next_name(value, index) -> name`. `TY_HELPER2` shape.
const G_NEXTNAME: u32 = 31;
/// `has_next(value, index) -> next_index`. `TY_HELPER2` shape.
const G_HASNEXT: u32 = 32;
/// `new_function(method_ptr) -> closure`. `TY_PTR1` shape.
const G_NEWFUNCTION: u32 = 33;
/// `apply_type(base, args_off, argc) -> applied`. `TY_BINOP` shape.
const G_APPLYTYPE: u32 = 34;
/// `construct_slot(source, index, args_off, argc) -> obj`. `TY_CALLMETHOD` shape.
const G_CONSTRUCTSLOT: u32 = 35;
/// `construct_prop(source, mn_ptr, args_off, argc) -> obj`. `TY_CP` shape.
const G_CONSTRUCTPROP: u32 = 36;
/// `call_fn(function, receiver, args_off, argc) -> result`. `TY_CP` shape.
const G_CALL: u32 = 37;
/// `new_object(pairs_off, num_pairs) -> obj`. `TY_NEWARRAY` shape.
const G_NEWOBJECT: u32 = 38;
/// `has_next_2(obj_reg, idx_reg, frame_off) -> bool`. `TY_HASNEXT2` shape.
const G_HASNEXT2: u32 = 39;
/// `throw_value(value) -> undefined` (stashes the error). `TY_PTR1` shape.
const G_THROW: u32 = 40;
/// `get_slot_checked(receiver, slot_id) -> value` (null-checks). `TY_HELPER2` shape.
const G_GSCHECKED: u32 = 41;
/// `mop_load(address, code) -> value` (domain-memory load / sign-extend). `TY_UNOP` shape.
const G_MOPLOAD: u32 = 42;
/// `mop_store(value, address, code) -> undefined` (domain-memory store). `TY_BINOP` shape.
const G_MOPSTORE: u32 = 43;
/// `get_property_ic(receiver, mn_ptr, cache_addr) -> value` — the property inline-cache MISS
/// handler. `TY_HELPER3` shape.
const G_GP_IC: u32 = 44;
/// `dm_desc_ptr() -> i64` — the domainMemory descriptor-cell address for inline `li*`/`si*`.
const G_DMDESC: u32 = 45;
/// `call_prop_ic(receiver, mn, args_off, argc, ic_addr) -> value` — monomorphic call IC.
const G_CALL_IC: u32 = 46;
/// `call_method_ic(receiver, disp_id, args_off, argc, ic_addr) -> value` — call IC by disp-id.
const G_CALL_METHOD_IC: u32 = 47;
/// `jit_enter(env) -> prev` — §8 in-WASM dispatch bracket: install the callee's cached ctx +
/// bump frame depth. `TY_PTR1` shape. Web-only emit (native binds a stub, never called).
const G_JIT_ENTER: u32 = 48;
/// `jit_leave(prev)` — end the [`G_JIT_ENTER`] bracket (drop depth + restore ctx). `TY_I64_VOID`.
const G_JIT_LEAVE: u32 = 49;
/// Number of imported helper table slots / index globals.
const NUM_HELPERS: u32 = 50;
/// The single defined function (`run`) — index 0 (no imported functions).
const F_RUN: u32 = 0;

/// Type indices (see the type section in [`compile`]).
const TY_HELPER2: u32 = 1; // (i64,i64)->i64  — gs/cr/gp
const TY_CP: u32 = 2; // (i64,i64,i64,i64)->i64 — cp
const TY_PERR: u32 = 3; // ()->i32 — perr
const TY_BINOP: u32 = 4; // (i64,i64,i32)->i64 — binop
const TY_UNOP: u32 = 5; // (i64,i32)->i64 — unop
const TY_TRUTHY: u32 = 6; // (i64)->i32 — truthy
const TY_SETPROP: u32 = 7; // (i64,i64,i64)->() — set_property
const TY_SETSLOT: u32 = 8; // (i64,i32,i64,i32)->() — set_slot
const TY_CALLMETHOD: u32 = 9; // (i64,i32,i64,i32)->i64 — call_method
const TY_NEWARRAY: u32 = 10; // (i64,i32)->i64 — new_array
const TY_I32_I64: u32 = 11; // (i32)->i64 — outer_scope / get_scope
const TY_PTR1: u32 = 12; // (i64)->i64 — script_globals / new_activation
const TY_I64_VOID: u32 = 13; // (i64)->() — push_scope
const TY_VOID: u32 = 14; // ()->() — pop_scope
const TY_HELPER3: u32 = 15; // (i64,i64,i64)->i64 — get_prop_index
const TY_HASNEXT2: u32 = 16; // (i32,i32,i64)->i64 — has_next_2
const TY_DMDESC: u32 = 17; // ()->i64 — dm_desc_ptr
const TY_CALL_IC: u32 = 18; // (i64,i64,i64,i64,i64)->i64 — call_prop_ic

/// Frame slot where a method's outgoing call-arg scratch begins (`i64.store`d args that
/// `cp` reads). Locals occupy `[0, CALL_SCRATCH_SLOT)`, outgoing args
/// `[CALL_SCRATCH_SLOT, 512)` — both within one `FRAME_STRIDE` (512 slots). Methods with
/// more locals decline (see [`compile`]).
const CALL_SCRATCH_SLOT: u32 = 256;

/// The most locals a compiled method can have (== [`CALL_SCRATCH_SLOT`]). Lets the caller
/// build the frame on the stack (no per-call heap allocation).
pub const MAX_LOCALS: usize = CALL_SCRATCH_SLOT as usize;

// type-0 params.
const P_ENV: u32 = 0;
const P_ARGC: u32 = 1;
const P_ARGS: u32 = 2;
/// Two declared i64 scratch locals (indices after the 3 params), for `setlocal`/`dup`/`swap`.
const SCRATCH64: u32 = 3;
const SCRATCH64_B: u32 = 4;
/// i32 local holding the current basic-block index for the dispatch loop (multi-block
/// methods). Defaults to 0 = the entry block.
const STATE: u32 = 5;
/// i64 local holding this run's domainMemory `[base, cap, len]` descriptor-cell ADDRESS, for
/// the inline `li*`/`si*` fast path. Set once at function entry (`dm_desc_ptr` helper) when the
/// method has integer MOP ops (else unused/0). Web only — native `li*`/`si*` stay helper.
const P_DM: u32 = 6;
/// First operand-stack SPILL local (index after `P_DM`). A block that crosses a boundary
/// with `d` operand values live spills them here (`SPILL_BASE + 0..d`) at its exit and the
/// successor reloads them at entry — how the JIT models expression-level control flow
/// (ternary, `&&`, `||`) whose value lives across a block edge.
const SPILL_BASE: u32 = 7;

/// A single op in a Phase-1/2 body. Extended in later phases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JitOp {
    /// Push local slot `i` (`this` = 0, params = 1..=argc) onto the operand stack.
    GetLocal(u32),
    /// Pop the top `Value` into local slot `i`.
    SetLocal(u32),
    /// Push a precomputed primitive `Value` (int/bool/null/undefined bits).
    PushBits(u64),
    /// Pop the receiver object, push `receiver.get_slot(id)` (null-safe: no throw).
    GetSlot(u32),
    /// Pop the receiver, push `receiver.get_slot(id)` after a null-check (may throw #1009) —
    /// for a `getslot` the verifier did NOT prove non-null.
    GetSlotChecked(u32),
    /// Pop the receiver, push `receiver.get_property(mn)` — the baked `Gc<Multiname>`
    /// pointer. Polymorphic (may hit a getter → reification + may throw).
    GetProperty(u64),
    /// Pop `argc` args + the receiver; push `receiver.call_property(mn, args)`. Carries the
    /// baked `Gc<Multiname>` pointer and the arg count. Reification + may throw.
    CallProperty(u64, u32),
    /// Pop `value` then `receiver`; `receiver.set_property(mn, value)` (baked `mn` ptr). Void.
    SetProperty(u64),
    /// Pop `value` then `receiver`; `receiver.set_slot(index, value, mode)`. `mode` = 0
    /// coerce / 1 coerce-int / 2 no-coerce. Void.
    SetSlot(u32, i32),
    /// Pop `argc` args + the receiver; push `receiver.call_method(disp_id, args)` (by-disp-id
    /// call). Carries `(disp_id, argc)`. Reification + may throw.
    CallMethod(u32, u32),
    /// Pop `num_args` elements; push a new `Array` of them. Reification; does not throw.
    NewArray(u32),
    /// Pop `num_args` args + the ctor; push `new ctor(args)`. Reification + may throw.
    Construct(u32),
    /// Pop the receiver; push `delete receiver.<mn>` (baked `mn` ptr) as a Boolean. May throw.
    DeleteProperty(u64),
    /// Pop `type` then `value`; push `value is Type` (Boolean). May throw (#1041).
    IsTypeLate,
    /// Pop `class` then `value`; push `value as Type` (value or null). May throw.
    AsTypeLate,
    /// Pop the receiver; push `super.<mn>` (baked `mn` ptr). May throw.
    GetSuper(u64),
    /// Pop `value` then `receiver`; `super.<mn> = value`. Void. May throw.
    SetSuper(u64),
    /// Pop `argc` args + the receiver; push `super.<mn>(args)`. May throw.
    CallSuper(u64, u32),
    /// Pop `argc` args + the receiver; `super(args)` (super_init). Result discarded. May throw.
    ConstructSuper(u32),
    /// Pop `argc` args + the receiver; push the baked native method's result. May throw (null).
    CallNative(u64, u32),
    /// Pop `name` then `object`; push `object[name]` (baked lazy `mn` ptr). May throw.
    GetPropertyFast(u64),
    /// Pop `value`, `name`, `object`; `object[name] = value`. Void. May throw.
    SetPropertyFast(u64),
    /// Push the `index`-th captured (outer) scope's object. No throw.
    OuterScope(u32),
    /// Push a baked script's globals object. May throw (lazy init).
    ScriptGlobals(u64),
    /// Push a fresh activation object for the baked class. No throw.
    NewActivation(u64),
    /// Pop the object; null-check + push it onto the local scope stack (`pushscope`). May throw.
    ScopePush,
    /// Pop the local scope stack (`popscope`). No stack effect, no throw.
    ScopePop,
    /// Push the `index`-th local scope's object (`getscopeobject`). No throw.
    GetScope(u32),
    /// Pop `name` then `value`; push `name in value` (Boolean). May throw (coercion).
    In,
    /// Pop `value` then `index`; push the enumerant value at `index` (`nextvalue`). May throw.
    NextValue,
    /// Pop `value` then `index`; push the enumerant name at `index` (`nextname`). May throw.
    NextName,
    /// Pop `value` then `index`; push the next enumerant index (`hasnext`). May throw.
    HasNext,
    /// Push a closure over the baked method + current scope chain (`newfunction`). No throw.
    NewFunction(u64),
    /// Pop `num_types` types + the base; push `base.<types>` (`applytype`). May throw.
    ApplyType(u32),
    /// Pop `argc` args + the source; push `new source.slot[index](args)` (`constructslot`).
    /// Carries `(index, argc)`. May throw.
    ConstructSlot(u32, u32),
    /// Pop `argc` args + the source; push `new source.<mn>(args)` (`constructprop`). Carries
    /// the baked `Gc<Multiname>` ptr and `argc`. May throw.
    ConstructProp(u64, u32),
    /// Pop `argc` args + receiver + function; push `function.call(receiver, args)` (`call`).
    /// May throw.
    Call(u32),
    /// Pop `2 × num_pairs` `[name, value]` slots; push a new dynamic object (`newobject`).
    /// May throw (name coercion).
    NewObject(u32),
    /// `hasnext2 obj_reg idx_reg`: advance the for..in cursor held in locals `obj_reg`/`idx_reg`
    /// (read AND written in the frame), push a Boolean (more?). Carries the two register
    /// indices. May throw (index coercion / enumeration).
    HasNext2(u32, u32),
    /// `throw`: pop the thrown value, stash it as the pending error, and `Return` — a
    /// terminator (the method has no exception handler, so the throw unwinds straight out).
    Throw,
    /// Pop the address (or value, for sign-extends); push a domain-memory load / sign-extend
    /// result. Carries the MOP code. May throw (#1506 / coercion).
    MopLoad(i32),
    /// Pop the address then the value; store to domain memory (`code`). Result discarded.
    /// May throw (#1506 / coercion).
    MopStore(i32),
    /// After a throwing i64-returning helper: if it returned the error SENTINEL (top of
    /// stack), return at once (`try_enter` propagates the stashed error); else restore the
    /// result. No `call_indirect` — a `SENTINEL_BITS` compare on the value.
    BailIfError,
    /// After a throwing VOID helper (no return value to sentinel-check — `set_property`,
    /// `set_slot`, `pushscope`, `set_super`): if a helper stashed an error, return `undefined`.
    /// Uses the `perr` flag call (there is no result to compare). Leaves the stack unchanged.
    BailIfPerr,
    /// Pop two values, push `binop(a, b, code)` — a dynamic binary operator (add/compare/…).
    /// May coerce (`valueOf`) → reification + may throw.
    BinOp(i32),
    /// Pop two operands PROVEN to be canonical inline `Number`s (by `translate`'s repr
    /// analysis), push `a OP b` computed as a pure `f64` op (ADD/SUBTRACT/MULTIPLY/DIVIDE).
    /// No runtime tag guard and no `binop` helper — pure `f64` arithmetic never runs `valueOf`,
    /// so it cannot throw (no `BailIfError` follows). The unguarded core of `BinOp`'s f64 path.
    BinOpNum(i32),
    /// Pop two operands PROVEN to share a numeric repr and push `a CMP b` as a native compare
    /// (`code` is a comparison binop code). `is_f64 = false` → both are `Int` boxes → signed i32
    /// compare; `is_f64 = true` → both canonical `Number` → f64 compare (NaN → false). Result is
    /// a `Bool` box. No runtime tag guard, no helper, no `BailIfError` — the hot `i < n` path.
    BinOpCmp(i32, bool),
    /// Pop one value, push `unop(a, code)` — a dynamic unary operator (negate/not/…).
    UnOp(i32),
    /// Coerce the top value to the baked return-type class pointer, leaving it on the stack
    /// (mid-body `coerce`, unlike `CoerceReturn` which then returns). Reuses the `cr` helper;
    /// may throw (#1034).
    Coerce(u64),
    /// Swap the top two values.
    Swap,
    /// Coerce the top value to the baked return-type class pointer, then RETURN it.
    /// A terminator (like `ReturnValue`) — always the last op emitted.
    CoerceReturn(u64),
    /// Discard the top of stack.
    Pop,
    /// Duplicate the top of stack.
    Dup,
    /// Return the top `Value`.
    ReturnValue,
    /// Return `undefined`.
    ReturnVoid,
}

/// The typed WASM register a promoted local lives in. `IntI32`/`BoolI32` share the same i32
/// declaration group and `i32.wrap` unbox; they differ only in the box tag `GetLocal` re-applies.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum RegKind {
    /// A canonical `Number` local → an `f64` register.
    F64,
    /// A provable `Int` local → an `i32` register (re-boxed with `VALUE_INT_MARK`).
    IntI32,
    /// A provable `Bool` local → an `i32` register (re-boxed with `VALUE_BOOL_MARK`).
    BoolI32,
}

/// A local promoted out of the memory frame into a typed WASM register (Phase 3/4).
#[derive(Clone, Copy)]
pub struct Promotion {
    /// The AVM2 local index.
    pub local: u32,
    /// Which register (and box tag) this local uses.
    pub kind: RegKind,
    /// Load the param's frame value at the prologue (a typed param), vs. the WASM-default `0`/
    /// `0.0` for a dead-on-entry accumulator (its `undefined` initial value is never read).
    pub init_from_frame: bool,
}

/// A basic block: a straight-line body plus how it hands off control.
pub struct Block {
    /// The block's straight-line ops. On entry the operand stack holds `entry_depth` values
    /// (reloaded from spill locals); a `Cond` terminator leaves the branch condition on top
    /// of the crossing values.
    pub ops: Vec<JitOp>,
    pub term: Term,
    /// Operand-stack depth on entry (values that crossed from a predecessor; reloaded from
    /// `SPILL_BASE`). `0` for straight-line / statement-level control flow.
    pub entry_depth: usize,
}

/// How a basic block transfers control. Block indices refer to positions in the `&[Block]`.
pub enum Term {
    /// The body already emitted a `Return` (`returnvalue`/`returnvoid`). Terminal.
    Return,
    /// Unconditional / implicit-fallthrough transfer to another block.
    Jump(usize),
    /// Pop the condition (`truthy`); go to `on_true` if truthy else `on_false`.
    Cond { on_true: usize, on_false: usize },
}

fn slot(i: u32) -> MemArg {
    MemArg { offset: (i as u64) * 8, align: VALUE_ALIGN, memory_index: 0 }
}

/// A `MemArg` for an arbitrary byte `offset` and `align` (log2 alignment) into the frame/main
/// memory — used by the inline `getslot` fast path's object-pointer + slot loads (and the §8
/// in-WASM call-IC guard/frame writes, which native tests validate — hence `test` too).
#[cfg(any(target_arch = "wasm32", test))]
fn mem_arg(offset: u64, align: u32) -> MemArg {
    MemArg { offset, align, memory_index: 0 }
}

/// The cached `jit_slots_layout()` — `(slots_ptr_off, slots_len_off)` byte offsets of the
/// slots slice header within an object's `ScriptObjectData` prefix, probed once. `None` if
/// the `Box<[T]>` word-order probe failed → the inline `getslot` falls back to the helper.
/// Whether `op` is a domainMemory load/store the JIT inlines: integer (`li8/16/32`,
/// `si8/16/32`) and float (`lf32/64`, `sf32/64`) raw byte accesses. Sign-extends (`sxi*`) touch
/// no memory and stay in the helper. A float LOAD whose bits collide with the box space, or a
/// float STORE of a non-numeric value, bails to the helper at RUNTIME (the guards below).
#[cfg(target_arch = "wasm32")]
fn is_inline_mop(op: &JitOp) -> bool {
    use crate::helpers::mop_code as mc;
    matches!(
        op,
        JitOp::MopLoad(mc::LI8 | mc::LI16 | mc::LI32 | mc::LF32 | mc::LF64)
            | JitOp::MopStore(mc::SI8 | mc::SI16 | mc::SI32 | mc::SF32 | mc::SF64)
    )
}

/// Codegen for the INLINE domainMemory `li*`/`si*`/`lf*`/`sf*` fast path (web only). Shared
/// guard prologue (address is a packed int, `P_DM != 0`, `addr + width <= len` read from the
/// descriptor cell) + a per-width load/store body; on any guard miss, bail to the helper.
#[cfg(target_arch = "wasm32")]
mod mop_emit {
    use super::{
        call_helper, mem_arg, BlockType, Function, Instruction as I, ValType, G_MOPLOAD,
        G_MOPSTORE, P_DM, SCRATCH64, SCRATCH64_B, TY_BINOP, TY_UNOP, UNDEFINED_BITS, VALUE_INT_MARK,
    };
    use crate::helpers::mop_code as mc;

    /// Sign + all-exponent + quiet-NaN bits: a word is a boxed Value (not an inline f64) iff
    /// `bits & BOX_MARK == BOX_MARK` (see core `value.rs`).
    const BOX_MARK: i64 = 0xFFF8_0000_0000_0000u64 as i64;

    /// `desc[0] + (addr as i32)` (i32) on the stack — the byte address of the access.
    fn base_plus_addr(body: &mut Function) {
        body.instruction(&I::LocalGet(P_DM));
        body.instruction(&I::I32WrapI64);
        body.instruction(&I::I32Load(mem_arg(0, 2))); // base = desc[0]
        body.instruction(&I::LocalGet(SCRATCH64));
        body.instruction(&I::I32WrapI64); // addr
        body.instruction(&I::I32Add);
    }

    /// i32 bool: `(addr as u32) + width <= len` (`desc[2]`), in i64 so the add can't overflow.
    fn in_bounds(body: &mut Function, width: i64) {
        body.instruction(&I::LocalGet(SCRATCH64));
        body.instruction(&I::I64Const(0xFFFF_FFFF));
        body.instruction(&I::I64And); // addr_u32 (zero-extended)
        body.instruction(&I::I64Const(width));
        body.instruction(&I::I64Add);
        body.instruction(&I::LocalGet(P_DM));
        body.instruction(&I::I32WrapI64);
        body.instruction(&I::I64Load32U(mem_arg(8, 2))); // len = desc[2]
        body.instruction(&I::I64LeU);
    }

    fn load_helper(body: &mut Function, code: i32) {
        body.instruction(&I::LocalGet(SCRATCH64));
        body.instruction(&I::I32Const(code));
        call_helper(body, G_MOPLOAD, TY_UNOP);
    }

    fn store_helper(body: &mut Function, code: i32) {
        body.instruction(&I::LocalGet(SCRATCH64_B));
        body.instruction(&I::LocalGet(SCRATCH64));
        body.instruction(&I::I32Const(code));
        call_helper(body, G_MOPSTORE, TY_BINOP);
    }

    /// Push SCRATCH64_B (a known-numeric Value) as an f64: inline double → reinterpret; else int
    /// → `f64.convert_i32_s`. Matches the helper's `coerce_to_number` on numeric operands.
    fn value_as_f64(body: &mut Function) {
        body.instruction(&I::LocalGet(SCRATCH64_B));
        body.instruction(&I::I64Const(BOX_MARK));
        body.instruction(&I::I64And);
        body.instruction(&I::I64Const(BOX_MARK));
        body.instruction(&I::I64Ne); // is_f64?
        body.instruction(&I::If(BlockType::Result(ValType::F64)));
        body.instruction(&I::LocalGet(SCRATCH64_B));
        body.instruction(&I::F64ReinterpretI64);
        body.instruction(&I::Else);
        body.instruction(&I::LocalGet(SCRATCH64_B));
        body.instruction(&I::I32WrapI64);
        body.instruction(&I::F64ConvertI32S);
        body.instruction(&I::End);
    }

    /// Emit the inline for `MopLoad(code)`. `false` (no inline) for sxi* (no memory access).
    pub fn load(body: &mut Function, code: i32) -> bool {
        let width: i64 = match code {
            mc::LI8 => 1,
            mc::LI16 => 2,
            mc::LI32 | mc::LF32 => 4,
            mc::LF64 => 8,
            _ => return false,
        };
        body.instruction(&I::LocalSet(SCRATCH64)); // address value
        // outer guard: (addr is int) & (P_DM != 0).
        body.instruction(&I::LocalGet(SCRATCH64));
        body.instruction(&I::I64Const(48));
        body.instruction(&I::I64ShrU);
        body.instruction(&I::I64Const(0xFFFB));
        body.instruction(&I::I64Eq);
        body.instruction(&I::LocalGet(P_DM));
        body.instruction(&I::I64Eqz);
        body.instruction(&I::I32Eqz);
        body.instruction(&I::I32And);
        body.instruction(&I::If(BlockType::Result(ValType::I64)));
        in_bounds(body, width);
        body.instruction(&I::If(BlockType::Result(ValType::I64)));
        match code {
            mc::LI8 | mc::LI16 | mc::LI32 => {
                let loadw = match code {
                    mc::LI8 => I::I32Load8U(mem_arg(0, 0)),
                    mc::LI16 => I::I32Load16U(mem_arg(0, 0)),
                    _ => I::I32Load(mem_arg(0, 0)),
                };
                base_plus_addr(body);
                body.instruction(&loadw);
                body.instruction(&I::I64ExtendI32U); // payload = value as u32
                body.instruction(&I::I64Const(VALUE_INT_MARK as i64));
                body.instruction(&I::I64Or);
            }
            _ => {
                // lf32/lf64: read the f64 bits. A `Number` is stored AS its raw bits unless they
                // collide with the box space — then the interpreter heap-boxes to preserve the
                // exact bits (`number_lossless`), which we can't do inline → helper.
                base_plus_addr(body);
                if code == mc::LF32 {
                    body.instruction(&I::F32Load(mem_arg(0, 0)));
                    body.instruction(&I::F64PromoteF32);
                    body.instruction(&I::I64ReinterpretF64);
                } else {
                    body.instruction(&I::I64Load(mem_arg(0, 0)));
                }
                body.instruction(&I::LocalTee(SCRATCH64_B)); // bits (SCRATCH64_B free on a load)
                body.instruction(&I::I64Const(BOX_MARK));
                body.instruction(&I::I64And);
                body.instruction(&I::I64Const(BOX_MARK));
                body.instruction(&I::I64Ne); // non-colliding?
                body.instruction(&I::If(BlockType::Result(ValType::I64)));
                body.instruction(&I::LocalGet(SCRATCH64_B)); // the double Value = its bits
                body.instruction(&I::Else);
                load_helper(body, code);
                body.instruction(&I::End);
            }
        }
        body.instruction(&I::Else);
        load_helper(body, code);
        body.instruction(&I::End);
        body.instruction(&I::Else);
        load_helper(body, code);
        body.instruction(&I::End);
        true
    }

    /// Emit the inline for `MopStore(code)`. `false` for sxi*/unknown codes.
    pub fn store(body: &mut Function, code: i32) -> bool {
        let width: i64 = match code {
            mc::SI8 => 1,
            mc::SI16 => 2,
            mc::SI32 | mc::SF32 => 4,
            mc::SF64 => 8,
            _ => return false,
        };
        let is_float = matches!(code, mc::SF32 | mc::SF64);
        body.instruction(&I::LocalSet(SCRATCH64)); // address (top)
        body.instruction(&I::LocalSet(SCRATCH64_B)); // value
        // outer guard: (addr int) & (value: int, or any numeric for float stores) & (P_DM != 0).
        body.instruction(&I::LocalGet(SCRATCH64));
        body.instruction(&I::I64Const(48));
        body.instruction(&I::I64ShrU);
        body.instruction(&I::I64Const(0xFFFB));
        body.instruction(&I::I64Eq); // addr int
        body.instruction(&I::LocalGet(SCRATCH64_B));
        body.instruction(&I::I64Const(48));
        body.instruction(&I::I64ShrU);
        body.instruction(&I::I64Const(0xFFFB));
        body.instruction(&I::I64Eq); // value int
        if is_float {
            body.instruction(&I::LocalGet(SCRATCH64_B));
            body.instruction(&I::I64Const(BOX_MARK));
            body.instruction(&I::I64And);
            body.instruction(&I::I64Const(BOX_MARK));
            body.instruction(&I::I64Ne); // OR value is inline double
            body.instruction(&I::I32Or);
        }
        body.instruction(&I::I32And);
        body.instruction(&I::LocalGet(P_DM));
        body.instruction(&I::I64Eqz);
        body.instruction(&I::I32Eqz);
        body.instruction(&I::I32And);
        body.instruction(&I::If(BlockType::Result(ValType::I64)));
        in_bounds(body, width);
        body.instruction(&I::If(BlockType::Result(ValType::I64)));
        base_plus_addr(body); // store address (base+addr)
        if is_float {
            value_as_f64(body);
            if code == mc::SF32 {
                body.instruction(&I::F32DemoteF64);
                body.instruction(&I::F32Store(mem_arg(0, 0)));
            } else {
                body.instruction(&I::F64Store(mem_arg(0, 0)));
            }
        } else {
            let storew = match code {
                mc::SI8 => I::I32Store8(mem_arg(0, 0)),
                mc::SI16 => I::I32Store16(mem_arg(0, 0)),
                _ => I::I32Store(mem_arg(0, 0)),
            };
            body.instruction(&I::LocalGet(SCRATCH64_B));
            body.instruction(&I::I32WrapI64); // value low 32 bits
            body.instruction(&storew);
        }
        body.instruction(&I::I64Const(UNDEFINED_BITS as i64));
        body.instruction(&I::Else);
        store_helper(body, code);
        body.instruction(&I::End);
        body.instruction(&I::Else);
        store_helper(body, code);
        body.instruction(&I::End);
        true
    }
}

#[cfg(target_arch = "wasm32")]
fn slots_layout() -> Option<(u32, u32)> {
    use std::sync::OnceLock;
    static LAYOUT: OnceLock<Option<(u32, u32)>> = OnceLock::new();
    *LAYOUT.get_or_init(ruffle_core::avm2::jit_slots_layout)
}

/// The cached byte offset of the `vtable` word within an object's `ScriptObjectData` — the
/// property inline cache's monomorphic class guard (`jit_vtable_offset`).
#[cfg(target_arch = "wasm32")]
fn vtable_offset() -> u32 {
    use std::sync::OnceLock;
    static V: OnceLock<u32> = OnceLock::new();
    *V.get_or_init(ruffle_core::avm2::jit_vtable_offset)
}

/// Allocates the next property inline-cache site and returns the baked byte address of its
/// `[vtable, slot]` cell (`ic_cache_base + site*8`), or `None` once the fixed cache is full
/// (then that `GetProperty` falls back to the plain `gp` helper).
#[cfg(target_arch = "wasm32")]
fn next_ic_site() -> Option<u64> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static SITE: AtomicUsize = AtomicUsize::new(0);
    let site = SITE.fetch_add(1, Ordering::Relaxed);
    if site >= crate::helpers::ic_cache_capacity() {
        return None;
    }
    Some((crate::helpers::ic_cache_base() + site * 8) as u64)
}

/// Allocates the next monomorphic CALL-cache site and returns the baked address of its
/// `[vtable_ptr, fm_ptr, env_ptr, run_idx]` cell (`call_ic_base + site * 4 * size_of::<usize>()`),
/// or `None` once the fixed cache is full (then that `CallProperty` uses the plain `cp` helper).
/// Unlike the property IC this is NOT `cfg(wasm32)`: the cell is read/written by the `call_prop_ic`
/// HELPER (native Rust), and the baked address is a native/wasm pointer the helper dereferences —
/// so it works on both targets (native → verify-testable). The extra two slots (`env_ptr`,
/// `run_idx`) drive the §8 in-WASM direct dispatch, read only by the web caller emit.
fn next_call_ic_site() -> Option<u64> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static SITE: AtomicUsize = AtomicUsize::new(0);
    let site = SITE.fetch_add(1, Ordering::Relaxed);
    if site >= crate::helpers::call_ic_capacity() {
        return None;
    }
    Some((crate::helpers::call_ic_base() + site * 4 * core::mem::size_of::<usize>()) as u64)
}

/// §8 in-WASM direct dispatch for a `callproperty`/`callmethod` site (WEB ONLY). Precondition:
/// the `argc` outgoing args are already `i64.store`d in this frame's scratch
/// (`[CALL_SCRATCH_SLOT, +argc)`) and the receiver is on top of the operand stack. Emits the
/// guarded fast path — receiver is an object whose vtable matches the cached one AND the cached
/// callee has a resolved in-WASM `run_idx` — then reserves the callee frame (`jit_enter(env)`,
/// which returns its byte offset or `0` on arena overflow), copies `[this, args]` in, and
/// `call_indirect`s the callee's `run(0, argc, base)` directly (no Rust bounce), closing with
/// `jit_leave()`. Every miss / overflow / non-object receiver falls back to the Rust call-IC
/// helper (`fallback_global`), which reads the args still in scratch — byte-identical to the
/// non-fast path. Leaves the call's i64 result on the stack (may be SENTINEL; a `BailIfError`
/// follows). `key` is the baked `mn_ptr` (callproperty) or `disp_id` (callmethod). `vt_off` is
/// the object-`vtable` word offset (`vtable_offset()` on web; a fixed value in native tests, which
/// only exercise the non-object fallback branch). Compiled on web (always) and in native tests
/// (so `WtModule::new` validates the exact bytes the browser runs — every branch's stack shape).
#[cfg(any(target_arch = "wasm32", test))]
fn emit_inwasm_call_ic(
    body: &mut Function,
    ic_addr: u64,
    key: i64,
    fallback_global: u32,
    argc: u32,
    vt_off: u64,
) {
    use Instruction as I;
    let ic32 = ic_addr as i32;
    let us = core::mem::size_of::<usize>() as u64; // cell slot stride (4 on wasm32)

    // Rust call-IC fallback: `helper(receiver, key, args_off = scratch, argc, ic_addr)` — reads
    // the args still in this frame's outgoing scratch (identical to the plain call-IC path).
    let emit_fallback = |body: &mut Function| {
        body.instruction(&I::LocalGet(SCRATCH64)); // receiver
        body.instruction(&I::I64Const(key));
        body.instruction(&I::LocalGet(P_ARGS));
        body.instruction(&I::I64ExtendI32U);
        body.instruction(&I::I64Const((CALL_SCRATCH_SLOT as i64) * 8));
        body.instruction(&I::I64Add);
        body.instruction(&I::I64Const(argc as i64));
        body.instruction(&I::I64Const(ic_addr as i64));
        call_helper(body, fallback_global, TY_CALL_IC);
    };

    // Save receiver; is it an object? (tag == 0xFFFD in the top 16 bits.)
    body.instruction(&I::LocalTee(SCRATCH64));
    body.instruction(&I::I64Const(48));
    body.instruction(&I::I64ShrU);
    body.instruction(&I::I64Const(0xFFFD));
    body.instruction(&I::I64Eq);
    body.instruction(&I::If(BlockType::Result(ValType::I64)));
    // Object: (receiver.vtable == cell.vtable) && (cell.run_idx != 0)?
    body.instruction(&I::LocalGet(SCRATCH64));
    body.instruction(&I::I32WrapI64);
    body.instruction(&I::I32Load(mem_arg(vt_off, 2))); // receiver.vtable
    body.instruction(&I::I32Const(ic32));
    body.instruction(&I::I32Load(mem_arg(0, 2))); // cell.vtable
    body.instruction(&I::I32Eq);
    body.instruction(&I::I32Const(ic32));
    body.instruction(&I::I32Load(mem_arg(3 * us, 2))); // cell.run_idx
    body.instruction(&I::I32Const(0));
    body.instruction(&I::I32Ne);
    body.instruction(&I::I32And);
    body.instruction(&I::If(BlockType::Result(ValType::I64)));
    // Eligible: `jit_enter(env, fm, args_off, argc) -> base` — validates the args already match the
    // callee's param types (else 0 → fall back), reserves the callee frame (0 on arena overflow),
    // and installs the ctx/depth/call-stack. `args_off` = this frame's outgoing scratch.
    body.instruction(&I::I32Const(ic32));
    body.instruction(&I::I32Load(mem_arg(2 * us, 2))); // cell.env
    body.instruction(&I::I64ExtendI32U);
    body.instruction(&I::I32Const(ic32));
    body.instruction(&I::I32Load(mem_arg(us, 2))); // cell.fm  (cell[1])
    body.instruction(&I::I64ExtendI32U);
    body.instruction(&I::LocalGet(P_ARGS)); // args_off = P_ARGS + CALL_SCRATCH_SLOT*8
    body.instruction(&I::I64ExtendI32U);
    body.instruction(&I::I64Const((CALL_SCRATCH_SLOT as i64) * 8));
    body.instruction(&I::I64Add);
    body.instruction(&I::I64Const(argc as i64));
    call_helper(body, G_JIT_ENTER, TY_CP);
    body.instruction(&I::LocalTee(SCRATCH64_B)); // save base
    body.instruction(&I::I64Eqz);
    body.instruction(&I::If(BlockType::Result(ValType::I64)));
    // Arena overflow: jit_enter changed nothing → fall back to the helper.
    emit_fallback(body);
    body.instruction(&I::Else);
    // Copy args (scratch -> callee frame base+8+k*8), then receiver -> base+0.
    for k in 0..argc {
        body.instruction(&I::LocalGet(SCRATCH64_B));
        body.instruction(&I::I32WrapI64); // base addr
        body.instruction(&I::LocalGet(P_ARGS));
        body.instruction(&I::I64Load(slot(CALL_SCRATCH_SLOT + k))); // arg k from scratch
        body.instruction(&I::I64Store(mem_arg(8 + (k as u64) * 8, 3)));
    }
    body.instruction(&I::LocalGet(SCRATCH64_B));
    body.instruction(&I::I32WrapI64);
    body.instruction(&I::LocalGet(SCRATCH64)); // receiver = callee's `this`
    body.instruction(&I::I64Store(mem_arg(0, 3)));
    // call_indirect callee.run(0, argc, base) through the main table (holds run + helpers).
    body.instruction(&I::I32Const(0)); // env arg (unused by `run`)
    body.instruction(&I::I32Const(argc as i32));
    body.instruction(&I::LocalGet(SCRATCH64_B));
    body.instruction(&I::I32WrapI64); // args_off = base
    body.instruction(&I::I32Const(ic32));
    body.instruction(&I::I32Load(mem_arg(3 * us, 2))); // run_idx (the table index)
    body.instruction(&I::CallIndirect { type_index: 0, table_index: T_HELPERS });
    // Close the bracket (pop call, drop depth, restore ctx); the i64 result stays on the stack.
    call_helper(body, G_JIT_LEAVE, TY_VOID);
    body.instruction(&I::End); // end base-overflow If
    body.instruction(&I::Else);
    emit_fallback(body); // object, but vtable / run_idx miss
    body.instruction(&I::End); // end vtable&run_idx If
    body.instruction(&I::Else);
    emit_fallback(body); // non-object receiver
    body.instruction(&I::End); // end is-object If
}

/// Native call-IC emission for `callproperty`/`callmethod`: the Rust `call_prop_ic`/
/// `call_method_ic` helper (`key` = `mn_ptr`/`disp_id`), reading args from this frame's scratch.
/// Native has no shared table to `call_indirect` a callee through (each method is its own wasmtime
/// instance), so it never takes the in-WASM path — EXCEPT under a unit-test flag, which forces the
/// web `emit_inwasm_call_ic` shape so `WtModule::new` validates the exact bytes the browser runs
/// (the fast path's every branch) and the non-object fallback executes.
#[cfg(not(target_arch = "wasm32"))]
fn emit_call_ic_native(body: &mut Function, key: i64, global: u32, argc: u32, ic_addr: u64) {
    use Instruction as I;
    #[cfg(test)]
    if test_force_inwasm_call_ic() {
        // vt_off is irrelevant here: tests drive a non-object receiver, so the vtable word is
        // never loaded (the outer `is-object` guard short-circuits to the fallback).
        emit_inwasm_call_ic(body, ic_addr, key, global, argc, 0);
        return;
    }
    body.instruction(&I::I64Const(key));
    body.instruction(&I::LocalGet(P_ARGS));
    body.instruction(&I::I64ExtendI32U);
    body.instruction(&I::I64Const((CALL_SCRATCH_SLOT as i64) * 8));
    body.instruction(&I::I64Add);
    body.instruction(&I::I64Const(argc as i64));
    body.instruction(&I::I64Const(ic_addr as i64));
    call_helper(body, global, TY_CALL_IC);
}

/// Unit-test toggle: force the native call-IC emission to use the web in-WASM shape (see
/// [`emit_call_ic_native`]). Off by default; a test sets it around a `compile()` call.
#[cfg(all(not(target_arch = "wasm32"), test))]
thread_local! {
    static FORCE_INWASM_CALL_IC: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
#[cfg(all(not(target_arch = "wasm32"), test))]
fn test_force_inwasm_call_ic() -> bool {
    FORCE_INWASM_CALL_IC.with(|c| c.get())
}

/// Emits the type-0 module for a body with `num_locals` locals. `Some(bytes)` on success;
/// the module imports the helpers + `env.memory` (shared frame memory) and exports `run`
/// (the entry) and re-exports `memory` (so the native runner's `Caller` can read the
/// outgoing call-arg scratch). Declines (`None`) if `num_locals` leaves no room for the
/// outgoing-arg scratch within one frame stride.
pub fn compile(
    blocks: &[Block],
    num_locals: usize,
    promoted: &[Promotion],
    undefined_init: &[u32],
) -> Option<Vec<u8>> {
    if num_locals > CALL_SCRATCH_SLOT as usize {
        return None; // locals would overlap the outgoing call-arg scratch
    }
    let mut module = Module::new();

    // Type 0: run(env, argc, args) -> i64. Type 1: gs/cr/gp (i64,i64)->i64.
    // Type 2: cp(receiver, mn, args_off, argc) -> i64. Type 3: perr() -> i32.
    let mut types = TypeSection::new();
    types
        .ty()
        .function([ValType::I32, ValType::I32, ValType::I32], [ValType::I64]);
    types.ty().function([ValType::I64, ValType::I64], [ValType::I64]);
    types.ty().function(
        [ValType::I64, ValType::I64, ValType::I64, ValType::I64],
        [ValType::I64],
    );
    types.ty().function([], [ValType::I32]);
    types.ty().function([ValType::I64, ValType::I64, ValType::I32], [ValType::I64]); // binop
    types.ty().function([ValType::I64, ValType::I32], [ValType::I64]); // unop
    types.ty().function([ValType::I64], [ValType::I32]); // truthy
    types.ty().function([ValType::I64, ValType::I64, ValType::I64], []); // set_property
    types.ty().function([ValType::I64, ValType::I32, ValType::I64, ValType::I32], []); // set_slot
    types
        .ty()
        .function([ValType::I64, ValType::I32, ValType::I64, ValType::I32], [ValType::I64]); // call_method
    types.ty().function([ValType::I64, ValType::I32], [ValType::I64]); // new_array
    types.ty().function([ValType::I32], [ValType::I64]); // outer_scope / get_scope
    types.ty().function([ValType::I64], [ValType::I64]); // script_globals / new_activation
    types.ty().function([ValType::I64], []); // push_scope
    types.ty().function([], []); // pop_scope
    types.ty().function([ValType::I64, ValType::I64, ValType::I64], [ValType::I64]); // get_prop_index
    types.ty().function([ValType::I32, ValType::I32, ValType::I64], [ValType::I64]); // has_next_2
    types.ty().function([], [ValType::I64]); // dm_desc_ptr
    types.ty().function(
        [ValType::I64, ValType::I64, ValType::I64, ValType::I64, ValType::I64],
        [ValType::I64],
    ); // call_prop_ic
    module.section(&types);

    // Imports (in this exact order — the native runner binds them positionally):
    //   table `env.table` (the helper funcref table, `call_indirect` target),
    //   i32 globals `env.{gs,cr,gp,cp,perr}_i` (each helper's index in that table),
    //   memory `env.memory` (the frame memory).
    // On WEB the memory import is the MAIN Ruffle module's linear memory — which is SHARED
    // (atomics/threads enabled), so the import MUST declare `shared: true` with a maximum,
    // or `WebAssembly.Instance()` rejects the mismatch (and, since a failed instantiate
    // isn't cached, retries every call → an `Instance` blow-up). On NATIVE the runner hands
    // each module its own non-shared wasmtime memory. `cfg!` picks the shape per target.
    let (mem_shared, mem_max) = if cfg!(target_arch = "wasm32") {
        (true, Some(65536)) // shared main memory: any max ≥ its own (≤ 4 GiB) accepts it
    } else {
        (false, Some(FRAME_PAGES as u64))
    };
    let mut imports = ImportSection::new();
    imports.import(
        "env",
        "table",
        EntityType::Table(TableType {
            element_type: RefType::FUNCREF,
            table64: false,
            minimum: 0,
            maximum: None,
            shared: false,
        }),
    );
    let idx_global = EntityType::Global(GlobalType {
        val_type: ValType::I32,
        mutable: false,
        shared: false,
    });
    imports.import("env", "gs_i", idx_global);
    imports.import("env", "cr_i", idx_global);
    imports.import("env", "gp_i", idx_global);
    imports.import("env", "cp_i", idx_global);
    imports.import("env", "perr_i", idx_global);
    imports.import("env", "binop_i", idx_global);
    imports.import("env", "unop_i", idx_global);
    imports.import("env", "truthy_i", idx_global);
    imports.import("env", "setprop_i", idx_global);
    imports.import("env", "setslot_i", idx_global);
    imports.import("env", "callmethod_i", idx_global);
    imports.import("env", "newarray_i", idx_global);
    imports.import("env", "outerscope_i", idx_global);
    imports.import("env", "scriptglobals_i", idx_global);
    imports.import("env", "newactivation_i", idx_global);
    imports.import("env", "pushscope_i", idx_global);
    imports.import("env", "popscope_i", idx_global);
    imports.import("env", "getscope_i", idx_global);
    imports.import("env", "construct_i", idx_global);
    imports.import("env", "delprop_i", idx_global);
    imports.import("env", "istype_i", idx_global);
    imports.import("env", "astype_i", idx_global);
    imports.import("env", "getsuper_i", idx_global);
    imports.import("env", "setsuper_i", idx_global);
    imports.import("env", "callsuper_i", idx_global);
    imports.import("env", "constructsuper_i", idx_global);
    imports.import("env", "callnative_i", idx_global);
    imports.import("env", "getpropfast_i", idx_global);
    imports.import("env", "setpropfast_i", idx_global);
    imports.import("env", "in_i", idx_global);
    imports.import("env", "nextvalue_i", idx_global);
    imports.import("env", "nextname_i", idx_global);
    imports.import("env", "hasnext_i", idx_global);
    imports.import("env", "newfunction_i", idx_global);
    imports.import("env", "applytype_i", idx_global);
    imports.import("env", "constructslot_i", idx_global);
    imports.import("env", "constructprop_i", idx_global);
    imports.import("env", "call_i", idx_global);
    imports.import("env", "newobject_i", idx_global);
    imports.import("env", "hasnext2_i", idx_global);
    imports.import("env", "throw_i", idx_global);
    imports.import("env", "gschecked_i", idx_global);
    imports.import("env", "mopload_i", idx_global);
    imports.import("env", "mopstore_i", idx_global);
    imports.import("env", "gp_ic_i", idx_global);
    imports.import("env", "dmdesc_i", idx_global);
    imports.import("env", "callpropic_i", idx_global);
    imports.import("env", "callmethodic_i", idx_global);
    imports.import("env", "jitenter_i", idx_global);
    imports.import("env", "jitleave_i", idx_global);
    imports.import(
        "env",
        "memory",
        EntityType::Memory(MemoryType {
            minimum: 1,
            maximum: mem_max,
            memory64: false,
            shared: mem_shared,
            page_size_log2: None,
        }),
    );
    module.section(&imports);

    let mut funcs = FunctionSection::new();
    funcs.function(0); // `run` : type 0
    module.section(&funcs);

    let mut exports = ExportSection::new();
    exports.export("run", ExportKind::Func, F_RUN);
    // Re-export the imported frame memory so the native runner's `cp` can reach it via
    // `Caller::get_export("memory")` (harmless on web).
    exports.export("memory", ExportKind::Memory, 0);
    module.section(&exports);

    // SCRATCH64/_B (2×i64), STATE (i32), P_DM (i64), then `max_spill` SPILL locals (i64), then the
    // PROMOTED registers (Phase 3/4: `Number` locals → f64, `Int` locals → i32, lifted out of the
    // memory frame). The spill i64 run is `1 (P_DM) + max_spill` after STATE, so the f64 group
    // starts at `SPILL_BASE + max_spill`, and the i32 group right after it.
    let max_spill = blocks.iter().map(|b| b.entry_depth).max().unwrap_or(0) as u32;
    let n_f64 = promoted.iter().filter(|p| p.kind == RegKind::F64).count() as u32;
    let n_i32 = promoted.iter().filter(|p| p.kind != RegKind::F64).count() as u32;
    let f64_base = SPILL_BASE + max_spill;
    let i32_base = f64_base + n_f64;
    // `promote[local_i] = Some((wasm register index, kind))` for a promoted local, else `None`.
    // Read by `emit_op`'s `GetLocal`/`SetLocal` to use the register instead of `i64.load`/`store`.
    // `Int`/`Bool` share the i32 declaration group (indexed after the f64 group).
    let mut promote: Vec<Option<(u32, RegKind)>> = vec![None; num_locals];
    let (mut f64_rank, mut i32_rank) = (0u32, 0u32);
    for p in promoted {
        let idx = if p.kind == RegKind::F64 {
            let r = f64_base + f64_rank;
            f64_rank += 1;
            r
        } else {
            let r = i32_base + i32_rank;
            i32_rank += 1;
            r
        };
        if let Some(slot) = promote.get_mut(p.local as usize) {
            *slot = Some((idx, p.kind));
        }
    }
    let local_decls = vec![
        (2u32, ValType::I64),
        (1u32, ValType::I32),
        (1u32 + max_spill, ValType::I64),
        (n_f64, ValType::F64),
        (n_i32, ValType::I32),
    ];
    let mut body = Function::new(local_decls);
    // Prologue: a promoted typed PARAM (`init_from_frame`) is loaded from the frame ONCE into its
    // register — f64 via `f64.reinterpret` (a canonical `Number`'s box bits ARE its f64 bits),
    // i32 via `i32.wrap` (an `Int`/`Bool` box's low 32 bits ARE the payload). A promoted
    // accumulator is dead-on-entry, so its `undefined` frame value is never read and the
    // WASM-default `0`/`0.0` is fine → no init. Thereafter `GetLocal`/`SetLocal` use the register.
    for p in promoted {
        if p.init_from_frame {
            let (idx, kind) = promote[p.local as usize].unwrap();
            body.instruction(&Instruction::LocalGet(P_ARGS));
            body.instruction(&Instruction::I64Load(slot(p.local)));
            if kind == RegKind::F64 {
                body.instruction(&Instruction::F64ReinterpretI64);
            } else {
                body.instruction(&Instruction::I32WrapI64);
            }
            body.instruction(&Instruction::LocalSet(idx));
        }
    }
    // Prologue `undefined`-init: the caller (`try_enter`) now writes ONLY `[this, params]` into
    // the frame — NOT the full-width `undefined` padding. So a NON-promoted local that may be
    // read before it is written (`undefined_init`, computed by liveness) must be set here, or it
    // would read stale arena data from a previous run at this depth. Usually empty.
    for &i in undefined_init {
        body.instruction(&Instruction::LocalGet(P_ARGS));
        body.instruction(&Instruction::I64Const(UNDEFINED_BITS as i64));
        body.instruction(&Instruction::I64Store(slot(i)));
    }
    // domainMemory inline (`li*`/`si*`) fast path: when the method has any INTEGER MOP op,
    // fetch the current domainMemory descriptor-cell address ONCE at entry into `P_DM` (the
    // inline reads base+len from the cell on each access). Web only — native keeps the helper,
    // so the reifying `dm_desc_ptr` call is elided there (`P_DM` stays 0, inline is cfg'd out).
    #[cfg(target_arch = "wasm32")]
    if blocks.iter().flat_map(|b| &b.ops).any(is_inline_mop) {
        call_helper(&mut body, G_DMDESC, TY_DMDESC);
        body.instruction(&Instruction::LocalSet(P_DM));
    }
    // Single straight-line block (ending in a return) → emit its ops directly. Otherwise
    // (branches/loops) → a `br_table` dispatch loop over the basic blocks.
    if blocks.len() == 1 && matches!(blocks[0].term, Term::Return) {
        for &op in &blocks[0].ops {
            emit_op(&mut body, op, &promote)?;
        }
    } else {
        emit_dispatch(&mut body, blocks, &promote)?;
    }
    // Fallthrough guard: yield `undefined` so the function is always valid and total (for
    // the dispatch loop this is unreachable, but it keeps the function well-typed).
    body.instruction(&Instruction::I64Const(UNDEFINED_BITS as i64));
    body.instruction(&Instruction::End);

    let mut code = CodeSection::new();
    code.function(&body);
    module.section(&code);

    Some(module.finish())
}

fn emit_op(body: &mut Function, op: JitOp, promote: &[Option<(u32, RegKind)>]) -> Option<()> {
    use Instruction as I;
    #[cfg(target_arch = "wasm32")]
    use crate::helpers::mop_code as mc;
    match op {
        JitOp::GetLocal(i) => {
            // Promoted: read the register and RE-BOX. F64 → the canonical `Number`'s box bits ARE
            // its f64 bits (`i64.reinterpret`). Int/Bool → `i64.extend_i32_u | tag`.
            match promote.get(i as usize) {
                Some(&Some((reg, RegKind::F64))) => {
                    body.instruction(&I::LocalGet(reg));
                    body.instruction(&I::I64ReinterpretF64);
                }
                Some(&Some((reg, kind))) => {
                    let mark = if kind == RegKind::BoolI32 {
                        VALUE_BOOL_MARK
                    } else {
                        VALUE_INT_MARK
                    };
                    body.instruction(&I::LocalGet(reg));
                    body.instruction(&I::I64ExtendI32U);
                    body.instruction(&I::I64Const(mark as i64));
                    body.instruction(&I::I64Or);
                }
                _ => {
                    body.instruction(&I::LocalGet(P_ARGS));
                    body.instruction(&I::I64Load(slot(i)));
                }
            }
        }
        JitOp::SetLocal(i) => {
            // Promoted: UNBOX the (proven `Number`/`Int`/`Bool`) i64 to the register — no
            // `i64.store` to the frame (the slot goes stale but is never read again). F64 →
            // `f64.reinterpret`; Int/Bool → `i32.wrap` (the box's low 32 bits are the payload).
            match promote.get(i as usize) {
                Some(&Some((reg, RegKind::F64))) => {
                    body.instruction(&I::F64ReinterpretI64);
                    body.instruction(&I::LocalSet(reg));
                }
                Some(&Some((reg, _))) => {
                    body.instruction(&I::I32WrapI64);
                    body.instruction(&I::LocalSet(reg));
                }
                _ => {
                    body.instruction(&I::LocalSet(SCRATCH64));
                    body.instruction(&I::LocalGet(P_ARGS));
                    body.instruction(&I::LocalGet(SCRATCH64));
                    body.instruction(&I::I64Store(slot(i)));
                }
            }
        }
        JitOp::PushBits(bits) => {
            body.instruction(&I::I64Const(bits as i64));
        }
        JitOp::GetSlot(index) => {
            // Stack: [.., receiver] — a verifier-proven non-null object. On the WEB build the
            // compiled module shares the main memory (where GC objects live), so read the slot
            // INLINE via the stable `jit_slots_layout` ABI (no helper `call_indirect`): the
            // object Value's payload is the 32-bit pointer to its `ScriptObjectData`; the slots
            // slice's data pointer is at `+slots_ptr_off`; slot `id`'s `Value` word is at
            // `data_ptr + id*8`. On NATIVE the wasm can't reach GC memory → the helper.
            #[cfg(target_arch = "wasm32")]
            let inlined = match slots_layout() {
                Some((slots_ptr_off, _)) => {
                    body.instruction(&I::I32WrapI64); // obj_ptr (payload = low 32 bits)
                    body.instruction(&I::I32Load(mem_arg(slots_ptr_off as u64, 2))); // slots data ptr
                    body.instruction(&I::I64Load(mem_arg(index as u64 * 8, 3))); // the Value word
                    true
                }
                None => false,
            };
            #[cfg(not(target_arch = "wasm32"))]
            let inlined = false;
            if !inlined {
                body.instruction(&I::I64Const(index as i64));
                call_helper(body, G_GS, TY_HELPER2);
            }
        }
        JitOp::GetSlotChecked(index) => {
            // Like `GetSlot`, but the receiver may be null/primitive. INLINE (web) only when it
            // is an object (tag `0xFFFD`); otherwise the helper (which throws #1009). The helper
            // also covers the native build and a failed layout probe.
            #[cfg(target_arch = "wasm32")]
            let inlined = match slots_layout() {
                Some((slots_ptr_off, _)) => {
                    body.instruction(&I::LocalTee(SCRATCH64)); // save receiver
                    body.instruction(&I::I64Const(48));
                    body.instruction(&I::I64ShrU);
                    body.instruction(&I::I64Const(0xFFFD)); // BOX_MARK | TAG_OBJECT<<48 >> 48
                    body.instruction(&I::I64Eq);
                    body.instruction(&I::If(BlockType::Result(ValType::I64)));
                    body.instruction(&I::LocalGet(SCRATCH64));
                    body.instruction(&I::I32WrapI64);
                    body.instruction(&I::I32Load(mem_arg(slots_ptr_off as u64, 2)));
                    body.instruction(&I::I64Load(mem_arg(index as u64 * 8, 3)));
                    body.instruction(&I::Else);
                    body.instruction(&I::LocalGet(SCRATCH64));
                    body.instruction(&I::I64Const(index as i64));
                    call_helper(body, G_GSCHECKED, TY_HELPER2);
                    body.instruction(&I::End);
                    true
                }
                None => false,
            };
            #[cfg(not(target_arch = "wasm32"))]
            let inlined = false;
            if !inlined {
                body.instruction(&I::I64Const(index as i64));
                call_helper(body, G_GSCHECKED, TY_HELPER2);
            }
        }
        JitOp::GetProperty(mn_ptr) => {
            // Stack: [.., receiver]. Monomorphic inline cache (web only): each site owns a
            // baked `[vtable:u32, slot:u32]` cell. If `receiver` is an object whose vtable
            // matches the cached one, read `receiver.slots[cached.slot]` inline — no crossing.
            // Otherwise `gp_ic(receiver, mn_ptr, cache_addr)` resolves it AND refills the cell
            // (or throws → SENTINEL, caught by the following `BailIfError`). The plain `gp`
            // helper covers the native build, a failed layout probe, and a full cache.
            #[cfg(target_arch = "wasm32")]
            let inlined = match (slots_layout(), next_ic_site()) {
                (Some((slots_ptr_off, _)), Some(cache_addr)) => {
                    let vt_off = vtable_offset();
                    let cache = cache_addr as i32; // wasm32 address (bit pattern = u32 addr)
                    body.instruction(&I::LocalTee(SCRATCH64)); // save receiver
                    // Guard: (receiver is object) && (receiver.vtable == cached.vtable).
                    // A non-object receiver has payload bits that won't alias a live vtable
                    // pointer AND its tag makes the low 32 bits a poor match; but to be safe
                    // we first check the object tag, then the vtable word.
                    body.instruction(&I::I64Const(48));
                    body.instruction(&I::I64ShrU);
                    body.instruction(&I::I64Const(0xFFFD)); // TAG_OBJECT
                    body.instruction(&I::I64Eq);
                    body.instruction(&I::If(BlockType::Result(ValType::I64)));
                    // Object: compare vtable.
                    body.instruction(&I::LocalGet(SCRATCH64));
                    body.instruction(&I::I32WrapI64); // obj_ptr
                    body.instruction(&I::I32Load(mem_arg(vt_off as u64, 2))); // obj.vtable
                    body.instruction(&I::I32Const(cache));
                    body.instruction(&I::I32Load(mem_arg(0, 2))); // cached.vtable
                    body.instruction(&I::I32Eq);
                    body.instruction(&I::If(BlockType::Result(ValType::I64)));
                    // HIT: receiver.slots[cached.slot].
                    body.instruction(&I::LocalGet(SCRATCH64));
                    body.instruction(&I::I32WrapI64);
                    body.instruction(&I::I32Load(mem_arg(slots_ptr_off as u64, 2))); // slots data ptr
                    body.instruction(&I::I32Const(cache));
                    body.instruction(&I::I32Load(mem_arg(4, 2))); // cached.slot
                    body.instruction(&I::I32Const(3));
                    body.instruction(&I::I32Shl); // slot * 8
                    body.instruction(&I::I32Add);
                    body.instruction(&I::I64Load(mem_arg(0, 3))); // the Value word
                    body.instruction(&I::Else);
                    // Object, vtable miss: helper (refills cache).
                    body.instruction(&I::LocalGet(SCRATCH64));
                    body.instruction(&I::I64Const(mn_ptr as i64));
                    body.instruction(&I::I64Const(cache_addr as i64));
                    call_helper(body, G_GP_IC, TY_HELPER3);
                    body.instruction(&I::End);
                    body.instruction(&I::Else);
                    // Non-object receiver: helper (null-checks / dynamic prop).
                    body.instruction(&I::LocalGet(SCRATCH64));
                    body.instruction(&I::I64Const(mn_ptr as i64));
                    body.instruction(&I::I64Const(cache_addr as i64));
                    call_helper(body, G_GP_IC, TY_HELPER3);
                    body.instruction(&I::End);
                    true
                }
                _ => false,
            };
            #[cfg(not(target_arch = "wasm32"))]
            let inlined = false;
            if !inlined {
                body.instruction(&I::I64Const(mn_ptr as i64));
                call_helper(body, G_GP, TY_HELPER2);
            }
        }
        JitOp::CallProperty(mn_ptr, argc) => {
            // Stack: [.., receiver, arg0, …, arg{N-1}] (arg{N-1} on top). Store each arg
            // into the frame's outgoing-arg scratch via pure-wasm `i64.store` — arg{k} at
            // slot `CALL_SCRATCH_SLOT + k` — popping top-down, leaving [.., receiver].
            for i in (0..argc).rev() {
                body.instruction(&I::LocalSet(SCRATCH64)); // pop arg{i}
                body.instruction(&I::LocalGet(P_ARGS)); // frame base addr
                body.instruction(&I::LocalGet(SCRATCH64)); // value
                body.instruction(&I::I64Store(slot(CALL_SCRATCH_SLOT + i)));
            }
            // Reads `argc` args from `args_off` (= frame base + scratch) in ONE crossing. When a
            // call-IC site is free, `call_prop_ic(receiver, mn, args_off, argc, ic_addr)` — which
            // dispatches a cached monomorphic method DIRECTLY, skipping `call_method_with_args`'s
            // resolve-IC; otherwise the plain `cp` helper.
            match next_call_ic_site() {
                // Web: a guarded IN-WASM `call_indirect` into the cached callee (no Rust bounce),
                // falling back to `call_prop_ic` on a miss. Native: the helper only (no shared
                // table to `call_indirect` through — see §8). Receiver is on the stack top.
                #[cfg(target_arch = "wasm32")]
                Some(ic_addr) => {
                    emit_inwasm_call_ic(body, ic_addr, mn_ptr as i64, G_CALL_IC, argc, vtable_offset() as u64)
                }
                #[cfg(not(target_arch = "wasm32"))]
                Some(ic_addr) => emit_call_ic_native(body, mn_ptr as i64, G_CALL_IC, argc, ic_addr),
                None => {
                    body.instruction(&I::I64Const(mn_ptr as i64));
                    body.instruction(&I::LocalGet(P_ARGS));
                    body.instruction(&I::I64ExtendI32U);
                    body.instruction(&I::I64Const((CALL_SCRATCH_SLOT as i64) * 8));
                    body.instruction(&I::I64Add);
                    body.instruction(&I::I64Const(argc as i64));
                    call_helper(body, G_CP, TY_CP);
                }
            }
        }
        JitOp::CallMethod(disp_id, argc) => {
            // Like CallProperty: store args to the outgoing scratch, then call by disp-id.
            for i in (0..argc).rev() {
                body.instruction(&I::LocalSet(SCRATCH64));
                body.instruction(&I::LocalGet(P_ARGS));
                body.instruction(&I::LocalGet(SCRATCH64));
                body.instruction(&I::I64Store(slot(CALL_SCRATCH_SLOT + i)));
            }
            // Stack: [.., receiver]. With a call-IC site free, `call_method_ic` dispatches a
            // cached monomorphic method DIRECTLY (skipping `call_method_with_args`'s resolve-IC)
            // — `disp_id`/`argc` baked as `i64` so it shares the all-i64 `TY_CALL_IC` shape.
            // Otherwise the plain `call_method` helper.
            match next_call_ic_site() {
                // Web: guarded IN-WASM direct dispatch (see `emit_inwasm_call_ic`); native: the
                // by-disp-id call-IC helper only. Receiver is on the stack top.
                #[cfg(target_arch = "wasm32")]
                Some(ic_addr) => emit_inwasm_call_ic(
                    body,
                    ic_addr,
                    disp_id as i64,
                    G_CALL_METHOD_IC,
                    argc,
                    vtable_offset() as u64,
                ),
                #[cfg(not(target_arch = "wasm32"))]
                Some(ic_addr) => {
                    emit_call_ic_native(body, disp_id as i64, G_CALL_METHOD_IC, argc, ic_addr)
                }
                None => {
                    body.instruction(&I::I32Const(disp_id as i32));
                    body.instruction(&I::LocalGet(P_ARGS));
                    body.instruction(&I::I64ExtendI32U);
                    body.instruction(&I::I64Const((CALL_SCRATCH_SLOT as i64) * 8));
                    body.instruction(&I::I64Add);
                    body.instruction(&I::I32Const(argc as i32));
                    call_helper(body, G_CALLMETHOD, TY_CALLMETHOD);
                }
            }
        }
        JitOp::NewArray(argc) => {
            // Store the `argc` elements to the outgoing scratch, then `new_array(off, argc)`.
            for i in (0..argc).rev() {
                body.instruction(&I::LocalSet(SCRATCH64));
                body.instruction(&I::LocalGet(P_ARGS));
                body.instruction(&I::LocalGet(SCRATCH64));
                body.instruction(&I::I64Store(slot(CALL_SCRATCH_SLOT + i)));
            }
            body.instruction(&I::LocalGet(P_ARGS));
            body.instruction(&I::I64ExtendI32U);
            body.instruction(&I::I64Const((CALL_SCRATCH_SLOT as i64) * 8));
            body.instruction(&I::I64Add);
            body.instruction(&I::I32Const(argc as i32));
            call_helper(body, G_NEWARRAY, TY_NEWARRAY);
        }
        JitOp::Construct(argc) => {
            // Stack: [.., ctor, arg0, …]. Store args to scratch, then `construct(ctor, off, n)`.
            for i in (0..argc).rev() {
                body.instruction(&I::LocalSet(SCRATCH64));
                body.instruction(&I::LocalGet(P_ARGS));
                body.instruction(&I::LocalGet(SCRATCH64));
                body.instruction(&I::I64Store(slot(CALL_SCRATCH_SLOT + i)));
            }
            body.instruction(&I::LocalGet(P_ARGS));
            body.instruction(&I::I64ExtendI32U);
            body.instruction(&I::I64Const((CALL_SCRATCH_SLOT as i64) * 8));
            body.instruction(&I::I64Add);
            body.instruction(&I::I32Const(argc as i32));
            call_helper(body, G_CONSTRUCT, TY_BINOP); // (i64,i64,i32)->i64
        }
        JitOp::OuterScope(index) => {
            body.instruction(&I::I32Const(index as i32));
            call_helper(body, G_OUTERSCOPE, TY_I32_I64);
        }
        JitOp::ScriptGlobals(script_ptr) => {
            body.instruction(&I::I64Const(script_ptr as i64));
            call_helper(body, G_SCRIPTGLOBALS, TY_PTR1);
        }
        JitOp::NewActivation(class_ptr) => {
            body.instruction(&I::I64Const(class_ptr as i64));
            call_helper(body, G_NEWACTIVATION, TY_PTR1);
        }
        JitOp::ScopePush => {
            // Stack: [.., object]. `push_scope(object)` consumes it (may throw on null).
            call_helper(body, G_PUSHSCOPE, TY_I64_VOID);
        }
        JitOp::ScopePop => {
            call_helper(body, G_POPSCOPE, TY_VOID);
        }
        JitOp::GetScope(index) => {
            body.instruction(&I::I32Const(index as i32));
            call_helper(body, G_GETSCOPE, TY_I32_I64);
        }
        JitOp::In => {
            // Stack: [.., name, value]. `op_in(name, value)` → Boolean.
            call_helper(body, G_IN, TY_HELPER2);
        }
        JitOp::NextValue => {
            // Stack: [.., value, index]. `next_value(value, index)` → enumerant value.
            call_helper(body, G_NEXTVALUE, TY_HELPER2);
        }
        JitOp::NextName => {
            call_helper(body, G_NEXTNAME, TY_HELPER2);
        }
        JitOp::HasNext => {
            call_helper(body, G_HASNEXT, TY_HELPER2);
        }
        JitOp::NewFunction(method_ptr) => {
            body.instruction(&I::I64Const(method_ptr as i64));
            call_helper(body, G_NEWFUNCTION, TY_PTR1);
        }
        JitOp::ApplyType(num_types) => {
            // Stack: [.., base, type0, …]. Store types to scratch, leaving [.., base], then
            // `apply_type(base, off, num)` (same shape as Construct).
            for i in (0..num_types).rev() {
                body.instruction(&I::LocalSet(SCRATCH64));
                body.instruction(&I::LocalGet(P_ARGS));
                body.instruction(&I::LocalGet(SCRATCH64));
                body.instruction(&I::I64Store(slot(CALL_SCRATCH_SLOT + i)));
            }
            body.instruction(&I::LocalGet(P_ARGS));
            body.instruction(&I::I64ExtendI32U);
            body.instruction(&I::I64Const((CALL_SCRATCH_SLOT as i64) * 8));
            body.instruction(&I::I64Add);
            body.instruction(&I::I32Const(num_types as i32));
            call_helper(body, G_APPLYTYPE, TY_BINOP); // (i64,i64,i32)->i64
        }
        JitOp::ConstructSlot(index, argc) => {
            // Stack: [.., source, arg0, …]. Store args, leaving [.., source], then
            // `construct_slot(source, index, off, argc)` (same shape as CallMethod).
            for i in (0..argc).rev() {
                body.instruction(&I::LocalSet(SCRATCH64));
                body.instruction(&I::LocalGet(P_ARGS));
                body.instruction(&I::LocalGet(SCRATCH64));
                body.instruction(&I::I64Store(slot(CALL_SCRATCH_SLOT + i)));
            }
            body.instruction(&I::I32Const(index as i32));
            body.instruction(&I::LocalGet(P_ARGS));
            body.instruction(&I::I64ExtendI32U);
            body.instruction(&I::I64Const((CALL_SCRATCH_SLOT as i64) * 8));
            body.instruction(&I::I64Add);
            body.instruction(&I::I32Const(argc as i32));
            call_helper(body, G_CONSTRUCTSLOT, TY_CALLMETHOD); // (i64,i32,i64,i32)->i64
        }
        JitOp::ConstructProp(mn_ptr, argc) => {
            // Stack: [.., source, arg0, …]. Store args, leaving [.., source], then
            // `construct_prop(source, mn_ptr, off, argc)` (same shape as CallProperty).
            for i in (0..argc).rev() {
                body.instruction(&I::LocalSet(SCRATCH64));
                body.instruction(&I::LocalGet(P_ARGS));
                body.instruction(&I::LocalGet(SCRATCH64));
                body.instruction(&I::I64Store(slot(CALL_SCRATCH_SLOT + i)));
            }
            body.instruction(&I::I64Const(mn_ptr as i64));
            body.instruction(&I::LocalGet(P_ARGS));
            body.instruction(&I::I64ExtendI32U);
            body.instruction(&I::I64Const((CALL_SCRATCH_SLOT as i64) * 8));
            body.instruction(&I::I64Add);
            body.instruction(&I::I64Const(argc as i64));
            call_helper(body, G_CONSTRUCTPROP, TY_CP); // (i64,i64,i64,i64)->i64
        }
        JitOp::Call(argc) => {
            // Stack: [.., function, receiver, arg0, …]. Store args, leaving
            // [.., function, receiver], then `call_fn(function, receiver, off, argc)`.
            for i in (0..argc).rev() {
                body.instruction(&I::LocalSet(SCRATCH64));
                body.instruction(&I::LocalGet(P_ARGS));
                body.instruction(&I::LocalGet(SCRATCH64));
                body.instruction(&I::I64Store(slot(CALL_SCRATCH_SLOT + i)));
            }
            body.instruction(&I::LocalGet(P_ARGS));
            body.instruction(&I::I64ExtendI32U);
            body.instruction(&I::I64Const((CALL_SCRATCH_SLOT as i64) * 8));
            body.instruction(&I::I64Add);
            body.instruction(&I::I64Const(argc as i64));
            call_helper(body, G_CALL, TY_CP); // (i64,i64,i64,i64)->i64
        }
        JitOp::NewObject(num_pairs) => {
            // Stack: [.., name0, value0, …]. Store all `2 × num_pairs` slots to scratch, then
            // `new_object(off, num_pairs)`.
            for i in (0..num_pairs * 2).rev() {
                body.instruction(&I::LocalSet(SCRATCH64));
                body.instruction(&I::LocalGet(P_ARGS));
                body.instruction(&I::LocalGet(SCRATCH64));
                body.instruction(&I::I64Store(slot(CALL_SCRATCH_SLOT + i)));
            }
            body.instruction(&I::LocalGet(P_ARGS));
            body.instruction(&I::I64ExtendI32U);
            body.instruction(&I::I64Const((CALL_SCRATCH_SLOT as i64) * 8));
            body.instruction(&I::I64Add);
            body.instruction(&I::I32Const(num_pairs as i32));
            call_helper(body, G_NEWOBJECT, TY_NEWARRAY); // (i64,i32)->i64
        }
        JitOp::HasNext2(obj_reg, idx_reg) => {
            // `has_next_2(obj_reg, idx_reg, frame_base)` — the helper reads/writes those two
            // frame locals directly and returns the Boolean (pushed here). No operand input.
            body.instruction(&I::I32Const(obj_reg as i32));
            body.instruction(&I::I32Const(idx_reg as i32));
            body.instruction(&I::LocalGet(P_ARGS));
            body.instruction(&I::I64ExtendI32U);
            call_helper(body, G_HASNEXT2, TY_HASNEXT2); // (i32,i32,i64)->i64
        }
        JitOp::Throw => {
            // Stack: [.., value]. `throw_value(value)` stashes the error, returns undefined;
            // then `Return` ends the run and `try_enter` propagates the stashed error.
            call_helper(body, G_THROW, TY_PTR1);
            body.instruction(&I::Return);
        }
        JitOp::MopLoad(code) => {
            // Stack: [.., address]. INLINE (web) domainMemory reads: integer (`li8/16/32`) and
            // float (`lf32/64`). When the address is a packed int, domainMemory is shareable
            // (`P_DM != 0`), and `addr + width <= len` (read from the descriptor cell), read the
            // bytes directly from `base + addr` — no helper crossing. A float load whose bits
            // collide with the box space bails to the helper (it heap-boxes). Any guard failing
            // (incl. OOB → the helper's #1506) takes `mop_load`. sxi* + native always helper.
            #[cfg(target_arch = "wasm32")]
            let inlined = mop_emit::load(body, code);
            #[cfg(not(target_arch = "wasm32"))]
            let inlined = false;
            if !inlined {
                body.instruction(&I::I32Const(code));
                call_helper(body, G_MOPLOAD, TY_UNOP); // (i64,i32)->i64
            }
        }
        JitOp::MopStore(code) => {
            // Stack: [.., value, address]. INLINE (web) domainMemory writes: integer
            // (`si8/16/32`, value must be a packed int) and float (`sf32/64`, value must be
            // numeric). When domainMemory is shareable and `addr + width <= len`, store the low
            // `width` bytes at `base + addr`; leave `undefined` (matching the helper's discarded
            // result). Any guard failing takes `mop_store` (which coerces / throws #1506).
            #[cfg(target_arch = "wasm32")]
            let inlined = mop_emit::store(body, code);
            #[cfg(not(target_arch = "wasm32"))]
            let inlined = false;
            if !inlined {
                body.instruction(&I::I32Const(code));
                call_helper(body, G_MOPSTORE, TY_BINOP); // (i64,i64,i32)->i64
            }
        }
        JitOp::DeleteProperty(mn_ptr) => {
            // Stack: [.., receiver]. `delete_property(receiver, mn)` → Boolean.
            body.instruction(&I::I64Const(mn_ptr as i64));
            call_helper(body, G_DELPROP, TY_HELPER2);
        }
        JitOp::GetPropertyFast(mn_ptr) => {
            // Stack: [.., object, name]. `get_prop_index(object, name, mn)` → value.
            body.instruction(&I::I64Const(mn_ptr as i64));
            call_helper(body, G_GETPROPFAST, TY_HELPER3);
        }
        JitOp::SetPropertyFast(mn_ptr) => {
            // Stack: [.., object, name, value]. `set_prop_index(object, name, value, mn)`.
            body.instruction(&I::I64Const(mn_ptr as i64));
            call_helper(body, G_SETPROPFAST, TY_CP); // (i64,i64,i64,i64)->i64 (undefined)
        }
        JitOp::CallNative(method_ptr, argc) => {
            // Stack: [.., receiver, args…]. Store args, then `call_native(receiver, m, off, n)`.
            for i in (0..argc).rev() {
                body.instruction(&I::LocalSet(SCRATCH64));
                body.instruction(&I::LocalGet(P_ARGS));
                body.instruction(&I::LocalGet(SCRATCH64));
                body.instruction(&I::I64Store(slot(CALL_SCRATCH_SLOT + i)));
            }
            body.instruction(&I::I64Const(method_ptr as i64));
            body.instruction(&I::LocalGet(P_ARGS));
            body.instruction(&I::I64ExtendI32U);
            body.instruction(&I::I64Const((CALL_SCRATCH_SLOT as i64) * 8));
            body.instruction(&I::I64Add);
            body.instruction(&I::I64Const(argc as i64));
            call_helper(body, G_CALLNATIVE, TY_CP); // (i64,i64,i64,i64)->i64
        }
        JitOp::GetSuper(mn_ptr) => {
            // Stack: [.., receiver]. `get_super(receiver, mn)` → value.
            body.instruction(&I::I64Const(mn_ptr as i64));
            call_helper(body, G_GETSUPER, TY_HELPER2);
        }
        JitOp::SetSuper(mn_ptr) => {
            // Stack: [.., receiver, value]. `set_super(receiver, mn, value)`.
            body.instruction(&I::LocalSet(SCRATCH64));
            body.instruction(&I::I64Const(mn_ptr as i64));
            body.instruction(&I::LocalGet(SCRATCH64));
            call_helper(body, G_SETSUPER, TY_SETPROP);
        }
        JitOp::CallSuper(mn_ptr, argc) => {
            // Stack: [.., receiver, args…]. Store args, then `call_super(receiver, mn, off, n)`.
            for i in (0..argc).rev() {
                body.instruction(&I::LocalSet(SCRATCH64));
                body.instruction(&I::LocalGet(P_ARGS));
                body.instruction(&I::LocalGet(SCRATCH64));
                body.instruction(&I::I64Store(slot(CALL_SCRATCH_SLOT + i)));
            }
            body.instruction(&I::I64Const(mn_ptr as i64));
            body.instruction(&I::LocalGet(P_ARGS));
            body.instruction(&I::I64ExtendI32U);
            body.instruction(&I::I64Const((CALL_SCRATCH_SLOT as i64) * 8));
            body.instruction(&I::I64Add);
            body.instruction(&I::I64Const(argc as i64));
            call_helper(body, G_CALLSUPER, TY_CP); // (i64,i64,i64,i64)->i64
        }
        JitOp::ConstructSuper(argc) => {
            // Stack: [.., receiver, args…]. Store args, then `construct_super(receiver, off, n)`.
            for i in (0..argc).rev() {
                body.instruction(&I::LocalSet(SCRATCH64));
                body.instruction(&I::LocalGet(P_ARGS));
                body.instruction(&I::LocalGet(SCRATCH64));
                body.instruction(&I::I64Store(slot(CALL_SCRATCH_SLOT + i)));
            }
            body.instruction(&I::LocalGet(P_ARGS));
            body.instruction(&I::I64ExtendI32U);
            body.instruction(&I::I64Const((CALL_SCRATCH_SLOT as i64) * 8));
            body.instruction(&I::I64Add);
            body.instruction(&I::I32Const(argc as i32));
            call_helper(body, G_CONSTRUCTSUPER, TY_BINOP); // (i64,i64,i32)->i64 (undefined)
        }
        JitOp::IsTypeLate => {
            // Stack: [.., value, type]. `is_type_late(value, type)` → Boolean.
            call_helper(body, G_ISTYPE, TY_HELPER2);
        }
        JitOp::AsTypeLate => {
            // Stack: [.., value, class]. `as_type_late(value, class)` → value|null.
            call_helper(body, G_ASTYPE, TY_HELPER2);
        }
        JitOp::SetProperty(mn_ptr) => {
            // Stack: [.., object, value]. Save `value`, then `set_property(object, mn, value)`.
            body.instruction(&I::LocalSet(SCRATCH64));
            body.instruction(&I::I64Const(mn_ptr as i64));
            body.instruction(&I::LocalGet(SCRATCH64));
            call_helper(body, G_SETPROP, TY_SETPROP);
        }
        JitOp::SetSlot(index, mode) => {
            // Stack: [.., object, value]. INLINE (web) the raw slot write when it needs neither
            // coercion NOR a GC write-barrier: mode 2 (`setslotnocoerce`) with a NON-Gc value,
            // or mode 1 (`setslotcoercei`) with an int (coerce_to_i32 of an int is identity). A
            // non-Gc value (tag `< 0xFFFC`: undefined/null/bool/int/Number) adds no Gc edge, so
            // the barrier the helper performs is unnecessary — write straight to
            // `obj.slots[index]`. Requires an Object receiver (else the helper throws #1009);
            // mode 0 (typed coercion), Gc values, non-object receivers, and native all take the
            // helper. `BailIfPerr` follows: the inline write stashes nothing, so it falls through.
            #[cfg(target_arch = "wasm32")]
            let inlined = match (slots_layout(), mode) {
                (Some((slots_ptr_off, _)), 1) | (Some((slots_ptr_off, _)), 2) => {
                    body.instruction(&I::LocalSet(SCRATCH64)); // value
                    body.instruction(&I::LocalSet(SCRATCH64_B)); // object
                    // guard: receiver is an Object (tag 0xFFFD).
                    body.instruction(&I::LocalGet(SCRATCH64_B));
                    body.instruction(&I::I64Const(48));
                    body.instruction(&I::I64ShrU);
                    body.instruction(&I::I64Const(0xFFFD));
                    body.instruction(&I::I64Eq);
                    // value guard: mode 1 → int (tag 0xFFFB); mode 2 → non-Gc (tag < 0xFFFC).
                    body.instruction(&I::LocalGet(SCRATCH64));
                    body.instruction(&I::I64Const(48));
                    body.instruction(&I::I64ShrU);
                    if mode == 1 {
                        body.instruction(&I::I64Const(0xFFFB));
                        body.instruction(&I::I64Eq);
                    } else {
                        body.instruction(&I::I64Const(0xFFFC));
                        body.instruction(&I::I64LtU);
                    }
                    body.instruction(&I::I32And);
                    body.instruction(&I::If(BlockType::Empty));
                    // in-range write (verifier guarantees the slot exists) — no barrier.
                    body.instruction(&I::LocalGet(SCRATCH64_B));
                    body.instruction(&I::I32WrapI64);
                    body.instruction(&I::I32Load(mem_arg(slots_ptr_off as u64, 2))); // slots data ptr
                    body.instruction(&I::LocalGet(SCRATCH64));
                    body.instruction(&I::I64Store(mem_arg(index as u64 * 8, 3)));
                    body.instruction(&I::Else);
                    body.instruction(&I::LocalGet(SCRATCH64_B));
                    body.instruction(&I::I32Const(index as i32));
                    body.instruction(&I::LocalGet(SCRATCH64));
                    body.instruction(&I::I32Const(mode));
                    call_helper(body, G_SETSLOT, TY_SETSLOT);
                    body.instruction(&I::End);
                    true
                }
                _ => false,
            };
            #[cfg(not(target_arch = "wasm32"))]
            let inlined = false;
            if !inlined {
                body.instruction(&I::LocalSet(SCRATCH64));
                body.instruction(&I::I32Const(index as i32));
                body.instruction(&I::LocalGet(SCRATCH64));
                body.instruction(&I::I32Const(mode));
                call_helper(body, G_SETSLOT, TY_SETSLOT);
            }
        }
        JitOp::BinOp(code) => {
            // Stack: [.., a, b]. Inline the int-int fast path for ops whose result is an
            // EXACT int/bool (no coercion, no overflow-to-double): comparisons, bitwise, and
            // the wrapping `*_i` ops. That removes the `call_indirect` to the `binop` helper
            // for the common loop-hot case (`i < n`, counters). Everything else (add/sub/mul
            // overflow, divide/modulo, shifts, string concat) still calls the helper.
            use crate::helpers::binop_code as bc;
            // Double×double fast path for numeric arithmetic: when BOTH operands are inline
            // `f64` (`bits & BOX_MARK != BOX_MARK`), the result is unambiguously `Number`
            // (float arithmetic never yields int/string/object), so compute it NATIVELY —
            // no `binop` helper `call_indirect`. This is the float-loop analogue of the int
            // fast path below; int/mixed/string operands still take the helper.
            const BOX_MARK: i64 = 0xFFF8_0000_0000_0000u64 as i64;
            const CANON_NAN: i64 = 0x7FF8_0000_0000_0000;
            let double_op: Option<I> = match code {
                bc::ADD => Some(I::F64Add),
                bc::SUBTRACT => Some(I::F64Sub),
                bc::MULTIPLY => Some(I::F64Mul),
                bc::DIVIDE => Some(I::F64Div),
                _ => None,
            };
            if let Some(fop) = double_op {
                body.instruction(&I::LocalSet(SCRATCH64)); // b (top)
                body.instruction(&I::LocalSet(SCRATCH64_B)); // a
                // guard: is_f64(a) & is_f64(b), where is_f64(x) = (x & BOX_MARK) != BOX_MARK.
                for l in [SCRATCH64_B, SCRATCH64] {
                    body.instruction(&I::LocalGet(l));
                    body.instruction(&I::I64Const(BOX_MARK));
                    body.instruction(&I::I64And);
                    body.instruction(&I::I64Const(BOX_MARK));
                    body.instruction(&I::I64Ne);
                }
                body.instruction(&I::I32And);
                body.instruction(&I::If(BlockType::Result(ValType::I64)));
                // both f64 → r = reinterpret(a) OP reinterpret(b), reinterpret back to bits.
                body.instruction(&I::LocalGet(SCRATCH64_B));
                body.instruction(&I::F64ReinterpretI64);
                body.instruction(&I::LocalGet(SCRATCH64));
                body.instruction(&I::F64ReinterpretI64);
                body.instruction(&fop);
                body.instruction(&I::I64ReinterpretF64);
                // Canonicalize a NaN result: a raw wasm NaN whose sign+exponent+quiet bits are
                // all set (`bits & BOX_MARK == BOX_MARK`) would alias the box space — replace
                // it with CANON_NAN so it reads back as `Number(NaN)`. (Finite/±Inf never hit
                // this; a non-colliding NaN already unpacks to `Number(NaN)`.)
                body.instruction(&I::LocalTee(SCRATCH64)); // rb (b no longer needed)
                body.instruction(&I::I64Const(BOX_MARK));
                body.instruction(&I::I64And);
                body.instruction(&I::I64Const(BOX_MARK));
                body.instruction(&I::I64Eq);
                body.instruction(&I::If(BlockType::Result(ValType::I64)));
                body.instruction(&I::I64Const(CANON_NAN));
                body.instruction(&I::Else);
                body.instruction(&I::LocalGet(SCRATCH64));
                body.instruction(&I::End);
                body.instruction(&I::Else);
                // Not both f64. For int arithmetic (ADD/SUB/MUL of two ints), compute in i64
                // and box as Integer when it fits i32, else as Number — EXACTLY like the
                // helper's `checked_{add,sub,mul}` (overflow → `(a as i64 OP b as i64) as f64`).
                // DIVIDE and every non-int case go to the helper.
                let int_arith: Option<I> = match code {
                    bc::ADD => Some(I::I64Add),
                    bc::SUBTRACT => Some(I::I64Sub),
                    bc::MULTIPLY => Some(I::I64Mul),
                    _ => None,
                };
                if let Some(iop) = int_arith {
                    // both int?  (a>>48 == 0xFFFB) & (b>>48 == 0xFFFB)
                    body.instruction(&I::LocalGet(SCRATCH64_B));
                    body.instruction(&I::I64Const(48));
                    body.instruction(&I::I64ShrU);
                    body.instruction(&I::I64Const(0xFFFB));
                    body.instruction(&I::I64Eq);
                    body.instruction(&I::LocalGet(SCRATCH64));
                    body.instruction(&I::I64Const(48));
                    body.instruction(&I::I64ShrU);
                    body.instruction(&I::I64Const(0xFFFB));
                    body.instruction(&I::I64Eq);
                    body.instruction(&I::I32And);
                    body.instruction(&I::If(BlockType::Result(ValType::I64)));
                    // r = (a as i64) OP (b as i64) — i32 operands, so no i64 overflow.
                    body.instruction(&I::LocalGet(SCRATCH64_B));
                    body.instruction(&I::I32WrapI64);
                    body.instruction(&I::I64ExtendI32S);
                    body.instruction(&I::LocalGet(SCRATCH64));
                    body.instruction(&I::I32WrapI64);
                    body.instruction(&I::I64ExtendI32S);
                    body.instruction(&iop);
                    body.instruction(&I::LocalTee(SCRATCH64_B)); // r (a no longer needed)
                    // fits i32?  (r >= i32::MIN) & (r <= i32::MAX)
                    body.instruction(&I::I64Const(i32::MIN as i64));
                    body.instruction(&I::I64GeS);
                    body.instruction(&I::LocalGet(SCRATCH64_B));
                    body.instruction(&I::I64Const(i32::MAX as i64));
                    body.instruction(&I::I64LeS);
                    body.instruction(&I::I32And);
                    body.instruction(&I::If(BlockType::Result(ValType::I64)));
                    // Integer(r): low 32 bits (the i32 payload) | INT_MARK.
                    body.instruction(&I::LocalGet(SCRATCH64_B));
                    body.instruction(&I::I64Const(0xFFFF_FFFF));
                    body.instruction(&I::I64And);
                    body.instruction(&I::I64Const(VALUE_INT_MARK as i64));
                    body.instruction(&I::I64Or);
                    body.instruction(&I::Else);
                    // Number(r as f64) — the overflow case is a finite, non-box-colliding double.
                    body.instruction(&I::LocalGet(SCRATCH64_B));
                    body.instruction(&I::F64ConvertI64S);
                    body.instruction(&I::I64ReinterpretF64);
                    body.instruction(&I::End);
                    body.instruction(&I::Else);
                    body.instruction(&I::LocalGet(SCRATCH64_B));
                    body.instruction(&I::LocalGet(SCRATCH64));
                    body.instruction(&I::I32Const(code));
                    call_helper(body, G_BINOP, TY_BINOP);
                    body.instruction(&I::End);
                } else {
                    body.instruction(&I::LocalGet(SCRATCH64_B));
                    body.instruction(&I::LocalGet(SCRATCH64));
                    body.instruction(&I::I32Const(code));
                    call_helper(body, G_BINOP, TY_BINOP);
                }
                body.instruction(&I::End); // close the both-f64 `if`
                return Some(()); // handled; skip the int/bool inline below
            }
            // URSHIFT on two ints: the UNSIGNED shift yields a `u32` — boxed as `int` when it
            // fits i32 (high bit clear), else `Number` (like `u32.into()`). Not both int → helper.
            if code == bc::URSHIFT {
                body.instruction(&I::LocalSet(SCRATCH64)); // b (count)
                body.instruction(&I::LocalSet(SCRATCH64_B)); // a
                for l in [SCRATCH64_B, SCRATCH64] {
                    body.instruction(&I::LocalGet(l));
                    body.instruction(&I::I64Const(48));
                    body.instruction(&I::I64ShrU);
                    body.instruction(&I::I64Const(0xFFFB));
                    body.instruction(&I::I64Eq);
                }
                body.instruction(&I::I32And);
                body.instruction(&I::If(BlockType::Result(ValType::I64)));
                // r_u32 = a >>> (count & 31); wasm `shr_u` already masks the count.
                body.instruction(&I::LocalGet(SCRATCH64_B));
                body.instruction(&I::I32WrapI64);
                body.instruction(&I::LocalGet(SCRATCH64));
                body.instruction(&I::I32WrapI64);
                body.instruction(&I::I32ShrU);
                body.instruction(&I::I64ExtendI32U); // r_u32 as i64 (0..2^32-1)
                body.instruction(&I::LocalTee(SCRATCH64_B));
                body.instruction(&I::I64Const(0x8000_0000));
                body.instruction(&I::I64LtU); // fits i32 (< 2^31)?
                body.instruction(&I::If(BlockType::Result(ValType::I64)));
                body.instruction(&I::LocalGet(SCRATCH64_B));
                body.instruction(&I::I64Const(VALUE_INT_MARK as i64));
                body.instruction(&I::I64Or); // Integer(r_u32)
                body.instruction(&I::Else);
                body.instruction(&I::LocalGet(SCRATCH64_B));
                body.instruction(&I::F64ConvertI64U);
                body.instruction(&I::I64ReinterpretF64); // Number(r_u32 as f64)
                body.instruction(&I::End);
                body.instruction(&I::Else);
                body.instruction(&I::LocalGet(SCRATCH64_B));
                body.instruction(&I::LocalGet(SCRATCH64));
                body.instruction(&I::I32Const(code));
                call_helper(body, G_BINOP, TY_BINOP);
                body.instruction(&I::End);
                return Some(());
            }
            // `(instruction, result-is-bool)`; `None` → no inline path, plain helper call.
            let inline: Option<(I, bool)> = match code {
                bc::BITAND => Some((I::I32And, false)),
                bc::BITOR => Some((I::I32Or, false)),
                bc::BITXOR => Some((I::I32Xor, false)),
                bc::EQUALS | bc::STRICT_EQUALS => Some((I::I32Eq, true)),
                bc::LESS_THAN => Some((I::I32LtS, true)),
                bc::LESS_EQUALS => Some((I::I32LeS, true)),
                bc::GREATER_THAN => Some((I::I32GtS, true)),
                bc::GREATER_EQUALS => Some((I::I32GeS, true)),
                bc::ADD_I => Some((I::I32Add, false)),
                bc::SUBTRACT_I => Some((I::I32Sub, false)),
                bc::MULTIPLY_I => Some((I::I32Mul, false)),
                // Shifts on two ints: the i32 result is an exact `int` (wasm `shl`/`shr_s` use
                // the count mod 32 = AS3's `& 0x1F`). URSHIFT is handled below (uint → maybe
                // Number). `coerce_to_u32(int_count) & 0x1F == count & 31`, so the count is
                // correct regardless of the second operand's sign.
                bc::LSHIFT => Some((I::I32Shl, false)),
                bc::RSHIFT => Some((I::I32ShrS, false)),
                _ => None,
            };
            match inline {
                None => {
                    body.instruction(&I::I32Const(code));
                    call_helper(body, G_BINOP, TY_BINOP);
                }
                Some((op_instr, is_bool)) => {
                    // NaN-box marks: an `Integer` is `0xFFFB<<48 | (i as u32)`, a `Bool` is
                    // `0xFFFA<<48 | (b as u1)` (see core `value.rs`). A value is a packed int
                    // iff `bits >> 48 == 0xFFFB` (that pattern is uniquely `TAG_INT`).
                    const INT_MARK: i64 = 0xFFFB_0000_0000_0000u64 as i64;
                    const BOOL_MARK: i64 = 0xFFFA_0000_0000_0000u64 as i64;
                    let mark = if is_bool { BOOL_MARK } else { INT_MARK };
                    body.instruction(&I::LocalSet(SCRATCH64)); // b (top)
                    body.instruction(&I::LocalSet(SCRATCH64_B)); // a
                    // both int?  (a>>48 == 0xFFFB) & (b>>48 == 0xFFFB)
                    body.instruction(&I::LocalGet(SCRATCH64_B));
                    body.instruction(&I::I64Const(48));
                    body.instruction(&I::I64ShrU);
                    body.instruction(&I::I64Const(0xFFFB));
                    body.instruction(&I::I64Eq);
                    body.instruction(&I::LocalGet(SCRATCH64));
                    body.instruction(&I::I64Const(48));
                    body.instruction(&I::I64ShrU);
                    body.instruction(&I::I64Const(0xFFFB));
                    body.instruction(&I::I64Eq);
                    body.instruction(&I::I32And);
                    body.instruction(&I::If(BlockType::Result(ValType::I64)));
                    // both int → compute in i32, re-box.
                    body.instruction(&I::LocalGet(SCRATCH64_B));
                    body.instruction(&I::I32WrapI64); // a as i32 (low 32 = payload)
                    body.instruction(&I::LocalGet(SCRATCH64));
                    body.instruction(&I::I32WrapI64); // b as i32
                    body.instruction(&op_instr);
                    body.instruction(&I::I64ExtendI32U); // payload = result as u32 (zero-extend)
                    body.instruction(&I::I64Const(mark));
                    body.instruction(&I::I64Or);
                    body.instruction(&I::Else);
                    // Not both int. For a COMPARISON, try both-DOUBLE (a native `f64` compare,
                    // NaN-correct: every f64 compare with NaN is false, matching AS3) before the
                    // helper; other ops (bitwise/`*_i`) go straight to the helper.
                    let double_cmp: Option<I> = match code {
                        bc::EQUALS | bc::STRICT_EQUALS => Some(I::F64Eq),
                        bc::LESS_THAN => Some(I::F64Lt),
                        bc::LESS_EQUALS => Some(I::F64Le),
                        bc::GREATER_THAN => Some(I::F64Gt),
                        bc::GREATER_EQUALS => Some(I::F64Ge),
                        _ => None,
                    };
                    if let Some(fcmp) = double_cmp {
                        for l in [SCRATCH64_B, SCRATCH64] {
                            body.instruction(&I::LocalGet(l));
                            body.instruction(&I::I64Const(BOX_MARK));
                            body.instruction(&I::I64And);
                            body.instruction(&I::I64Const(BOX_MARK));
                            body.instruction(&I::I64Ne);
                        }
                        body.instruction(&I::I32And);
                        body.instruction(&I::If(BlockType::Result(ValType::I64)));
                        body.instruction(&I::LocalGet(SCRATCH64_B));
                        body.instruction(&I::F64ReinterpretI64);
                        body.instruction(&I::LocalGet(SCRATCH64));
                        body.instruction(&I::F64ReinterpretI64);
                        body.instruction(&fcmp);
                        body.instruction(&I::I64ExtendI32U);
                        body.instruction(&I::I64Const(BOOL_MARK));
                        body.instruction(&I::I64Or);
                        body.instruction(&I::Else);
                        body.instruction(&I::LocalGet(SCRATCH64_B));
                        body.instruction(&I::LocalGet(SCRATCH64));
                        body.instruction(&I::I32Const(code));
                        call_helper(body, G_BINOP, TY_BINOP);
                        body.instruction(&I::End);
                    } else {
                        body.instruction(&I::LocalGet(SCRATCH64_B));
                        body.instruction(&I::LocalGet(SCRATCH64));
                        body.instruction(&I::I32Const(code));
                        call_helper(body, G_BINOP, TY_BINOP);
                    }
                    body.instruction(&I::End);
                }
            }
        }
        JitOp::BinOpNum(code) => {
            // Stack: [.., a, b] — both PROVEN canonical inline `f64`s (translate's repr
            // analysis). The unguarded core of `BinOp`'s f64 fast path: reinterpret, compute,
            // canonicalize a box-colliding NaN result. No guard, no helper, no `BailIfError`.
            use crate::helpers::binop_code as bc;
            const BOX_MARK: i64 = 0xFFF8_0000_0000_0000u64 as i64;
            const CANON_NAN: i64 = 0x7FF8_0000_0000_0000;
            let fop = match code {
                bc::ADD => I::F64Add,
                bc::SUBTRACT => I::F64Sub,
                bc::MULTIPLY => I::F64Mul,
                bc::DIVIDE => I::F64Div,
                _ => return None, // translate emits `BinOpNum` only for these four
            };
            body.instruction(&I::LocalSet(SCRATCH64)); // b (top)
            body.instruction(&I::LocalSet(SCRATCH64_B)); // a
            body.instruction(&I::LocalGet(SCRATCH64_B));
            body.instruction(&I::F64ReinterpretI64);
            body.instruction(&I::LocalGet(SCRATCH64));
            body.instruction(&I::F64ReinterpretI64);
            body.instruction(&fop);
            body.instruction(&I::I64ReinterpretF64);
            // Canonicalize a NaN result whose bits alias the box space (see `BinOp`).
            body.instruction(&I::LocalTee(SCRATCH64));
            body.instruction(&I::I64Const(BOX_MARK));
            body.instruction(&I::I64And);
            body.instruction(&I::I64Const(BOX_MARK));
            body.instruction(&I::I64Eq);
            body.instruction(&I::If(BlockType::Result(ValType::I64)));
            body.instruction(&I::I64Const(CANON_NAN));
            body.instruction(&I::Else);
            body.instruction(&I::LocalGet(SCRATCH64));
            body.instruction(&I::End);
        }
        JitOp::BinOpCmp(code, is_f64) => {
            // Stack: [.., a, b] — both PROVEN the same numeric repr. Native same-type compare
            // (unguarded core of `BinOp`'s comparison fast paths) → a `Bool` box. No `BailIfError`.
            use crate::helpers::binop_code as bc;
            const BOOL_MARK: i64 = 0xFFFA_0000_0000_0000u64 as i64;
            body.instruction(&I::LocalSet(SCRATCH64)); // b (top)
            body.instruction(&I::LocalSet(SCRATCH64_B)); // a
            body.instruction(&I::LocalGet(SCRATCH64_B));
            if is_f64 {
                body.instruction(&I::F64ReinterpretI64);
            } else {
                body.instruction(&I::I32WrapI64); // Int box → i32 payload
            }
            body.instruction(&I::LocalGet(SCRATCH64));
            if is_f64 {
                body.instruction(&I::F64ReinterpretI64);
            } else {
                body.instruction(&I::I32WrapI64);
            }
            let cmp = match (code, is_f64) {
                (bc::EQUALS, false) | (bc::STRICT_EQUALS, false) => I::I32Eq,
                (bc::LESS_THAN, false) => I::I32LtS,
                (bc::LESS_EQUALS, false) => I::I32LeS,
                (bc::GREATER_THAN, false) => I::I32GtS,
                (bc::GREATER_EQUALS, false) => I::I32GeS,
                (bc::EQUALS, true) | (bc::STRICT_EQUALS, true) => I::F64Eq,
                (bc::LESS_THAN, true) => I::F64Lt,
                (bc::LESS_EQUALS, true) => I::F64Le,
                (bc::GREATER_THAN, true) => I::F64Gt,
                (bc::GREATER_EQUALS, true) => I::F64Ge,
                _ => return None, // translate emits `BinOpCmp` only for the six comparisons
            };
            body.instruction(&cmp); // → i32 boolean (0/1)
            body.instruction(&I::I64ExtendI32U);
            body.instruction(&I::I64Const(BOOL_MARK));
            body.instruction(&I::I64Or); // Bool box
        }
        JitOp::UnOp(code) => {
            // Stack: [.., a]. Inline `coerce_i`/`coerce_u` when `a` is already an int (the
            // common redundant-coerce case): `ToInt32(int)` is identity; `ToUint32(int)` is
            // identity for i >= 0, else `Number(i as u32)`. A double operand needs full
            // ToInt32/ToUint32 (non-trapping float→int, unavailable on this target) → helper.
            use crate::helpers::unop_code as uc;
            if code == uc::COERCE_I || code == uc::COERCE_U {
                body.instruction(&I::LocalSet(SCRATCH64)); // a
                body.instruction(&I::LocalGet(SCRATCH64));
                body.instruction(&I::I64Const(48));
                body.instruction(&I::I64ShrU);
                body.instruction(&I::I64Const(0xFFFB));
                body.instruction(&I::I64Eq); // a is int?
                body.instruction(&I::If(BlockType::Result(ValType::I64)));
                if code == uc::COERCE_I {
                    body.instruction(&I::LocalGet(SCRATCH64)); // ToInt32(int) = the int, unchanged
                } else {
                    // ToUint32: i >= 0 → Integer(i) unchanged; i < 0 → Number(i as u32).
                    body.instruction(&I::LocalGet(SCRATCH64));
                    body.instruction(&I::I64Const(0xFFFF_FFFF));
                    body.instruction(&I::I64And); // payload = i as u32 (zero-extended)
                    body.instruction(&I::LocalTee(SCRATCH64_B));
                    body.instruction(&I::I64Const(0x8000_0000));
                    body.instruction(&I::I64LtU); // i >= 0 (fits i32)?
                    body.instruction(&I::If(BlockType::Result(ValType::I64)));
                    body.instruction(&I::LocalGet(SCRATCH64)); // Integer(i) unchanged
                    body.instruction(&I::Else);
                    body.instruction(&I::LocalGet(SCRATCH64_B));
                    body.instruction(&I::F64ConvertI64U);
                    body.instruction(&I::I64ReinterpretF64); // Number(i as u32 as f64)
                    body.instruction(&I::End);
                }
                body.instruction(&I::Else);
                body.instruction(&I::LocalGet(SCRATCH64));
                body.instruction(&I::I32Const(code));
                call_helper(body, G_UNOP, TY_UNOP);
                body.instruction(&I::End);
            } else {
                // Stack: [.., a]. Push the op-code and `unop(a, code)`.
                body.instruction(&I::I32Const(code));
                call_helper(body, G_UNOP, TY_UNOP);
            }
        }
        JitOp::Coerce(class_ptr) => {
            // Stack: [.., value]. `cr(value, class_ptr)` coerces, leaving it on the stack.
            body.instruction(&I::I64Const(class_ptr as i64));
            call_helper(body, G_CR, TY_HELPER2);
        }
        JitOp::Swap => {
            body.instruction(&I::LocalSet(SCRATCH64)); // b
            body.instruction(&I::LocalSet(SCRATCH64_B)); // a
            body.instruction(&I::LocalGet(SCRATCH64)); // b
            body.instruction(&I::LocalGet(SCRATCH64_B)); // a
        }
        JitOp::BailIfError => {
            // Stack: [.., result] from a throwing i64-returning helper. If it returned the
            // error SENTINEL, `Return` it (try_enter's `take_error` surfaces the stashed
            // error) — otherwise restore the result and fall through. NO `call_indirect`
            // (unlike the old `perr` flag call): just a compare on the value already on hand.
            body.instruction(&I::LocalTee(SCRATCH64));
            body.instruction(&I::I64Const(SENTINEL_BITS as i64));
            body.instruction(&I::I64Eq);
            body.instruction(&I::If(BlockType::Empty));
            body.instruction(&I::I64Const(SENTINEL_BITS as i64));
            body.instruction(&I::Return);
            body.instruction(&I::End);
            body.instruction(&I::LocalGet(SCRATCH64)); // restore the (non-error) result
        }
        JitOp::BailIfPerr => {
            // For a VOID helper (no result to sentinel-check): `if perr() { return undefined }`.
            // The operand stack is untouched on the no-error path; the taken branch diverges.
            call_helper(body, G_PERR, TY_PERR);
            body.instruction(&I::If(BlockType::Empty));
            body.instruction(&I::I64Const(UNDEFINED_BITS as i64));
            body.instruction(&I::Return);
            body.instruction(&I::End);
        }
        JitOp::CoerceReturn(class_ptr) => {
            // Stack: [.., value]. `cr(value, class_ptr)` coerces (reifying an Activation),
            // then return the coerced value.
            body.instruction(&I::I64Const(class_ptr as i64));
            call_helper(body, G_CR, TY_HELPER2);
            body.instruction(&I::Return);
        }
        JitOp::Pop => {
            body.instruction(&I::Drop);
        }
        JitOp::Dup => {
            body.instruction(&I::LocalTee(SCRATCH64));
            body.instruction(&I::LocalGet(SCRATCH64));
        }
        JitOp::ReturnValue => {
            body.instruction(&I::Return);
        }
        JitOp::ReturnVoid => {
            body.instruction(&I::I64Const(UNDEFINED_BITS as i64));
            body.instruction(&I::Return);
        }
    }
    let _ = (P_ENV, P_ARGC);
    Some(())
}

/// Emits an indirect call to a helper: pushes its table index (an imported i32 global) and
/// `call_indirect`s through `T_HELPERS`. The call args must already be on the stack; the
/// index goes on top last (as `call_indirect` requires).
fn call_helper(body: &mut Function, global: u32, type_index: u32) {
    body.instruction(&Instruction::GlobalGet(global));
    body.instruction(&Instruction::CallIndirect { type_index, table_index: T_HELPERS });
}

/// Inline `coerce_to_boolean` for a branch condition already saved in `SCRATCH64`; leaves the
/// i32 boolean on the stack (what the `truthy` helper returned). Handles the packed primitives
/// directly — Number (`!(x == ±0 || NaN)`), Bool (its bit), int (`!= 0`), Object (always true) —
/// and falls to the helper for undefined/null/String/boxed-double. Pure NaN-box bit logic (no
/// memory layout), so it runs on native too and is validated by `RUFFLE_JIT=verify`.
fn emit_truthy(body: &mut Function) {
    use Instruction as I;
    const BOX_MARK: i64 = 0xFFF8_0000_0000_0000u64 as i64;
    // is_f64: (cond & BOX_MARK) != BOX_MARK.
    body.instruction(&I::LocalGet(SCRATCH64));
    body.instruction(&I::I64Const(BOX_MARK));
    body.instruction(&I::I64And);
    body.instruction(&I::I64Const(BOX_MARK));
    body.instruction(&I::I64Ne);
    body.instruction(&I::If(BlockType::Result(ValType::I32)));
    // Number: truthy = !(x == ±0 || x is NaN). `bits << 1 == 0` catches +0 and -0; `x != x` is
    // NaN — neither needs an f64 constant.
    body.instruction(&I::LocalGet(SCRATCH64));
    body.instruction(&I::I64Const(1));
    body.instruction(&I::I64Shl);
    body.instruction(&I::I64Eqz); // ±0
    body.instruction(&I::LocalGet(SCRATCH64));
    body.instruction(&I::F64ReinterpretI64);
    body.instruction(&I::LocalGet(SCRATCH64));
    body.instruction(&I::F64ReinterpretI64);
    body.instruction(&I::F64Ne); // NaN
    body.instruction(&I::I32Or);
    body.instruction(&I::I32Eqz);
    body.instruction(&I::Else);
    // Boxed: dispatch on tag = cond >> 48.
    body.instruction(&I::LocalGet(SCRATCH64));
    body.instruction(&I::I64Const(48));
    body.instruction(&I::I64ShrU);
    body.instruction(&I::LocalSet(SCRATCH64_B)); // tag
    body.instruction(&I::LocalGet(SCRATCH64_B));
    body.instruction(&I::I64Const(0xFFFA)); // bool
    body.instruction(&I::I64Eq);
    body.instruction(&I::If(BlockType::Result(ValType::I32)));
    body.instruction(&I::LocalGet(SCRATCH64));
    body.instruction(&I::I32WrapI64); // the bool bit (payload is 0/1)
    body.instruction(&I::Else);
    body.instruction(&I::LocalGet(SCRATCH64_B));
    body.instruction(&I::I64Const(0xFFFB)); // int
    body.instruction(&I::I64Eq);
    body.instruction(&I::If(BlockType::Result(ValType::I32)));
    body.instruction(&I::LocalGet(SCRATCH64));
    body.instruction(&I::I32WrapI64);
    body.instruction(&I::I32Const(0));
    body.instruction(&I::I32Ne); // int != 0
    body.instruction(&I::Else);
    body.instruction(&I::LocalGet(SCRATCH64_B));
    body.instruction(&I::I64Const(0xFFFD)); // object
    body.instruction(&I::I64Eq);
    body.instruction(&I::If(BlockType::Result(ValType::I32)));
    body.instruction(&I::I32Const(1));
    body.instruction(&I::Else);
    // undefined / null / String / boxed-double → helper.
    body.instruction(&I::LocalGet(SCRATCH64));
    call_helper(body, G_TRUTHY, TY_TRUTHY);
    body.instruction(&I::End);
    body.instruction(&I::End);
    body.instruction(&I::End);
    body.instruction(&I::End);
}

/// Emits the basic-block dispatch loop: a `loop` wrapping nested `block`s, dispatched by a
/// `br_table` on the `STATE` local (current block index). Each block body runs, then its
/// terminator sets `STATE` to the successor and `br`s back to the loop (re-dispatch), or the
/// body already returned. Requires every block to enter with an empty operand stack.
///
/// Layout (n blocks): `loop { block B0 { … block B_{n-1} { block Bdef {
/// br_table[B0..B_{n-1}] Bdef } unreachable } <blk n-1> } … <blk 0> } <guard>`. Blocks are
/// emitted in REVERSE, so at block `k` the enclosing `loop` sits at branch depth `k`.
fn emit_dispatch(body: &mut Function, blocks: &[Block], promote: &[Option<(u32, RegKind)>]) -> Option<()> {
    use Instruction as I;
    let n = blocks.len();
    if n == 0 {
        return None;
    }
    body.instruction(&I::Loop(BlockType::Empty));
    for _ in 0..=n {
        body.instruction(&I::Block(BlockType::Empty)); // n target blocks + 1 default
    }
    // Dispatch: state `k` branches to `B_k` (branch depth `n - k`); out-of-range → default.
    body.instruction(&I::LocalGet(STATE));
    let targets: Vec<u32> = (0..n).map(|k| (n - k) as u32).collect();
    body.instruction(&I::BrTable(targets.into(), 0));
    body.instruction(&I::End); // Bdef
    body.instruction(&I::Unreachable); // state is always valid → never reached
    for k in (0..n).rev() {
        body.instruction(&I::End); // close B_k; block k's code follows
        // Reload the `entry_depth` values that crossed into this block from its predecessor.
        for d in 0..blocks[k].entry_depth {
            body.instruction(&I::LocalGet(SPILL_BASE + d as u32));
        }
        for &op in &blocks[k].ops {
            emit_op(body, op, promote)?;
        }
        match blocks[k].term {
            Term::Return => {} // the body already emitted a `Return`
            Term::Jump(j) => {
                if j >= n {
                    return None;
                }
                // Spill the crossing operand values (the successor's `entry_depth`) top-down.
                for d in (0..blocks[j].entry_depth).rev() {
                    body.instruction(&I::LocalSet(SPILL_BASE + d as u32));
                }
                body.instruction(&I::I32Const(j as i32));
                body.instruction(&I::LocalSet(STATE));
                body.instruction(&I::Br(k as u32)); // the loop is `k` levels out
            }
            Term::Cond { on_true, on_false } => {
                if on_true >= n || on_false >= n {
                    return None;
                }
                // Stack: [crossing…, cond]. Pop the condition, spill the crossing values (both
                // successors share the same `entry_depth`), then `truthy(cond)` → branch.
                let cross = blocks[on_true].entry_depth;
                body.instruction(&I::LocalSet(SCRATCH64)); // pop cond
                for d in (0..cross).rev() {
                    body.instruction(&I::LocalSet(SPILL_BASE + d as u32));
                }
                emit_truthy(body); // cond is in SCRATCH64 → i32 boolean on the stack
                body.instruction(&I::If(BlockType::Empty));
                body.instruction(&I::I32Const(on_true as i32));
                body.instruction(&I::LocalSet(STATE));
                body.instruction(&I::Br((k + 1) as u32)); // +1: inside the `if`
                body.instruction(&I::Else);
                body.instruction(&I::I32Const(on_false as i32));
                body.instruction(&I::LocalSet(STATE));
                body.instruction(&I::Br((k + 1) as u32));
                body.instruction(&I::End);
            }
        }
    }
    body.instruction(&I::End); // close the loop
    Some(())
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use wasmtime::{
        Caller, Engine, Func, Global, GlobalType, HeapType, Instance, Memory, MemoryType,
        Module as WtModule, Mutability, Ref, RefType, Store, Table, TableType, Val, ValType as WtV,
    };

    /// Compiles `ops`, writes `frame` at offset 0 of the imported memory, calls
    /// `run(0, argc, 0)`, and returns the raw result bits. Helpers are provided via an
    /// imported funcref table + index globals — the same `call_indirect` shape the runners use.
    fn run(ops: &[JitOp], frame: &[u64]) -> u64 {
        run_blocks(vec![Block { ops: ops.to_vec(), term: Term::Return, entry_depth: 0 }], frame)
    }

    /// Like [`run`] but for an explicit block list — exercises the dispatch loop.
    fn run_blocks(blocks: Vec<Block>, frame: &[u64]) -> u64 {
        run_blocks_p(blocks, frame, &[])
    }

    /// [`run_blocks`] with an explicit promoted-locals set (Phase 3/4).
    fn run_blocks_p(blocks: Vec<Block>, frame: &[u64], promoted: &[Promotion]) -> u64 {
        let bytes = compile(&blocks, frame.len(), promoted, &[]).expect("compiles");
        let engine = Engine::default();
        let module = WtModule::new(&engine, &bytes).expect("valid module");
        let mut store = Store::new(&engine, ());
        // Test helpers: `gs` returns `obj_bits + slot_id`; `cr` is identity (no coercion).
        let gs = Func::wrap(&mut store, |o: i64, s: i64| -> i64 { o.wrapping_add(s) });
        let cr = Func::wrap(&mut store, |v: i64, _class: i64| -> i64 { v });
        let gp = Func::wrap(&mut store, |r: i64, mn: i64| -> i64 { r.wrapping_add(mn) });
        // `cp` reads the `argc` args from memory at `args_off` and returns receiver + Σargs
        // — proving the in-wasm `i64.store` arg-passing round-trips through the scratch.
        let cp = Func::wrap(
            &mut store,
            |mut caller: Caller<'_, ()>, r: i64, _mn: i64, off: i64, argc: i64| -> i64 {
                let mem = caller
                    .get_export("memory")
                    .and_then(|e| e.into_memory())
                    .expect("re-exported memory");
                let mut sum = r;
                for j in 0..argc as usize {
                    let mut b = [0u8; 8];
                    mem.read(&caller, off as usize + j * 8, &mut b).expect("read arg");
                    sum = sum.wrapping_add(i64::from_le_bytes(b));
                }
                sum
            },
        );
        // `perr` never signals an error in these codegen tests.
        let perr = Func::wrap(&mut store, || -> i32 { 0 });
        // `binop`/`unop` test stubs: `binop` returns a+b+op, `unop` returns a+op.
        let binop = Func::wrap(&mut store, |a: i64, b: i64, op: i32| -> i64 {
            a.wrapping_add(b).wrapping_add(op as i64)
        });
        let unop = Func::wrap(&mut store, |a: i64, op: i32| -> i64 { a.wrapping_add(op as i64) });
        // `truthy` test stub: nonzero bits → true.
        let truthy = Func::wrap(&mut store, |a: i64| -> i32 { (a != 0) as i32 });
        // `setprop`/`setslot` test stubs: void, no-op. `callmethod` stub: receiver + disp_id.
        let setprop = Func::wrap(&mut store, |_r: i64, _m: i64, _v: i64| {});
        let setslot = Func::wrap(&mut store, |_r: i64, _i: i32, _v: i64, _mode: i32| {});
        let callmethod = Func::wrap(&mut store, |r: i64, id: i32, _off: i64, _n: i32| -> i64 {
            r.wrapping_add(id as i64)
        });
        // `new_array` stub: returns the element count (proves argc reached it).
        let newarray = Func::wrap(&mut store, |_off: i64, n: i32| -> i64 { n as i64 });
        // scope-op stubs: `outer_scope` returns index; `script_globals`/`new_activation` echo ptr.
        let outerscope = Func::wrap(&mut store, |i: i32| -> i64 { i as i64 });
        let scriptglobals = Func::wrap(&mut store, |p: i64| -> i64 { p });
        let newactivation = Func::wrap(&mut store, |p: i64| -> i64 { p });
        // scope-stack stubs: no-ops; `get_scope` echoes the index.
        let pushscope = Func::wrap(&mut store, |_b: i64| {});
        let popscope = Func::wrap(&mut store, || {});
        let getscope = Func::wrap(&mut store, |i: i32| -> i64 { i as i64 });
        // `construct` stub: ctor + argc (proves args reached it).
        let construct = Func::wrap(&mut store, |ctor: i64, _off: i64, n: i32| -> i64 {
            ctor.wrapping_add(n as i64)
        });
        // `delete_property`/`is_type`/`as_type` stubs: sum the two operands.
        let delprop = Func::wrap(&mut store, |r: i64, m: i64| -> i64 { r.wrapping_add(m) });
        let istype = Func::wrap(&mut store, |v: i64, t: i64| -> i64 { v.wrapping_add(t) });
        let astype = Func::wrap(&mut store, |v: i64, c: i64| -> i64 { v.wrapping_add(c) });
        // super stubs.
        let getsuper = Func::wrap(&mut store, |r: i64, m: i64| -> i64 { r.wrapping_add(m) });
        let setsuper = Func::wrap(&mut store, |_r: i64, _m: i64, _v: i64| {});
        let callsuper = Func::wrap(&mut store, |r: i64, m: i64, _o: i64, n: i64| -> i64 {
            r.wrapping_add(m).wrapping_add(n)
        });
        let constructsuper = Func::wrap(&mut store, |r: i64, _o: i64, n: i32| -> i64 {
            r.wrapping_add(n as i64)
        });
        let callnative = Func::wrap(&mut store, |r: i64, m: i64, _o: i64, n: i64| -> i64 {
            r.wrapping_add(m).wrapping_add(n)
        });
        // `get/set_prop_index` stubs: sum the operands.
        let getpropfast = Func::wrap(&mut store, |o: i64, n: i64, m: i64| -> i64 {
            o.wrapping_add(n).wrapping_add(m)
        });
        let setpropfast =
            Func::wrap(&mut store, |o: i64, _n: i64, _v: i64, _m: i64| -> i64 { o });
        // New-op stubs: sum the scalar operands (proves the operands reached the slot).
        let op_in = Func::wrap(&mut store, |n: i64, v: i64| -> i64 { n.wrapping_add(v) });
        let nextvalue = Func::wrap(&mut store, |v: i64, i: i64| -> i64 { v.wrapping_add(i) });
        let nextname = Func::wrap(&mut store, |v: i64, i: i64| -> i64 { v.wrapping_add(i) });
        let hasnext = Func::wrap(&mut store, |v: i64, i: i64| -> i64 { v.wrapping_add(i) });
        let newfunction = Func::wrap(&mut store, |p: i64| -> i64 { p });
        let applytype = Func::wrap(&mut store, |base: i64, _o: i64, n: i32| -> i64 {
            base.wrapping_add(n as i64)
        });
        let constructslot = Func::wrap(&mut store, |s: i64, index: i32, _o: i64, n: i32| -> i64 {
            s.wrapping_add(index as i64).wrapping_add(n as i64)
        });
        let constructprop = Func::wrap(&mut store, |s: i64, m: i64, _o: i64, n: i64| -> i64 {
            s.wrapping_add(m).wrapping_add(n)
        });
        let call = Func::wrap(&mut store, |f: i64, r: i64, _o: i64, n: i64| -> i64 {
            f.wrapping_add(r).wrapping_add(n)
        });
        let newobject = Func::wrap(&mut store, |_o: i64, n: i32| -> i64 { n as i64 });
        let hasnext2 =
            Func::wrap(&mut store, |o: i32, i: i32, off: i64| -> i64 { o as i64 + i as i64 + off });
        let throw = Func::wrap(&mut store, |v: i64| -> i64 { v });
        let gschecked = Func::wrap(&mut store, |o: i64, s: i64| -> i64 { o.wrapping_add(s) });
        let mopload = Func::wrap(&mut store, |a: i64, c: i32| -> i64 { a.wrapping_add(c as i64) });
        let mopstore =
            Func::wrap(&mut store, |v: i64, a: i64, c: i32| -> i64 { v.wrapping_add(a).wrapping_add(c as i64) });
        let gp_ic = Func::wrap(&mut store, |r: i64, _m: i64, _c: i64| -> i64 { r });
        let dm_desc = Func::wrap(&mut store, || -> i64 { 0 });
        // Mirrors the `cp` stub (reads args, returns receiver + Σargs) so the CallProperty
        // memory-round-trip test still validates through the call-IC path; ignores mn + ic_addr.
        let call_prop_ic = Func::wrap(
            &mut store,
            |mut caller: Caller<'_, ()>, r: i64, _mn: i64, off: i64, argc: i64, _ic: i64| -> i64 {
                let mem = caller
                    .get_export("memory")
                    .and_then(|e| e.into_memory())
                    .expect("re-exported memory");
                let mut sum = r;
                for j in 0..argc as usize {
                    let mut b = [0u8; 8];
                    mem.read(&caller, off as usize + j * 8, &mut b).expect("read arg");
                    sum = sum.wrapping_add(i64::from_le_bytes(b));
                }
                sum
            },
        );
        // `call_method_ic` stub: receiver + disp_id + Σargs (proves disp_id + the args round-trip
        // both reach the helper through the call-IC path); ignores ic_addr.
        let call_method_ic = Func::wrap(
            &mut store,
            |mut caller: Caller<'_, ()>, r: i64, d: i64, off: i64, argc: i64, _ic: i64| -> i64 {
                let mem = caller
                    .get_export("memory")
                    .and_then(|e| e.into_memory())
                    .expect("re-exported memory");
                let mut sum = r.wrapping_add(d);
                for j in 0..argc as usize {
                    let mut b = [0u8; 8];
                    mem.read(&caller, off as usize + j * 8, &mut b).expect("read arg");
                    sum = sum.wrapping_add(i64::from_le_bytes(b));
                }
                sum
            },
        );
        // §8 in-WASM dispatch bracket stubs: `jit_enter` echoes `env` back as `prev`, `jit_leave`
        // is a no-op (these unit tests never emit the in-WASM call path, but the slots must exist
        // so the helper index globals line up with `NUM_HELPERS`).
        let jit_enter =
            Func::wrap(&mut store, |_env: i64, _fm: i64, _off: i64, _argc: i64| -> i64 { 0 });
        let jit_leave = Func::wrap(&mut store, || {});
        // Helper table (50 slots); index globals name each.
        let table = Table::new(
            &mut store,
            TableType::new(RefType::new(true, HeapType::Func), 50, Some(50)),
            Ref::Func(None),
        )
        .expect("helper table");
        let helpers_tbl = [
            gs, cr, gp, cp, perr, binop, unop, truthy, setprop, setslot, callmethod, newarray,
            outerscope, scriptglobals, newactivation, pushscope, popscope, getscope, construct,
            delprop, istype, astype, getsuper, setsuper, callsuper, constructsuper, callnative,
            getpropfast, setpropfast, op_in, nextvalue, nextname, hasnext, newfunction, applytype,
            constructslot, constructprop, call, newobject, hasnext2, throw, gschecked, mopload,
            mopstore, gp_ic, dm_desc, call_prop_ic, call_method_ic, jit_enter, jit_leave,
        ];
        for (i, f) in helpers_tbl.into_iter().enumerate() {
            table.set(&mut store, i as u64, Ref::Func(Some(f))).expect("set helper");
        }
        let idx = |store: &mut Store<()>, i: i32| {
            Global::new(store, GlobalType::new(WtV::I32, Mutability::Const), Val::I32(i))
                .expect("index global")
        };
        let g: Vec<_> = (0..50).map(|i| idx(&mut store, i)).collect();
        let mem = Memory::new(&mut store, MemoryType::new(FRAME_PAGES, Some(FRAME_PAGES)))
            .expect("frame memory");
        let mut imports: Vec<wasmtime::Extern> = vec![table.into()];
        imports.extend(g.into_iter().map(Into::into));
        imports.push(mem.into());
        let instance = Instance::new(&mut store, &module, &imports).expect("instantiates");
        let mut buf = Vec::with_capacity(frame.len() * 8);
        for &v in frame {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        mem.write(&mut store, 0, &buf).expect("write frame");
        let run = instance
            .get_typed_func::<(i32, i32, i32), i64>(&mut store, "run")
            .expect("run export");
        let argc = frame.len().saturating_sub(1) as i32;
        run.call(&mut store, (0, argc, 0)).expect("runs") as u64
    }

    #[test]
    fn getlocal_returns_the_argument_verbatim() {
        let out = run(
            &[JitOp::GetLocal(1), JitOp::ReturnValue],
            &[UNDEFINED_BITS, VALUE_INT_MARK | 42],
        );
        assert_eq!(out, VALUE_INT_MARK | 42);
    }

    #[test]
    fn pushbits_returns_the_constant() {
        let out = run(&[JitOp::PushBits(VALUE_INT_MARK | 7), JitOp::ReturnValue], &[UNDEFINED_BITS]);
        assert_eq!(out, VALUE_INT_MARK | 7);
    }

    #[test]
    fn setlocal_then_getlocal_roundtrips() {
        let out = run(
            &[
                JitOp::PushBits(NULL_BITS),
                JitOp::SetLocal(2),
                JitOp::GetLocal(2),
                JitOp::ReturnValue,
            ],
            &[UNDEFINED_BITS, VALUE_INT_MARK | 1, UNDEFINED_BITS],
        );
        assert_eq!(out, NULL_BITS);
    }

    #[test]
    fn dup_leaves_a_copy() {
        let out = run(
            &[JitOp::GetLocal(1), JitOp::Dup, JitOp::Pop, JitOp::ReturnValue],
            &[UNDEFINED_BITS, VALUE_BOOL_MARK | 1],
        );
        assert_eq!(out, VALUE_BOOL_MARK | 1);
    }

    #[test]
    fn returnvoid_yields_undefined() {
        let out = run(&[JitOp::ReturnVoid], &[UNDEFINED_BITS]);
        assert_eq!(out, UNDEFINED_BITS);
    }

    #[test]
    fn getslot_calls_the_helper_with_receiver_and_id() {
        // getlocal1 (receiver=100); getslot 5; returnvalue → passthrough gs = 100 + 5.
        let out = run(
            &[JitOp::GetLocal(1), JitOp::GetSlot(5), JitOp::ReturnValue],
            &[UNDEFINED_BITS, 100],
        );
        assert_eq!(out, 105);
    }

    #[test]
    fn callproperty_stores_args_to_memory_and_calls_the_helper() {
        // getlocal1 (receiver=100); push two args (i64.store'd into scratch); callproperty
        // (mn=7, 2); bail (perr=0, no-op); returnvalue → test `cp` reads the args back and
        // returns receiver + Σargs = 100 + 11 + 22 = 133, proving the memory round-trip.
        let out = run(
            &[
                JitOp::GetLocal(1),
                JitOp::PushBits(11),
                JitOp::PushBits(22),
                JitOp::CallProperty(7, 2),
                JitOp::BailIfError,
                JitOp::ReturnValue,
            ],
            &[UNDEFINED_BITS, 100],
        );
        assert_eq!(out, 133);
    }

    #[test]
    fn dump_inwasm_wat() {
        // Throwaway: emit a module whose body does a `callmethod` via the in-WASM IC shape and
        // write the bytes so `wasm-tools print` can show the exact guard. (Debug aid.)
        FORCE_INWASM_CALL_IC.with(|c| c.set(true));
        let blocks = vec![Block {
            ops: vec![
                JitOp::GetLocal(1),
                JitOp::PushBits(5),
                JitOp::CallMethod(3, 1),
                JitOp::BailIfError,
                JitOp::ReturnValue,
            ],
            term: Term::Return,
            entry_depth: 0,
        }];
        let bytes = compile(&blocks, 2, &[], &[]).expect("compiles");
        FORCE_INWASM_CALL_IC.with(|c| c.set(false));
        std::fs::write("/tmp/claude-1000/-home-jaroslavb-rufflepostatnicich-jestedalsiruffle-jitruffle/9fc2f868-b600-4693-b922-7a4947cd13f8/scratchpad/inwasm.wasm", &bytes).unwrap();
    }

    #[test]
    fn inwasm_call_ic_module_validates_and_falls_back() {
        // §8: force the WEB in-WASM call-IC shape on native so `WtModule::new` VALIDATES the exact
        // bytes the browser runs — the guard, the `jit_enter`/`jit_leave` bracket, the arg copy,
        // and the `call_indirect` (every branch's stack/type shape). A NON-object receiver (100)
        // takes the fallback branch, so `call_prop_ic`/`call_method_ic` (stubs = recv[+disp]+Σargs)
        // execute and prove the fallback path round-trips too. (The in-WASM HIT branch is validated
        // structurally here; its runtime behaviour is covered by the web `shared_verified` gate.)
        FORCE_INWASM_CALL_IC.with(|c| c.set(true));
        let prop = run(
            &[
                JitOp::GetLocal(1),
                JitOp::PushBits(11),
                JitOp::PushBits(22),
                JitOp::CallProperty(7, 2),
                JitOp::BailIfError,
                JitOp::ReturnValue,
            ],
            &[UNDEFINED_BITS, 100],
        );
        let meth = run(
            &[
                JitOp::GetLocal(1),
                JitOp::PushBits(5),
                JitOp::CallMethod(3, 1),
                JitOp::BailIfError,
                JitOp::ReturnValue,
            ],
            &[UNDEFINED_BITS, 100],
        );
        FORCE_INWASM_CALL_IC.with(|c| c.set(false));
        assert_eq!(prop, 133); // 100 + 11 + 22 via call_prop_ic fallback
        assert_eq!(meth, 108); // 100 + 3 (disp_id) + 5 via call_method_ic fallback
    }

    #[test]
    fn binop_passes_both_operands_and_the_code() {
        // push 10, 20; binop(code=5); bail; return → stub `binop` = a + b + op = 10+20+5.
        let out = run(
            &[
                JitOp::PushBits(10),
                JitOp::PushBits(20),
                JitOp::BinOp(5),
                JitOp::BailIfError,
                JitOp::ReturnValue,
            ],
            &[UNDEFINED_BITS],
        );
        assert_eq!(out, 35);
    }

    #[test]
    fn binopnum_computes_unguarded_f64() {
        // The unguarded f64 core: same results as `BinOp`'s guarded f64 path, but no runtime
        // tag check and no `BailIfError` (translate proves both operands canonical `Number`).
        let dbl = |x: f64| x.to_bits();
        let bin = |code: i32, a: f64, b: f64| {
            f64::from_bits(run(
                &[JitOp::GetLocal(1), JitOp::GetLocal(2), JitOp::BinOpNum(code), JitOp::ReturnValue],
                &[UNDEFINED_BITS, dbl(a), dbl(b)],
            ))
        };
        use crate::helpers::binop_code as bc;
        assert_eq!(bin(bc::ADD, 1.5, 2.25), 3.75);
        assert_eq!(bin(bc::SUBTRACT, 1.5, 2.25), -0.75);
        assert_eq!(bin(bc::MULTIPLY, 1.5, 2.0), 3.0);
        assert_eq!(bin(bc::DIVIDE, 3.0, 2.0), 1.5);
        assert_eq!(bin(bc::DIVIDE, 1.0, 0.0), f64::INFINITY);
        // A NaN result is canonicalized to a non-box-colliding double.
        let raw = run(
            &[JitOp::GetLocal(1), JitOp::GetLocal(2), JitOp::BinOpNum(bc::DIVIDE), JitOp::ReturnValue],
            &[UNDEFINED_BITS, dbl(0.0), dbl(0.0)],
        );
        assert!(f64::from_bits(raw).is_nan());
        assert_ne!(raw & 0xFFF8_0000_0000_0000, 0xFFF8_0000_0000_0000, "must not alias box space");
        // Chained: (a*b)*c stays a valid `Number` across ops (result feeds the next BinOpNum).
        let out = f64::from_bits(run(
            &[
                JitOp::GetLocal(1),
                JitOp::GetLocal(2),
                JitOp::BinOpNum(bc::MULTIPLY),
                JitOp::GetLocal(3),
                JitOp::BinOpNum(bc::MULTIPLY),
                JitOp::ReturnValue,
            ],
            &[UNDEFINED_BITS, dbl(2.0), dbl(3.0), dbl(4.0)],
        ));
        assert_eq!(out, 24.0);
    }

    #[test]
    fn binopcmp_native_unguarded_compare() {
        use crate::helpers::binop_code as bc;
        let tru = VALUE_BOOL_MARK | 1;
        let fal = VALUE_BOOL_MARK;
        let ib = |v: i32| VALUE_INT_MARK | (v as u32 as u64);
        // Two `Int`s → signed i32 compare.
        let icmp = |code: i32, a: i32, b: i32| {
            run(
                &[JitOp::GetLocal(1), JitOp::GetLocal(2), JitOp::BinOpCmp(code, false), JitOp::ReturnValue],
                &[UNDEFINED_BITS, ib(a), ib(b)],
            )
        };
        assert_eq!(icmp(bc::LESS_THAN, 3, 5), tru);
        assert_eq!(icmp(bc::LESS_THAN, 5, 3), fal);
        assert_eq!(icmp(bc::LESS_THAN, -2, 1), tru, "signed"); // signed: -2 < 1
        assert_eq!(icmp(bc::GREATER_EQUALS, 4, 4), tru);
        assert_eq!(icmp(bc::EQUALS, 7, 7), tru);
        assert_eq!(icmp(bc::EQUALS, 7, 8), fal);
        // Two canonical `Number`s → f64 compare (NaN → false).
        let dbl = |x: f64| x.to_bits();
        let fcmp = |code: i32, a: f64, b: f64| {
            run(
                &[JitOp::GetLocal(1), JitOp::GetLocal(2), JitOp::BinOpCmp(code, true), JitOp::ReturnValue],
                &[UNDEFINED_BITS, dbl(a), dbl(b)],
            )
        };
        assert_eq!(fcmp(bc::LESS_THAN, 1.5, 2.5), tru);
        assert_eq!(fcmp(bc::GREATER_THAN, 3.0, 1.0), tru);
        assert_eq!(fcmp(bc::EQUALS, 2.0, 2.0), tru);
        assert_eq!(fcmp(bc::LESS_THAN, f64::NAN, 1.0), fal, "NaN compares false");
        assert_eq!(fcmp(bc::EQUALS, f64::NAN, f64::NAN), fal);
    }

    #[test]
    fn promoted_bool_local_round_trips_through_i32_register() {
        // Local 1 is a promoted `Bool` local → an i32 register (box tag `VALUE_BOOL_MARK`).
        let bb = |v: bool| VALUE_BOOL_MARK | (v as u64);
        // b = true; return b.
        let out = run_blocks_p(
            vec![Block {
                ops: vec![JitOp::PushBits(bb(true)), JitOp::SetLocal(1), JitOp::GetLocal(1), JitOp::ReturnValue],
                term: Term::Return,
                entry_depth: 0,
            }],
            &[UNDEFINED_BITS, UNDEFINED_BITS],
            &[Promotion { local: 1, kind: RegKind::BoolI32, init_from_frame: false }],
        );
        assert_eq!(out, bb(true));
        // A `Bool` param read from the frame register.
        let out = run_blocks_p(
            vec![Block {
                ops: vec![JitOp::GetLocal(1), JitOp::ReturnValue],
                term: Term::Return,
                entry_depth: 0,
            }],
            &[UNDEFINED_BITS, bb(false)],
            &[Promotion { local: 1, kind: RegKind::BoolI32, init_from_frame: true }],
        );
        assert_eq!(out, bb(false));
    }

    #[test]
    fn promoted_number_param_reads_from_register() {
        // Local 1 is a promoted read-only `Number` param: the prologue loads it from the frame
        // into an f64 WASM local, and `GetLocal(1)` reads that register (no `i64.load`). The
        // round-trip must be bit-identical to the boxed frame value.
        let n = 3.5f64.to_bits();
        let out = run_blocks_p(
            vec![Block {
                ops: vec![JitOp::GetLocal(1), JitOp::ReturnValue],
                term: Term::Return,
                entry_depth: 0,
            }],
            &[UNDEFINED_BITS, n],
            &[Promotion { local: 1, kind: RegKind::F64, init_from_frame: true }],
        );
        assert_eq!(out, n, "promoted param must read back its exact Number bits");

        // Arithmetic on the promoted param: x * x, x a canonical Number → BinOpNum.
        let out = f64::from_bits(run_blocks_p(
            vec![Block {
                ops: vec![
                    JitOp::GetLocal(1),
                    JitOp::GetLocal(1),
                    JitOp::BinOpNum(crate::helpers::binop_code::MULTIPLY),
                    JitOp::ReturnValue,
                ],
                term: Term::Return,
                entry_depth: 0,
            }],
            &[UNDEFINED_BITS, n],
            &[Promotion { local: 1, kind: RegKind::F64, init_from_frame: true }],
        ));
        assert_eq!(out, 3.5 * 3.5);
    }

    #[test]
    fn promoted_writable_local_round_trips_through_register() {
        // Local 1 is a promoted WRITABLE local (init_from_frame = false → the f64 register
        // starts at the WASM default 0.0). Storing a canonical `Number` and reading it back must
        // be bit-identical; a `BinOpNum` accumulate must land back in the register.
        let a = 2.5f64.to_bits();
        let b = 4.0f64.to_bits();
        // s = a; s = s + b; return s   →  a + b.
        let out = f64::from_bits(run_blocks_p(
            vec![Block {
                ops: vec![
                    JitOp::PushBits(a),
                    JitOp::SetLocal(1),
                    JitOp::GetLocal(1),
                    JitOp::PushBits(b),
                    JitOp::BinOpNum(crate::helpers::binop_code::ADD),
                    JitOp::SetLocal(1),
                    JitOp::GetLocal(1),
                    JitOp::ReturnValue,
                ],
                term: Term::Return,
                entry_depth: 0,
            }],
            &[UNDEFINED_BITS, UNDEFINED_BITS],
            &[Promotion { local: 1, kind: RegKind::F64, init_from_frame: false }],
        ));
        assert_eq!(out, 2.5 + 4.0);
    }

    #[test]
    fn promoted_int_local_round_trips_through_i32_register() {
        // Local 1 is a promoted `Int` accumulator → an i32 register (init 0). Storing an `Int`
        // box and reading it back is bit-identical; an int-int `ADD_I` accumulate lands in the
        // register. `int` box = `VALUE_INT_MARK | (v as u32)`.
        let ib = |v: i32| VALUE_INT_MARK | (v as u32 as u64);
        use crate::helpers::binop_code as bc;
        // c = 5; c = c + 10; return c   →  15.
        let out = run_blocks_p(
            vec![Block {
                ops: vec![
                    JitOp::PushBits(ib(5)),
                    JitOp::SetLocal(1),
                    JitOp::GetLocal(1),
                    JitOp::PushBits(ib(10)),
                    JitOp::BinOp(bc::ADD_I),
                    JitOp::SetLocal(1),
                    JitOp::GetLocal(1),
                    JitOp::ReturnValue,
                ],
                term: Term::Return,
                entry_depth: 0,
            }],
            &[UNDEFINED_BITS, UNDEFINED_BITS],
            &[Promotion { local: 1, kind: RegKind::IntI32, init_from_frame: false }],
        );
        assert_eq!(out, ib(15));

        // A promoted `Int` PARAM (init_from_frame): read from the frame register, negate-wrap.
        let out = run_blocks_p(
            vec![Block {
                ops: vec![JitOp::GetLocal(1), JitOp::ReturnValue],
                term: Term::Return,
                entry_depth: 0,
            }],
            &[UNDEFINED_BITS, ib(-42)],
            &[Promotion { local: 1, kind: RegKind::IntI32, init_from_frame: true }],
        );
        assert_eq!(out, ib(-42), "signed i32 payload round-trips");
    }

    #[test]
    fn unop_passes_the_operand_and_the_code() {
        // push 100; unop(code=2); bail; return → stub `unop` = a + op = 100+2.
        let out = run(
            &[JitOp::PushBits(100), JitOp::UnOp(2), JitOp::BailIfError, JitOp::ReturnValue],
            &[UNDEFINED_BITS],
        );
        assert_eq!(out, 102);
    }

    #[test]
    fn scope_push_and_get_balance_the_stack() {
        // push an object; scopepush consumes it (stub no-op); getscope 3 (stub echoes 3);
        // return → 3. Proves ScopePush pops one and GetScope pushes one (stack balanced).
        let out = run(
            &[
                JitOp::PushBits(VALUE_INT_MARK | 9),
                JitOp::ScopePush,
                JitOp::BailIfPerr,
                JitOp::GetScope(3),
                JitOp::ReturnValue,
            ],
            &[UNDEFINED_BITS],
        );
        assert_eq!(out, 3);
    }

    #[test]
    fn callmethod_passes_receiver_and_disp_id() {
        // getlocal1 (receiver=100); callmethod(disp_id=5, 0 args); bail; return →
        // stub `call_method` = receiver + disp_id = 105.
        let out = run(
            &[
                JitOp::GetLocal(1),
                JitOp::CallMethod(5, 0),
                JitOp::BailIfError,
                JitOp::ReturnValue,
            ],
            &[UNDEFINED_BITS, 100],
        );
        assert_eq!(out, 105);
    }

    #[test]
    fn set_property_and_slot_pop_object_and_value() {
        // getlocal1 (object); push value; setproperty — pops both (void stub); then setslot
        // similarly; finally return a constant. Proves the write ops balance the stack.
        let out = run(
            &[
                JitOp::GetLocal(1),
                JitOp::PushBits(99),
                JitOp::SetProperty(0),
                JitOp::BailIfPerr,
                JitOp::GetLocal(1),
                JitOp::PushBits(7),
                JitOp::SetSlot(3, 2),
                JitOp::BailIfPerr,
                JitOp::PushBits(VALUE_INT_MARK | 5),
                JitOp::ReturnValue,
            ],
            &[UNDEFINED_BITS, 12345],
        );
        assert_eq!(out, VALUE_INT_MARK | 5);
    }

    #[test]
    fn cond_branch_picks_the_right_block() {
        // block0: getlocal1 (cond); Cond{true->1, false->2}
        // block1: return 111 ; block2: return 222.  truthy stub: nonzero -> true.
        let blocks = || {
            vec![
                Block { ops: vec![JitOp::GetLocal(1)], term: Term::Cond { on_true: 1, on_false: 2 }, entry_depth: 0 },
                Block { ops: vec![JitOp::PushBits(111), JitOp::ReturnValue], term: Term::Return, entry_depth: 0 },
                Block { ops: vec![JitOp::PushBits(222), JitOp::ReturnValue], term: Term::Return, entry_depth: 0 },
            ]
        };
        assert_eq!(run_blocks(blocks(), &[UNDEFINED_BITS, 5]), 111); // truthy
        assert_eq!(run_blocks(blocks(), &[UNDEFINED_BITS, 0]), 222); // falsy
    }

    #[test]
    fn truthy_inlines_primitive_conditions() {
        // Branch on local 1 (see `cond_branch`): truthy → 111, falsy → 222. Exercises the
        // inline `emit_truthy` for bool/int/Number/object (undefined/null/String would take the
        // helper — omitted here, the stub isn't real `ToBoolean`).
        let blocks = || {
            vec![
                Block { ops: vec![JitOp::GetLocal(1)], term: Term::Cond { on_true: 1, on_false: 2 }, entry_depth: 0 },
                Block { ops: vec![JitOp::PushBits(111), JitOp::ReturnValue], term: Term::Return, entry_depth: 0 },
                Block { ops: vec![JitOp::PushBits(222), JitOp::ReturnValue], term: Term::Return, entry_depth: 0 },
            ]
        };
        let t = |v: u64| run_blocks(blocks(), &[UNDEFINED_BITS, v]);
        let int = |n: i32| 0xFFFB_0000_0000_0000u64 | (n as u32 as u64);
        let dbl = |x: f64| x.to_bits();
        assert_eq!(t(0xFFFA_0000_0000_0000u64 | 1), 111); // Bool(true)
        assert_eq!(t(0xFFFA_0000_0000_0000u64), 222); // Bool(false)
        assert_eq!(t(int(7)), 111);
        assert_eq!(t(int(-1)), 111);
        assert_eq!(t(int(0)), 222);
        assert_eq!(t(dbl(1.5)), 111);
        assert_eq!(t(dbl(-3.0)), 111);
        assert_eq!(t(dbl(0.0)), 222);
        assert_eq!(t(dbl(-0.0)), 222); // -0 is falsy
        assert_eq!(t(dbl(f64::NAN)), 222); // NaN is falsy
        assert_eq!(t(0xFFFD_0000_0000_0000u64 | 0x1234), 111); // Object → always true
    }

    #[test]
    fn binop_inlines_the_int_int_fast_path() {
        // Packed reprs (core `value.rs`): int = 0xFFFB<<48 | (i as u32); bool = 0xFFFA<<48 | b.
        let int = |n: i32| 0xFFFB_0000_0000_0000u64 | (n as u32 as u64);
        let tru = 0xFFFA_0000_0000_0000u64 | 1;
        let fal = 0xFFFA_0000_0000_0000u64;
        let bin = |code: i32, a: i32, b: i32| {
            run(
                &[JitOp::GetLocal(1), JitOp::GetLocal(2), JitOp::BinOp(code), JitOp::ReturnValue],
                &[UNDEFINED_BITS, int(a), int(b)],
            )
        };
        use crate::helpers::binop_code as bc;
        // Comparisons → bool (loop conditions).
        assert_eq!(bin(bc::LESS_THAN, 3, 5), tru);
        assert_eq!(bin(bc::LESS_THAN, 5, 3), fal);
        assert_eq!(bin(bc::GREATER_EQUALS, 5, 5), tru);
        assert_eq!(bin(bc::EQUALS, -2, -2), tru);
        // Wrapping int arithmetic + bitwise → int (a negative result re-boxes correctly).
        assert_eq!(bin(bc::ADD_I, 3, 5), int(8));
        assert_eq!(bin(bc::SUBTRACT_I, 3, 5), int(-2));
        assert_eq!(bin(bc::BITAND, 6, 3), int(2));
    }

    #[test]
    fn binop_inlines_the_double_double_fast_path() {
        // A `Number` is stored as its raw f64 bits (core `value.rs`): both operands inline
        // f64 → compute natively, no helper. Result reads back as the correct `Number`.
        let dbl = |x: f64| x.to_bits();
        let bin = |code: i32, a: f64, b: f64| {
            f64::from_bits(run(
                &[JitOp::GetLocal(1), JitOp::GetLocal(2), JitOp::BinOp(code), JitOp::ReturnValue],
                &[UNDEFINED_BITS, dbl(a), dbl(b)],
            ))
        };
        use crate::helpers::binop_code as bc;
        assert_eq!(bin(bc::ADD, 1.5, 2.25), 3.75);
        assert_eq!(bin(bc::SUBTRACT, 1.5, 2.25), -0.75);
        assert_eq!(bin(bc::MULTIPLY, 1.5, 2.0), 3.0);
        assert_eq!(bin(bc::DIVIDE, 3.0, 2.0), 1.5);
        assert_eq!(bin(bc::DIVIDE, 1.0, 0.0), f64::INFINITY);
        assert_eq!(bin(bc::DIVIDE, -1.0, 0.0), f64::NEG_INFINITY);
        // A NaN result is canonicalized to a NON-box-colliding double (reads back as NaN).
        let raw = run(
            &[JitOp::GetLocal(1), JitOp::GetLocal(2), JitOp::BinOp(bc::DIVIDE), JitOp::ReturnValue],
            &[UNDEFINED_BITS, dbl(0.0), dbl(0.0)],
        );
        assert!(f64::from_bits(raw).is_nan());
        assert_ne!(raw & 0xFFF8_0000_0000_0000, 0xFFF8_0000_0000_0000, "must not alias box space");
        // Double comparisons → Bool (native f64 compare; NaN comparisons are all false).
        let cmp = |code: i32, a: f64, b: f64| {
            run(
                &[JitOp::GetLocal(1), JitOp::GetLocal(2), JitOp::BinOp(code), JitOp::ReturnValue],
                &[UNDEFINED_BITS, dbl(a), dbl(b)],
            )
        };
        let tru = 0xFFFA_0000_0000_0000u64 | 1;
        let fal = 0xFFFA_0000_0000_0000u64;
        assert_eq!(cmp(bc::LESS_THAN, 1.0, 2.0), tru);
        assert_eq!(cmp(bc::LESS_THAN, 2.0, 1.0), fal);
        assert_eq!(cmp(bc::LESS_EQUALS, 2.0, 2.0), tru);
        assert_eq!(cmp(bc::GREATER_THAN, 3.5, 1.0), tru);
        assert_eq!(cmp(bc::GREATER_EQUALS, 2.0, 2.0), tru);
        assert_eq!(cmp(bc::EQUALS, 1.5, 1.5), tru);
        assert_eq!(cmp(bc::EQUALS, 1.5, 2.5), fal);
        assert_eq!(cmp(bc::EQUALS, f64::NAN, f64::NAN), fal); // NaN != NaN
        assert_eq!(cmp(bc::LESS_THAN, f64::NAN, 1.0), fal); // NaN comparisons all false
    }

    #[test]
    fn binop_inlines_general_int_arithmetic() {
        // ADD/SUBTRACT/MULTIPLY of two ints: Integer when it fits i32, else Number — matching
        // the helper's `checked_{add,sub,mul}` (overflow → `(a as i64 OP b as i64) as f64`).
        let int = |n: i32| 0xFFFB_0000_0000_0000u64 | (n as u32 as u64);
        let bin = |code: i32, a: i32, b: i32| {
            run(
                &[JitOp::GetLocal(1), JitOp::GetLocal(2), JitOp::BinOp(code), JitOp::ReturnValue],
                &[UNDEFINED_BITS, int(a), int(b)],
            )
        };
        use crate::helpers::binop_code as bc;
        // Fits i32 → Integer.
        assert_eq!(bin(bc::ADD, 3, 5), int(8));
        assert_eq!(bin(bc::SUBTRACT, 3, 5), int(-2));
        assert_eq!(bin(bc::MULTIPLY, 4, 5), int(20));
        assert_eq!(bin(bc::MULTIPLY, -4, 5), int(-20));
        // Overflow → Number (raw f64 bits), matching `(a as i64 OP b) as f64`.
        assert_eq!(bin(bc::ADD, i32::MAX, 1), ((i32::MAX as i64 + 1) as f64).to_bits());
        assert_eq!(bin(bc::SUBTRACT, i32::MIN, 1), ((i32::MIN as i64 - 1) as f64).to_bits());
        assert_eq!(bin(bc::MULTIPLY, 100_000, 100_000), ((100_000i64 * 100_000) as f64).to_bits());
    }

    #[test]
    fn binop_inlines_int_shifts_and_coerce() {
        let int = |n: i32| 0xFFFB_0000_0000_0000u64 | (n as u32 as u64);
        let bin = |code: i32, a: i32, b: i32| {
            run(
                &[JitOp::GetLocal(1), JitOp::GetLocal(2), JitOp::BinOp(code), JitOp::ReturnValue],
                &[UNDEFINED_BITS, int(a), int(b)],
            )
        };
        let un = |code: i32, a: u64| {
            run(&[JitOp::GetLocal(1), JitOp::UnOp(code), JitOp::ReturnValue], &[UNDEFINED_BITS, a])
        };
        use crate::helpers::binop_code as bc;
        use crate::helpers::unop_code as uc;
        // Shifts (int×int): count masks to `& 31`; `<<`/`>>` → Integer, `>>>` → maybe Number.
        assert_eq!(bin(bc::LSHIFT, 1, 4), int(16));
        assert_eq!(bin(bc::LSHIFT, 1, 33), int(2)); // 33 & 31 == 1
        assert_eq!(bin(bc::RSHIFT, -16, 2), int(-4)); // arithmetic (signed)
        assert_eq!(bin(bc::URSHIFT, 16, 2), int(4));
        assert_eq!(bin(bc::URSHIFT, -1, 0), (4_294_967_295u32 as f64).to_bits()); // > i32::MAX → Number
        // coerce_i / coerce_u on an int operand.
        assert_eq!(un(uc::COERCE_I, int(7)), int(7)); // ToInt32(int) = identity
        assert_eq!(un(uc::COERCE_U, int(7)), int(7)); // ToUint32(non-neg int) = identity
        assert_eq!(un(uc::COERCE_U, int(-1)), (4_294_967_295u32 as f64).to_bits()); // → Number
    }

    #[test]
    fn value_crosses_a_cond_edge_via_spill() {
        // Ternary `cond ? X : Y`: block0 pushes X (bits 111) then the condition, so X is LIVE
        // across the `Cond` edge (entry_depth 1 on both successors). block1 reloads X and
        // returns it; block2 reloads X, drops it, returns Y (222). Validates spill/reload.
        let blocks = || {
            vec![
                Block {
                    ops: vec![JitOp::PushBits(111), JitOp::GetLocal(1)],
                    term: Term::Cond { on_true: 1, on_false: 2 },
                    entry_depth: 0,
                },
                // reloads the crossed X (111) → returns it.
                Block { ops: vec![JitOp::ReturnValue], term: Term::Return, entry_depth: 1 },
                // reloads X, drops it, returns Y (222).
                Block {
                    ops: vec![JitOp::Pop, JitOp::PushBits(222), JitOp::ReturnValue],
                    term: Term::Return,
                    entry_depth: 1,
                },
            ]
        };
        assert_eq!(run_blocks(blocks(), &[UNDEFINED_BITS, 5]), 111); // truthy → X
        assert_eq!(run_blocks(blocks(), &[UNDEFINED_BITS, 0]), 222); // falsy → Y
    }

    #[test]
    fn loop_with_back_edge_runs_and_locals_persist() {
        // block0: -> block1
        // block1 (header): getlocal1; Cond{true->2 (body), false->3 (exit)}
        // block2 (body): local1 = 0; -> block1   (back-edge; local write persists in memory)
        // block3 (exit): return local1
        // frame local1=7 -> enters body once, sets 0, header sees 0 -> exit -> returns 0.
        let blocks = vec![
            Block { ops: vec![], term: Term::Jump(1), entry_depth: 0 },
            Block { ops: vec![JitOp::GetLocal(1)], term: Term::Cond { on_true: 2, on_false: 3 }, entry_depth: 0 },
            Block {
                ops: vec![JitOp::PushBits(0), JitOp::SetLocal(1)],
                term: Term::Jump(1),
                entry_depth: 0,
            },
            Block { ops: vec![JitOp::GetLocal(1), JitOp::ReturnValue], term: Term::Return, entry_depth: 0 },
        ];
        assert_eq!(run_blocks(blocks, &[UNDEFINED_BITS, 7]), 0);
    }

    #[test]
    fn callpropvoid_discards_the_result() {
        // A void call (result popped) followed by returning the argument verbatim proves
        // the module still validates with the extra `Pop` and the operand stack is balanced.
        let out = run(
            &[
                JitOp::GetLocal(1),
                JitOp::CallProperty(0, 0),
                JitOp::BailIfError,
                JitOp::Pop,
                JitOp::GetLocal(1),
                JitOp::ReturnValue,
            ],
            &[UNDEFINED_BITS, 42],
        );
        assert_eq!(out, 42);
    }
}
