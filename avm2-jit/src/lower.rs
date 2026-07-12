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
    BlockType, CodeSection, ConstExpr, Elements, ElementSection, EntityType, ExportKind,
    ExportSection, Function, FunctionSection, GlobalType, ImportSection, Instruction, MemArg,
    MemoryType, Module, NameMap, NameSection, RefType, TableSection, TableType, TypeSection, ValType,
};

/// Bit pattern OR-ed onto a 32-bit integer to form an AVM2 int [`Value`].
/// MUST match `ruffle_core::avm2::value`'s `BOX_MARK | (TAG_INT << 48)`.
pub(crate) const VALUE_INT_MARK: u64 = 0xFFFB_0000_0000_0000;
const VALUE_ALIGN: u32 = 3;
// WASM locals. Five params first — `run(state_ptr, dm_base, dm_len, regs_ptr,
// regs_len)`: `dm_base` (offset into memory 1) + `dm_len` describe the current
// domainMemory for the inline `li*`/`si*` fast path; `regs_ptr`/`regs_len`
// (offset/bytes in memory 1 = Ruffle's own memory, web) locate the register
// snapshot the wasm32 prologue copies into the frame itself — replacing the
// per-call JS `Uint8Array` copy, which a profile showed at ~10% of the worker.
// Declared scratch follows (3× i32, 2× i64, 1× f64);
// `Function::new([(3, I32), (2, I64), (1, F64)])` must match.
const STATE_PTR: u32 = 0;
const DM_BASE: u32 = 1;
const DM_LEN: u32 = 2;
const REGS_PTR: u32 = 3;
const REGS_LEN: u32 = 4;
/// 6th `run` param: memory-1 byte offset of this method's property-IC cache
/// buffer (a JIT-owned `Box<[IcCell]>`; see [`emit_getprop_ic`]). 0 when the
/// method has no IC site (`Manifest::has_ic == false`) — then no emitted code
/// reads it.
const IC_BASE: u32 = 5;
const SCRATCH: u32 = 6;
/// Second i32 scratch (inline `si*` holds the address and the value at once).
const SCRATCH2: u32 = 7;
const BLOCK: u32 = 8;
/// i64 scratch for boxed-`Value` stores.
const SCRATCH64: u32 = 9;
/// Second i64 scratch (used by `swap`, which must hold both operands).
const SCRATCH64_2: u32 = 10;
/// f64 scratch for double boxing.
const SCRATCH_F64: u32 = 11;
/// Base index of the operand-stack spill pool: `SPILL_POOL` i64 locals used to
/// carry live boxed operands across a basic-block boundary (the `br_table`
/// dispatch can't keep the wasm operand stack live across blocks). A branch spills
/// its live values here; the target block reloads them. Only the boxed path uses
/// them (its stack values are i64); the int/double paths declare them but never
/// spill (int values are i32; the double path is branch-free).
const SPILL_BASE: u32 = 12;
/// Max operand-stack height carryable across a branch. Ternary/short-circuit carry
/// 1; deeper cross-branch expressions are rare — a method exceeding this declines.
const SPILL_POOL: u32 = 8;

/// Bytes per JIT frame nesting level in memory 0 (`state_ptr = depth * FRAME_STRIDE`).
/// The in-wasm JIT→JIT direct call ([`emit_call_direct`]) places the callee frame at
/// `state_ptr + FRAME_STRIDE`, so this MUST equal the runner's frame stride
/// (`runner::web::STRIDE`, which is defined as this).
pub(crate) const FRAME_STRIDE: u32 = 4096;

/// Total committed bytes of the shared JIT frame memory (memory 0), = the runner's
/// `web::MEM_BYTES` (`PAGES * 65536`). A JIT→JIT direct call to a **non-leaf** callee
/// (one that itself calls) can chain frames without going through the runner's
/// per-`run` depth-decline check, so [`emit_call_direct`] bounds-checks the callee
/// frame against this before the `call_indirect` (over → falls to the `cm` helper,
/// which re-enters `run` and declines to the interpreter). MUST equal
/// `runner::web::MEM_BYTES`.
pub(crate) const FRAME_MEM_BYTES: u32 = 64 * 65536;

/// The canonical NaN bit pattern Ruffle stores for a NaN `Number` (must match
/// `core::avm2::value`'s `CANON_NAN`).
const CANON_NAN: u64 = 0x7FF8_0000_0000_0000;

/// Bit pattern of `Value::Undefined` (`BOX_MARK | (TAG_UNDEFINED << 48)`, must
/// match `core::avm2::value`). What a `returnvoid` method returns.
pub(crate) const UNDEFINED_BITS: u64 = 0xFFF8_0000_0000_0000;

/// NaN-box marker (sign + all exponent + quiet-NaN bit): a word is a boxed
/// (non-`Number`) `Value` iff `bits & BOX_MARK == BOX_MARK`. Must match
/// `core::avm2::value::BOX_MARK`.
const BOX_MARK: u64 = 0xFFF8_0000_0000_0000;
/// Mask isolating `BOX_MARK | tag` (the top 16 bits) so an int box can be
/// identified by `bits & VALUE_TAG_MASK == VALUE_INT_MARK`.
pub(crate) const VALUE_TAG_MASK: u64 = 0xFFFF_0000_0000_0000;
/// A `Boolean` `Value`'s base bits (`BOX_MARK | TAG_BOOL << 48`); the `0`/`1`
/// payload is OR-ed in. Must match `core::avm2::value` (`TAG_BOOL = 2`).
pub(crate) const VALUE_BOOL_MARK: u64 = 0xFFFA_0000_0000_0000;
/// Bit position of the 3-bit box tag (`bits >> TAG_SHIFT & 0x7`). Must match
/// `core::avm2::value::TAG_SHIFT`.
const TAG_SHIFT: u64 = 48;
/// Tag of a **heap-boxed double**: a colliding negative-quiet-NaN `Number` whose
/// exact bits can't live inline, so `Value::number_lossless` stashes them in a
/// `Gc<f64>` and stores the pointer. Must match `core::avm2::value::TAG_BOXED_DOUBLE`.
const TAG_BOXED_DOUBLE: u64 = 6;
/// An `Object` `Value`'s top 16 bits (`(BOX_MARK | TAG_OBJECT << 48) >> 48`,
/// `TAG_OBJECT = 5`): a word is object-boxed iff `bits >> 48 == 0xFFFD`. The
/// low 48 bits are then the raw `Gc` data pointer (see `Value::pack`).
const VALUE_OBJECT_TAG16: i64 = 0xFFFD;

/// The probed `(slots_ptr_off, slots_len_off)` layout for the **inline `getslot`
/// fast path** — byte offsets of the slots slice's data pointer / length within
/// an object's `ScriptObjectData` prefix (see `ruffle_core`'s
/// `jit_slots_layout`). Web only: memory 1 is Ruffle's real heap there, so an
/// object pointer can be chased directly; native (wasmi) memory 1 is a mock, so
/// `getslot` stays a helper call (`None`), like the inline dm gate.
fn slot_layout() -> Option<(u32, u32)> {
    #[cfg(all(test, not(target_arch = "wasm32")))]
    if let Some(forced) = tests_slot_layout::get() {
        return Some(forced);
    }
    if cfg!(target_arch = "wasm32") {
        static LAYOUT: std::sync::OnceLock<Option<(u32, u32)>> = std::sync::OnceLock::new();
        *LAYOUT.get_or_init(ruffle_core::avm2::jit_slots_layout)
    } else {
        None
    }
}

/// Byte offset of the object's `vtable` pointer within its `ScriptObjectData`
/// prefix — the property-IC class guard's key (see [`emit_getprop_ic`]). Same
/// web-only gating as [`slot_layout`] (`None` natively → the IC is never
/// emitted); forced in tests alongside the slot layout.
fn vtable_offset() -> Option<u32> {
    #[cfg(all(test, not(target_arch = "wasm32")))]
    if let Some(off) = tests_slot_layout::get_vtable() {
        return Some(off);
    }
    if cfg!(target_arch = "wasm32") {
        static VOFF: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
        Some(*VOFF.get_or_init(ruffle_core::avm2::jit_vtable_offset))
    } else {
        None
    }
}

/// Byte offset of an object's `instance_class` `Gc` word (web only) — the direct-call
/// exact-class arg guard reads it. `None` natively (direct calls are web-only). No test
/// hook: the codegen tests pass the offset to `emit_call_direct` directly.
fn instance_class_offset() -> Option<u32> {
    if cfg!(target_arch = "wasm32") {
        static OFF: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
        Some(*OFF.get_or_init(ruffle_core::avm2::jit_instance_class_offset))
    } else {
        None
    }
}

/// Whether the property inline cache can be emitted: both the slot layout and the
/// vtable-offset probes must be available (web, or forced in a test). When false
/// (native production), `translate` keeps the generic `GetProperty` helper path.
pub(crate) fn ic_available() -> bool {
    slot_layout().is_some() && vtable_offset().is_some()
}

/// Test hook: forces a fake [`slot_layout`] / [`vtable_offset`] so the
/// (wasm32-gated) inline `getslot` + property-IC codegen can be exercised
/// natively under wasmi with a mock memory 1.
#[cfg(all(test, not(target_arch = "wasm32")))]
pub(crate) mod tests_slot_layout {
    use std::cell::Cell;
    thread_local! {
        static FORCED: Cell<Option<(u32, u32)>> = const { Cell::new(None) };
        static FORCED_VTABLE: Cell<Option<u32>> = const { Cell::new(None) };
    }
    pub fn get() -> Option<(u32, u32)> {
        FORCED.with(|c| c.get())
    }
    pub fn force(layout: Option<(u32, u32)>) {
        FORCED.with(|c| c.set(layout));
    }
    pub fn get_vtable() -> Option<u32> {
        FORCED_VTABLE.with(|c| c.get())
    }
    pub fn force_vtable(off: Option<u32>) {
        FORCED_VTABLE.with(|c| c.set(off));
    }
}

/// Function indices of the special arity-1 helpers implicitly called by certain
/// ops (`HELPERS[5..8]`). Keep in sync with [`crate::helpers::HELPERS`];
/// `helper_count` guarantees the method imports `h0..=h{index}` when the op is
/// present, so these equal the helpers' function indices.
pub(crate) const TO_BOOLEAN: u32 = 5; // boxed branch condition
const PUSH_SCOPE: u32 = 6; // real scope push
const GET_SCOPE_OBJECT: u32 = 7; // local scope read
pub(crate) const COERCE_U: u32 = 11; // ToUint32 (coerce_u) helper fallback
pub(crate) const COERCE_I: u32 = 12; // ToInt32 (coerce_i) helper fallback
pub(crate) const COERCE_RETURN: u32 = 16; // returnvalue return-type coercion
pub(crate) const GET_SCRIPT_GLOBALS: u32 = 17; // getscriptglobals (pre-resolved bits by index)
pub(crate) const GET_PUSH_STRING: u32 = 18; // pushstring (pre-resolved string Value bits by index)
const THROW: u32 = 19; // throw (stash the thrown Value as a pending error, then return)
const DISPATCH_EXC: u32 = 20; // route a pending exception through the handler table
const NEW_CATCH: u32 = 21; // newcatch (build the catch scope object)
const POP_CAUGHT: u32 = 22; // catch-block entry: fetch the caught exception value
const POP_SCOPE: u32 = 23; // popscope (pop the real scope stack)
const GET_OUTER_SCOPE: u32 = 24; // getouterscope (read a captured/outer scope by index)
pub(crate) const COERCE_S: u32 = 25; // coerces (ToString; throwing toString → PENDING_ERROR)
pub(crate) const DM_LOADF32: u32 = 26; // lf32 fallback (inline dm miss → storage read)
pub(crate) const DM_LOADF64: u32 = 27; // lf64 fallback
pub(crate) const INCREMENT_I: u32 = 28; // increment_i (wrapping ToInt32+1) — inline inc's non-int fallback
pub(crate) const DECREMENT_I: u32 = 29; // decrement_i (wrapping ToInt32-1) — inline dec's non-int fallback
pub(crate) const INCREMENT_NUM: u32 = 0; // increment (ToNumber+1) — `inclocal`
pub(crate) const DECREMENT_NUM: u32 = 1; // decrement (ToNumber-1) — `declocal`
const HN2_INDEX: u32 = 30; // hasnext2: fetch the stashed index-register output
const HN2_OBJECT: u32 = 31; // hasnext2: fetch the stashed object-register output

/// Pushes the `Value` (i64) bits of the `f64` already held in `SCRATCH_F64` —
/// canonicalizing NaN to `CANON_NAN` so it doesn't collide with Ruffle's
/// boxed-value NaN space. Reads only `SCRATCH_F64`, so it leaves whatever the
/// operand stack held on entry untouched (used by the peek-and-store path).
fn emit_box_scratch_f64(body: &mut Function) {
    body.instruction(&Instruction::LocalGet(SCRATCH_F64));
    body.instruction(&Instruction::LocalGet(SCRATCH_F64));
    body.instruction(&Instruction::F64Ne); // v != v  → 1 if NaN
    body.instruction(&Instruction::If(BlockType::Result(ValType::I64)));
    body.instruction(&Instruction::I64Const(CANON_NAN as i64));
    body.instruction(&Instruction::Else);
    body.instruction(&Instruction::LocalGet(SCRATCH_F64));
    body.instruction(&Instruction::I64ReinterpretF64);
    body.instruction(&Instruction::End);
}

/// Emits: pop an `f64`, push its `Value` bits (i64). `LocalSet` (not `Tee`)
/// *consumes* the input `f64` so this is a true net `f64 → i64` with no leftover.
fn emit_box_double(body: &mut Function) {
    body.instruction(&Instruction::LocalSet(SCRATCH_F64));
    emit_box_scratch_f64(body);
}

/// How many Rust helper functions this method's module imports (`= max
/// CallHelper index + 1`, `0` if it calls none). Imported as `("env","h{i}")`
/// type `(i64)->i64` at WASM function indices `0..N` (before `run`, which is
/// index `N`). Kept minimal — an int-path method imports 0, so it needs no
/// helper support (works on the web runner, which can't yet vend helpers). The
/// runner binds `i` to [`crate::helpers::HELPERS`]`[i]`.
pub(crate) fn helper_count(ops: &[JitOp]) -> u32 {
    // Imports are contiguous `h0..=h{max}`, so the count is `max invoked index + 1`.
    let mask = helper_mask(ops);
    if mask == 0 { 0 } else { mask.ilog2() + 1 }
}

/// Bit `i` set iff this method actually **invokes** helper `h{i}` (see
/// [`crate::helpers::HELPERS`]). Unlike [`helper_count`] (which is `max index + 1`,
/// covering imported-but-unused slots), this is the exact set of helpers whose Rust
/// code runs — what [`Manifest::directable`] checks against [`SAFE_HELPER_MASK`] to
/// decide whether a callee's helper calls are all activation-context-only (safe under
/// a direct entry) or touch per-method/per-activation state (not).
fn helper_mask(ops: &[JitOp]) -> u32 {
    ops.iter()
        .filter_map(|op| match op {
            JitOp::CallHelper(i) => Some(*i),
            // Some ops implicitly call a fixed arity-1 helper.
            JitOp::IfTrueBoxed(_) | JitOp::IfFalseBoxed(_) => Some(TO_BOOLEAN),
            JitOp::PushScopeReal => Some(PUSH_SCOPE),
            JitOp::PopScopeReal => Some(POP_SCOPE),
            JitOp::GetScopeObject(_) => Some(GET_SCOPE_OBJECT),
            JitOp::GetOuterScope(_) => Some(GET_OUTER_SCOPE),
            JitOp::CoerceString => Some(COERCE_S),
            // Inline dm ops need their helper FALLBACK imported (taken when the
            // access misses the shared reservation — incl. an unshared
            // domainMemory, where `dm_len == 0`).
            JitOp::DmLoad(w) => Some(match *w {
                1 => DM_LOAD8,
                2 => DM_LOAD16,
                _ => DM_LOAD32,
            }),
            JitOp::DmLoadF(w) => Some(if *w == 4 { DM_LOADF32 } else { DM_LOADF64 }),
            JitOp::GetScriptGlobals(_) => Some(GET_SCRIPT_GLOBALS),
            JitOp::PushString(_) => Some(GET_PUSH_STRING),
            JitOp::Throw => Some(THROW),
            // A method with exception handlers always has `newcatch`; key the whole
            // dispatch machinery (`dispatch_exc`, `new_catch`, `pop_caught`) on the
            // highest of them so `h0..=h22` are imported whenever handlers are present.
            // (`POP_CAUGHT` is unsafe, so the mask excludes such methods too.)
            JitOp::NewCatch(_) => Some(POP_CAUGHT),
            JitOp::ReturnValueCoerced => Some(COERCE_RETURN),
            // The inline coerce's fallback needs its arity-1 helper imported.
            JitOp::CoerceInt(true) => Some(COERCE_I),
            JitOp::CoerceInt(false) => Some(COERCE_U),
            JitOp::CoerceBool => Some(TO_BOOLEAN),
            // The inline int inc/dec's non-int fallback (wrapping ToInt32 ± 1).
            JitOp::IncrementIBoxed | JitOp::IncDecLocalIValue(_, true) => Some(INCREMENT_I),
            JitOp::DecrementIBoxed | JitOp::IncDecLocalIValue(_, false) => Some(DECREMENT_I),
            // Number-semantics local inc/dec goes through h0/h1.
            JitOp::IncDecLocalNum(_, true) => Some(INCREMENT_NUM),
            JitOp::IncDecLocalNum(_, false) => Some(DECREMENT_NUM),
            // `hasnext2` fetches its two register outputs via h30/h31.
            JitOp::HasNext2(..) => Some(HN2_OBJECT),
            // `lookupswitch` coerces its selector to an i32 via the `coerce_i`
            // (ToInt32) helper before the `br_table`, so it must import `h0..=h12`.
            JitOp::LookupSwitch(_) => Some(COERCE_I),
            _ => None,
        })
        .fold(0u32, |m, i| m | (1 << i))
}

/// Helpers (`h{i}`) that read the current activation ONLY as a GC/domain **context
/// carrier** — no per-method table (multinames / coerce-classes / script-globals /
/// push-strings / return-type) and no per-activation scope stack. These are sound to
/// call from a **direct-entered** method: the caller's activation left in place is a
/// valid context, and a `valueOf`/`toString` re-entry they trigger is frame-safe under
/// the direct-call `DEPTH` bump ([`emit_call_direct`]). Bit `i` = `h{i}` is safe.
///
/// Included: `increment`/`decrement`/`negate`/`bit_not`/`not`/`to_boolean` (0–5),
/// `coerce_u`/`coerce_i`/`sxi1`/`sxi8`/`sxi16` (11–15), `coerce_s` (25),
/// `increment_i`/`decrement_i` (28–29). EXCLUDED (need state a direct entry doesn't set
/// up): scope (6/7/23/24), `dm` (8/9/10/26/27), `coerce_return` (16),
/// `get_script_globals` (17), `get_push_string` (18), the exception machinery (19–22),
/// and `hasnext2` (30/31).
pub(crate) const SAFE_HELPER_MASK: u32 = 0b11_1111        // 0..=5
    | (1 << PUSH_SCOPE)                                    // 6  \
    | (1 << GET_SCOPE_OBJECT)                              // 7   > scope stack (see SCOPE_HELPER_MASK)
    | (0b111 << DM_LOAD8)                                  // 8,9,10 (dm_load8/16/32 fallback)
    | (0b1_1111 << 11)                                     // 11..=15
    | (1 << POP_SCOPE)                                     // 23 /
    | (1 << COERCE_S)                                      // 25
    | (0b11 << DM_LOADF32)                                 // 26,27 (dm_load f32/f64 fallback)
    | (0b11 << INCREMENT_I);                               // 28,29

/// Helpers touching the shared activation's scope stack (`pushscope`/`getscopeobject`/
/// `popscope`). Safe in a direct entry ONLY when the caller retargets `scope_depth`
/// around the `call_indirect` (`emit_call_direct`'s `sen`/`sle`, gated on the callee's
/// scope flag). `getouterscope` (24) is NOT here — it reads the bound outer scope chain,
/// which is the CALLER's under a direct entry and can't be fixed by `scope_enter`.
pub(crate) const SCOPE_HELPER_MASK: u32 =
    (1 << PUSH_SCOPE) | (1 << GET_SCOPE_OBJECT) | (1 << POP_SCOPE);

/// [`JitOp::VCall`] kinds (bit `1 << kind`) that are SAFE under a direct entry: they
/// read only the shared activation as GC/domain context (valid on a direct call) plus
/// their own operand/`imm` — never a per-method `RUN_CTX` table, `bound_superclass_object`,
/// or the scope chain. Their `imm` is `0` or an object-level slot id (`constructslot`),
/// so NO baking is needed. EXCLUDED (still non-directable): the multiname/native/namespace
/// table kinds (`constructprop`/`callnative`/`deleteproperty`/`setprop*`/`pushnamespace` —
/// would need `imm` baked like `getproperty`), the SUPER kinds (`call/get/set_super` read
/// the running method's `bound_superclass_object`), and the SCOPE-capturing kinds
/// (`newclass`/`newfunction`/`newactivation` — `create_scopechain` reads the live scope).
pub(crate) const SAFE_VCALL_MASK: u32 = (1 << vc::CONSTRUCT_SLOT)
    | (1 << vc::CONSTRUCT)
    | (1 << vc::APPLY_TYPE)
    | (1 << vc::NEW_ARRAY)
    | (1 << vc::NEW_OBJECT)
    | (1 << vc::NEXT_VALUE)
    | (1 << vc::IN)
    | (1 << vc::TYPE_OF)
    | (1 << vc::COERCE_D)
    | (1 << vc::CONVERT_S);
// dm helpers are "safe" for a direct entry: they read the current activation's
// domainMemory (a GC/domain context, not a per-method table) and coerce the address
// (a `valueOf` re-entry, frame-safe under the direct-call DEPTH bump). Sound because all
// real domainMemory content is single-domain (Starling / CrossBridge), so the caller's
// domainMemory — whose `dm_base` `emit_call_direct` forwards — is the callee's too.

/// Whether a boxed method is **helper-dominated** — its work is mostly
/// JS-boundary crossings (`getproperty`/`getslot`/`callmethod`/generic helpers) the
/// JIT can't inline, with little inline compute to offset them. For such a method
/// the interpreter (fast native vtable dispatch, no per-call reg copy or boundary
/// crossing) beats the JIT, so [`crate::compile_method`] declines it.
///
/// Conservative by design: only the *expensive, unavoidable* crossings count against
/// the method, the inline ops the JIT genuinely accelerates (numeric compares,
/// bitwise, coerce, boxed-int arithmetic, inline domainMemory) count for it, and
/// neutral ops (local moves, pushes, control flow, returns) are ignored. A method is
/// declined only when it has several crossings *and* they clearly outweigh its inline
/// compute (`crossings > 2 × wins`) — so any method doing meaningful inline work,
/// including the FlasCC domainMemory hot paths, still compiles.
pub(crate) fn helper_dominated(ops: &[JitOp]) -> bool {
    let mut crossings = 0usize;
    let mut wins = 0usize;
    for &op in ops {
        match op {
            // Expensive JS-boundary crossings the JIT can't inline (a helper `call`).
            JitOp::GetProperty(_)
            | JitOp::GetSlot(_, _)
            | JitOp::CallMethod(..)
            | JitOp::CallHelper(_)
            | JitOp::CallHelper2(_)
            | JitOp::CallHelper3(_, _)
            | JitOp::VCall(..) => crossings += 1,
            // Inline compute the JIT does natively (no crossing) — a genuine win over
            // the interpreter. The inline compares/bitwise/coerce/arith take a helper
            // only on their rare non-numeric fallback, so they count as wins.
            JitOp::CmpNum(_)
            | JitOp::BitOpInt(_)
            | JitOp::ArithInt(_)
            | JitOp::ArithNum(_)
            | JitOp::CoerceInt(_)
            | JitOp::CoerceBool
            | JitOp::IncDecLocalIValue(_, _)
            | JitOp::AddIBoxed
            | JitOp::SubtractIBoxed
            | JitOp::MultiplyIBoxed
            | JitOp::IncrementIBoxed
            | JitOp::DecrementIBoxed
            | JitOp::DmLoad(_)
            | JitOp::DmStore(_)
            | JitOp::DmLoadF(_)
            | JitOp::DmStoreF(_) => wins += 1,
            // Neutral (local moves, pushes, dup/swap, control flow, returns): ignored.
            _ => {}
        }
    }
    crossings >= 4 && crossings > wins * 2
}

/// Whether this method uses the arity-2 getproperty import (`("env","gp")`).
pub(crate) fn has_getprop(ops: &[JitOp]) -> bool {
    ops.iter().any(|op| matches!(op, JitOp::GetProperty(_)))
}

/// Whether this method uses the arity-3 getpropertyfast import (`("env","gpf")`).
pub(crate) fn has_getprop_fast(ops: &[JitOp]) -> bool {
    ops.iter().any(|op| matches!(op, JitOp::GetPropertyFast(_, _)))
}

/// Whether this method uses the arity-2 getslot import (`("env","gs")`).
pub(crate) fn has_getslot(ops: &[JitOp]) -> bool {
    ops.iter().any(|op| matches!(op, JitOp::GetSlot(_, _)))
}

/// Whether this method uses the arity-2 findprop import (`("env","fp")`).
pub(crate) fn has_findprop(ops: &[JitOp]) -> bool {
    ops.iter().any(|op| matches!(op, JitOp::FindProp(..)))
}

/// How many arity-2 two-stack helper imports (`t{i}`) this method uses
/// (`= max CallHelper2 index + 1`, `0` if none).
fn helper2_count(ops: &[JitOp]) -> u32 {
    ops.iter()
        .filter_map(|op| match op {
            // Both the explicit helper call and the inline compare's fallback need
            // the `t{i}` import present.
            JitOp::CallHelper2(i)
            | JitOp::CmpNum(i)
            | JitOp::BitOpInt(i)
            | JitOp::ArithInt(i)
            | JitOp::ArithNum(i) => Some(i + 1),
            // The inline int arithmetic's non-int fallback (wrapping ToInt32 op).
            JitOp::AddIBoxed => Some(H2_ADD_I + 1),
            JitOp::SubtractIBoxed => Some(H2_SUBTRACT_I + 1),
            JitOp::MultiplyIBoxed => Some(H2_MULTIPLY_I + 1),
            JitOp::HasNext2(..) => Some(H2_HAS_NEXT2 + 1),
            _ => None,
        })
        .max()
        .unwrap_or(0)
}

/// Whether this method makes any `callmethod` (needs the call imports: the
/// `call_method` helper, the `push_call_arg` spill, and the `pending_error` check).
fn has_call(ops: &[JitOp]) -> bool {
    ops.iter().any(|op| matches!(op, JitOp::CallMethod(..)))
}

/// Whether this method makes any JIT→JIT direct call (`callmethod` behind the
/// call-cache) — needs the `cmi` (call+refill) import + the shared `pca`/`perr`.
fn has_call_direct(ops: &[JitOp]) -> bool {
    ops.iter().any(|op| matches!(op, JitOp::CallMethodDirect(..)))
}

/// Whether this method makes any `callproperty`/`callpropvoid` (needs the `cp`
/// call_property import + the shared `pca`/`perr` call imports).
fn has_callprop(ops: &[JitOp]) -> bool {
    ops.iter().any(|op| matches!(op, JitOp::CallProperty(..)))
}

/// Whether this method makes any `constructsuper` (needs the `csup` import + the
/// shared `pca`/`perr` call imports).
fn has_construct_super(ops: &[JitOp]) -> bool {
    ops.iter().any(|op| matches!(op, JitOp::ConstructSuper(_)))
}

/// Whether this method makes any `Op::Call` (function-value call — needs the `callv`
/// import + the shared `pca`/`perr` call imports).
fn has_call_value(ops: &[JitOp]) -> bool {
    ops.iter().any(|op| matches!(op, JitOp::CallValue(_)))
}

/// Whether the method makes any variadic call (`callmethod`/`callproperty`/
/// `constructsuper`/`call`) — gates the shared `pca` (arg spill) and `perr` (error bail).
/// Whether any op is a `coerces` (throwing `toString`) — it needs the `perr`
/// import + a post-op error bail/dispatch, like a call.
fn has_coerce_s(ops: &[JitOp]) -> bool {
    ops.iter().any(|op| matches!(op, JitOp::CoerceString))
}

/// Whether any op is a `coerce <class>` — it needs the `coerce` import, the `perr`
/// import, and a post-op error bail/dispatch (a failing coercion throws `#1034`).
fn has_coerce(ops: &[JitOp]) -> bool {
    // `ReturnValueCoerceBaked` also calls the `coerce` helper (with a baked class ptr).
    ops.iter()
        .any(|op| matches!(op, JitOp::Coerce(_) | JitOp::ReturnValueCoerceBaked(_)))
}

fn has_any_call(ops: &[JitOp]) -> bool {
    has_call(ops)
        || has_call_direct(ops)
        || has_callprop(ops)
        || has_construct_super(ops)
        || has_call_value(ops)
}

/// Whether this method inlines domainMemory (`li*`/`si*`) — needs memory 1
/// (Ruffle's own linear memory) imported + the `dm_base`/`dm_len` run params.
fn has_dm(ops: &[JitOp]) -> bool {
    ops.iter().any(|op| {
        matches!(
            op,
            JitOp::DmLoad(_) | JitOp::DmStore(_) | JitOp::DmLoadF(_) | JitOp::DmStoreF(_)
        )
    })
}

// dm helper function indices (keep in sync with `translate`'s `HELPER_DM_LOAD*`) and
// the `dm_store` HELPER3 kind (`translate::DM_STORE`).
pub(crate) const DM_LOAD8: u32 = 8;
pub(crate) const DM_LOAD16: u32 = 9;
pub(crate) const DM_LOAD32: u32 = 10;
const DM_STORE_KIND: u32 = 3;

/// Whether `op` is a **helper** domainMemory load/store (`li*`→`CallHelper`,
/// `si*`→`CallHelper3` kind `DM_STORE`) — these throw `#1506` on OOB via
/// `PENDING_ERROR`, so they must be followed by a `perr` bail. The **inline**
/// (`DmLoad`/`DmStore`) path still swallows OOB (returns `undefined`/skips) and is
/// deliberately excluded: a `perr` call after every inline dm op would be a
/// JS-boundary crossing per op on web, defeating the whole point of inlining.
fn is_dm_op(op: JitOp) -> bool {
    matches!(
        op,
        JitOp::CallHelper(DM_LOAD8 | DM_LOAD16 | DM_LOAD32) | JitOp::CallHelper3(DM_STORE_KIND, _)
    )
}

/// Whether any op is a domainMemory access (inline or helper) — gates the `perr`
/// import/bail (dm ops throw `#1506` out of band, like `callmethod`).
fn dm_throws(ops: &[JitOp]) -> bool {
    ops.iter().any(|&op| is_dm_op(op))
}

/// The arity-2 helper kinds that can **throw** via `PENDING_ERROR`:
/// `astypelate`/`istypelate` (a non-class type operand throws `#1041`/`#1009`/`#1010`,
/// matching the interpreter). Keep in sync with `translate`'s `AS_TYPE_LATE`/
/// `IS_TYPE_LATE` and [`crate::helpers::HELPERS2`].
const H2_AS_TYPE_LATE: u32 = 16;
const H2_IS_TYPE_LATE: u32 = 17;

/// The wrapping-int (`_i`) arithmetic helper kinds — the `AddIBoxed`/
/// `SubtractIBoxed`/`MultiplyIBoxed` inline ops' non-int fallback (`ToInt32` both
/// operands, wrapping i32 op). Keep in sync with [`crate::helpers::HELPERS2`].
pub(crate) const H2_ADD_I: u32 = 19;
pub(crate) const H2_SUBTRACT_I: u32 = 20;
pub(crate) const H2_MULTIPLY_I: u32 = 21;
/// `hasnext2` (the enumeration step): arity-2 helper returning the Boolean; the
/// two register outputs ride the `HN2_INDEX`/`HN2_OBJECT` fetchers. Throwing.
const H2_HAS_NEXT2: u32 = 22;

/// Whether `op` is a throwing arity-2 helper (`astypelate`/`istypelate`) — these
/// stash a thrown error in `PENDING_ERROR`, so they must be followed by a `perr`
/// bail/dispatch, like `coerce`/`coerces`.
fn is_throwing_h2(op: JitOp) -> bool {
    matches!(op, JitOp::CallHelper2(H2_AS_TYPE_LATE | H2_IS_TYPE_LATE))
}

/// Whether any op is a throwing arity-2 helper — gates the `perr` import/bail.
fn h2_throws(ops: &[JitOp]) -> bool {
    ops.iter().any(|&op| is_throwing_h2(op))
}

/// The ternary setslot helper kinds (`HELPERS3[0..=2]`). Keep in sync with
/// `translate`'s `SET_SLOT`/`SET_SLOT_NO_COERCE`/`SET_SLOT_COERCE_I`.
const SETSLOT_KIND_MAX: u32 = 2;

/// Whether `op` is a throwing property/slot access (`getproperty`/`getpropertyfast`/
/// `getslot`/the setslot kinds) — a null receiver, sealed-object miss, throwing
/// getter, or failing trait coercion throws (`#1009`/`#1010`/`#1069`/`#1034`, …)
/// via `PENDING_ERROR`, exactly like the interpreter, so these must be followed by
/// a `perr` bail/dispatch. (Swallowing them to `undefined` let the bogus value flow
/// on into slots/args and corrupt game state far from the fault.)
fn is_throwing_prop(op: JitOp) -> bool {
    matches!(
        op,
        JitOp::GetProperty(_)
            | JitOp::GetPropertyFast(_, _)
            | JitOp::GetSlot(_, _)
            | JitOp::FindProp(..)
            | JitOp::CallHelper3(0..=SETSLOT_KIND_MAX, _)
    )
}

/// Whether `op` can throw out of band (a call whose error routes through
/// `PENDING_ERROR`, or a dm op that can `#1506`). Such an op inside an exception
/// handler range needs the same dispatch as `throw`, which isn't wired — so
/// `compile_method` declines those methods.
pub(crate) fn is_throwing_call_or_dm(op: JitOp) -> bool {
    is_self_bailing_call(op)
        || matches!(op, JitOp::CoerceString | JitOp::Coerce(_) | JitOp::CoerceInt(_) | JitOp::HasNext2(..))
        || is_dm_op(op)
        || is_throwing_h2(op)
        || is_throwing_prop(op)
}

/// The call ops — throwing ops whose `emit_linear` arm emits its own inline `perr`
/// bail when the method has no handlers (`lay.inline_perr`); every other throwing
/// op gets its bail/dispatch from the compile loop. `GetSlot` joined them for the
/// inline fast path: its bail lives only in the helper-fallback arms, so the
/// (non-throwing) inline slot load skips the per-op `perr` crossing entirely.
fn is_self_bailing_call(op: JitOp) -> bool {
    matches!(
        op,
        JitOp::CallMethod(..)
            | JitOp::CallMethodDirect(..)
            | JitOp::CallProperty(..)
            | JitOp::ConstructSuper(_)
            | JitOp::CallValue(_)
            | JitOp::VCall(..)
            | JitOp::GetSlot(_, _)
    )
}

/// Emits the post-throw bail: `if pending_error() { return undefined }`. Shared by
/// `callmethod` and every dm op (both propagate a thrown error out of band via
/// `PENDING_ERROR`; `try_run` takes it after the run). `Return` is stack-polymorphic.
fn emit_perr_bail(body: &mut Function, perr_index: u32) {
    body.instruction(&Instruction::Call(perr_index));
    body.instruction(&Instruction::If(BlockType::Empty));
    body.instruction(&Instruction::I64Const(UNDEFINED_BITS as i64));
    body.instruction(&Instruction::Return);
    body.instruction(&Instruction::End);
}

/// Emits exception dispatch for a thrown error already stashed in `PENDING_ERROR`:
/// `dispatch_exc(op_idx)` → if a handler caught it (target >= 0), an if-chain over
/// `catch_bbs` (`(target_offset, bb)`) sets `BLOCK` and `Br(br_to_loop)` re-enters
/// the dispatch loop (→ the catch block); otherwise `Return` (propagate). The
/// caller must have cleared the wasm operand stack first. `br_to_loop` is the depth
/// from inside this emitter's own (caught) `If` out to the main loop.
fn emit_dispatch_core(
    body: &mut Function,
    op_idx: usize,
    catch_bbs: &[(i64, i32)],
    br_to_loop: u32,
) {
    body.instruction(&Instruction::I64Const(op_idx as i64));
    body.instruction(&Instruction::Call(DISPATCH_EXC));
    body.instruction(&Instruction::LocalSet(SCRATCH64));
    body.instruction(&Instruction::LocalGet(SCRATCH64));
    body.instruction(&Instruction::I64Const(0));
    body.instruction(&Instruction::I64GeS); // caught (target >= 0)?
    body.instruction(&Instruction::If(BlockType::Empty));
    for &(target, bb) in catch_bbs {
        body.instruction(&Instruction::LocalGet(SCRATCH64));
        body.instruction(&Instruction::I64Const(target));
        body.instruction(&Instruction::I64Eq);
        body.instruction(&Instruction::If(BlockType::Empty));
        body.instruction(&Instruction::I32Const(bb));
        body.instruction(&Instruction::LocalSet(BLOCK));
        body.instruction(&Instruction::End);
    }
    body.instruction(&Instruction::Br(br_to_loop));
    body.instruction(&Instruction::Else);
    body.instruction(&Instruction::I64Const(UNDEFINED_BITS as i64));
    body.instruction(&Instruction::Return);
    body.instruction(&Instruction::End);
}

/// [`JitOp::VCall`] kinds — the generic variadic-helper family (the single `vc`
/// import, `helpers::vcall(a, imm, spill, kind)`). Each kind mirrors one core op
/// the boxed path previously declined; extra stack operands beyond `a` are
/// spilled via `pca` (top-first) and drained by the helper. Every kind can throw
/// via `PENDING_ERROR` (→ the shared perr bail/dispatch, like a call).
pub(crate) mod vc {
    /// `constructslot`: `a`=receiver, `imm`=slot id, `spill`=argc → constructed object.
    pub const CONSTRUCT_SLOT: u32 = 0;
    /// `construct`: `a`=ctor value, `spill`=argc → constructed object.
    pub const CONSTRUCT: u32 = 1;
    /// `constructprop` (non-lazy): `a`=source, `imm`=mn `k`, `spill`=argc → object.
    pub const CONSTRUCT_PROP: u32 = 2;
    /// `callsuper` (non-lazy): `a`=receiver, `imm`=mn `k`, `spill`=argc → result.
    pub const CALL_SUPER: u32 = 3;
    /// `callnative`: `a`=receiver, `imm`=natives-table `k`, `spill`=argc → result
    /// (dropped for the void form).
    pub const CALL_NATIVE: u32 = 4;
    /// `applytype`: `a`=base, `spill`=num_types → the applied (parameterized) class.
    pub const APPLY_TYPE: u32 = 5;
    /// `newarray`: no receiver, `spill`=argc → the array object.
    pub const NEW_ARRAY: u32 = 6;
    /// `newobject`: no receiver, `spill`=2·argc (name/value pairs) → the object.
    pub const NEW_OBJECT: u32 = 7;
    /// `getsuper` (non-lazy): `a`=receiver, `imm`=mn `k` → the super property value.
    pub const GET_SUPER: u32 = 8;
    /// `setsuper` (non-lazy): `a`=receiver, `imm`=mn `k`, `spill`=1 (value). Void.
    pub const SET_SUPER: u32 = 9;
    /// `deleteproperty` (non-lazy): `a`=receiver, `imm`=mn `k` → Boolean.
    pub const DELETE_PROPERTY: u32 = 10;
    /// `nextvalue`: `a`=object, `spill`=1 (index) → the enumerant value.
    pub const NEXT_VALUE: u32 = 11;
    /// `in`: `a`=name value, `spill`=1 (object) → Boolean.
    pub const IN: u32 = 12;
    /// `setproperty` (static mn): `a`=receiver, `imm`=mn `k`, `spill`=1 (value). Void.
    pub const SET_PROP_STATIC: u32 = 13;
    /// `setproperty` (fast/lazy-name): `a`=receiver, `imm`=mn template `k`,
    /// `spill`=2 (name, value). Void.
    pub const SET_PROP_FAST: u32 = 14;
    /// `newclass`: `a`=base class value, `imm`=coerce-class-table `k` → ClassObject.
    pub const NEW_CLASS: u32 = 15;
    /// `newactivation`: no receiver, `imm`=coerce-class-table `k` → the activation object.
    pub const NEW_ACTIVATION: u32 = 16;
    /// `typeof`: `a`=value → the type-name String.
    pub const TYPE_OF: u32 = 17;
    /// `newfunction`: no receiver, `imm`=coerce-class-table `k` (an erased
    /// `Method`) → the FunctionObject closure (captures the current scope).
    pub const NEW_FUNCTION: u32 = 21;
    /// `pushnamespace`: no receiver, `imm`=namespace-table `k` → NamespaceObject.
    pub const PUSH_NAMESPACE: u32 = 18;
    /// `coerce_d` (`ToNumber`): `a`=value → Number.
    pub const COERCE_D: u32 = 19;
    /// `convert_s` (`ToString`): `a`=value → String.
    pub const CONVERT_S: u32 = 20;

    /// Whether the kind pops a real receiver/base/value operand as `a` — the
    /// no-receiver kinds make the emitter push a dummy `0` instead.
    pub fn has_receiver(kind: u32) -> bool {
        !matches!(kind, NEW_ARRAY | NEW_OBJECT | NEW_ACTIVATION | PUSH_NAMESPACE | NEW_FUNCTION)
    }
}

/// How many distinct ternary (arity-3) helper kinds exist. Keep in sync with
/// [`crate::helpers::HELPERS3`] (setslot ×3 + domainMemory int store + float store).
const HELPER3_KINDS: usize = 5;

/// The `dm_store_f` ternary kind (`HELPERS3[4]`) — the inline `sf32`/`sf64`
/// fallback (there is no `Op`-level helper translation for float dm; only the
/// inline path's miss branch calls it).
const DM_STORE_F_KIND: u32 = 4;

/// The import/helper usage of a compiled method — everything
/// [`crate::runner::run`] needs to bind the emitted module's imports, computed
/// once at compile time and cached alongside the bytes (so the runner never
/// re-derives it from `parsed_code`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Manifest {
    /// Number of arity-1 `h{i}` helper imports (`= helper_count`, `= max invoked index
    /// + 1`; imports are contiguous so this covers imported-but-unused slots too).
    pub num_helpers: u32,
    /// Bit `i` set iff helper `h{i}` is actually INVOKED (`= helper_mask`). Finer than
    /// `num_helpers`: [`Manifest::directable`] checks it against [`SAFE_HELPER_MASK`].
    pub helper_mask: u32,
    /// Whether the arity-2 `gp` (getproperty) import is used.
    pub has_getprop: bool,
    /// Whether the arity-2 `gs` (getslot) import is used.
    pub has_getslot: bool,
    /// Whether the property inline-cache miss import `gpi` is used (a
    /// [`JitOp::GetPropertyIc`] is present).
    pub has_ic: bool,
    /// Whether the arity-3 `gpf` (getpropertyfast) import is used.
    pub has_getprop_fast: bool,
    /// Whether the arity-2 `fp` (findprop, `findpropstrict`/`findproperty`) import
    /// is used.
    pub has_findprop: bool,
    /// Number of arity-2 two-stack `t{i}` helper imports (`= helper2_count`).
    pub num_helpers2: u32,
    /// Bit `k` set iff ternary helper kind `k` (`HELPERS3[k]`) is used.
    pub set3_mask: u32,
    /// Whether the `cm` (call_method) import is used (`= has_call`).
    pub has_call: bool,
    /// Whether the `cmi` (call_method + call-cache refill) import is used
    /// (`= has_call_direct`) — the JIT→JIT direct-call miss path.
    pub has_call_direct: bool,
    /// Whether the `cp` (call_property) import is used (`= has_callprop`).
    pub has_callprop: bool,
    /// Whether the `csup` (construct_super) import is used (`= has_construct_super`).
    pub has_construct_super: bool,
    /// Whether the `callv` (call_value, `Op::Call`) import is used (`= has_call_value`).
    pub has_call_value: bool,
    /// Whether memory 1 (`dm`, Ruffle's own memory) is imported for inline
    /// domainMemory (`= has_dm`).
    pub has_dm: bool,
    /// Whether any dm op (inline or helper) is present → the `perr` import is needed
    /// (dm OOB throws `#1506` out of band).
    pub dm_throws: bool,
    /// Whether the method has a `ReturnValueCoerced` op → `try_run` must resolve the
    /// declared return type for the `coerce_return` helper. Lets the hot path skip the
    /// per-call signature resolution for the (common) methods that don't need it.
    pub has_coerced_return: bool,
    /// Whether the method has a `GetScriptGlobals` op → `try_run` builds the per-run
    /// pre-resolved script-globals table (via `with_script_globals`).
    pub has_script_globals: bool,
    /// Whether the method has a `PushString` op → `try_run` builds the per-run
    /// pre-resolved string table (via `with_push_strings`).
    pub has_push_strings: bool,
    /// Whether the method has a `CoerceString` op → needs the `perr` import (a
    /// throwing `toString` propagates via `PENDING_ERROR`).
    pub has_coerce_s: bool,
    /// Whether the method has a `Coerce` op → needs the `coerce` import + the `perr`
    /// import (a failing coercion, `#1034`, propagates via `PENDING_ERROR`), and
    /// `try_run` builds the per-run class table (via `with_coerce_classes`).
    pub has_coerce: bool,
    /// Whether the method has a throwing arity-2 helper (`astypelate`/`istypelate`)
    /// → needs the `perr` import (a non-class type operand throws `#1041` et al.
    /// via `PENDING_ERROR`, matching the interpreter).
    pub h2_throws: bool,
    /// Whether the method has any [`JitOp::VCall`] → needs the `vc` import, the
    /// `pca` spill import, and the `perr` import (every kind can throw out of band).
    pub has_vcall: bool,
    /// Bitmask (`1 << kind`) of the distinct [`JitOp::VCall`] kinds the method uses.
    /// Directability keys off this: a method whose kinds are ALL in [`SAFE_VCALL_MASK`]
    /// stays a direct-call target (see [`Manifest::directable`]); `0` ⟺ `!has_vcall`.
    pub vcall_mask: u32,
    /// Whether the method has an inline `CoerceInt` (`coerce_i`/`coerce_u`) → needs
    /// the `perr` import: an `Object`→primitive coercion whose `valueOf`/`toString`
    /// don't yield a primitive throws `#1050` via `PENDING_ERROR` (the helper's
    /// fallback), so it's dispatched/bailed like any other throwing op.
    pub has_coerce_int: bool,
    /// Whether the method has a `HasNext2` (enumeration step) → needs the `perr`
    /// import: a throwing `ToInt32`/`get_next_enumerant` rides `PENDING_ERROR`.
    pub has_hasnext2: bool,
}

impl Manifest {
    /// Whether the module imports `perr` (pending_error) — the **single source of
    /// truth** for the import's presence, shared by the import section, [`layout`],
    /// and both runners (native + web) so their binding order can never desync.
    /// True when any op can throw out of band via `PENDING_ERROR`: calls, dm ops,
    /// `coerce`/`coerces`, `astypelate`/`istypelate`, and the property/slot
    /// accessors (`gp`/`gpf`/`gs` + the setslot kinds).
    /// Field-wise max/OR union — [`compile_generation`] lays every member out
    /// against the union of their manifests. The exhaustive destructure means a
    /// newly added `Manifest` field FAILS TO COMPILE here until it is folded —
    /// a forgotten field once emitted generation members calling an undeclared
    /// import (invalid module → every install failed → the entry-slot pool
    /// exhausted → methods silently fell back to the slow JS entry).
    pub(crate) fn union_with(&mut self, m: &Manifest) {
        let Manifest {
            num_helpers,
            helper_mask,
            has_getprop,
            has_getslot,
            has_ic,
            has_getprop_fast,
            has_findprop,
            num_helpers2,
            set3_mask,
            has_call,
            has_call_direct,
            has_callprop,
            has_construct_super,
            has_call_value,
            has_dm,
            dm_throws,
            has_coerced_return,
            has_script_globals,
            has_push_strings,
            has_coerce_s,
            has_coerce,
            h2_throws,
            has_vcall,
            vcall_mask,
            has_coerce_int,
            has_hasnext2,
        } = *m;
        self.num_helpers = self.num_helpers.max(num_helpers);
        self.helper_mask |= helper_mask;
        self.has_getprop |= has_getprop;
        self.has_getslot |= has_getslot;
        self.has_ic |= has_ic;
        self.has_getprop_fast |= has_getprop_fast;
        self.has_findprop |= has_findprop;
        self.num_helpers2 = self.num_helpers2.max(num_helpers2);
        self.set3_mask |= set3_mask;
        self.has_call |= has_call;
        self.has_call_direct |= has_call_direct;
        self.has_callprop |= has_callprop;
        self.has_construct_super |= has_construct_super;
        self.has_call_value |= has_call_value;
        self.has_dm |= has_dm;
        self.dm_throws |= dm_throws;
        self.has_coerced_return |= has_coerced_return;
        self.has_script_globals |= has_script_globals;
        self.has_push_strings |= has_push_strings;
        self.has_coerce_s |= has_coerce_s;
        self.has_coerce |= has_coerce;
        self.h2_throws |= h2_throws;
        self.has_vcall |= has_vcall;
        self.vcall_mask |= vcall_mask;
        self.has_coerce_int |= has_coerce_int;
        self.has_hasnext2 |= has_hasnext2;
    }

    pub fn needs_perr(&self) -> bool {
        self.has_call
            || self.has_call_direct
            || self.has_callprop
            || self.has_construct_super
            || self.has_call_value
            || self.has_vcall
            || self.dm_throws
            || self.has_coerce_s
            || self.has_coerce
            || self.has_coerce_int
            || self.has_hasnext2
            || self.h2_throws
            || self.has_getprop
            || self.has_getprop_fast
            || self.has_getslot
            || self.has_findprop
            || self.set3_mask & ((1 << (SETSLOT_KIND_MAX + 1)) - 1) != 0
    }

    /// Whether a JITed method with this manifest is a **direct-call target** for the
    /// in-wasm JIT→JIT `call_indirect` path (§8 / step 4): the caller places the
    /// callee's frame at `state_ptr + STRIDE` and calls its `run` wasm→wasm, without
    /// building an interpreter `Activation`.
    ///
    /// A direct entry does NOT establish the callee's own `Activation`, so the callee
    /// must need none of the state one carries: **scopes** (`findprop`/`getscriptglobals`),
    /// the **property inline cache**, **coercions** (`coerce`/`coerce_s`/`coerce_int`/a
    /// typed `returnvalue`), **domainMemory** (`dm`), or the generic unary/binary helpers
    /// (`h{i}`/`t{i}`) that read `activation()`. Those stay excluded.
    ///
    /// **Calls ARE allowed** (`has_call`/`has_call_direct` — the non-leaf case): a callee
    /// may invoke other methods via the `cm` helper or a nested direct call.
    /// [`emit_call_direct`] save+bump+restores the runner's `DEPTH` frame counter around
    /// every direct `call_indirect` (the `dp` global → memory 1), so a re-entrant `run`
    /// the callee triggers frames ABOVE it rather than aliasing its live frame; and the
    /// site bounds-checks the callee frame so an unbounded direct-recursion chain declines
    /// to the interpreter instead of running off the memory. The `cm`/`call_method` helper
    /// uses the current activation only as a GC/domain **context carrier** (it resolves the
    /// method off the receiver's vtable and builds the callee's own activation), so the
    /// outer activation left in place by the direct entry is correct.
    ///
    /// `getslots_all_null_safe`: every `GetSlot` in the callee is verifier-proven
    /// not-null (`VerifiedMethodInfo::null_safe_getslots`). A `GetSlot` already
    /// implies an OBJECT receiver (the verifier only emits it after resolving the
    /// property to a slot on a known class), so the inline fast path — which reads
    /// the slot directly, `activation()`-free (see `helpers::get_slot`) — always
    /// runs; null-safety additionally rules out the slow throw path. So a leaf whose
    /// only "helper" is `getslot` is directable exactly when all its getslots are
    /// null-safe. (`gs` is still IMPORTED for the defensive non-object fallback, but
    /// that arm is dead here — hence `has_getslot` is allowed under the flag.)
    pub(crate) fn directable(&self, getslots_all_null_safe: bool) -> bool {
        // `set_slot_no_coerce` (kind 1) IS allowed: it writes a slot with the GC write
        // barrier and never re-enters AS3 (no coercion → no `valueOf`), so it needs no
        // nested frame — it unblocks field-copy/setter methods as direct-call targets. The
        // coercing `set_slot` (0) and `set_slot_coerce_i` (2) can re-enter, dm stores (3–4)
        // are `has_dm`, so only bit 1 is cleared.
        // `set_slot_no_coerce` (1) and the dm STORES (3,4) are allowed; the coercing
        // `set_slot` (0) and `set_slot_coerce_i` (2) can re-enter AS3 unsafely.
        (self.set3_mask & !((1 << 1) | (1 << DM_STORE_KIND) | (1 << DM_STORE_F_KIND))) == 0
            // `has_call`/`has_call_direct` (a `cm` helper call or a nested JIT→JIT direct
            // call) ARE now allowed: [`emit_call_direct`] saves+bumps+restores the Rust-side
            // `DEPTH` frame counter (via the `dp` global, memory 1) around every direct
            // `call_indirect`, so a non-leaf callee's re-entrant `run`s (a `cm` → `run`, or a
            // nested direct call's miss) frame ABOVE the callee, not aliasing its live frame;
            // and the site bounds-checks the callee frame so an unbounded direct-recursion
            // chain declines to the interpreter rather than running off the memory. What stays
            // excluded is everything that needs the callee's OWN activation state (scopes,
            // property IC, coercions, typed returns, dm) — a direct entry doesn't establish it.
            // Arity-1 helpers: allowed iff every INVOKED helper is context-only
            // (`SAFE_HELPER_MASK` — numeric coercions/`to_boolean`/`coerce_s`); a scope,
            // dm, script-globals, push-string, coerce-return or exception helper needs
            // per-method/per-activation state a direct entry doesn't establish → excluded.
            && (self.helper_mask & !SAFE_HELPER_MASK) == 0
            // Arity-2 compare/arithmetic helpers (`t{i}`, [`crate::helpers::HELPERS2`])
            // are all context-only — `coerce_to_number`/`abstract_eq` etc. read the
            // activation only as a GC/domain context and a `valueOf` re-entry is
            // frame-safe under the direct-call `DEPTH` bump — so they're ALLOWED. (The
            // throwing `astypelate`/`istypelate` set `h2_throws`, still excluded below.)
            // `getproperty` (`gp`) and its inline cache (`gpi` miss) ARE now allowed: the
            // emitter BAKES the `Gc<Multiname>` pointer at each site (`mn_list[k]`), so the
            // helper reads no per-method `RUN_CTX.multinames` table — it needs only the
            // activation as context (valid under a direct entry) and its IC cell is the
            // callee's (`ic_base` threaded through the direct call). A triggered getter
            // re-enters via `run` and is frame-safe under the DEPTH bump.
            && (!self.has_getslot || getslots_all_null_safe)
            // `getpropertyfast` (`arr[i]` — pushes runtime params onto the activation
            // stack) and `findproperty` (`fp` — searches the SCOPE chain, which a direct
            // entry leaves as the caller's) stay excluded.
            && !self.has_getprop_fast
            && !self.has_findprop
            && !self.has_callprop
            && !self.has_construct_super
            && !self.has_call_value
            // `vcall` (the generic variadic-helper family) IS now allowed WHEN every kind
            // used is context-only (`SAFE_VCALL_MASK`: newarray/newobject/construct/
            // applytype/nextvalue/in/typeof/coerce_d/convert_s/constructslot). Those read
            // only the shared activation as GC/domain context (valid under a direct entry)
            // and their operands — no `RUN_CTX` table, superclass, or scope. The remaining
            // kinds (multiname/native/namespace table refs, `*_super`, `new{class,function,
            // activation}`) still block: they need `imm` baking or the callee's own frame.
            && (self.vcall_mask & !SAFE_VCALL_MASK) == 0
            // `has_coerce` (`Op::Coerce{class}`) IS now allowed: the emitter BAKES the
            // `Class` pointer at each site (`coerce_classes[k]`), so the `coerce` helper
            // reads no per-method `RUN_CTX.coerce_classes` table — it needs only the
            // activation as context (valid under a direct entry). `coerce_to_type` on an
            // object may re-enter (`valueOf`), which is frame-safe under the DEPTH bump.
            && !self.h2_throws
            && !self.has_hasnext2
            && !self.has_script_globals
            && !self.has_push_strings
            && !self.has_coerced_return
        // `has_dm` / `dm_throws` (inline `li*`/`si*` + the `dm_load` fallback) ARE now
        // allowed: the dm ops read the current activation's domainMemory (context, not a
        // per-method table) and the caller-forwarded `dm_base` is the callee's too under
        // single-domain content (see `SAFE_HELPER_MASK`). An OOB `#1506` rides
        // `PENDING_ERROR` to the caller's `perr` bail, and a `valueOf` address coercion is
        // frame-safe under the DEPTH bump.
    }

    /// Whether the method invokes a scope-stack helper ([`SCOPE_HELPER_MASK`]) — a
    /// directable such method needs its caller to `scope_enter`/`scope_leave` around the
    /// direct call (flagged into the cell's `ic_base` high bit; see `emit_call_direct`).
    pub(crate) fn uses_scope(&self) -> bool {
        self.helper_mask & SCOPE_HELPER_MASK != 0
    }

    /// The FIRST reason this manifest is not a direct-call target, or `None` if it is
    /// — the direct-call analogue of the JIT decline histogram (see
    /// `WasmJit::record_nondirectable`). Checks mirror [`Manifest::directable`]'s order,
    /// so `directable() == reason.is_none()`; it names which remaining feature keeps a
    /// JITed method paying the `exec`/Activation call path instead of a direct call.
    pub(crate) fn directable_decline_reason(
        &self,
        getslots_all_null_safe: bool,
    ) -> Option<&'static str> {
        if self.set3_mask & !((1 << 1) | (1 << DM_STORE_KIND) | (1 << DM_STORE_F_KIND)) != 0 {
            return Some("set_slot_coerce");
        }
        let unsafe_helpers = self.helper_mask & !SAFE_HELPER_MASK;
        if unsafe_helpers != 0 {
            // Name the lowest unsafe helper's category so the histogram tells scope/
            // exceptions (FUNDAMENTAL — need the callee's own activation) apart from
            // coerced-return / script-globals / push-string (BAKEABLE like `coerce`).
            return Some(match unsafe_helpers.trailing_zeros() {
                24 => "helper_getouterscope",                  // bound outer scope (not fixable)
                8 | 9 | 10 | 26 | 27 => "helper_dm_load",      // inline dm fallbacks
                16 => "helper_coerced_return",                 // RUN_CTX.return_type (bakeable)
                17 => "helper_script_globals",                 // pre-resolved globals (bakeable)
                18 => "helper_push_string",                    // pre-resolved strings (bakeable)
                19 | 20 | 21 | 22 => "helper_exception",       // throw/dispatch/newcatch/popcaught
                30 | 31 => "helper_hasnext2",
                _ => "helper_unsafe_other",
            });
        }
        if self.has_getslot && !getslots_all_null_safe {
            return Some("getslot_not_null_safe");
        }
        if self.has_getprop_fast {
            return Some("getpropfast");
        }
        if self.has_findprop {
            return Some("findprop_scope");
        }
        if self.has_callprop {
            return Some("callprop");
        }
        if self.has_construct_super {
            return Some("construct_super");
        }
        if self.has_call_value {
            return Some("call_value");
        }
        let unsafe_vcall = self.vcall_mask & !SAFE_VCALL_MASK;
        if unsafe_vcall != 0 {
            // Name the lowest unsafe kind's category so the histogram separates the
            // BAKEABLE table kinds (mn/native/namespace `imm`, like `getproperty`) from
            // the FUNDAMENTAL super/scope kinds (need the callee's own frame).
            return Some(match unsafe_vcall.trailing_zeros() {
                k if k == vc::CALL_SUPER || k == vc::GET_SUPER || k == vc::SET_SUPER => {
                    "vcall_super"
                }
                k if k == vc::NEW_CLASS || k == vc::NEW_FUNCTION || k == vc::NEW_ACTIVATION => {
                    "vcall_scope"
                }
                _ => "vcall_table", // constructprop/callnative/deleteproperty/setprop*/pushnamespace
            });
        }
        if self.h2_throws {
            return Some("astype_istype");
        }
        if self.has_hasnext2 {
            return Some("hasnext2");
        }
        if self.has_script_globals {
            return Some("script_globals");
        }
        if self.has_push_strings {
            return Some("push_strings");
        }
        if self.has_coerced_return {
            return Some("coerced_return");
        }
        None
    }
}

/// Computes the [`Manifest`] for a compiled op slice.
pub(crate) fn manifest(ops: &[JitOp]) -> Manifest {
    let mut set3_mask = 0u32;
    for op in ops {
        match op {
            JitOp::CallHelper3(h, _) => set3_mask |= 1 << h,
            // Inline dm stores need their ternary fallback imported (taken when
            // the access misses the shared reservation).
            JitOp::DmStore(_) => set3_mask |= 1 << DM_STORE_KIND,
            JitOp::DmStoreF(_) => set3_mask |= 1 << DM_STORE_F_KIND,
            _ => {}
        }
    }
    Manifest {
        num_helpers: helper_count(ops),
        helper_mask: helper_mask(ops),
        has_getprop: has_getprop(ops),
        has_getslot: has_getslot(ops),
        has_ic: ops.iter().any(|op| matches!(op, JitOp::GetPropertyIc(..))),
        has_getprop_fast: has_getprop_fast(ops),
        has_findprop: has_findprop(ops),
        num_helpers2: helper2_count(ops),
        set3_mask,
        has_call: has_call(ops),
        has_call_direct: has_call_direct(ops),
        has_callprop: has_callprop(ops),
        has_construct_super: has_construct_super(ops),
        has_call_value: has_call_value(ops),
        has_dm: has_dm(ops),
        dm_throws: dm_throws(ops),
        has_coerced_return: ops.iter().any(|op| matches!(op, JitOp::ReturnValueCoerced)),
        has_script_globals: ops.iter().any(|op| matches!(op, JitOp::GetScriptGlobals(_))),
        has_push_strings: ops.iter().any(|op| matches!(op, JitOp::PushString(_))),
        has_coerce_s: has_coerce_s(ops),
        has_coerce: has_coerce(ops),
        h2_throws: h2_throws(ops),
        has_vcall: ops.iter().any(|op| matches!(op, JitOp::VCall(..))),
        vcall_mask: ops.iter().fold(0u32, |m, op| match op {
            JitOp::VCall(kind, ..) => m | (1u32 << kind),
            _ => m,
        }),
        has_coerce_int: ops.iter().any(|op| matches!(op, JitOp::CoerceInt(_))),
        has_hasnext2: ops.iter().any(|op| matches!(op, JitOp::HasNext2(..))),
    }
}

/// Imported-function-index layout. The arity-1 helpers occupy `0..num_helpers`,
/// then (if used) `gp`, `gs`, then the used ternary helpers in kind order, then
/// the exported `run`. Indices for unused imports are meaningless placeholders.
#[derive(Clone)]
struct Layout {
    gp_index: u32,
    gs_index: u32,
    /// `gpi` (property inline-cache miss) import (valid only when `has_ic`).
    gpi_index: u32,
    /// `get_property_fast` import (arity-3; valid only when `has_getprop_fast`).
    gpf_index: u32,
    /// `find_prop` import (arity-2 `(k, strict) -> object`; valid only when
    /// `has_findprop`).
    fp_index: u32,
    /// Base function index of the arity-2 two-stack `t{i}` helpers.
    t_base: u32,
    set3_index: [u32; HELPER3_KINDS],
    /// `call_method` import (valid only when `has_call`).
    call_index: u32,
    /// `cmi` (call_method + call-cache refill) import (valid only when
    /// `has_call_direct`) — follows `cm`.
    cmi_index: u32,
    /// `sen`/`sle` (direct-call scope enter/leave) imports (valid only when `has_call_direct`).
    scope_enter_index: u32,
    scope_leave_index: u32,
    /// `call_property` import (valid only when `has_callprop`).
    callprop_index: u32,
    /// `construct_super` import (valid only when `has_construct_super`).
    csup_index: u32,
    /// `call_value` import (valid only when `has_call_value`).
    callv_index: u32,
    /// Shared arg-spill (`pca`, when either call kind is present) + error-bail
    /// (`perr`, when either call kind or a throwing dm op is present).
    pca_index: u32,
    perr_index: u32,
    /// `coerce` import (arity-2 `(value, class_idx) -> result`; valid only when
    /// `has_coerce`). Follows `perr` (a `coerce` implies `perr` is present).
    coerce_index: u32,
    /// `vc` import (arity-4 `(a, imm, spill, kind) -> result`; valid only when
    /// `has_vcall`). Follows `coerce`.
    vc_index: u32,
    run_index: u32,
    /// Whether call ops emit their own inline `perr` bail (`return` on a thrown
    /// error). Set `false` for methods with exception handlers, where the compile
    /// loop instead emits handler-aware dispatch after each throwable op.
    inline_perr: bool,
}

fn layout(ops: &[JitOp]) -> Layout {
    layout_of(&manifest(ops))
}

/// [`layout`] from an already-computed manifest — a GENERATION module (many
/// methods, one import section) computes a **union** manifest and lays every
/// member's body out against it.
fn layout_of(m: &Manifest) -> Layout {
    let mut next = m.num_helpers;
    let gp_index = next;
    next += m.has_getprop as u32;
    let gs_index = next;
    next += m.has_getslot as u32;
    let gpi_index = next;
    next += m.has_ic as u32;
    let gpf_index = next;
    next += m.has_getprop_fast as u32;
    // `fp` (find_prop) follows the property-read imports.
    let fp_index = next;
    next += m.has_findprop as u32;
    let t_base = next;
    next += m.num_helpers2;
    let mut set3_index = [0u32; HELPER3_KINDS];
    for (k, slot) in set3_index.iter_mut().enumerate() {
        *slot = next;
        next += ((m.set3_mask >> k) & 1) as u32;
    }
    // Call imports follow the set helpers: `cm` (call_method), `cp` (call_property),
    // `csup` (construct_super), then the shared `pca` (arg spill) and `perr` (error bail).
    // `vcall` spills its extra operands through the same `pca`, so it gates it too.
    let any_call =
        m.has_call || m.has_call_direct || m.has_callprop || m.has_construct_super || m.has_call_value || m.has_vcall;
    // `cm` (plain call_method) is imported for plain `CallMethod` ops AND for the
    // direct-call SENTINEL short-circuit (a known-non-directable callee calls `cm`
    // without re-refilling — see `emit_call_direct`), so `has_call_direct` needs it too.
    let call_index = next;
    next += (m.has_call || m.has_call_direct) as u32;
    let cmi_index = next;
    next += m.has_call_direct as u32;
    // `sen`/`sle` (scope enter/leave) follow `cmi` — imported together under `has_call_direct`.
    let scope_enter_index = next;
    next += m.has_call_direct as u32;
    let scope_leave_index = next;
    next += m.has_call_direct as u32;
    let callprop_index = next;
    next += m.has_callprop as u32;
    let csup_index = next;
    next += m.has_construct_super as u32;
    let callv_index = next;
    next += m.has_call_value as u32;
    let pca_index = next;
    next += any_call as u32;
    let perr_index = next;
    next += m.needs_perr() as u32;
    // `coerce` (arity-2) follows `perr`, matching the import section's order.
    let coerce_index = next;
    next += m.has_coerce as u32;
    // `vc` (arity-4, the generic variadic helper) follows `coerce`.
    let vc_index = next;
    next += m.has_vcall as u32;
    Layout {
        gp_index,
        gs_index,
        gpi_index,
        gpf_index,
        fp_index,
        t_base,
        set3_index,
        call_index,
        cmi_index,
        scope_enter_index,
        scope_leave_index,
        callprop_index,
        csup_index,
        callv_index,
        pca_index,
        perr_index,
        coerce_index,
        vc_index,
        run_index: next,
        inline_perr: true,
    }
}

/// A `lookupswitch`'s resolved targets (op indices). Stored in a side-table
/// passed to [`compile`] because [`JitOp`] is `Copy` and can't hold the `Vec`;
/// [`JitOp::LookupSwitch`] carries a `u32` index into a `&[SwitchTable]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SwitchTable {
    /// Op index taken when the selector is out of `cases` range.
    pub default: usize,
    /// Op index per case value (`cases[selector]`).
    pub cases: Box<[usize]>,
}

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

    // --- Boxed-`Value` ops (raw NaN-boxed `u64` on the WASM stack, no unboxing).
    // These carry object/GC-aware values, which arithmetic ops can't; they feed
    // the imported Rust helpers that do GC-aware work. A method uses either the
    // int fast path or this boxed path (mixing i32/i64 on one stack is future work).
    /// Push local `i`'s raw 8-byte `Value` (no `i32.wrap`).
    GetLocalValue(u32),
    /// Pop a raw `Value`; store it into local `i`.
    SetLocalValue(u32),
    /// Pop a raw `Value` argument, call imported helper `index` (`(i64)->i64`),
    /// push its raw `Value` result. The runner binds helpers to Rust functions
    /// that reach the current `Activation` (GC stays in Rust).
    CallHelper(u32),
    /// Pop two raw `Value`s `(v1, v2)`, call arity-2 two-stack helper `index`
    /// ([`crate::helpers::HELPERS2`] — compares), push its `Value` result. Net -1.
    CallHelper2(u32),
    /// **Inline** boxed numeric compare with a helper fallback. Pops `(v1, v2)`;
    /// if both are numeric (`int` or `Number`) does the `f64` comparison inline and
    /// pushes the boxed `Boolean` result — else falls back to two-stack helper
    /// `index` ([`crate::helpers::HELPERS2`], same `CMP_*` index). `index` selects
    /// the comparison (`0`=eq `1`=lt `2`=le `3`=gt `4`=ge). Net -1. `f64.{lt,le,
    /// gt,ge,eq}` return `0` for a NaN operand, exactly matching the helper's
    /// `abstract_lt`/`abstract_eq` NaN semantics.
    CmpNum(u32),
    /// **Inline** boxed bitwise op with a helper fallback. Pops `(v1, v2)`; if both
    /// are int-boxed does the `i32` bitwise op inline and re-boxes the int result —
    /// else falls back to two-stack helper `index` ([`crate::helpers::HELPERS2`],
    /// same `BIT_*`/shift index: `5`=and `6`=or `7`=xor `8`=lshift `9`=rshift). The
    /// inline path is only taken for int operands (whose low 32 bits are already the
    /// `ToInt32` value); a `Number` operand needs the helper's wrapping `ToInt32`
    /// (`i32.trunc_sat` would saturate, not wrap). Shifts mask the count by `0x1F`.
    /// Net -1.
    BitOpInt(u32),
    /// **Inline** boxed generic numeric arithmetic (`multiply`/`subtract`) with a
    /// helper fallback. Pops `(v1, v2)`; if both are int-boxed does the op in `i64`
    /// and boxes the result as an `int` when it fits `i32`, else as a `Number` —
    /// exactly matching the helper's checked-int-else-Number semantics. Falls back to
    /// two-stack helper `index` ([`crate::helpers::HELPERS2`], `11`=multiply
    /// `12`=subtract) for non-int operands. Numeric-but-not-both-int operands take
    /// a middle inline path: the f64 op, boxed as a `Number`. Net -1.
    ArithInt(u32),
    /// **Inline** boxed numeric arithmetic whose result is ALWAYS a `Number`
    /// (`divide`, [`crate::helpers::HELPERS2`] `13`): both operands numeric → the
    /// f64 op inline (boxed `Number`, NaN canonicalized), else the two-stack
    /// helper fallback. Net -1.
    ArithNum(u32),
    /// **Inline** boxed `coerce_i`/`coerce_u` with a helper fallback. Pops one
    /// `Value`; if it is int-boxed passes it through unchanged for `coerce_i` (an
    /// int's `ToInt32` is itself), or for `coerce_u` when it is also non-negative
    /// (`ToUint32` == the int) — else falls back to the arity-1 helper (`coerce_i`
    /// = [`COERCE_I`], `coerce_u` = [`COERCE_U`]). `signed` selects `coerce_i`. Net 0.
    CoerceInt(bool),
    /// `coerceb`: pop a `Value`, push `ToBoolean(v)` as a `Boolean` `Value` (the
    /// `to_boolean` helper [`TO_BOOLEAN`] → raw `0`/`1`, OR-ed with `VALUE_BOOL_MARK`).
    CoerceBool,
    /// Swap the top two raw `Value`s (`swap`).
    SwapValue,
    /// Return the top raw `Value` as-is (already boxed).
    ReturnValueBoxed,
    /// Return the top `Value` **coerced to the method's declared return type** (via
    /// the `coerce_return` helper reading a thread-local; `#1034` on failure). Used
    /// when `returnvalue` carries a non-`*` `return_type` — the raw value can differ
    /// (e.g. `:uint` of `-10` → `4294967286`, `:Vector.<T>` of a generic vector).
    ReturnValueCoerced,
    /// Like [`JitOp::ReturnValueCoerced`] but with the return-type `Class` pointer BAKED
    /// as an immediate (`compile_method` rewrites `ReturnValueCoerced` to this once it
    /// resolves): coerces the top `Value` to that class via the `coerce` helper (no
    /// per-method `RUN_CTX.return_type` read), then returns — so a typed-return method is
    /// a direct-call target. Sets `has_coerce` (imports `coerce`), not `has_coerced_return`.
    ReturnValueCoerceBaked(u64),
    /// Return the given `Value` bits (`returnvoid` coerced to the method's declared
    /// `return_type`: `undefined`/`0`/`false`/`null`) — ignores the operand stack.
    ReturnVoidBoxed(u64),
    /// Duplicate the top raw `Value` (boxed `dup`; the i64 counterpart of [`JitOp::Dup`]).
    DupValue,
    /// Store the top `Value` into local `i`, *keeping it on the stack* (`storelocal`
    /// — the verifier's `setlocal i; getlocal i` fusion; boxed counterpart of
    /// [`JitOp::StoreLocalDouble`]).
    StoreLocalValue(u32),
    /// **Inline** domainMemory load (`li8`/`li16`/`li32`): pop an address `Value`,
    /// push the loaded int `Value`. Emitted as an inline bounds check + `i32.load`
    /// (width `1`/`2`/`4`; `1`/`2` zero-extend, `4` is the raw i32) on **memory 1**
    /// (Ruffle's own linear memory, holding domainMemory at `dm_base`) — no helper
    /// crossing. OOB → `undefined`, matching the interpreter. Width is the `u32`.
    DmLoad(u32),
    /// **Inline** domainMemory store (`si8`/`si16`/`si32`): pop `(value, addr)` and
    /// write `value`'s low `width` bytes to `dm_base + addr` on memory 1. OOB is
    /// skipped, matching the interpreter. Width (`1`/`2`/`4`) is the `u32`.
    DmStore(u32),
    /// **Inline** domainMemory *float* load (`lf32`/`lf64`): pop an addr `Value`,
    /// read `width` bytes (`4`=f32/`8`=f64) from `dm_base + addr` on memory 1,
    /// promote to `f64`, push a `Number` `Value`. OOB → `undefined`. Width is the `u32`.
    DmLoadF(u32),
    /// **Inline** domainMemory *float* store (`sf32`/`sf64`): pop `(value, addr)`,
    /// coerce `value` to `f64` (int/number), write `width` bytes (`4`=f32/`8`=f64) to
    /// `dm_base + addr`. OOB skipped. Width is the `u32`.
    DmStoreF(u32),
    // Boxed int arithmetic: the operands are int-typed `Value`s (`VALUE_INT_MARK |
    // i32`), guaranteed by the verifier (it only emits `*i` ops on ints). Each
    // `i32.wrap`s its operand(s), does the i32 op, and re-boxes the int result.
    /// Pop two int `Value`s; push `a + b` / `a - b` / `a * b` as an int `Value`.
    AddIBoxed,
    SubtractIBoxed,
    MultiplyIBoxed,
    /// Pop one int `Value`; push `v + 1` / `v - 1` as an int `Value`.
    IncrementIBoxed,
    DecrementIBoxed,
    /// Pop the receiver `Value`, read the method's `k`-th multiname, and push
    /// `receiver.get_property(mn)`'s raw `Value` (via the arity-2 `gp` import).
    GetProperty(u32),
    /// `getproperty` behind a monomorphic **inline cache** (`u32` = multiname
    /// `k`, `u32` = per-method IC site index): an inline vtable-class guard loads
    /// the cached slot directly on a hit, else calls the `gpi` helper which
    /// resolves and refills the cell (see [`emit_getprop_ic`]). Same net stack
    /// effect as [`JitOp::GetProperty`] — pop receiver, push value. Only emitted
    /// when the inline object layout is available (`slot_layout().is_some()`).
    GetPropertyIc(u32, u32),
    /// `getpropertyfast`: a dynamic-name property read. Stack `[.., receiver, name]`;
    /// calls the arity-3 `gpf` (`get_property_fast`) helper with `(receiver, name, k)`
    /// (`k` = the lazy multiname template), pushing the result. Net -1.
    /// `getpropertyfast` (dynamic index). `bool` = `returns_number` (a
    /// `Vector.<Number>` element) — phase-2 unbox may keep the result f64.
    GetPropertyFast(u32, bool),
    /// `findpropstrict`/`findproperty` (`bool` = strict): resolve the method's
    /// `k`-th multiname through the scope chain and push the scope object that has
    /// it (via the arity-2 `fp` import; `(k, strict) -> object`). Net +1 — takes no
    /// operand. Strict throws `#1065` on a miss (→ perr bail/dispatch); non-strict
    /// falls back to the global scope. The FlasCC C-call idiom
    /// (`findpropstrict F_fn; …; callpropvoid F_fn`) and `getlex`'s first half.
    FindProp(u32, bool),
    /// Pop the receiver `Value` and push `receiver.get_slot(slot_id)`'s raw
    /// `Value` (via the arity-2 `gs` import). The verifier's resolved form of a
    /// typed property read — a direct slot fetch, no multiname lookup.
    /// `getslot`: pop receiver, push slot value. The `bool` is `returns_number`
    /// (optimizer proved the slot class is `Number`) — phase-2 unbox may keep it f64.
    GetSlot(u32, bool),
    /// Push a boxed `int` `Value` constant (`VALUE_INT_MARK | v`). The boxed
    /// counterpart of `PushInt` (raw `Value` on the stack, not a raw i32).
    PushIntValue(i32),
    /// Push a boxed `Value` constant given its raw NaN-boxed bits — for primitive
    /// constant pushes (`pushuint`/`pushdouble`/`pushnull`/`pushtrue`/`pushfalse`/
    /// `pushundefined`) whose bits are computed at translate time (no `Gc`).
    PushConst(u64),
    /// Pop the receiver `Value` and the value `Value` (top), and write the value
    /// into the receiver's slot via arity-3 ternary helper `helper` (a
    /// [`crate::helpers::HELPERS3`] index — setslot variants), passing `imm` (the
    /// slot id) as the 3rd arg. Consumes both stack operands (net -2).
    CallHelper3(u32, u32),
    /// Pop one `Value`; branch to `target` if `ToBoolean(v)` is true. The boxed
    /// counterpart of [`JitOp::IfTrue`] — coerces the `Value` to a condition via
    /// the `to_boolean` helper ([`TO_BOOLEAN`]) before branching.
    IfTrueBoxed(usize),
    /// Pop one `Value`; branch to `target` if `ToBoolean(v)` is false.
    IfFalseBoxed(usize),
    /// `pushstring`: push the `k`-th string constant's `Value` bits. Rewritten to
    /// [`JitOp::PushBits`] in `compile_method` (the string's `Gc` is non-moving and
    /// alive for the method's lifetime — same as a baked multiname — so its bits ARE
    /// bakeable). Kept as the pre-rewrite form. Net +1.
    PushString(u32),
    /// Push a pre-resolved NaN-boxed `Value`'s raw bits as an immediate `i64.const` —
    /// no helper, no per-method table. `GetScriptGlobals`/`PushString` are rewritten to
    /// this once their bits resolve at compile time (baked like a multiname pointer);
    /// making a method that only used those helpers a direct-call target. Net +1.
    PushBits(u64),
    /// `throw`: pop a `Value`, stash `Error::from_value(v)` as the pending error
    /// (via the `throw_value` helper [`THROW`]), and `Return`. `try_run` propagates
    /// the pending error. A terminator. Sound only when the method has no exception
    /// handlers (else the throw would be caught locally) — gated in `compile_method`.
    Throw,
    /// `newcatch`: push the catch scope object for exception handler `index` (via
    /// the `new_catch` helper [`NEW_CATCH`]). Net +1.
    NewCatch(u32),
    /// `popscope` when the method reads scopes: pop the real scope stack (via the
    /// `pop_scope` helper [`POP_SCOPE`]). No operand-stack effect. (When scopes are
    /// unused, `popscope` translates to `Nop`.)
    PopScopeReal,
    /// `getouterscope index`: push the `index`-th outer (captured) scope's values
    /// object (via the `get_outer_scope` helper [`GET_OUTER_SCOPE`]). Net +1.
    GetOuterScope(u32),
    /// `coerces`: `ToString(v)` (via the `coerce_s` helper [`COERCE_S`]). A throwing
    /// `toString` sets `PENDING_ERROR`, so it's followed by a perr bail/dispatch.
    /// Net 0.
    CoerceString,
    /// `coerce <class>`: `ToType(v, class)` (via the `coerce` import). The `u32`
    /// indexes the per-run class table (built in `try_run` from the method's
    /// `Op::Coerce` classes, in op order), installed via `with_coerce_classes`. A
    /// failing coercion (`#1034`) sets `PENDING_ERROR`, so it's followed by a perr
    /// bail/dispatch like a throwing call. Net 0.
    Coerce(u32),
    /// `inclocal_i`/`declocal_i` on a boxed int local: `local[index] ±= 1` in place
    /// (unbox the int, add/sub 1, re-box). No operand-stack effect. `bool` = increment.
    IncDecLocalIValue(u32, bool),
    /// `inclocal`/`declocal` (*Number* semantics): `local[i] = ToNumber(local[i]) ± 1`
    /// in place via the `increment`/`decrement` helper. No operand-stack effect.
    IncDecLocalNum(u32, bool),
    /// `hasnext2 object_reg index_reg`: the enumeration step. Loads both register
    /// values, calls the `hasnext2` arity-2 helper (Boolean result, register
    /// outputs stashed), stores the outputs back into the two locals via the
    /// `HN2_INDEX`/`HN2_OBJECT` fetchers, pushes the Boolean. Net +1. Throwing.
    HasNext2(u32, u32),
    /// `lookupswitch`: pop an int selector; branch to `cases[selector]` (op index),
    /// or `default` if the selector is out of range. The `u32` indexes the
    /// compile's [`SwitchTable`] side-table (the case list can't live in a `Copy`
    /// `JitOp`). Multi-target, so [`JitOp::target`] returns `None` for it and
    /// [`basic_block_leaders`] reads the side-table to find its successors.
    LookupSwitch(u32),
    /// Pop a `Value` and push it onto the real Activation scope stack (`pushscope`,
    /// via the `push_scope` helper). Used only when the method also reads scopes.
    PushScopeReal,
    /// Push local scope `index` (`getscopeobject`, via the `get_scope_object`
    /// helper reading the Activation's scope frame).
    GetScopeObject(u32),
    /// `getscriptglobals`: push the `k`-th script's global object. The bits are
    /// **pre-resolved** in `try_run` (which has the context) into a per-run table;
    /// this just calls the `get_script_globals` helper ([`GET_SCRIPT_GLOBALS`]) with
    /// the immediate `k`, which reads that table — no receiver, no throw. Net +1.
    GetScriptGlobals(u32),
    /// `callmethod index argc`: a resolved (disp-id) method call. Pops `argc`
    /// args (spilled to a thread-local via `push_call_arg`) + the receiver, calls
    /// the `call_method` helper, and pushes the result iff `push`. Emits a
    /// pending-error check after the call so a thrown error bails the whole method
    /// (see `crate::helpers::call_method`). Fields: `(disp_id, argc, push)`.
    CallMethod(u32, u32, bool),
    /// `callmethod` behind a monomorphic JIT→JIT **direct-call cache** (`u32` = disp
    /// id, `u32` = argc, `bool` = push, `u32` = per-method call-cache site): an inline
    /// vtable guard `call_indirect`s a directable callee's compiled `run` on a hit,
    /// else the `cmi` helper does the call and refills the cell (see
    /// [`emit_call_direct`]). Same net stack effect as [`JitOp::CallMethod`]. Only
    /// emitted when the inline object layout is available (`ic_available()`).
    CallMethodDirect(u32, u32, bool, u32),
    /// `callproperty`/`callpropvoid`: a multiname method call. Like [`JitOp::CallMethod`]
    /// but resolves the `k`-th multiname (the run's multiname table) instead of a
    /// disp-id — spills `argc` args (`push_call_arg`), calls the `call_property`
    /// helper (`cp` import), pushes the result iff `push` (`callproperty` pushes,
    /// `callpropvoid` discards), and bails on a pending error. Only **non-lazy**
    /// multinames compile (translate declines lazy ones). Fields: `(mn_k, argc, push)`.
    CallProperty(u32, u32, bool),
    /// `constructsuper argc`: invoke the superclass constructor on the receiver.
    /// Spills `argc` args, calls the `csup` (`construct_super`) helper with the
    /// receiver + argc, discards the void result, and bails on a pending error. Net
    /// `-(argc + 1)` (consumes receiver + args, pushes nothing).
    ConstructSuper(u32),
    /// `call argc` (`Op::Call`): call a **function value** on the stack. Stack is
    /// `[.., function, receiver, args…]`; spills `argc` args, calls the `callv`
    /// (`call_value`) helper with `(function, receiver, argc)`, pushes the result,
    /// and bails on a pending error. Net `-(argc + 1)`.
    CallValue(u32),
    /// The generic variadic-helper family (`(kind, imm, spill, push)`): one op per
    /// [`vc`] kind — `constructslot`/`newclass`/`setproperty`/`applytype`/… Spills
    /// `spill` extra operands (`pca`, top-first), pops the receiver `a` (or pushes a
    /// dummy `0` for the no-receiver kinds), calls the `vc` import with
    /// `(a, imm, spill, kind)`, pushes the result iff `push`, and bails/dispatches
    /// on a pending error (every kind can throw). `imm` is kind-specific: a slot id,
    /// a multiname/coerce-class/natives/namespace table index, or unused.
    VCall(u32, u32, u32, bool),

    // --- Double fast path (unboxed `f64` on the WASM stack). Like the int fast
    // path but for `Number` values: numeric ops are emitted *inline* (native
    // `f64` arithmetic), no helper calls. A method uses this path only when every
    // value it touches is provably a `Number` (see `analysis::double_sound`).
    /// Push local `i` unboxed as `f64` (`f64.reinterpret` of the slot bits).
    GetLocalDouble(u32),
    /// Pop an `f64`, box it (NaN→CANON_NAN), store into local `i`.
    SetLocalDouble(u32),
    /// Box the top `f64` and store it into local `i`, *leaving it on the stack*
    /// (`storelocal` — the verifier's fusion of `setlocal i; getlocal i`).
    StoreLocalDouble(u32),
    /// Push an `f64` immediate (the value's bit pattern).
    PushDouble(u64),
    /// Pop two `f64`s; push their sum / difference / product / quotient.
    AddD,
    SubtractD,
    MultiplyD,
    DivideD,
    /// Pop one `f64`; push `+1` / `-1` / negation.
    IncrementD,
    DecrementD,
    NegateD,
    /// Pop an `f64`, box it as a `Number` `Value`, and return it.
    ReturnDouble,
}

impl JitOp {
    /// The branch target if this op is a (conditional or unconditional) branch.
    pub(crate) fn target(self) -> Option<usize> {
        match self {
            JitOp::Jump(t)
            | JitOp::IfLt(t)
            | JitOp::IfGe(t)
            | JitOp::IfFalse(t)
            | JitOp::IfTrue(t)
            | JitOp::IfTrueBoxed(t)
            | JitOp::IfFalseBoxed(t) => Some(t),
            _ => None,
        }
    }

    /// Whether this op ends a basic block (branch or return).
    pub(crate) fn is_terminator(self) -> bool {
        matches!(
            self,
            JitOp::Jump(_)
                | JitOp::IfLt(_)
                | JitOp::IfGe(_)
                | JitOp::IfFalse(_)
                | JitOp::IfTrue(_)
                | JitOp::IfTrueBoxed(_)
                | JitOp::IfFalseBoxed(_)
                | JitOp::LookupSwitch(_)
                | JitOp::Throw
                | JitOp::ReturnValue
                | JitOp::ReturnValueBoxed
                | JitOp::ReturnValueCoerced
                | JitOp::ReturnVoidBoxed(_)
                | JitOp::ReturnDouble
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

/// Whether check-elision (type-specialization phase 1) runs: on a proven-numeric
/// `ArithInt`, drop the runtime `is_numeric` guard + `valueOf` helper fallback.
/// Sound whenever the [`crate::typed`] repr analysis is correct (guarded by the
/// arity safety net) — the differential verify suite is the final gate.
const JIT_CHECK_ELISION: bool = true;

/// Check-elided [`JitOp::ArithInt`] (`multiply`/`subtract`/`add`, index 11/12/18):
/// the type analysis proved BOTH operands numeric (`int` box or `Number`), so the
/// runtime `is_numeric` guard and the string/object `valueOf` helper fallback are
/// dropped. Byte-for-byte identical to the general `ArithInt` lowering EXCEPT the
/// tier-1 `else` goes straight to the f64 numeric path. Same boxing (int-if-fits
/// else Number), so `is int` semantics are preserved. Caller checks `depth >= 2`
/// and does `depth -= 1`.
/// Unbox `local` to `f64`, choosing the heap-box-safe path
/// ([`emit_numval_boxsafe`]) iff `boxsafe` (the operand is a possibly-heap-boxed
/// `NumberBoxed`), else the cheaper branchless [`emit_numval`].
fn emit_numval_sel(body: &mut Function, local: u32, boxsafe: bool) {
    if boxsafe {
        emit_numval_boxsafe(body, local);
    } else {
        emit_numval(body, local);
    }
}

fn emit_arith_int_elided(body: &mut Function, index: u32, a_boxsafe: bool, b_boxsafe: bool) {
    body.instruction(&Instruction::LocalSet(SCRATCH64)); // B
    body.instruction(&Instruction::LocalSet(SCRATCH64_2)); // A
    // Tier 1: both int-boxed → i64 op, boxed int-if-fits-else-Number.
    emit_is_int(body, SCRATCH64_2);
    emit_is_int(body, SCRATCH64);
    body.instruction(&Instruction::I32And);
    body.instruction(&Instruction::If(BlockType::Result(ValType::I64)));
    body.instruction(&Instruction::LocalGet(SCRATCH64_2));
    body.instruction(&Instruction::I32WrapI64);
    body.instruction(&Instruction::I64ExtendI32S);
    body.instruction(&Instruction::LocalGet(SCRATCH64));
    body.instruction(&Instruction::I32WrapI64);
    body.instruction(&Instruction::I64ExtendI32S);
    body.instruction(&match index {
        11 => Instruction::I64Mul,
        18 => Instruction::I64Add,
        _ => Instruction::I64Sub,
    });
    body.instruction(&Instruction::LocalSet(SCRATCH64_2));
    body.instruction(&Instruction::LocalGet(SCRATCH64_2));
    body.instruction(&Instruction::LocalGet(SCRATCH64_2));
    body.instruction(&Instruction::I64Extend32S);
    body.instruction(&Instruction::I64Eq);
    body.instruction(&Instruction::If(BlockType::Result(ValType::I64)));
    body.instruction(&Instruction::LocalGet(SCRATCH64_2));
    body.instruction(&Instruction::I32WrapI64);
    emit_box_int(body);
    body.instruction(&Instruction::Else);
    body.instruction(&Instruction::LocalGet(SCRATCH64_2));
    body.instruction(&Instruction::F64ConvertI64S);
    body.instruction(&Instruction::I64ReinterpretF64);
    body.instruction(&Instruction::End);
    // ELIDED else: operands proven numeric → the f64 path directly (no is_numeric
    // guard, no helper fallback). A `NumberBoxed` operand (`a_boxsafe`/`b_boxsafe`)
    // is unboxed heap-box-safely; the result is a canonical `Number`.
    body.instruction(&Instruction::Else);
    emit_numval_sel(body, SCRATCH64_2, a_boxsafe);
    emit_numval_sel(body, SCRATCH64, b_boxsafe);
    body.instruction(&match index {
        11 => Instruction::F64Mul,
        18 => Instruction::F64Add,
        _ => Instruction::F64Sub,
    });
    emit_box_double(body);
    body.instruction(&Instruction::End);
}

/// Phase-2 physical unboxing: keep proven-`Number` values as raw `f64` on the
/// wasm operand stack across a run, boxing only at boundaries. OFF by default —
/// enable only after the differential verify suite confirms it (a bug here is the
/// silent-corruption class this whole approach guards against).
const JIT_UNBOX: bool = true;

/// Boxes the top `k` operand-stack slots that are physically `f64` (per `phys`)
/// back to boxed `Value` (`i64`), so a subsequent op that expects boxed operands
/// (anything but the numeric-arith fast path) sees `i64`. Uses the spill pool as
/// scratch — pop the top `k` (boxing `f64` ones with `emit_box_double`, which is
/// the sound re-box for a `Number`), then re-push in order. `None` (decline the
/// method) if `k` exceeds the spill pool AND an `f64` is present — too deep to box.
fn box_top(body: &mut Function, phys: &mut [bool], depth: i32, k: u32) -> Option<()> {
    let d = depth.max(0) as usize;
    let k = (k as usize).min(d);
    if k == 0 || !(0..k).any(|j| phys[d - 1 - j]) {
        return Some(());
    }
    if k > SPILL_POOL as usize {
        return None;
    }
    for j in 0..k {
        if phys[d - 1 - j] {
            emit_box_double(body); // f64 top -> i64
        }
        body.instruction(&Instruction::LocalSet(SPILL_BASE + (k - 1 - j) as u32));
    }
    for t in 0..k {
        body.instruction(&Instruction::LocalGet(SPILL_BASE + t as u32));
    }
    for j in 0..k {
        phys[d - 1 - j] = false;
    }
    Some(())
}

/// Physical-unboxing `ArithInt` (`multiply`/`subtract`/`add`) on proven-`Number`
/// operands: consume each operand as `f64` — directly if it is already physically
/// `f64` (`phys_*`), else `emit_numval` it — do the `f64` op, and LEAVE the result
/// unboxed on the stack (the caller marks it `phys = true`). No boxing: the value
/// stays `f64` until a boundary boxes it. Clobbers `SCRATCH64`/`SCRATCH_F64`.
fn emit_arith_number_phys(
    body: &mut Function,
    index: u32,
    phys_a: bool,
    phys_b: bool,
    a_boxsafe: bool,
    b_boxsafe: bool,
) {
    // Stash B (top) into SCRATCH_F64 as f64.
    if phys_b {
        body.instruction(&Instruction::LocalSet(SCRATCH_F64));
    } else {
        body.instruction(&Instruction::LocalSet(SCRATCH64));
        emit_numval_sel(body, SCRATCH64, b_boxsafe);
        body.instruction(&Instruction::LocalSet(SCRATCH_F64));
    }
    // A is now on top; ensure f64.
    if !phys_a {
        body.instruction(&Instruction::LocalSet(SCRATCH64));
        emit_numval_sel(body, SCRATCH64, a_boxsafe);
    }
    body.instruction(&Instruction::LocalGet(SCRATCH_F64));
    body.instruction(&match index {
        11 => Instruction::F64Mul,
        18 => Instruction::F64Add,
        _ => Instruction::F64Sub,
    });
}

/// Phase-2 pure-`Number` `ArithInt` (`multiply`/`subtract`/`add`, 11/12/18): BOTH
/// operands are proven `Number` (an `f64` box, never int-tagged), so even the
/// `is_int` tier is dead — emit the f64 path directly. The result is a `Number`
/// (these ops always yield one), so `emit_box_double` is the correct re-box.
/// Caller checks `depth >= 2` and does `depth -= 1`.
fn emit_arith_number(body: &mut Function, index: u32) {
    body.instruction(&Instruction::LocalSet(SCRATCH64)); // B
    body.instruction(&Instruction::LocalSet(SCRATCH64_2)); // A
    emit_numval(body, SCRATCH64_2);
    emit_numval(body, SCRATCH64);
    body.instruction(&match index {
        11 => Instruction::F64Mul,
        18 => Instruction::F64Add,
        _ => Instruction::F64Sub,
    });
    emit_box_double(body);
}

/// Size (bytes) of one property-IC cache cell in memory 1: `{ class_word: u32
/// @+0, slot_id: u32 @+4 }`. `class_word == 0` is the "never-observed /
/// uncacheable" sentinel — a real object's vtable pointer is never null, so a
/// zeroed cell always misses the guard and routes to the `gpi` helper, which
/// (re)fills it.
pub(crate) const IC_CELL_SIZE: u32 = 8;

/// Emits the monomorphic property **inline cache** for a `getproperty` site
/// (§7): pops the receiver `Value` bits (top of stack), pushes the property
/// `Value` bits — net stack effect 0, identical to the generic `gp` path.
///
/// The class guard is the receiver object's **vtable pointer** (stable per
/// class; every non-dynamic property dispatch resolves through it), compared
/// against the per-site cache cell at `cell_addr` in memory 1:
/// - **object receiver AND `obj.vtable == cell.class`** → the cached slot is
///   this class's fixed layout, so load `slots_ptr + cell.slot_id*8` directly
///   (no range check — the class identity guarantees the slot is in range);
/// - **otherwise** (non-object, or a class miss) → call the `gpi` helper
///   `(receiver_bits, k, cell_addr) -> value_bits`, which resolves the multiname
///   via the interpreter and, as a Rust-side effect, refills the cell at
///   `cell_addr` (a proven fixed slot → `{vtable, slot_id}`; anything else → the
///   uncacheable sentinel).
///
/// The per-site cache cell lives at `ic_base + site*IC_CELL_SIZE` in memory 1,
/// where `ic_base` is a WASM local (the `run` param carrying this method's
/// JIT-owned cache buffer offset); `site*IC_CELL_SIZE + field` folds into the
/// load's compile-time `MemArg` offset. `scratch_i32`/`scratch_i64` are
/// caller-provided WASM-local indices (the frame pool's `SCRATCH`/`SCRATCH64` in
/// real codegen; the unit test passes its own) so the guard can be validated in
/// isolation. All object/cell traffic is on memory 1 (Ruffle's own heap on web,
/// a mock under wasmi natively), exactly like the inline `getslot` fast path.
#[allow(clippy::too_many_arguments)]
fn emit_getprop_ic(
    body: &mut Function,
    slots_ptr_off: u32,
    vtable_off: u32,
    ic_base_local: u32,
    // The baked `Gc<Multiname>` POINTER for this site (from `mn_list[k]`), not the index.
    mn: u64,
    site: u32,
    gpi_index: u32,
    scratch_i32: u32,
    scratch_i64: u32,
) {
    use Instruction::*;
    let mem1 = |offset: u32, align: u32| MemArg { offset: offset as u64, align, memory_index: 1 };
    let cell = site * IC_CELL_SIZE;
    // The miss path: `gpi(receiver_bits, mn_ptr, cell_addr) -> value_bits`, where
    // `cell_addr = ic_base + site*IC_CELL_SIZE` is this site's cell address in
    // memory 1. Passing the absolute cell address (not the site index) lets
    // `gp_ic` refill the cell with a plain store, needing no side channel for the
    // base. Emitted twice (non-object receiver; class-guard miss) so neither arm
    // keeps the receiver on the wasm stack across the branch.
    let emit_miss = |body: &mut Function| {
        body.instruction(&LocalGet(scratch_i64)); // receiver bits
        body.instruction(&I64Const(mn as i64));
        body.instruction(&LocalGet(ic_base_local));
        body.instruction(&I32Const(cell as i32));
        body.instruction(&I32Add); // cell_addr = ic_base + site*IC_CELL_SIZE
        body.instruction(&Call(gpi_index));
    };

    body.instruction(&LocalSet(scratch_i64)); // scratch_i64 = receiver Value bits
    // Object-tagged? `(bits >> 48) == 0xFFFD` (see `VALUE_OBJECT_TAG16`).
    body.instruction(&LocalGet(scratch_i64));
    body.instruction(&I64Const(48));
    body.instruction(&I64ShrU);
    body.instruction(&I64Const(VALUE_OBJECT_TAG16));
    body.instruction(&I64Eq);
    body.instruction(&If(BlockType::Result(ValType::I64)));
    // Object: pointer = the payload's low 32 bits (wasm32 pointers).
    body.instruction(&LocalGet(scratch_i64));
    body.instruction(&I32WrapI64);
    body.instruction(&LocalSet(scratch_i32)); // scratch_i32 = obj ptr (memory 1)
    // Guard: `obj.vtable == cell.class`?
    body.instruction(&LocalGet(scratch_i32));
    body.instruction(&I32Load(mem1(vtable_off, 2))); // obj.vtable
    body.instruction(&LocalGet(ic_base_local));
    body.instruction(&I32Load(mem1(cell, 2))); // cell.class
    body.instruction(&I32Eq);
    body.instruction(&If(BlockType::Result(ValType::I64)));
    // HIT: slots[cell.slot_id] bits = load(memory1, slots_ptr + slot_id*8).
    body.instruction(&LocalGet(scratch_i32));
    body.instruction(&I32Load(mem1(slots_ptr_off, 2))); // slots_ptr
    body.instruction(&LocalGet(ic_base_local));
    body.instruction(&I32Load(mem1(cell + 4, 2))); // cell.slot_id
    body.instruction(&I32Const(8));
    body.instruction(&I32Mul);
    body.instruction(&I32Add);
    body.instruction(&I64Load(mem1(0, 3)));
    body.instruction(&Else);
    emit_miss(body); // class-guard miss
    body.instruction(&End);
    body.instruction(&Else);
    emit_miss(body); // non-object receiver
    body.instruction(&End);
}

/// Max args a direct call may carry when the callee has **typed** params: each needs
/// a per-arg check-word pair in the cache cell (below), so the cell — and the number
/// of inline checks emitted — is bounded. A call with more args, or an all-untyped
/// callee with more args, stays on the coercing helper.
pub(crate) const MAX_DIRECT_ARGC: u32 = 8;

/// `scope_enter_index` sentinel telling [`emit_call_direct`] to emit NO scope
/// enter/leave (the isolated codegen tests, whose hand-built modules don't import
/// `sen`/`sle`). Real modules always pass the real import index.
pub(crate) const NO_SCOPE_HELPER: u32 = u32::MAX;

/// Size (bytes) of one JIT→JIT call-target inline-cache cell in memory 1. The 16-byte
/// header `{ cached_vtable: u32 @+0, callee_run_index: u32 @+4, callee_ic_base: u32
/// @+8, member_idx: u32 @+12 }` is followed by [`MAX_DIRECT_ARGC`] **arg check-word
/// pairs** `{ mask_and_want: u64, expected: u64 }` (16 bytes each, at `@+16 + i*16`),
/// written by `refill_call_cell` from the callee's param types. The HIT path checks each
/// provided arg: a primitive/untyped param with `((arg & (mask_and_want & !1)) ==
/// expected) == (mask_and_want & 1)`, or — when `mask_and_want == CALL_CLASS_MARK` — a
/// concrete-class param with `arg == null || obj.instance_class == expected`. Both are the
/// runtime form of `Value::coerces_identically_to`, so an uncoerced direct pass is
/// bit-identical to the interpreter; a mismatch misses to the coercing helper.
/// `cached_vtable == 0` (fresh) or `callee_run_index == CALL_SENTINEL` → helper.
pub(crate) const CALL_CELL_SIZE: u32 = 16 + MAX_DIRECT_ARGC * 16;

/// `callee_run_index` value meaning "observed but NOT a direct-call target" (the
/// callee isn't a compiled directable leaf) — the guard's vtable may match but this
/// forces the helper path, without re-resolving directability every call.
pub(crate) const CALL_SENTINEL: u32 = 0xFFFF_FFFF;

/// Value of an arg check-word's `mask_and_want` marking an **exact-class** param (rather
/// than a uniform tag check): the `expected` word then holds the class's `Gc` pointer and
/// the HIT guard checks `arg == null || obj.instance_class == expected`. `2` can't be a
/// real `mask_and_want` (tag masks only set the top 16 bits, the uint sign bit, and the
/// want bit 0 — never bit 1), so it's an unambiguous discriminator.
pub(crate) const CALL_CLASS_MARK: u64 = 2;

/// Bit pattern of `Value::Null` (`BOX_MARK | (TAG_NULL=1 << 48)`, must match
/// `core::avm2::value`). A `null` arg coerces identically to any class param.
const NULL_BITS: u64 = 0xFFF9_0000_0000_0000;

/// `member_idx` cell value (`@+12`) meaning the target is a STANDALONE method (its
/// own 6-param `run` in the shared table). Any other value = the amalgam-member index
/// of a GENERATION target, reached via that amalgam's 7-param dispatcher.
pub(crate) const CALL_NO_MEMBER: u32 = 0xFFFF_FFFF;

/// Emits the in-wasm JIT→JIT **direct call** for a `CallMethod(disp_id, argc, push)`
/// to a directable leaf (§8 / step 4): a monomorphic call-target cache guards the
/// receiver's vtable; on a hit the args are written straight into the callee's frame
/// at `state_ptr + STRIDE` and its compiled `run` is invoked **wasm→wasm** via
/// `call_indirect` (no interpreter `Activation`, no JS crossing); on a miss the
/// generic `cm` (`call_method`) helper runs (which also refills the cell — a
/// directable callee → `{vtable, run_index, ic_base}`, else the `CALL_SENTINEL`).
///
/// Stack in: `[.., receiver, arg0, .., arg{argc-1}]` (top = last arg); stack out:
/// the result `Value` iff `push`. `scratch_sp`/`scratch_ptr` (i32) and
/// `scratch_bits` (i64) are caller-provided frame-pool locals. The callee frame is
/// memory 0 (`state_ptr` space); the vtable + cache cell are memory 1.
#[allow(clippy::too_many_arguments)]
fn emit_call_direct(
    body: &mut Function,
    argc: u32,
    disp_id: u32,
    vtable_off: u32,
    instance_class_off: u32,
    cell_base_local: u32,
    cell_offset: u32,
    state_ptr_local: u32,
    dm_base_local: u32,
    dm_len_local: u32,
    run_type_index: u32,
    dispatch_type_index: u32,
    miss_index: u32,
    cm_index: u32,
    pca_index: u32,
    scope_enter_index: u32,
    scope_leave_index: u32,
    scratch_sp: u32,
    scratch_ptr: u32,
    scratch_bits: u32,
    scratch_scope: u32,
    stride: u32,
    mem_bytes: u32,
    push: bool,
) {
    use Instruction::*;
    // Byte offset of this site's cell within the shared cache buffer (`cell_base` =
    // `ic_base`; the call region follows the property-IC cells).
    let cell = cell_offset;
    let mem1 = |o: u32, a: u32| MemArg { offset: o as u64, align: a, memory_index: 1 };
    let frame = |o: u32, a: u32| MemArg { offset: o as u64, align: a, memory_index: 0 };

    // callee_sp = caller state_ptr + STRIDE (the next disjoint frame slice).
    body.instruction(&LocalGet(state_ptr_local));
    body.instruction(&I32Const(stride as i32));
    body.instruction(&I32Add);
    body.instruction(&LocalSet(scratch_sp));

    // Write receiver + args into the callee frame, popping the wasm stack top-first:
    // arg{argc-1} → slot `argc`, …, receiver → slot 0. After the loop `scratch_bits`
    // holds the receiver bits (the last op written).
    for i in (0..=argc).rev() {
        body.instruction(&LocalSet(scratch_bits));
        body.instruction(&LocalGet(scratch_sp));
        body.instruction(&LocalGet(scratch_bits));
        body.instruction(&I64Store(frame(i * 8, 3)));
    }

    // Miss: reconstruct the generic call. Spill args top-first (arg{argc-1}…arg0) via
    // `pca`, reading them back out of the callee frame, then leave `receiver, disp,
    // argc` on the stack for the call — shared by both miss tails below.
    let emit_spill = |body: &mut Function| {
        for i in (1..=argc).rev() {
            body.instruction(&LocalGet(scratch_sp));
            body.instruction(&I64Load(frame(i * 8, 3)));
            body.instruction(&Call(pca_index));
        }
        body.instruction(&LocalGet(scratch_bits)); // receiver bits
        body.instruction(&I64Const(disp_id as i64));
        body.instruction(&I64Const(argc as i64));
    };
    // REFILL miss (fresh cell / vtable changed): `cmi(receiver, disp, argc, cell_addr)`
    // — does the generic call AND (re)resolves + refills this site's cell, so a later
    // same-class call can take the HIT (or the plain-`cm` short-circuit).
    let emit_miss_refill = |body: &mut Function| {
        emit_spill(body);
        body.instruction(&LocalGet(cell_base_local));
        body.instruction(&I32Const(cell as i32));
        body.instruction(&I32Add); // cell_addr for the refill
        body.instruction(&Call(miss_index));
    };
    // SENTINEL miss (cell says this class's callee is NOT directable): plain
    // `cm(receiver, disp, argc)` — the generic call WITHOUT re-running the refill
    // (which would just re-record the same sentinel every call — the measured
    // `call_method_ic` thrash). The vtable already matched, so this stays monomorphic.
    let emit_miss_plain = |body: &mut Function| {
        emit_spill(body);
        body.instruction(&Call(cm_index));
    };

    // Guard: object receiver? (nested so the vtable load never runs on a non-object)
    body.instruction(&LocalGet(scratch_bits));
    body.instruction(&I64Const(48));
    body.instruction(&I64ShrU);
    body.instruction(&I64Const(VALUE_OBJECT_TAG16));
    body.instruction(&I64Eq);
    body.instruction(&If(BlockType::Result(ValType::I64)));
    body.instruction(&LocalGet(scratch_bits));
    body.instruction(&I32WrapI64);
    body.instruction(&LocalSet(scratch_ptr)); // obj ptr (memory 1)
    // obj.vtable == cell.vtable ? (monomorphic — the cached class still matches)
    body.instruction(&LocalGet(scratch_ptr));
    body.instruction(&I32Load(mem1(vtable_off, 2)));
    body.instruction(&LocalGet(cell_base_local));
    body.instruction(&I32Load(mem1(cell, 2)));
    body.instruction(&I32Eq);
    body.instruction(&If(BlockType::Result(ValType::I64)));
    // cell.run_index != CALL_SENTINEL (callee is a directable target) → HIT; else the
    // callee for this (matched) class is known non-directable → plain `cm`, NO refill.
    body.instruction(&LocalGet(cell_base_local));
    body.instruction(&I32Load(mem1(cell + 4, 2)));
    body.instruction(&I32Const(CALL_SENTINEL as i32));
    body.instruction(&I32Ne);
    body.instruction(&If(BlockType::Result(ValType::I64)));
    // Arg type guard: a direct call writes the caller's args UNCOERCED, sound only when
    // each already matches its param type (`Value::coerces_identically_to`). Check each
    // provided arg against its per-arg cell check-word pair (`{mask_and_want, expected}`
    // at `@+16 + i*16`): `match_i = ((arg & (m & !1)) == expected) == (m & 1)`. AND them;
    // all match → HIT (uncoerced write is bit-identical to the interpreter), any mismatch
    // → the coercing `cm` (the cell stays, so a later matching-typed call HITs again).
    // An all-untyped callee wrote ANY words (mask 0, want 1) → every check is vacuously
    // true. A site with more args than the cell can hold is never a directable target
    // (see `direct_target`), so force a miss rather than read past the cell.
    if argc > MAX_DIRECT_ARGC {
        body.instruction(&I32Const(0));
    } else {
        for i in 0..argc {
            let e = cell + 16 + i * 16; // arg i's {mask_and_want, expected} pair
            let arg = |body: &mut Function| {
                body.instruction(&LocalGet(scratch_sp));
                body.instruction(&I64Load(frame((i + 1) * 8, 3))); // arg i (slot 0 = receiver)
            };
            // Selector: an exact-class param (`mask_and_want == CALL_CLASS_MARK`) needs an
            // object-model check; any other word is a uniform tag check.
            body.instruction(&LocalGet(cell_base_local));
            body.instruction(&I64Load(mem1(e, 3)));
            body.instruction(&I64Const(CALL_CLASS_MARK as i64));
            body.instruction(&I64Eq);
            body.instruction(&If(BlockType::Result(ValType::I32)));
            // CLASS check: `arg == null` OR (arg is object AND `obj.instance_class` ==
            // the cached class pointer) — the runtime form of `coerces_identically_to`.
            arg(body);
            body.instruction(&I64Const(48));
            body.instruction(&I64ShrU);
            body.instruction(&I64Const(VALUE_OBJECT_TAG16));
            body.instruction(&I64Eq); // object?
            body.instruction(&If(BlockType::Result(ValType::I32)));
            arg(body);
            body.instruction(&I32WrapI64); // obj ptr (memory 1)
            body.instruction(&I32Load(mem1(instance_class_off, 2))); // obj.instance_class
            body.instruction(&LocalGet(cell_base_local));
            body.instruction(&I32Load(mem1(e + 8, 2))); // cached class ptr (low word of expected)
            body.instruction(&I32Eq);
            body.instruction(&Else);
            arg(body);
            body.instruction(&I64Const(NULL_BITS as i64));
            body.instruction(&I64Eq); // non-object: only null matches a class param
            body.instruction(&End);
            body.instruction(&Else);
            // TAG check: `((arg & (mask & !1)) == expected) == (mask & 1)`.
            arg(body);
            body.instruction(&LocalGet(cell_base_local));
            body.instruction(&I64Load(mem1(e, 3))); // mask_and_want
            body.instruction(&I64Const(!1i64)); // clear the want bit → mask
            body.instruction(&I64And);
            body.instruction(&I64And); // arg & mask
            body.instruction(&LocalGet(cell_base_local));
            body.instruction(&I64Load(mem1(e + 8, 3))); // expected
            body.instruction(&I64Eq); // (arg & mask) == expected → i32 0/1
            body.instruction(&LocalGet(cell_base_local));
            body.instruction(&I64Load(mem1(e, 3)));
            body.instruction(&I64Const(1));
            body.instruction(&I64And);
            body.instruction(&I32WrapI64); // want → i32
            body.instruction(&I32Eq); // (eq == want)
            body.instruction(&End); // close class-vs-tag selector
            if i > 0 {
                body.instruction(&I32And); // fold into running all_match
            }
        }
        if argc == 0 {
            body.instruction(&I32Const(1)); // no args → vacuously match
        }
    }
    // In-bounds guard, AND-ed into the HIT condition: a NON-LEAF callee can chain
    // direct frames without the runner's per-`run` depth-decline check, so the callee
    // frame `[callee_sp, callee_sp + stride)` must lie within the committed frame
    // memory. Over → fall to the plain `cm` (miss), which re-enters `run` and declines
    // to the interpreter (bounded). Leaf callees always pass (one frame within the
    // guard band), so this only ever fires on deep non-leaf/direct recursion.
    body.instruction(&LocalGet(scratch_sp));
    body.instruction(&I32Const(stride as i32));
    body.instruction(&I32Add);
    body.instruction(&I32Const(mem_bytes as i32));
    body.instruction(&I32LeU);
    body.instruction(&I32And); // all_match & in_bounds
    body.instruction(&If(BlockType::Result(ValType::I64)));
    // Scope ENTER (conditional): the callee's `ic_base` high bit (@+8) flags a scope-using
    // callee. If set, retarget the shared activation's `scope_depth` to the callee's frame
    // (`sen` → old depth into `scratch_scope`) so its `getscopeobject`/`pushscope` read its
    // own scopes; `sle` after the call restores. The `ic_base` VALUE is masked (bit 31
    // cleared) where it's passed to the callee below. `emit_scope` is false only in the
    // isolated codegen tests (no `sen`/`sle` imports); real modules always emit it.
    let emit_scope = scope_enter_index != NO_SCOPE_HELPER;
    if emit_scope {
        body.instruction(&LocalGet(cell_base_local));
        body.instruction(&I32Load(mem1(cell + 8, 2)));
        body.instruction(&I32Const(31));
        body.instruction(&I32ShrU); // scope flag (0/1)
        body.instruction(&If(BlockType::Empty));
        body.instruction(&Call(scope_enter_index));
        body.instruction(&LocalSet(scratch_scope)); // saved caller scope_depth
        body.instruction(&End);
    }
    // Save + bump `DEPTH` (memory 1 @ `dp`) around the call. The callee runs at
    // `callee_sp = state_ptr + stride`; setting `DEPTH = DEPTH + 1` makes it equal the
    // callee's own frame depth, so any re-entrant `run` the callee triggers (a `cm`
    // helper call, or a nested direct call's miss) frames ABOVE the callee rather than
    // aliasing its live frame. `scratch_ptr` (free past the vtable guard) holds the old
    // value; the restore after the call rebalances it (no drift, even under recursion).
    body.instruction(&GlobalGet(DEPTH_PTR_GLOBAL));
    body.instruction(&I32Load(mem1(0, 2)));
    body.instruction(&LocalSet(scratch_ptr)); // saved DEPTH
    body.instruction(&GlobalGet(DEPTH_PTR_GLOBAL));
    body.instruction(&LocalGet(scratch_ptr));
    body.instruction(&I32Const(1));
    body.instruction(&I32Add);
    body.instruction(&I32Store(mem1(0, 2))); // DEPTH = saved + 1
    // HIT: invoke the callee's compiled `run`. `regs_ptr`/`regs_len` = 0 → its prologue
    // copies nothing; its args are the frame slots we just wrote. `member_idx` (@+12)
    // selects the entry shape: `CALL_NO_MEMBER` → a STANDALONE method's own 6-param
    // `run`; else a GENERATION member reached via its amalgam's 7-param dispatcher
    // (member index prepended). Both slots live in the shared table.
    body.instruction(&LocalGet(cell_base_local));
    body.instruction(&I32Load(mem1(cell + 12, 2)));
    body.instruction(&I32Const(CALL_NO_MEMBER as i32));
    body.instruction(&I32Eq);
    body.instruction(&If(BlockType::Result(ValType::I64)));
    // Standalone: run(callee_sp, dm_base, dm_len, regs_ptr=0, regs_len=(argc+1)*8, ic_base).
    // `regs_ptr = 0` tells the callee prologue this is a DIRECT call; `regs_len` = the
    // bytes we pre-wrote (this + argc args), from which the prologue zeroes any locals
    // beyond them to `undefined`.
    body.instruction(&LocalGet(scratch_sp));
    body.instruction(&LocalGet(dm_base_local));
    body.instruction(&LocalGet(dm_len_local));
    body.instruction(&I32Const(0)); // regs_ptr = 0 (direct-call marker)
    body.instruction(&I32Const(((argc + 1) * 8) as i32)); // regs_len = pre-written bytes
    body.instruction(&LocalGet(cell_base_local));
    body.instruction(&I32Load(mem1(cell + 8, 2)));
    body.instruction(&I32Const(0x7FFF_FFFF));
    body.instruction(&I32And); // callee ic_base (clear the scope flag in bit 31)
    body.instruction(&LocalGet(cell_base_local));
    body.instruction(&I32Load(mem1(cell + 4, 2))); // run slot
    body.instruction(&CallIndirect { type_index: run_type_index, table_index: 0 });
    body.instruction(&Else);
    // Generation member: dispatch(member_idx, callee_sp, dm_base, dm_len, regs_ptr=0,
    // regs_len=(argc+1)*8, ic_base) — same direct-call prologue signal as standalone.
    body.instruction(&LocalGet(cell_base_local));
    body.instruction(&I32Load(mem1(cell + 12, 2))); // member_idx
    body.instruction(&LocalGet(scratch_sp));
    body.instruction(&LocalGet(dm_base_local));
    body.instruction(&LocalGet(dm_len_local));
    body.instruction(&I32Const(0)); // regs_ptr = 0 (direct-call marker)
    body.instruction(&I32Const(((argc + 1) * 8) as i32)); // regs_len = pre-written bytes
    body.instruction(&LocalGet(cell_base_local));
    body.instruction(&I32Load(mem1(cell + 8, 2)));
    body.instruction(&I32Const(0x7FFF_FFFF));
    body.instruction(&I32And); // callee ic_base (clear the scope flag in bit 31)
    body.instruction(&LocalGet(cell_base_local));
    body.instruction(&I32Load(mem1(cell + 4, 2))); // dispatcher slot
    body.instruction(&CallIndirect { type_index: dispatch_type_index, table_index: 0 });
    body.instruction(&End); // close: standalone vs generation
    // Restore `DEPTH` to the caller's value (the call result i64 stays on the stack —
    // the store consumes only its own address/value operands).
    body.instruction(&GlobalGet(DEPTH_PTR_GLOBAL));
    body.instruction(&LocalGet(scratch_ptr));
    body.instruction(&I32Store(mem1(0, 2))); // DEPTH = saved
    // Scope LEAVE (conditional, same flag as enter): pop the callee's scopes + restore the
    // caller's `scope_depth`. The call result i64 stays on the stack (the flag/`sle`
    // operands sit above it).
    if emit_scope {
        body.instruction(&LocalGet(cell_base_local));
        body.instruction(&I32Load(mem1(cell + 8, 2)));
        body.instruction(&I32Const(31));
        body.instruction(&I32ShrU);
        body.instruction(&If(BlockType::Empty));
        body.instruction(&LocalGet(scratch_scope));
        body.instruction(&Call(scope_leave_index));
        body.instruction(&End);
    }
    body.instruction(&Else);
    emit_miss_plain(body); // arg type mismatch → coercing cm, cell kept for next call
    body.instruction(&End); // close: args match
    body.instruction(&Else);
    emit_miss_plain(body); // vtable matched but SENTINEL → plain cm, NO refill
    body.instruction(&End); // close: run_index != SENTINEL
    body.instruction(&Else);
    emit_miss_refill(body); // vtable mismatch (fresh / megamorphic) → cmi (call + refill)
    body.instruction(&End); // close: vtable match
    body.instruction(&Else);
    emit_miss_refill(body); // non-object receiver → cmi
    body.instruction(&End); // close: object receiver

    if !push {
        body.instruction(&Drop);
    }
}

/// Emits a non-branch op onto `body`, updating the compile-time operand-stack
/// `depth`. Returns `None` if the op isn't a linear op or underflows.
fn emit_linear(
    body: &mut Function,
    op: JitOp,
    depth: &mut i32,
    lay: &Layout,
    // Byte offset of the property-IC region's end = start of the call-cache region in
    // the shared per-method buffer (`= num_ic_sites * IC_CELL_SIZE`). Only read by
    // `CallMethodDirect`.
    call_region_offset: u32,
    // Resolved multiname pointers, indexed by an mn-bearing op's `k` (see `compile_full`).
    mn_list: &[*const ()],
    // Resolved coerce-class pointers, indexed by a `Coerce` op's `k` (see `compile_full`).
    coerce_classes: &[*const ()],
) -> Option<()> {
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
        // --- Boxed-`Value` (i64) ops.
        JitOp::GetLocalValue(i) => {
            body.instruction(&Instruction::LocalGet(STATE_PTR));
            body.instruction(&Instruction::I64Load(slot(i)));
            *depth += 1;
        }
        JitOp::SetLocalValue(i) => {
            if *depth < 1 {
                return None;
            }
            body.instruction(&Instruction::LocalSet(SCRATCH64));
            body.instruction(&Instruction::LocalGet(STATE_PTR));
            body.instruction(&Instruction::LocalGet(SCRATCH64));
            body.instruction(&Instruction::I64Store(slot(i)));
            *depth -= 1;
        }
        JitOp::CallHelper(index) => {
            if *depth < 1 {
                return None;
            }
            // Helper `i` is function index `i` (they precede `run`). Pops the i64
            // argument, pushes the i64 result — net depth unchanged.
            body.instruction(&Instruction::Call(index));
        }
        JitOp::CallHelper2(index) => {
            if *depth < 2 {
                return None;
            }
            // Two-stack arity-2 helper: pops (v1, v2), pushes the result — net -1.
            body.instruction(&Instruction::Call(lay.t_base + index));
            *depth -= 1;
        }
        JitOp::CmpNum(index) => {
            if *depth < 2 {
                return None;
            }
            // Stack: [v1, v2]. Stash into i64 scratch (A = v1, B = v2).
            body.instruction(&Instruction::LocalSet(SCRATCH64)); // B = v2
            body.instruction(&Instruction::LocalSet(SCRATCH64_2)); // A = v1
            // Both numeric? → inline f64 compare; else → helper fallback.
            emit_is_numeric(body, SCRATCH64_2);
            emit_is_numeric(body, SCRATCH64);
            body.instruction(&Instruction::I32And);
            body.instruction(&Instruction::If(BlockType::Result(ValType::I64)));
            emit_numval(body, SCRATCH64_2); // a
            emit_numval(body, SCRATCH64); // b
            body.instruction(&match index {
                0 => Instruction::F64Eq,
                1 => Instruction::F64Lt,
                2 => Instruction::F64Le,
                3 => Instruction::F64Gt,
                _ => Instruction::F64Ge,
            });
            // Box the i32 `0`/`1` as a `Boolean` `Value`.
            body.instruction(&Instruction::I64ExtendI32U);
            body.instruction(&Instruction::I64Const(VALUE_BOOL_MARK as i64));
            body.instruction(&Instruction::I64Or);
            body.instruction(&Instruction::Else);
            body.instruction(&Instruction::LocalGet(SCRATCH64_2)); // v1
            body.instruction(&Instruction::LocalGet(SCRATCH64)); // v2
            body.instruction(&Instruction::Call(lay.t_base + index));
            body.instruction(&Instruction::End);
            *depth -= 1;
        }
        JitOp::BitOpInt(index) => {
            if *depth < 2 {
                return None;
            }
            // Stack: [v1, v2]. Stash (A = v1, B = v2).
            body.instruction(&Instruction::LocalSet(SCRATCH64)); // B = v2
            body.instruction(&Instruction::LocalSet(SCRATCH64_2)); // A = v1

            // Given two `i32` operands (a, b) on the stack, emit the shift-count
            // mask (shifts only), the i32 op, and box the result. `urshift`
            // (index 10) is logical and its `uint` result needs uint-boxing; the
            // others are signed-i32 results.
            let emit_op_and_box = |body: &mut Function| {
                if matches!(index, 8 | 9 | 10) {
                    // Shift count masked by 0x1F (matching AVM2 / the helper). The
                    // low 5 bits agree between `ToInt32` and `ToUint32`.
                    body.instruction(&Instruction::I32Const(0x1F));
                    body.instruction(&Instruction::I32And);
                }
                body.instruction(&match index {
                    5 => Instruction::I32And,
                    6 => Instruction::I32Or,
                    7 => Instruction::I32Xor,
                    8 => Instruction::I32Shl,
                    9 => Instruction::I32ShrS,   // rshift (arithmetic)
                    _ => Instruction::I32ShrU,   // 10 = urshift (logical)
                });
                if index == 10 {
                    emit_urshift_box(body);
                } else {
                    emit_box_int(body);
                }
            };

            // Tier 1 — both int-boxed: their low 32 bits are already the `ToInt32`
            // value (`urshift` reads them as `ToUint32`; same bits), so no coercion.
            emit_is_int(body, SCRATCH64_2);
            emit_is_int(body, SCRATCH64);
            body.instruction(&Instruction::I32And);
            body.instruction(&Instruction::If(BlockType::Result(ValType::I64)));
            body.instruction(&Instruction::LocalGet(SCRATCH64_2)); // a
            body.instruction(&Instruction::I32WrapI64);
            body.instruction(&Instruction::LocalGet(SCRATCH64)); // b
            body.instruction(&Instruction::I32WrapI64);
            emit_op_and_box(body);
            body.instruction(&Instruction::Else);
            // Tier 2 — both numeric with `|v| < 2^63`: pure-wasm wrapping `ToInt32`
            // inline (no helper). This is the hot CrossBridge case, where bitwise
            // ops are applied to `Number`-boxed pointers/ints. Values outside that
            // range (or non-numeric) fall through to the helper.
            emit_is_int32_able(body, SCRATCH64_2);
            emit_is_int32_able(body, SCRATCH64);
            body.instruction(&Instruction::I32And);
            body.instruction(&Instruction::If(BlockType::Result(ValType::I64)));
            emit_to_int32(body, SCRATCH64_2); // a = ToInt32(A)
            emit_to_int32(body, SCRATCH64); // b = ToInt32(B)
            emit_op_and_box(body);
            body.instruction(&Instruction::Else);
            // Tier 3 — helper fallback (string/object operands, or out-of-range
            // `Number`s needing the exact wrapping `ToInt32`).
            body.instruction(&Instruction::LocalGet(SCRATCH64_2)); // v1
            body.instruction(&Instruction::LocalGet(SCRATCH64)); // v2
            body.instruction(&Instruction::Call(lay.t_base + index));
            body.instruction(&Instruction::End);
            body.instruction(&Instruction::End);
            *depth -= 1;
        }
        JitOp::ArithInt(index) => {
            if *depth < 2 {
                return None;
            }
            // Stack: [v1, v2]. Stash (A = v1, B = v2).
            body.instruction(&Instruction::LocalSet(SCRATCH64)); // B = v2
            body.instruction(&Instruction::LocalSet(SCRATCH64_2)); // A = v1
            emit_is_int(body, SCRATCH64_2);
            emit_is_int(body, SCRATCH64);
            body.instruction(&Instruction::I32And);
            body.instruction(&Instruction::If(BlockType::Result(ValType::I64)));
            // Widen both ints to i64, op, then box as `int` if it fits `i32` else as
            // `Number` — matching the helper's checked-int-else-Number semantics.
            body.instruction(&Instruction::LocalGet(SCRATCH64_2)); // A
            body.instruction(&Instruction::I32WrapI64);
            body.instruction(&Instruction::I64ExtendI32S); // a i64
            body.instruction(&Instruction::LocalGet(SCRATCH64)); // B
            body.instruction(&Instruction::I32WrapI64);
            body.instruction(&Instruction::I64ExtendI32S); // b i64
            body.instruction(&match index {
                11 => Instruction::I64Mul,
                18 => Instruction::I64Add, // add (numeric arms of `Value::add`)
                _ => Instruction::I64Sub,  // 12 = subtract
            });
            body.instruction(&Instruction::LocalSet(SCRATCH64_2)); // r = a op b
            // Fits i32? `r == sign_extend_32(r)`.
            body.instruction(&Instruction::LocalGet(SCRATCH64_2));
            body.instruction(&Instruction::LocalGet(SCRATCH64_2));
            body.instruction(&Instruction::I64Extend32S);
            body.instruction(&Instruction::I64Eq);
            body.instruction(&Instruction::If(BlockType::Result(ValType::I64)));
            body.instruction(&Instruction::LocalGet(SCRATCH64_2));
            body.instruction(&Instruction::I32WrapI64);
            emit_box_int(body);
            body.instruction(&Instruction::Else);
            // Number: reinterpret the f64 of the (finite, non-NaN) result. An int
            // product/difference is never NaN, so no canonicalization is needed.
            body.instruction(&Instruction::LocalGet(SCRATCH64_2));
            body.instruction(&Instruction::F64ConvertI64S);
            body.instruction(&Instruction::I64ReinterpretF64);
            body.instruction(&Instruction::End);
            body.instruction(&Instruction::Else);
            // **Numeric middle path**: not both ints, but both numeric (int or
            // `Number`) → the f64 op inline, boxed as a `Number` (NaN canonicalized
            // — `inf * 0` / `inf - inf` are NaN). Matches the helper's
            // `ToNumber × ToNumber → Value::from(f64)` for numeric operands
            // (numeric `ToNumber` has no side effects, so coercion order is moot).
            emit_is_numeric(body, SCRATCH64_2);
            emit_is_numeric(body, SCRATCH64);
            body.instruction(&Instruction::I32And);
            body.instruction(&Instruction::If(BlockType::Result(ValType::I64)));
            emit_numval(body, SCRATCH64_2); // a
            emit_numval(body, SCRATCH64); // b
            body.instruction(&match index {
                11 => Instruction::F64Mul,
                18 => Instruction::F64Add, // add (numeric middle path)
                _ => Instruction::F64Sub,  // 12 = subtract
            });
            emit_box_double(body);
            body.instruction(&Instruction::Else);
            // Helper fallback (string/object operands — `valueOf` coercion).
            body.instruction(&Instruction::LocalGet(SCRATCH64_2)); // v1
            body.instruction(&Instruction::LocalGet(SCRATCH64)); // v2
            body.instruction(&Instruction::Call(lay.t_base + index));
            body.instruction(&Instruction::End);
            body.instruction(&Instruction::End);
            *depth -= 1;
        }
        JitOp::ArithNum(index) => {
            if *depth < 2 {
                return None;
            }
            // Stack: [v1, v2]. Stash (A = v1, B = v2). Both numeric → the f64 op
            // inline, boxed as a `Number` (the result is ALWAYS a `Number` for these
            // ops — `divide` — matching the helper); else the helper fallback.
            body.instruction(&Instruction::LocalSet(SCRATCH64)); // B = v2
            body.instruction(&Instruction::LocalSet(SCRATCH64_2)); // A = v1
            emit_is_numeric(body, SCRATCH64_2);
            emit_is_numeric(body, SCRATCH64);
            body.instruction(&Instruction::I32And);
            body.instruction(&Instruction::If(BlockType::Result(ValType::I64)));
            emit_numval(body, SCRATCH64_2); // a
            emit_numval(body, SCRATCH64); // b
            body.instruction(&match index {
                13 => Instruction::F64Div,
                _ => return None, // only `divide` for now (`modulo` has no wasm f64 rem)
            });
            emit_box_double(body); // NaN/±inf canonicalized `Number` bits
            body.instruction(&Instruction::Else);
            body.instruction(&Instruction::LocalGet(SCRATCH64_2)); // v1
            body.instruction(&Instruction::LocalGet(SCRATCH64)); // v2
            body.instruction(&Instruction::Call(lay.t_base + index));
            body.instruction(&Instruction::End);
            *depth -= 1;
        }
        JitOp::CoerceInt(signed) => {
            if *depth < 1 {
                return None;
            }
            // Pop the `Value`, stash it. int-boxed → passthrough (`coerce_i`: an int's
            // `ToInt32` is itself; `coerce_u`: `ToUint32` == the int when non-negative)
            // — else the arity-1 helper.
            body.instruction(&Instruction::LocalSet(SCRATCH64));
            emit_is_int(body, SCRATCH64);
            if !signed {
                // coerce_u fast path also requires the int to be non-negative
                // (a negative int's `ToUint32` differs — it's `> i32::MAX`).
                body.instruction(&Instruction::LocalGet(SCRATCH64));
                body.instruction(&Instruction::I32WrapI64);
                body.instruction(&Instruction::I32Const(0));
                body.instruction(&Instruction::I32GeS);
                body.instruction(&Instruction::I32And);
            }
            body.instruction(&Instruction::If(BlockType::Result(ValType::I64)));
            body.instruction(&Instruction::LocalGet(SCRATCH64)); // passthrough
            body.instruction(&Instruction::Else);
            // **Numeric middle path**, pure wasm: numeric with |x| < 2^63 → inline
            // `ToInt32` (`trunc_sat` + `wrap`). `coerce_i` boxes it as int;
            // `coerce_u`'s `ToUint32` is the same 32 bits — non-negative → int box,
            // negative → the `uint` is 2^31..2^32, boxed as `Number`
            // (`f64.convert_i32_u`), matching `Value::from(u32)`. This covers the
            // hot `Number`-boxed results (and `coerce_u` of a negative int)
            // without a helper crossing; NaN/±Inf/huge/non-numeric → the helper.
            emit_is_int32_able(body, SCRATCH64);
            body.instruction(&Instruction::If(BlockType::Result(ValType::I64)));
            emit_to_int32(body, SCRATCH64);
            if signed {
                emit_box_int(body);
            } else {
                body.instruction(&Instruction::LocalSet(SCRATCH)); // t = ToInt32
                body.instruction(&Instruction::LocalGet(SCRATCH));
                emit_box_int(body); // val1: int box (t >= 0)
                body.instruction(&Instruction::LocalGet(SCRATCH));
                body.instruction(&Instruction::F64ConvertI32U);
                body.instruction(&Instruction::I64ReinterpretF64); // val2: Number box
                body.instruction(&Instruction::LocalGet(SCRATCH));
                body.instruction(&Instruction::I32Const(0));
                body.instruction(&Instruction::I32GeS); // selector: t >= 0
                body.instruction(&Instruction::Select);
            }
            body.instruction(&Instruction::Else);
            body.instruction(&Instruction::LocalGet(SCRATCH64));
            body.instruction(&Instruction::Call(if signed { COERCE_I } else { COERCE_U }));
            body.instruction(&Instruction::End);
            body.instruction(&Instruction::End);
            // net 0 (pop one, push one)
        }
        JitOp::CoerceBool => {
            if *depth < 1 {
                return None;
            }
            // `ToBoolean(v)` (inline for Boolean/int boxes, `to_boolean` helper
            // otherwise) boxed as a `Boolean` `Value`. Net 0.
            emit_to_boolean_i32(body);
            body.instruction(&Instruction::I64ExtendI32U);
            body.instruction(&Instruction::I64Const(VALUE_BOOL_MARK as i64));
            body.instruction(&Instruction::I64Or);
        }
        JitOp::SwapValue => {
            if *depth < 2 {
                return None;
            }
            // Swap the top two i64s via two i64 scratch locals. (Must NOT round-trip
            // through an f64 local: a boxed `Value` is a NaN bit-pattern, and some
            // engines canonicalize NaN payloads on f64 local traffic — corrupting it.)
            body.instruction(&Instruction::LocalSet(SCRATCH64)); // [a], SCRATCH64 = b
            body.instruction(&Instruction::LocalSet(SCRATCH64_2)); // [], SCRATCH64_2 = a
            body.instruction(&Instruction::LocalGet(SCRATCH64)); // [b]
            body.instruction(&Instruction::LocalGet(SCRATCH64_2)); // [b, a]
        }
        JitOp::DupValue => {
            if *depth < 1 {
                return None;
            }
            // i64 dup (boxed `Value`): stash the top and re-push it twice.
            body.instruction(&Instruction::LocalTee(SCRATCH64));
            body.instruction(&Instruction::LocalGet(SCRATCH64));
            *depth += 1;
        }
        JitOp::StoreLocalValue(i) => {
            if *depth < 1 {
                return None;
            }
            // Peek-and-store a raw `Value`: keep it on the stack (via `Tee`) and
            // also write it into the local. Net operand-stack effect 0.
            body.instruction(&Instruction::LocalTee(SCRATCH64));
            body.instruction(&Instruction::LocalGet(STATE_PTR));
            body.instruction(&Instruction::LocalGet(SCRATCH64));
            body.instruction(&Instruction::I64Store(slot(i)));
        }
        JitOp::AddIBoxed | JitOp::SubtractIBoxed | JitOp::MultiplyIBoxed => {
            if *depth < 2 {
                return None;
            }
            // Stack [v1, v2]. Stash (A = v1, B = v2). Both int-boxed → the wrapping
            // i32 op inline; else → the `_i` helper (wrapping `ToInt32` of both — a
            // `uint`/`Number`-boxed operand's low 32 box bits are NOT its `ToInt32`,
            // they're f64-bit garbage; blindly wrapping them corrupted PRNG state in
            // the `rng` test).
            body.instruction(&Instruction::LocalSet(SCRATCH64)); // B = v2
            body.instruction(&Instruction::LocalSet(SCRATCH64_2)); // A = v1
            emit_is_int(body, SCRATCH64_2);
            emit_is_int(body, SCRATCH64);
            body.instruction(&Instruction::I32And);
            body.instruction(&Instruction::If(BlockType::Result(ValType::I64)));
            body.instruction(&Instruction::LocalGet(SCRATCH64_2)); // A
            body.instruction(&Instruction::I32WrapI64); // a i32
            body.instruction(&Instruction::LocalGet(SCRATCH64)); // B
            body.instruction(&Instruction::I32WrapI64); // b i32
            let arith = match op {
                JitOp::AddIBoxed => Instruction::I32Add,
                JitOp::SubtractIBoxed => Instruction::I32Sub,
                _ => Instruction::I32Mul,
            };
            body.instruction(&arith);
            emit_box_int(body);
            body.instruction(&Instruction::Else);
            // **Numeric middle path**, pure wasm: both numeric with |x| < 2^63 →
            // inline `ToInt32` (`trunc_sat` + `wrap`), wrapping i32 op, int box.
            // Covers the hot `uint`-in-`Number`-box operands without a helper
            // crossing; NaN/±Inf/huge (and non-numerics) → the `_i` helper.
            emit_is_int32_able(body, SCRATCH64_2);
            emit_is_int32_able(body, SCRATCH64);
            body.instruction(&Instruction::I32And);
            body.instruction(&Instruction::If(BlockType::Result(ValType::I64)));
            emit_to_int32(body, SCRATCH64_2); // a
            emit_to_int32(body, SCRATCH64); // b
            body.instruction(&arith);
            emit_box_int(body);
            body.instruction(&Instruction::Else);
            body.instruction(&Instruction::LocalGet(SCRATCH64_2)); // v1
            body.instruction(&Instruction::LocalGet(SCRATCH64)); // v2
            body.instruction(&Instruction::Call(
                lay.t_base
                    + match op {
                        JitOp::AddIBoxed => H2_ADD_I,
                        JitOp::SubtractIBoxed => H2_SUBTRACT_I,
                        _ => H2_MULTIPLY_I,
                    },
            ));
            body.instruction(&Instruction::End);
            body.instruction(&Instruction::End);
            *depth -= 1;
        }
        JitOp::IncrementIBoxed | JitOp::DecrementIBoxed => {
            if *depth < 1 {
                return None;
            }
            // int-boxed → inline wrapping i32 `±1`; else → the `increment_i`/
            // `decrement_i` helper (wrapping `ToInt32 ± 1` — see `AddIBoxed`).
            body.instruction(&Instruction::LocalSet(SCRATCH64));
            emit_is_int(body, SCRATCH64);
            body.instruction(&Instruction::If(BlockType::Result(ValType::I64)));
            body.instruction(&Instruction::LocalGet(SCRATCH64));
            body.instruction(&Instruction::I32WrapI64);
            body.instruction(&Instruction::I32Const(1));
            body.instruction(&match op {
                JitOp::IncrementIBoxed => Instruction::I32Add,
                _ => Instruction::I32Sub,
            });
            emit_box_int(body);
            body.instruction(&Instruction::Else);
            body.instruction(&Instruction::LocalGet(SCRATCH64));
            body.instruction(&Instruction::Call(if matches!(op, JitOp::IncrementIBoxed) {
                INCREMENT_I
            } else {
                DECREMENT_I
            }));
            body.instruction(&Instruction::End);
        }
        JitOp::PushScopeReal => {
            if *depth < 1 {
                return None;
            }
            // Pop the scope `Value` into the real scope stack via `push_scope`;
            // drop its dummy result. Net -1.
            body.instruction(&Instruction::Call(PUSH_SCOPE));
            body.instruction(&Instruction::Drop);
            *depth -= 1;
        }
        JitOp::PopScopeReal => {
            // Pop the real scope stack; no operand-stack effect. Pass a dummy arg,
            // drop the dummy result.
            body.instruction(&Instruction::I64Const(0));
            body.instruction(&Instruction::Call(POP_SCOPE));
            body.instruction(&Instruction::Drop);
        }
        JitOp::GetScopeObject(index) => {
            // Push local scope `index` via `get_scope_object` (index immediate).
            body.instruction(&Instruction::I64Const(index as i64));
            body.instruction(&Instruction::Call(GET_SCOPE_OBJECT));
            *depth += 1;
        }
        JitOp::GetOuterScope(index) => {
            // Push outer (captured) scope `index` via `get_outer_scope`.
            body.instruction(&Instruction::I64Const(index as i64));
            body.instruction(&Instruction::Call(GET_OUTER_SCOPE));
            *depth += 1;
        }
        JitOp::CoerceString => {
            if *depth < 1 {
                return None;
            }
            // `ToString(v)`; a throwing `toString` stashes into `PENDING_ERROR` — the
            // compile loop emits the perr bail/dispatch after (like a call). Net 0.
            body.instruction(&Instruction::Call(COERCE_S));
        }
        JitOp::Coerce(k) => {
            if *depth < 1 {
                return None;
            }
            // Stack: [.., value]. Push the class POINTER (baked from `coerce_classes[k]`,
            // not the index) and call the arity-2 `coerce` import: pops (value, class_ptr),
            // pushes `ToType(value, class)` — net 0. Baking the pointer frees `coerce` from
            // reading the per-method `RUN_CTX.coerce_classes` table, so a `Coerce{class}`
            // method can be a direct-call target (a direct entry doesn't build that table).
            // A failing coercion (`#1034`) stashes into `PENDING_ERROR`; the compile loop
            // emits the perr bail/dispatch after. `?` declines if the table lacks the entry
            // (never for a real method; guards the convenience wrappers).
            let class = *coerce_classes.get(k as usize)?;
            body.instruction(&Instruction::I64Const(class as usize as i64));
            body.instruction(&Instruction::Call(lay.coerce_index));
        }
        JitOp::IncDecLocalIValue(index, inc) => {
            // `local[index] ±= 1` in place: load the local, `±1` (inline when
            // int-boxed, the wrapping-`ToInt32` helper otherwise — see
            // `IncrementIBoxed`), re-box, store. No operand-stack effect.
            body.instruction(&Instruction::LocalGet(STATE_PTR)); // store address
            body.instruction(&Instruction::LocalGet(STATE_PTR));
            body.instruction(&Instruction::I64Load(slot(index)));
            body.instruction(&Instruction::LocalSet(SCRATCH64));
            emit_is_int(body, SCRATCH64);
            body.instruction(&Instruction::If(BlockType::Result(ValType::I64)));
            body.instruction(&Instruction::LocalGet(SCRATCH64));
            body.instruction(&Instruction::I32WrapI64);
            body.instruction(&Instruction::I32Const(1));
            body.instruction(&if inc {
                Instruction::I32Add
            } else {
                Instruction::I32Sub
            });
            body.instruction(&Instruction::I64ExtendI32U);
            body.instruction(&Instruction::I64Const(VALUE_INT_MARK as i64));
            body.instruction(&Instruction::I64Or);
            body.instruction(&Instruction::Else);
            body.instruction(&Instruction::LocalGet(SCRATCH64));
            body.instruction(&Instruction::Call(if inc { INCREMENT_I } else { DECREMENT_I }));
            body.instruction(&Instruction::End);
            body.instruction(&Instruction::I64Store(slot(index)));
        }
        JitOp::IncDecLocalNum(index, inc) => {
            // `inclocal`/`declocal`: `local[index] = ToNumber(local[index]) ± 1` via
            // the arity-1 `increment`/`decrement` helper (full `Value` semantics; a
            // throwing coercion is swallowed there, like the stack forms). No
            // operand-stack effect.
            body.instruction(&Instruction::LocalGet(STATE_PTR)); // store address
            body.instruction(&Instruction::LocalGet(STATE_PTR));
            body.instruction(&Instruction::I64Load(slot(index)));
            body.instruction(&Instruction::Call(if inc { INCREMENT_NUM } else { DECREMENT_NUM }));
            body.instruction(&Instruction::I64Store(slot(index)));
        }
        JitOp::HasNext2(obj_reg, idx_reg) => {
            // The enumeration step: `hasnext2 object_reg index_reg`. Load both
            // register values, call the arity-2 `hasnext2` helper → the Boolean
            // (left on the stack), then store the helper's two stashed register
            // outputs back into the locals via the `hn2_*` fetchers. On the
            // early-false/error paths the stash holds the inputs, so the stores
            // are identity; a throwing step rides `PENDING_ERROR` and the compile
            // loop's perr bail runs after this arm (before the Boolean is used).
            // Net +1.
            body.instruction(&Instruction::LocalGet(STATE_PTR));
            body.instruction(&Instruction::I64Load(slot(obj_reg)));
            body.instruction(&Instruction::LocalGet(STATE_PTR));
            body.instruction(&Instruction::I64Load(slot(idx_reg)));
            body.instruction(&Instruction::Call(lay.t_base + H2_HAS_NEXT2));
            body.instruction(&Instruction::LocalGet(STATE_PTR));
            body.instruction(&Instruction::I64Const(0));
            body.instruction(&Instruction::Call(HN2_INDEX));
            body.instruction(&Instruction::I64Store(slot(idx_reg)));
            body.instruction(&Instruction::LocalGet(STATE_PTR));
            body.instruction(&Instruction::I64Const(0));
            body.instruction(&Instruction::Call(HN2_OBJECT));
            body.instruction(&Instruction::I64Store(slot(obj_reg)));
            *depth += 1;
        }
        JitOp::PushBits(bits) => {
            // A pre-resolved `Value`'s raw bits, baked as an immediate — no helper, no
            // table. Net +1. (`GetScriptGlobals`/`PushString` are rewritten to this.)
            body.instruction(&Instruction::I64Const(bits as i64));
            *depth += 1;
        }
        JitOp::GetScriptGlobals(k) => {
            // Fallback (un-rewritten): push the `k`-th script's global object via the
            // arity-1 helper. Net +1. `compile_method` normally rewrites this to `PushBits`.
            body.instruction(&Instruction::I64Const(k as i64));
            body.instruction(&Instruction::Call(GET_SCRIPT_GLOBALS));
            *depth += 1;
        }
        JitOp::PushString(k) => {
            // Push the `k`-th string constant's pre-resolved `Value` bits: pass `k`
            // as the arity-1 helper's argument (a plain table read). Net +1.
            body.instruction(&Instruction::I64Const(k as i64));
            body.instruction(&Instruction::Call(GET_PUSH_STRING));
            *depth += 1;
        }
        JitOp::NewCatch(index) => {
            // Push the catch scope object for exception handler `index`. Net +1.
            body.instruction(&Instruction::I64Const(index as i64));
            body.instruction(&Instruction::Call(NEW_CATCH));
            *depth += 1;
        }
        JitOp::GetProperty(k) => {
            if *depth < 1 {
                return None;
            }
            // Stack: [.., receiver]. Push the multiname POINTER (baked from `mn_list[k]`,
            // not the index) and call the arity-2 `gp` helper: pops (receiver, mn), pushes
            // the result — net depth 0. Baking the pointer frees `gp` from reading the
            // per-method `RUN_CTX.multinames` table, so a `getproperty` method can be a
            // direct-call target (a direct entry doesn't build that table). `?` declines
            // (→ interpreter) if `mn_list` lacks the entry — never happens for a real
            // method (its table is complete); guards the convenience wrappers.
            let mn = *mn_list.get(k as usize)?;
            body.instruction(&Instruction::I64Const(mn as usize as i64));
            body.instruction(&Instruction::Call(lay.gp_index));
        }
        JitOp::GetPropertyIc(k, site) => {
            if *depth < 1 {
                return None;
            }
            // Inline monomorphic cache: guard the receiver's vtable, load the
            // cached slot on a hit, else `gpi(receiver, mn, site)`. Net depth 0.
            // The layout is present whenever translate emits this op; `?` is a
            // belt-and-braces bail (an emitter bug, not a semantic path). The miss
            // passes the baked multiname POINTER (see `GetProperty`).
            let (ptr_off, _len_off) = slot_layout()?;
            let vtable_off = vtable_offset()?;
            let mn = *mn_list.get(k as usize)?;
            emit_getprop_ic(
                body,
                ptr_off,
                vtable_off,
                IC_BASE,
                mn as usize as u64,
                site,
                lay.gpi_index,
                SCRATCH,
                SCRATCH64,
            );
        }
        JitOp::FindProp(k, strict) => {
            // No operand: push (k, strict) and call the arity-2 `fp` helper, which
            // resolves the multiname through the scope chain and pushes the scope
            // object — net +1. A strict miss (`#1065`) rides `PENDING_ERROR` and is
            // bailed/dispatched by the compile loop (like `gp`, via
            // `is_throwing_call_or_dm`).
            body.instruction(&Instruction::I64Const(k as i64));
            body.instruction(&Instruction::I64Const(strict as i64));
            body.instruction(&Instruction::Call(lay.fp_index));
            *depth += 1;
        }
        JitOp::GetPropertyFast(k, _) => {
            if *depth < 2 {
                return None;
            }
            // Stack: [.., receiver, name]. Push the multiname index and call the
            // arity-3 `gpf` helper: pops (receiver, name, k), pushes the result — net -1.
            body.instruction(&Instruction::I64Const(k as i64));
            body.instruction(&Instruction::Call(lay.gpf_index));
            *depth -= 1;
        }
        JitOp::GetSlot(slot_id, _) => {
            if *depth < 1 {
                return None;
            }
            // Stack: [.., receiver] — net depth 0 on every path.
            if let Some((ptr_off, len_off)) = slot_layout() {
                // **Inline fast path** (web): an object receiver's payload is the
                // raw `Gc` data pointer, whose `ScriptObjectData` prefix holds the
                // slots slice — chase it directly on memory 1 (Ruffle's own heap):
                //   object-boxed?  → slots_len > slot_id?  → load slots[slot_id]
                // Anything else (null/primitive receiver → throw path; an
                // out-of-range id → the helper's own panic) falls back to `gs`.
                // The fast path can't throw, so the perr bail lives only in the
                // fallback arms (`GetSlot` is self-bailing; see the compile loop).
                let emit_gs_fallback = |body: &mut Function| {
                    body.instruction(&Instruction::LocalGet(SCRATCH64));
                    body.instruction(&Instruction::I64Const(slot_id as i64));
                    body.instruction(&Instruction::Call(lay.gs_index));
                    if lay.inline_perr {
                        emit_perr_bail(body, lay.perr_index);
                    }
                };
                body.instruction(&Instruction::LocalSet(SCRATCH64)); // receiver bits
                body.instruction(&Instruction::LocalGet(SCRATCH64));
                body.instruction(&Instruction::I64Const(48));
                body.instruction(&Instruction::I64ShrU);
                body.instruction(&Instruction::I64Const(VALUE_OBJECT_TAG16));
                body.instruction(&Instruction::I64Eq);
                body.instruction(&Instruction::If(BlockType::Result(ValType::I64)));
                // Object: pointer = the payload's low 32 bits (wasm32 pointers).
                body.instruction(&Instruction::LocalGet(SCRATCH64));
                body.instruction(&Instruction::I32WrapI64);
                body.instruction(&Instruction::LocalSet(SCRATCH));
                // In range? `slots_len > slot_id` (unsigned).
                body.instruction(&Instruction::LocalGet(SCRATCH));
                body.instruction(&Instruction::I32Load(MemArg {
                    offset: len_off as u64,
                    align: 2,
                    memory_index: 1,
                }));
                body.instruction(&Instruction::I32Const(slot_id as i32));
                body.instruction(&Instruction::I32GtU);
                body.instruction(&Instruction::If(BlockType::Result(ValType::I64)));
                // slots_ptr → the raw `Value` bits at `slots_ptr + slot_id * 8`.
                body.instruction(&Instruction::LocalGet(SCRATCH));
                body.instruction(&Instruction::I32Load(MemArg {
                    offset: ptr_off as u64,
                    align: 2,
                    memory_index: 1,
                }));
                body.instruction(&Instruction::I64Load(MemArg {
                    offset: slot_id as u64 * 8,
                    align: 3,
                    memory_index: 1,
                }));
                body.instruction(&Instruction::Else);
                emit_gs_fallback(body);
                body.instruction(&Instruction::End);
                body.instruction(&Instruction::Else);
                emit_gs_fallback(body);
                body.instruction(&Instruction::End);
            } else {
                // Helper path (native): push the slot id and call the arity-2 `gs`
                // helper: pops (receiver, slot_id), pushes the result.
                body.instruction(&Instruction::I64Const(slot_id as i64));
                body.instruction(&Instruction::Call(lay.gs_index));
                if lay.inline_perr {
                    emit_perr_bail(body, lay.perr_index);
                }
            }
        }
        JitOp::PushIntValue(v) => {
            // A boxed `int` `Value`: `VALUE_INT_MARK | (v as u32)`.
            body.instruction(&Instruction::I64Const(
                (VALUE_INT_MARK | (v as u32 as u64)) as i64,
            ));
            *depth += 1;
        }
        JitOp::PushConst(bits) => {
            // A boxed primitive `Value` constant (bits baked at translate time).
            body.instruction(&Instruction::I64Const(bits as i64));
            *depth += 1;
        }
        JitOp::CallHelper3(helper, imm) => {
            if *depth < 2 {
                return None;
            }
            // Stack: [.., receiver, value]. Push the immediate (slot id) and call
            // the arity-3 ternary helper: pops (receiver, value, imm), pushes a
            // dummy result which we drop — net depth -2 (a `void` store).
            body.instruction(&Instruction::I64Const(imm as i64));
            body.instruction(&Instruction::Call(lay.set3_index[helper as usize]));
            body.instruction(&Instruction::Drop);
            *depth -= 2;
        }
        JitOp::DmLoad(width) => {
            if *depth < 1 {
                return None;
            }
            // domainMemory is byte-addressed and may be unaligned, so `align: 0`.
            let ma = MemArg { offset: 0, align: 0, memory_index: 1 };
            // Address = pure-wasm `ToInt32` of the addr `Value` — handles an
            // *int-boxed* addr AND a `Number`-boxed one (CrossBridge models the
            // 32-bit machine's pointers as `Number`, so nearly every FlasCC/Lua dm
            // access has a Number address; the old `is_int`-only fast path bailed
            // on all of them, making `dm_load*`/`dm_store` the profile's #1 cost).
            // `emit_to_int32` is exact under `emit_is_int32_able` (numeric AND
            // `|v| < 2^63`), matching the helper's `coerce_to_i32`; NaN/±Inf/huge
            // fail that guard and fall to the helper (which throws #1506 on a real
            // OOB). This also subsumes the earlier #1506 garbage-low-bits guard.
            body.instruction(&Instruction::LocalSet(SCRATCH64)); // SCRATCH64 = addr Value
            emit_to_int32(body, SCRATCH64);
            body.instruction(&Instruction::LocalSet(SCRATCH)); // addr i32 (ToInt32)
            // Fast path: numeric ToInt32-able addr AND desc != 0 AND in-bounds.
            emit_is_int32_able(body, SCRATCH64);
            // The descriptor cell (see core `SharedByteBuffer::desc_ptr`): DM_BASE is
            // the ADDRESS of a stable `[base: u32, cap: u32]` pair, re-read on
            // every access so a growth reallocation (the buffer has NO
            // reservation) is observed immediately — even by a frame that never
            // exits. `DM_BASE == 0` = domainMemory unavailable (the cap load
            // then reads memory1[4..8] — harmless garbage, masked by the `!= 0`
            // conjunct).
            body.instruction(&Instruction::LocalGet(DM_BASE));
            body.instruction(&Instruction::I32Const(0));
            body.instruction(&Instruction::I32Ne);
            body.instruction(&Instruction::I32And);
            // In-bounds? `(addr as u64) + width <= cap` (64-bit: no wrap).
            body.instruction(&Instruction::LocalGet(SCRATCH));
            body.instruction(&Instruction::I64ExtendI32U);
            body.instruction(&Instruction::I64Const(width as i64));
            body.instruction(&Instruction::I64Add);
            body.instruction(&Instruction::LocalGet(DM_BASE));
            body.instruction(&Instruction::I32Load(MemArg { offset: 4, align: 2, memory_index: 1 }));
            body.instruction(&Instruction::I64ExtendI32U);
            body.instruction(&Instruction::I64LeU);
            body.instruction(&Instruction::I32And);
            body.instruction(&Instruction::If(BlockType::Result(ValType::I64)));
            // load memory1[base + addr], zero-extend, box as an int `Value`.
            body.instruction(&Instruction::LocalGet(DM_BASE));
            body.instruction(&Instruction::I32Load(MemArg { offset: 0, align: 2, memory_index: 1 }));
            body.instruction(&Instruction::LocalGet(SCRATCH));
            body.instruction(&Instruction::I32Add);
            body.instruction(&match width {
                1 => Instruction::I32Load8U(ma),
                2 => Instruction::I32Load16U(ma),
                _ => Instruction::I32Load(ma),
            });
            body.instruction(&Instruction::I64ExtendI32U);
            body.instruction(&Instruction::I64Const(VALUE_INT_MARK as i64));
            body.instruction(&Instruction::I64Or);
            body.instruction(&Instruction::Else);
            // Miss (non-int addr, OOB of the reservation, or `dm_len == 0`): fall
            // back to the dm helper on the ORIGINAL addr `Value` — it `coerce_to_i32`s
            // it and routes through real storage (throwing #1506 on a genuine OOB).
            body.instruction(&Instruction::LocalGet(SCRATCH64));
            body.instruction(&Instruction::Call(match width {
                1 => DM_LOAD8,
                2 => DM_LOAD16,
                _ => DM_LOAD32,
            }));
            body.instruction(&Instruction::End);
            // net: pop addr (-1), push result (+1) → depth unchanged.
        }
        JitOp::DmStore(width) => {
            if *depth < 2 {
                return None;
            }
            let ma = MemArg { offset: 0, align: 0, memory_index: 1 };
            // Stack: [value, addr] (addr on top). Both addr and stored value go
            // through pure-wasm `ToInt32` (int OR Number-boxed — see `DmLoad`);
            // CrossBridge Number-boxes both, so the old int-only path bailed to the
            // helper on every store. `emit_to_int32` is exact under
            // `emit_is_int32_able`, matching the helper's `coerce_to_i32`.
            body.instruction(&Instruction::LocalSet(SCRATCH64)); // SCRATCH64 = addr Value
            emit_to_int32(body, SCRATCH64);
            body.instruction(&Instruction::LocalSet(SCRATCH)); // addr i32
            body.instruction(&Instruction::LocalSet(SCRATCH64_2)); // SCRATCH64_2 = value Value
            emit_to_int32(body, SCRATCH64_2);
            body.instruction(&Instruction::LocalSet(SCRATCH2)); // value i32
            // Fast path: ToInt32-able addr AND value AND desc != 0 AND in-bounds.
            emit_is_int32_able(body, SCRATCH64);
            emit_is_int32_able(body, SCRATCH64_2);
            body.instruction(&Instruction::I32And);
            body.instruction(&Instruction::LocalGet(DM_BASE));
            body.instruction(&Instruction::I32Const(0));
            body.instruction(&Instruction::I32Ne);
            body.instruction(&Instruction::I32And);
            body.instruction(&Instruction::LocalGet(SCRATCH));
            body.instruction(&Instruction::I64ExtendI32U);
            body.instruction(&Instruction::I64Const(width as i64));
            body.instruction(&Instruction::I64Add);
            body.instruction(&Instruction::LocalGet(DM_BASE));
            body.instruction(&Instruction::I32Load(MemArg { offset: 4, align: 2, memory_index: 1 }));
            body.instruction(&Instruction::I64ExtendI32U);
            body.instruction(&Instruction::I64LeU);
            body.instruction(&Instruction::I32And);
            body.instruction(&Instruction::If(BlockType::Empty));
            body.instruction(&Instruction::LocalGet(DM_BASE));
            body.instruction(&Instruction::I32Load(MemArg { offset: 0, align: 2, memory_index: 1 }));
            body.instruction(&Instruction::LocalGet(SCRATCH));
            body.instruction(&Instruction::I32Add); // address
            body.instruction(&Instruction::LocalGet(SCRATCH2)); // value
            body.instruction(&match width {
                1 => Instruction::I32Store8(ma),
                2 => Instruction::I32Store16(ma),
                _ => Instruction::I32Store(ma),
            });
            body.instruction(&Instruction::Else);
            // Miss: fall back to the ternary `dm_store` helper on the ORIGINAL
            // `Value`s (it coerces). `(value, addr, width)`.
            body.instruction(&Instruction::LocalGet(SCRATCH64_2)); // value Value
            body.instruction(&Instruction::LocalGet(SCRATCH64)); // addr Value
            body.instruction(&Instruction::I64Const(width as i64));
            body.instruction(&Instruction::Call(lay.set3_index[DM_STORE_KIND as usize]));
            body.instruction(&Instruction::Drop); // dummy result
            body.instruction(&Instruction::End);
            *depth -= 2;
        }
        JitOp::DmLoadF(width) => {
            if *depth < 1 {
                return None;
            }
            let ma = MemArg { offset: 0, align: 0, memory_index: 1 };
            // Address = pure-wasm `ToInt32` (int OR Number-boxed — see `DmLoad`).
            body.instruction(&Instruction::LocalSet(SCRATCH64)); // SCRATCH64 = addr Value
            emit_to_int32(body, SCRATCH64);
            body.instruction(&Instruction::LocalSet(SCRATCH)); // addr i32 (ToInt32)
            // Fast path: ToInt32-able addr AND desc != 0 AND in-bounds (see `DmLoad`).
            emit_is_int32_able(body, SCRATCH64);
            body.instruction(&Instruction::LocalGet(DM_BASE));
            body.instruction(&Instruction::I32Const(0));
            body.instruction(&Instruction::I32Ne);
            body.instruction(&Instruction::I32And);
            body.instruction(&Instruction::LocalGet(SCRATCH));
            body.instruction(&Instruction::I64ExtendI32U);
            body.instruction(&Instruction::I64Const(width as i64));
            body.instruction(&Instruction::I64Add);
            body.instruction(&Instruction::LocalGet(DM_BASE));
            body.instruction(&Instruction::I32Load(MemArg { offset: 4, align: 2, memory_index: 1 }));
            body.instruction(&Instruction::I64ExtendI32U);
            body.instruction(&Instruction::I64LeU);
            body.instruction(&Instruction::I32And);
            body.instruction(&Instruction::If(BlockType::Result(ValType::I64)));
            // load f32/f64 from memory1[base + addr], promote to f64, box as Number.
            body.instruction(&Instruction::LocalGet(DM_BASE));
            body.instruction(&Instruction::I32Load(MemArg { offset: 0, align: 2, memory_index: 1 }));
            body.instruction(&Instruction::LocalGet(SCRATCH));
            body.instruction(&Instruction::I32Add);
            if width == 4 {
                body.instruction(&Instruction::F32Load(ma));
                body.instruction(&Instruction::F64PromoteF32);
            } else {
                body.instruction(&Instruction::F64Load(ma));
            }
            // LOSSLESS box (not `emit_box_double`, which canonicalizes NaN):
            // FlasCC copies raw memory through `lf*`/`sf*`, so a NaN payload must
            // survive. Non-colliding bits box inline; a colliding negative
            // quiet-NaN (`bits & BOX_MARK == BOX_MARK`) re-loads via the
            // `number_lossless` helper (heap-boxes). Mirrors core `op_lf64`.
            body.instruction(&Instruction::I64ReinterpretF64);
            body.instruction(&Instruction::LocalTee(SCRATCH64_2)); // keep the bits
            body.instruction(&Instruction::I64Const(BOX_MARK as i64));
            body.instruction(&Instruction::I64And);
            body.instruction(&Instruction::I64Const(BOX_MARK as i64));
            body.instruction(&Instruction::I64Ne); // non-colliding?
            body.instruction(&Instruction::If(BlockType::Result(ValType::I64)));
            body.instruction(&Instruction::LocalGet(SCRATCH64_2)); // inline raw bits
            body.instruction(&Instruction::Else);
            body.instruction(&Instruction::LocalGet(SCRATCH64)); // original addr Value
            body.instruction(&Instruction::Call(if width == 4 { DM_LOADF32 } else { DM_LOADF64 }));
            body.instruction(&Instruction::End);
            body.instruction(&Instruction::Else);
            // Miss (non-int addr / OOB / unshared): fall back to the float dm load
            // helper on the ORIGINAL addr `Value` (it coerces; #1506 on genuine OOB).
            body.instruction(&Instruction::LocalGet(SCRATCH64));
            body.instruction(&Instruction::Call(if width == 4 { DM_LOADF32 } else { DM_LOADF64 }));
            body.instruction(&Instruction::End);
            // net: pop addr (-1), push result (+1) → depth unchanged.
        }
        JitOp::DmStoreF(width) => {
            if *depth < 2 {
                return None;
            }
            let ma = MemArg { offset: 0, align: 0, memory_index: 1 };
            // Stack: [value, addr] (addr on top). The addr goes through pure-wasm
            // `ToInt32` (int OR Number-boxed — CrossBridge Number-boxes it; see
            // `DmLoad`). The stored value goes through `emit_numval`, which already
            // handles int OR Number.
            body.instruction(&Instruction::LocalSet(SCRATCH64_2)); // SCRATCH64_2 = addr Value
            emit_to_int32(body, SCRATCH64_2);
            body.instruction(&Instruction::LocalSet(SCRATCH)); // addr i32 (ToInt32)
            body.instruction(&Instruction::LocalSet(SCRATCH64)); // value `Value` bits
            // Fast path: ToInt32-able addr AND value is inline-numeric (int or f64)
            // AND desc != 0 AND in-bounds. A boxed double (colliding NaN) or any
            // non-numeric box isn't `emit_numval`-safe, so it falls to the coercing
            // `dm_store_f` helper (which derefs a boxed double).
            emit_is_int32_able(body, SCRATCH64_2);
            // value inline-numeric?  is_f64(value) || is_int(value)
            body.instruction(&Instruction::LocalGet(SCRATCH64));
            body.instruction(&Instruction::I64Const(BOX_MARK as i64));
            body.instruction(&Instruction::I64And);
            body.instruction(&Instruction::I64Const(BOX_MARK as i64));
            body.instruction(&Instruction::I64Ne); // is_f64(value)
            emit_is_int(body, SCRATCH64);
            body.instruction(&Instruction::I32Or);
            body.instruction(&Instruction::I32And); // int addr AND value numeric
            body.instruction(&Instruction::LocalGet(DM_BASE));
            body.instruction(&Instruction::I32Const(0));
            body.instruction(&Instruction::I32Ne);
            body.instruction(&Instruction::I32And);
            body.instruction(&Instruction::LocalGet(SCRATCH));
            body.instruction(&Instruction::I64ExtendI32U);
            body.instruction(&Instruction::I64Const(width as i64));
            body.instruction(&Instruction::I64Add);
            body.instruction(&Instruction::LocalGet(DM_BASE));
            body.instruction(&Instruction::I32Load(MemArg { offset: 4, align: 2, memory_index: 1 }));
            body.instruction(&Instruction::I64ExtendI32U);
            body.instruction(&Instruction::I64LeU);
            body.instruction(&Instruction::I32And);
            body.instruction(&Instruction::If(BlockType::Empty));
            body.instruction(&Instruction::LocalGet(DM_BASE));
            body.instruction(&Instruction::I32Load(MemArg { offset: 0, align: 2, memory_index: 1 }));
            body.instruction(&Instruction::LocalGet(SCRATCH));
            body.instruction(&Instruction::I32Add); // address
            emit_numval(body, SCRATCH64); // value `Value` (int/number) → f64
            if width == 4 {
                body.instruction(&Instruction::F32DemoteF64);
                body.instruction(&Instruction::F32Store(ma));
            } else {
                body.instruction(&Instruction::F64Store(ma));
            }
            body.instruction(&Instruction::Else);
            // Miss (non-int addr / OOB / unshared): fall back to the ternary
            // `dm_store_f` helper on the ORIGINAL `Value`s (it coerces).
            // `(value Value bits, addr Value, width)`.
            body.instruction(&Instruction::LocalGet(SCRATCH64));
            body.instruction(&Instruction::LocalGet(SCRATCH64_2));
            body.instruction(&Instruction::I64Const(width as i64));
            body.instruction(&Instruction::Call(lay.set3_index[DM_STORE_F_KIND as usize]));
            body.instruction(&Instruction::Drop); // dummy result
            body.instruction(&Instruction::End);
            *depth -= 2;
        }
        JitOp::CallMethod(id, argc, push) => {
            // Stack: [.., receiver, arg0, .., arg{argc-1}] (arg{argc-1} on top).
            if *depth < argc as i32 + 1 {
                return None;
            }
            // Spill the args (each `Call` pops one, top-first → `push_call_arg`).
            for _ in 0..argc {
                body.instruction(&Instruction::Call(lay.pca_index));
            }
            // Stack: [.., receiver]. Call `call_method(receiver, id, argc)`.
            body.instruction(&Instruction::I64Const(id as i64));
            body.instruction(&Instruction::I64Const(argc as i64));
            body.instruction(&Instruction::Call(lay.call_index));
            // Stack: [.., result]. Drop it for the void form (`push_return_value`
            // false); `callpropvoid`/discarded results use this.
            if !push {
                body.instruction(&Instruction::Drop);
            }
            // Error check: if the call threw, bail out of the whole method now.
            // `try_run` sees the pending error and propagates it; the returned
            // `undefined` here is ignored. `Return` is stack-polymorphic.
            if lay.inline_perr {
                emit_perr_bail(body, lay.perr_index);
            }
            *depth -= argc as i32 + 1;
            if push {
                *depth += 1;
            }
        }
        JitOp::CallMethodDirect(disp_id, argc, push, site) => {
            // Stack: [.., receiver, arg0, .., arg{argc-1}]. `emit_call_direct` consumes
            // them (writes them into the callee frame), guards the receiver's vtable,
            // and either `call_indirect`s the callee's `run` (hit) or `cmi(receiver,
            // disp, argc, cell_addr)` (miss, which also refills the cell). A non-leaf
            // callee reached on the hit path can throw — its own `cm` stashes the error in
            // `PENDING_ERROR`/`PENDING_FLAG` and it bails — so the `perr` bail after the
            // call propagates it out of the caller too (a no-op when nothing threw).
            if *depth < argc as i32 + 1 {
                return None;
            }
            let vtable_off = vtable_offset()?;
            let instance_class_off = instance_class_offset()?;
            let cell_offset = call_region_offset + site * CALL_CELL_SIZE;
            emit_call_direct(
                body,
                argc,
                disp_id,
                vtable_off,
                instance_class_off,
                IC_BASE,
                cell_offset,
                STATE_PTR,
                DM_BASE,
                DM_LEN,
                /*run_type_index*/ 0,
                /*dispatch_type_index*/ 6,
                lay.cmi_index,
                lay.call_index,
                lay.pca_index,
                lay.scope_enter_index,
                lay.scope_leave_index,
                SCRATCH,
                SCRATCH2,
                SCRATCH64,
                BLOCK,
                FRAME_STRIDE,
                FRAME_MEM_BYTES,
                push,
            );
            if lay.inline_perr {
                emit_perr_bail(body, lay.perr_index);
            }
            *depth -= argc as i32 + 1;
            if push {
                *depth += 1;
            }
        }
        JitOp::CallProperty(k, argc, push) => {
            // Same shape as `CallMethod`, but calls `call_property(receiver, k, argc)`
            // (the `k`-th multiname) instead of `call_method(receiver, id, argc)`.
            // Stack: [.., receiver, arg0, .., arg{argc-1}].
            if *depth < argc as i32 + 1 {
                return None;
            }
            for _ in 0..argc {
                body.instruction(&Instruction::Call(lay.pca_index));
            }
            body.instruction(&Instruction::I64Const(k as i64));
            body.instruction(&Instruction::I64Const(argc as i64));
            body.instruction(&Instruction::Call(lay.callprop_index));
            if !push {
                body.instruction(&Instruction::Drop);
            }
            if lay.inline_perr {
                emit_perr_bail(body, lay.perr_index);
            }
            *depth -= argc as i32 + 1;
            if push {
                *depth += 1;
            }
        }
        JitOp::ConstructSuper(argc) => {
            // Stack: [.., receiver, arg0, .., arg{argc-1}]. Spill the args, then call
            // `construct_super(receiver, argc)` (drops the void dummy). Net -(argc+1).
            if *depth < argc as i32 + 1 {
                return None;
            }
            for _ in 0..argc {
                body.instruction(&Instruction::Call(lay.pca_index));
            }
            body.instruction(&Instruction::I64Const(argc as i64));
            body.instruction(&Instruction::Call(lay.csup_index));
            body.instruction(&Instruction::Drop);
            if lay.inline_perr {
                emit_perr_bail(body, lay.perr_index);
            }
            *depth -= argc as i32 + 1;
        }
        JitOp::CallValue(argc) => {
            // Stack: [.., function, receiver, arg0, .., arg{argc-1}]. Spill args, then
            // call `call_value(function, receiver, argc)`. Consumes function+receiver+
            // args (argc+2), pushes the result → net -(argc+1).
            if *depth < argc as i32 + 2 {
                return None;
            }
            for _ in 0..argc {
                body.instruction(&Instruction::Call(lay.pca_index));
            }
            // Stack: [.., function, receiver]. Push argc; the call pops (argc, receiver,
            // function) → helper `(function, receiver, argc)`.
            body.instruction(&Instruction::I64Const(argc as i64));
            body.instruction(&Instruction::Call(lay.callv_index));
            if lay.inline_perr {
                emit_perr_bail(body, lay.perr_index);
            }
            *depth -= argc as i32 + 1;
        }
        JitOp::VCall(kind, imm, spill, push) => {
            // Stack: [.., a?, x0, .., x{spill-1}] (`a` present unless a no-receiver
            // kind). Spill the extra operands (top-first, like a call's args), push a
            // dummy `a` for the no-receiver kinds, then call
            // `vc(a, imm, spill, kind)`. Push the result iff `push`; every kind can
            // throw out of band → the shared perr bail/dispatch follows.
            let has_recv = vc::has_receiver(kind);
            let consumed = spill as i32 + has_recv as i32;
            if *depth < consumed {
                return None;
            }
            for _ in 0..spill {
                body.instruction(&Instruction::Call(lay.pca_index));
            }
            if !has_recv {
                body.instruction(&Instruction::I64Const(0));
            }
            body.instruction(&Instruction::I64Const(imm as i64));
            body.instruction(&Instruction::I64Const(spill as i64));
            body.instruction(&Instruction::I64Const(kind as i64));
            body.instruction(&Instruction::Call(lay.vc_index));
            if !push {
                body.instruction(&Instruction::Drop);
            }
            if lay.inline_perr {
                emit_perr_bail(body, lay.perr_index);
            }
            *depth -= consumed;
            if push {
                *depth += 1;
            }
        }
        // --- Double fast path (unboxed f64).
        JitOp::GetLocalDouble(i) => {
            body.instruction(&Instruction::LocalGet(STATE_PTR));
            body.instruction(&Instruction::I64Load(slot(i)));
            body.instruction(&Instruction::F64ReinterpretI64);
            *depth += 1;
        }
        JitOp::SetLocalDouble(i) => {
            if *depth < 1 {
                return None;
            }
            emit_box_double(body); // f64 -> i64 bits
            body.instruction(&Instruction::LocalSet(SCRATCH64));
            body.instruction(&Instruction::LocalGet(STATE_PTR));
            body.instruction(&Instruction::LocalGet(SCRATCH64));
            body.instruction(&Instruction::I64Store(slot(i)));
            *depth -= 1;
        }
        JitOp::StoreLocalDouble(i) => {
            if *depth < 1 {
                return None;
            }
            // Peek-and-store: keep the `f64` on the stack (via `Tee`) while boxing
            // a copy from `SCRATCH_F64` into the local. Net operand-stack effect 0.
            body.instruction(&Instruction::LocalTee(SCRATCH_F64));
            body.instruction(&Instruction::LocalGet(STATE_PTR));
            emit_box_scratch_f64(body); // push boxed i64 (reads SCRATCH_F64)
            body.instruction(&Instruction::I64Store(slot(i)));
        }
        JitOp::PushDouble(bits) => {
            body.instruction(&Instruction::F64Const(f64::from_bits(bits)));
            *depth += 1;
        }
        JitOp::AddD | JitOp::SubtractD | JitOp::MultiplyD | JitOp::DivideD => {
            if *depth < 2 {
                return None;
            }
            body.instruction(&match op {
                JitOp::AddD => Instruction::F64Add,
                JitOp::SubtractD => Instruction::F64Sub,
                JitOp::MultiplyD => Instruction::F64Mul,
                _ => Instruction::F64Div,
            });
            *depth -= 1;
        }
        JitOp::IncrementD | JitOp::DecrementD => {
            if *depth < 1 {
                return None;
            }
            body.instruction(&Instruction::F64Const(1.0));
            body.instruction(&match op {
                JitOp::IncrementD => Instruction::F64Add,
                _ => Instruction::F64Sub,
            });
        }
        JitOp::NegateD => {
            if *depth < 1 {
                return None;
            }
            body.instruction(&Instruction::F64Neg);
        }
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

/// Boxes a raw `urshift` result (the i32 on top of the stack, whose bits are the
/// unsigned `ToUint32` result in `0..2^32`) as a `Value`, matching the helper:
/// a `uint` that fits AVM2 `int` (`< 2^31`) → an `int` box; otherwise (`>= 2^31`,
/// which can't be a signed `i32`) → a `Number` box. The `Number` value is always
/// a finite positive integer in `[2^31, 2^32)`, so a raw `f64` reinterpret is a
/// valid box (never NaN/`-0`, never in the boxed-double collision range — its
/// sign bit is `0`). Clobbers `SCRATCH64`.
fn emit_urshift_box(body: &mut Function) {
    body.instruction(&Instruction::I64ExtendI32U); // uint as i64 (zero-extended)
    body.instruction(&Instruction::LocalTee(SCRATCH64));
    body.instruction(&Instruction::I64Const(0x8000_0000)); // 2^31
    body.instruction(&Instruction::I64GeU);
    body.instruction(&Instruction::If(BlockType::Result(ValType::I64)));
    // >= 2^31: doesn't fit `int` → `Number`.
    body.instruction(&Instruction::LocalGet(SCRATCH64));
    body.instruction(&Instruction::F64ConvertI64U); // exact for uint < 2^32
    body.instruction(&Instruction::I64ReinterpretF64);
    body.instruction(&Instruction::Else);
    // < 2^31: fits `int`.
    body.instruction(&Instruction::LocalGet(SCRATCH64));
    body.instruction(&Instruction::I32WrapI64);
    emit_box_int(body);
    body.instruction(&Instruction::End);
}

/// Pops the top `Value` (i64), pushes its `ToBoolean` as an i32 `0`/`1` — inline
/// for the hot cases (a `Boolean`-boxed comparison result, an int box), falling
/// back to the `to_boolean` helper (h5) for everything else (Number, string,
/// object, null, undefined). The helper crossing per branch/`coerceb` was ~1 s of
/// an OpenTTD gameplay profile. Clobbers `SCRATCH64`.
fn emit_to_boolean_i32(body: &mut Function) {
    body.instruction(&Instruction::LocalSet(SCRATCH64));
    // Boolean-boxed → the payload bit.
    body.instruction(&Instruction::LocalGet(SCRATCH64));
    body.instruction(&Instruction::I64Const(VALUE_TAG_MASK as i64));
    body.instruction(&Instruction::I64And);
    body.instruction(&Instruction::I64Const(VALUE_BOOL_MARK as i64));
    body.instruction(&Instruction::I64Eq);
    body.instruction(&Instruction::If(BlockType::Result(ValType::I32)));
    body.instruction(&Instruction::LocalGet(SCRATCH64));
    body.instruction(&Instruction::I32WrapI64);
    body.instruction(&Instruction::I32Const(1));
    body.instruction(&Instruction::I32And);
    body.instruction(&Instruction::Else);
    // int-boxed → non-zero payload.
    emit_is_int(body, SCRATCH64);
    body.instruction(&Instruction::If(BlockType::Result(ValType::I32)));
    body.instruction(&Instruction::LocalGet(SCRATCH64));
    body.instruction(&Instruction::I32WrapI64);
    body.instruction(&Instruction::I32Const(0));
    body.instruction(&Instruction::I32Ne);
    body.instruction(&Instruction::Else);
    // Everything else (Number incl. NaN/-0, string, object, null, undefined):
    // the helper implements the full `ToBoolean`.
    body.instruction(&Instruction::LocalGet(SCRATCH64));
    body.instruction(&Instruction::Call(TO_BOOLEAN));
    body.instruction(&Instruction::I32WrapI64);
    body.instruction(&Instruction::End);
    body.instruction(&Instruction::End);
}

/// Pushes i32 `1` if the `Value` in `local` is int-boxed, else `0`
/// (`is_int = (x & VALUE_TAG_MASK) == VALUE_INT_MARK`).
fn emit_is_int(body: &mut Function, local: u32) {
    body.instruction(&Instruction::LocalGet(local));
    body.instruction(&Instruction::I64Const(VALUE_TAG_MASK as i64));
    body.instruction(&Instruction::I64And);
    body.instruction(&Instruction::I64Const(VALUE_INT_MARK as i64));
    body.instruction(&Instruction::I64Eq);
}

/// Pushes i32 `1` if the `Value` in `local` is numeric (an `int` box **or** a
/// `Number`), else `0`. `is_f64 = (x & BOX_MARK) != BOX_MARK`;
/// `is_int = (x & VALUE_TAG_MASK) == VALUE_INT_MARK`.
fn emit_is_numeric(body: &mut Function, local: u32) {
    body.instruction(&Instruction::LocalGet(local));
    body.instruction(&Instruction::I64Const(BOX_MARK as i64));
    body.instruction(&Instruction::I64And);
    body.instruction(&Instruction::I64Const(BOX_MARK as i64));
    body.instruction(&Instruction::I64Ne); // is_f64
    body.instruction(&Instruction::LocalGet(local));
    body.instruction(&Instruction::I64Const(VALUE_TAG_MASK as i64));
    body.instruction(&Instruction::I64And);
    body.instruction(&Instruction::I64Const(VALUE_INT_MARK as i64));
    body.instruction(&Instruction::I64Eq); // is_int
    body.instruction(&Instruction::I32Or);
}

/// Pushes i32 `1` if the `Value` in `local` is numeric with `|numval| < 2^63` —
/// the precondition for the pure-wasm `ToInt32` ([`emit_to_int32`]). NaN and ±Inf
/// fail the `<` compare, so they (like non-numerics) fall to the helper path.
/// (On a non-numeric value `emit_numval` reinterprets pointer bits as f64 —
/// garbage but pure, and the `is_numeric` conjunct zeroes the result.)
fn emit_is_int32_able(body: &mut Function, local: u32) {
    emit_is_numeric(body, local);
    emit_numval(body, local);
    body.instruction(&Instruction::F64Abs);
    body.instruction(&Instruction::F64Const(9_223_372_036_854_775_808.0)); // 2^63
    body.instruction(&Instruction::F64Lt);
    body.instruction(&Instruction::I32And);
}

/// Pushes the wrapping ECMA `ToInt32` (i32) of the numeric `Value` in `local`,
/// pure wasm — no helper crossing. Caller must guarantee
/// [`emit_is_int32_able`]: then `i64.trunc_sat_f64_s` is exact (no saturation)
/// and its low 32 bits are exactly `ToInt32` (mod 2^32, two's complement) —
/// including the hot `uint`-in-`Number`-box case (2^31..2^32).
fn emit_to_int32(body: &mut Function, local: u32) {
    emit_numval(body, local);
    body.instruction(&Instruction::I64TruncSatF64S);
    body.instruction(&Instruction::I32WrapI64);
}

/// Pushes the `f64` numeric value of the (caller-guaranteed numeric) `Value` in
/// `local`: `Number` → `f64.reinterpret`, `int` → `f64.convert_i32_s` of the low
/// 32 bits. Branchless via `select` on `is_f64` (both candidate values are pure,
/// so computing the unused one is harmless — no trap, no memory effect).
fn emit_numval(body: &mut Function, local: u32) {
    // number candidate (select's val1, taken when is_f64)
    body.instruction(&Instruction::LocalGet(local));
    body.instruction(&Instruction::F64ReinterpretI64);
    // int candidate (val2)
    body.instruction(&Instruction::LocalGet(local));
    body.instruction(&Instruction::I32WrapI64);
    body.instruction(&Instruction::F64ConvertI32S);
    // selector: is_f64(local)
    body.instruction(&Instruction::LocalGet(local));
    body.instruction(&Instruction::I64Const(BOX_MARK as i64));
    body.instruction(&Instruction::I64And);
    body.instruction(&Instruction::I64Const(BOX_MARK as i64));
    body.instruction(&Instruction::I64Ne);
    body.instruction(&Instruction::Select);
}

/// Like [`emit_numval`] but also correct for a **heap-boxed double**
/// (`TAG_BOXED_DOUBLE`): a colliding negative-quiet-NaN `Number` that
/// `Value::number_lossless` stashed in a `Gc<f64>`. Its inline bits are a pointer,
/// so `emit_numval`'s reinterpret/int-convert would read garbage; here the
/// boxed-double case instead `f64.load`s the pointee (memory 1 = Ruffle's heap on
/// web, where the `Gc` lives; the exposed pointer is a low-32-bit offset on wasm32).
///
/// Never traps: the `f64.load` fires ONLY when `tag == TAG_BOXED_DOUBLE`, the sole
/// tag whose payload is a live `f64` pointer (produced only by `number_lossless`).
/// Every other input — inline `f64`, `int`, or a swallowed-OOB `Undefined` (the
/// inline dm path returns `undefined` on OOB, see [`is_dm_op`]) — takes a pure
/// arithmetic branch (`Undefined`→`0.0`). Non-numerics reinterpret to garbage,
/// pure and memory-safe, exactly like [`emit_numval`]'s contract.
pub(crate) fn emit_numval_boxsafe(body: &mut Function, local: u32) {
    // is_f64 = (bits & BOX_MARK) != BOX_MARK
    body.instruction(&Instruction::LocalGet(local));
    body.instruction(&Instruction::I64Const(BOX_MARK as i64));
    body.instruction(&Instruction::I64And);
    body.instruction(&Instruction::I64Const(BOX_MARK as i64));
    body.instruction(&Instruction::I64Ne);
    body.instruction(&Instruction::If(BlockType::Result(ValType::F64)));
    // inline `Number`: the bits ARE the f64.
    body.instruction(&Instruction::LocalGet(local));
    body.instruction(&Instruction::F64ReinterpretI64);
    body.instruction(&Instruction::Else);
    // boxed? tag == TAG_BOXED_DOUBLE (tag = (bits >> TAG_SHIFT) & 0x7)
    body.instruction(&Instruction::LocalGet(local));
    body.instruction(&Instruction::I64Const(TAG_SHIFT as i64));
    body.instruction(&Instruction::I64ShrU);
    body.instruction(&Instruction::I64Const(0x7));
    body.instruction(&Instruction::I64And);
    body.instruction(&Instruction::I64Const(TAG_BOXED_DOUBLE as i64));
    body.instruction(&Instruction::I64Eq);
    body.instruction(&Instruction::If(BlockType::Result(ValType::F64)));
    // heap-boxed double: deref the `Gc<f64>` (pointer = low 32 bits on wasm32).
    body.instruction(&Instruction::LocalGet(local));
    body.instruction(&Instruction::I32WrapI64);
    body.instruction(&Instruction::F64Load(MemArg {
        offset: 0,
        align: VALUE_ALIGN,
        memory_index: 1,
    }));
    body.instruction(&Instruction::Else);
    // int box (or swallowed-OOB `Undefined` → 0.0): convert the low 32 bits.
    body.instruction(&Instruction::LocalGet(local));
    body.instruction(&Instruction::I32WrapI64);
    body.instruction(&Instruction::F64ConvertI32S);
    body.instruction(&Instruction::End);
    body.instruction(&Instruction::End);
}

/// Lowers `ops` to a WASM module exporting `run(state_ptr: i32) -> i64` (the
/// returned `Value`'s bits), importing the shared memory as `("env", "memory")`.
/// Returns `None` for anything unsupported.
/// Basic-block leaders: op 0, every branch target, and every op after a
/// terminator. `None` if any branch target is out of range. Shared by [`compile`]
/// and the int-soundness analysis so they agree on the block structure.
pub(crate) fn basic_block_leaders(ops: &[JitOp]) -> Option<Vec<usize>> {
    basic_block_leaders_sw(ops, &[])
}

/// Switch-aware [`basic_block_leaders`]: `switches` supplies each
/// [`JitOp::LookupSwitch`]'s targets (its `default` + every case), which are
/// block leaders too.
pub(crate) fn basic_block_leaders_sw(ops: &[JitOp], switches: &[SwitchTable]) -> Option<Vec<usize>> {
    let mut leaders = BTreeSet::new();
    leaders.insert(0usize);
    for (i, op) in ops.iter().enumerate() {
        if let Some(t) = op.target() {
            if t >= ops.len() {
                return None;
            }
            leaders.insert(t);
        }
        if let JitOp::LookupSwitch(idx) = *op {
            let sw = switches.get(idx as usize)?;
            for &t in std::iter::once(&sw.default).chain(sw.cases.iter()) {
                if t >= ops.len() {
                    return None;
                }
                leaders.insert(t);
            }
        }
        if op.is_terminator() && i + 1 < ops.len() {
            leaders.insert(i + 1);
        }
    }
    Some(leaders.into_iter().collect())
}

/// Spills the top `d` boxed (i64) operands into the spill pool, bottom → `SPILL[0]`
/// … top → `SPILL[d-1]`, leaving the wasm operand stack empty for a branch.
fn emit_spill(body: &mut Function, d: i32) {
    for j in (0..d).rev() {
        body.instruction(&Instruction::LocalSet(SPILL_BASE + j as u32));
    }
}

/// Reloads `d` spilled operands back onto the stack in their original order.
fn emit_reload(body: &mut Function, d: i32) {
    for j in 0..d {
        body.instruction(&Instruction::LocalGet(SPILL_BASE + j as u32));
    }
}

/// Records `d` as basic block `bb`'s operand-stack entry depth, or `None` if a
/// previously recorded value disagrees (an unverifiable stack shape → bail).
fn record_entry(entry_depth: &mut [i32], bb: usize, d: i32) -> Option<()> {
    match entry_depth[bb] {
        -1 => {
            entry_depth[bb] = d;
            Some(())
        }
        prev if prev == d => Some(()),
        _ => None,
    }
}

/// Whether `ops` use the raw-i32 int fast-path model (its operand-stack values are
/// i32, so the i64 spill pool doesn't apply — a cross-branch stack there declines).
fn uses_int_model(ops: &[JitOp]) -> bool {
    ops.iter().any(|o| {
        matches!(
            o,
            JitOp::GetLocal(_)
                | JitOp::SetLocal(_)
                | JitOp::PushInt(_)
                | JitOp::AddI
                | JitOp::SubtractI
                | JitOp::MultiplyI
                | JitOp::IncrementI
                | JitOp::DecrementI
                | JitOp::IncLocalI(_)
                | JitOp::DecLocalI(_)
                | JitOp::LessThan
                | JitOp::LessEquals
                | JitOp::GreaterThan
                | JitOp::GreaterEquals
                | JitOp::Equals
                | JitOp::IfLt(_)
                | JitOp::IfGe(_)
                | JitOp::IfFalse(_)
                | JitOp::IfTrue(_)
                | JitOp::ReturnValue
        )
    })
}

/// A method's exception handler (op indices), for the JIT's exception dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExcRange {
    /// First op covered by the handler (inclusive).
    pub from: usize,
    /// First op past the handler (exclusive).
    pub to: usize,
    /// Op index jumped to on a caught exception (the handler entry).
    pub target: usize,
}

pub fn compile(ops: &[JitOp]) -> Option<Vec<u8>> {
    compile_full(ops, &[], &[], &[], &[], &[], &[], 0)
}

/// Switch-aware [`compile`]: `switches` supplies the targets for each
/// [`JitOp::LookupSwitch`] op (see [`SwitchTable`]).
pub fn compile_with_switches(ops: &[JitOp], switches: &[SwitchTable]) -> Option<Vec<u8>> {
    compile_full(ops, switches, &[], &[], &[], &[], &[], 0)
}

/// Full [`compile`]: also takes the method's exception handlers. A throw inside a
/// handler range — whether an explicit `throw` or an out-of-band error from a
/// throwing call/dm op — dispatches through core `handle_err` (jumps to the catch
/// block, or propagates); catch-target blocks materialize the caught value on entry.
pub fn compile_full(
    ops: &[JitOp],
    switches: &[SwitchTable],
    exceptions: &[ExcRange],
    numeric_seed: &[bool],
    // Resolved `Gc<Multiname>` pointers, one per mn-bearing op in op order (see
    // `multiname_table`). `getproperty`/IC bake `mn_list[k]` as an immediate so the
    // helper needs no per-method `RUN_CTX.multinames` table (making it directable).
    // Empty on the switch/int-path convenience wrappers (which never emit `getproperty`).
    mn_list: &[*const ()],
    // Resolved `Class` pointers per `Coerce` op (see `coerce_class_table`), baked as an
    // immediate at each `Coerce` site so the `coerce` helper needs no per-method
    // `RUN_CTX.coerce_classes` table — making a `Coerce{class}` method directable.
    coerce_classes: &[*const ()],
    // Pre-coerced default `Value` bits per param + the first-optional index (see
    // `compile_method`); the direct-call prologue fills missing optional slots from these.
    param_defaults: &[u64],
    required_count: u32,
) -> Option<Vec<u8>> {
    if ops.is_empty() {
        return None;
    }
    let m = manifest(ops);
    let mut lay = layout_of(&m);
    // With handlers, throwable ops dispatch (in the compile loop) rather than emit
    // their own inline `perr` return — so suppress the call arms' inline bail.
    lay.inline_perr = exceptions.is_empty();
    let body = emit_body(
        ops,
        switches,
        exceptions,
        &lay,
        numeric_seed,
        mn_list,
        coerce_classes,
        param_defaults,
        required_count,
    )?;

    // Assemble module.
    let mut module = Module::new();
    module.section(&emit_types());
    module.section(&emit_imports(&m));

    let mut functions = FunctionSection::new();
    functions.function(0); // `run` has type 0
    module.section(&functions);
    let mut exports = ExportSection::new();
    // `run` follows all imported functions (arity-1 helpers + `gp`/`gs` + set).
    exports.export("run", ExportKind::Func, lay.run_index);
    module.section(&exports);
    let mut code = CodeSection::new();
    code.function(&body);
    module.section(&code);
    Some(module.finish())
}

/// One member method of a GENERATION module (see [`compile_generation`]) — the
/// same inputs its standalone [`compile_full`] compilation used.
pub struct GenMember<'a> {
    pub ops: &'a [JitOp],
    pub switches: &'a [SwitchTable],
    pub exceptions: &'a [ExcRange],
    /// AVM2 method name, emitted into the module's wasm name section for profilers.
    pub name: &'a str,
    /// Per-local "is numeric on entry" (param types) — seeds check-elision.
    pub numeric_seed: &'a [bool],
    /// This member's resolved multiname pointers (see [`compile_full`]'s `mn_list`).
    pub mn_list: &'a [*const ()],
    /// This member's resolved coerce-class pointers (see [`compile_full`]).
    pub coerce_classes: &'a [*const ()],
    /// This member's pre-coerced param defaults + first-optional index (see
    /// [`compile_method`]) — the direct-call prologue fills missing optional slots.
    pub param_defaults: &'a [u64],
    pub required_count: u32,
}

/// Compiles MANY methods into ONE module (an "amalgam" generation): every member
/// becomes a wasm function laid out against the members' UNION manifest, an
/// internal funcref table holds them all, and the exported `run` is a DISPATCHER
/// `(method_idx, state_ptr, dm_base, dm_len, regs_ptr, regs_len) -> Value` that
/// `call_indirect`s member `idx`. One module + one instance + one entry-table
/// slot then serve every member — instead of a page-granular executable-memory
/// allocation per method (the browser's "failed to allocate executable memory
/// for module" once thousands of tiny modules pile up), an instance-cache entry
/// per method, and a reserved-slot per method.
///
/// Returns the module bytes + the union manifest (which imports to bind).
/// Member `i` dispatches as `method_idx == i`. Every member must already have
/// compiled standalone, so a decline here means the inputs changed — `None`.
pub fn compile_generation(members: &[GenMember<'_>]) -> Option<(Vec<u8>, Manifest)> {
    if members.is_empty() {
        return None;
    }
    // Union manifest: field-wise max/OR over the members (exhaustive — see
    // `Manifest::union_with`), so `layout_of` gives every body the same import
    // indices.
    let mut u = Manifest::default();
    for mem in members {
        u.union_with(&manifest(mem.ops));
    }
    let lay = layout_of(&u);

    let mut code = CodeSection::new();
    for mem in members {
        let mut member_lay = lay.clone();
        // Per-member: with handlers, throwable ops dispatch instead of inline-bailing.
        member_lay.inline_perr = mem.exceptions.is_empty();
        code.function(&emit_body(
            mem.ops,
            mem.switches,
            mem.exceptions,
            &member_lay,
            mem.numeric_seed,
            mem.mn_list,
            mem.coerce_classes,
            mem.param_defaults,
            mem.required_count,
        )?);
    }
    // The dispatcher: push the six `run` args (incl. `ic_base`), then the method
    // index, and `call_indirect` through the internal member table (type 0 = the
    // method signature, so the type check always passes). The member table is table 0,
    // UNLESS a member makes a direct call — then the shared `__indirect_function_table`
    // is imported as table 0 and the member table shifts to table 1.
    let member_table = member_table_index(&u);
    let mut disp = Function::new([]);
    for arg in 1..=6u32 {
        disp.instruction(&Instruction::LocalGet(arg));
    }
    disp.instruction(&Instruction::LocalGet(0)); // method_idx selects the table entry
    disp.instruction(&Instruction::CallIndirect { type_index: 0, table_index: member_table });
    disp.instruction(&Instruction::End);
    code.function(&disp);

    let n = members.len() as u32;
    let mut module = Module::new();
    module.section(&emit_types());
    module.section(&emit_imports(&u));
    let mut functions = FunctionSection::new();
    for _ in 0..n {
        functions.function(0); // members have the method type
    }
    functions.function(6); // the dispatcher
    module.section(&functions);
    // The internal member table, filled by an active element segment below.
    let mut tables = TableSection::new();
    tables.table(TableType {
        element_type: RefType::FUNCREF,
        table64: false,
        minimum: n as u64,
        maximum: Some(n as u64),
        shared: false,
    });
    module.section(&tables);
    let mut exports = ExportSection::new();
    // Members occupy function indices `run_index..run_index + n` (after the
    // imports); the dispatcher follows them.
    exports.export("run", ExportKind::Func, lay.run_index + n);
    // Also export each member directly as `m{i}`, so `install_generation` can pull a
    // member's funcref into its OWN permanent shared-table slot — then BOTH a direct call
    // (`emit_call_direct`) AND the normal `run` dispatch reach the member's 6-param `run`
    // straight, with no dispatcher `call_indirect` + member-index branch. Each member `run`
    // is regs-prologue-shaped exactly like a standalone module's, so a direct call
    // (regs_len = 0, args pre-written to the frame) works identically. The runner slots
    // members budget-permitting (directable first); the rest fall back to the dispatcher.
    for i in 0..n {
        exports.export(&format!("m{i}"), ExportKind::Func, lay.run_index + i);
    }
    module.section(&exports);
    let mut elements = ElementSection::new();
    let member_funcs: Vec<u32> = (lay.run_index..lay.run_index + n).collect();
    elements.active(
        // The member table (table 0, or table 1 when the shared table is imported).
        Some(member_table),
        &ConstExpr::i32_const(0),
        Elements::Functions(member_funcs.into()),
    );
    module.section(&elements);
    module.section(&code);

    // Name section: label each member with its AVM2 method name (prefixed by its
    // dispatch index for uniqueness/correlation) plus the dispatcher, so profilers
    // show real names instead of `wasm-function[N]`. Imports occupy `0..run_index`;
    // members are `run_index..run_index + n`; the dispatcher is last.
    let mut func_names = NameMap::new();
    for (i, mem) in members.iter().enumerate() {
        let idx = lay.run_index + i as u32;
        let label = if mem.name.is_empty() {
            format!("jit[{i}]")
        } else {
            format!("jit[{i}]:{}", mem.name)
        };
        func_names.append(idx, &label);
    }
    func_names.append(lay.run_index + n, "jit_dispatch");
    let mut names = NameSection::new();
    names.functions(&func_names);
    module.section(&names);

    Some((module.finish(), u))
}

/// The shared type section: type 0 = method `run`, 1..5 = the helper-import
/// signatures, 6 = the generation dispatcher (leading method index).
fn emit_types() -> TypeSection {
    let mut types = TypeSection::new();
    types.ty().function(
        [ValType::I32, ValType::I32, ValType::I32, ValType::I32, ValType::I32, ValType::I32],
        [ValType::I64],
    ); // type 0: run(state_ptr, dm_base, dm_len, regs_ptr, regs_len, ic_base)->Value
    types.ty().function([ValType::I64], [ValType::I64]); // type 1: helper(Value)->Value
    types.ty().function([ValType::I64, ValType::I64], [ValType::I64]); // type 2: gp/gs(recv,k)->Value
    types
        .ty()
        .function([ValType::I64, ValType::I64, ValType::I64], [ValType::I64]); // type 3: set(recv,val,imm)->dummy / call_method(recv,id,argc)->result
    types.ty().function([ValType::I64], []); // type 4: push_call_arg(Value)->()
    types.ty().function([], [ValType::I32]); // type 5: pending_error()->i32
    types.ty().function(
        [
            ValType::I32,
            ValType::I32,
            ValType::I32,
            ValType::I32,
            ValType::I32,
            ValType::I32,
            ValType::I32,
        ],
        [ValType::I64],
    ); // type 6: dispatch(method_idx, state_ptr, dm_base, dm_len, regs_ptr, regs_len, ic_base)->Value
    types.ty().function(
        [ValType::I64, ValType::I64, ValType::I64, ValType::I64],
        [ValType::I64],
    ); // type 7: vc(a, imm, spill, kind)->result (the generic variadic helper)
    types
        .ty()
        .function([ValType::I64, ValType::I64, ValType::I32], [ValType::I64]); // type 8: gpi(receiver, k, cell_addr)->Value (property inline cache miss)
    types.ty().function(
        [ValType::I64, ValType::I64, ValType::I64, ValType::I32],
        [ValType::I64],
    ); // type 9: cmi(receiver, disp, argc, cell_addr)->Value (direct-call miss + refill)
    types.ty().function([ValType::I32], []); // type 10: scope_leave(old_depth)->()
    types
}
/// Type index of `scope_leave(i32)->()`. `scope_enter()->i32` reuses type 5 (`perr`).
const SCOPE_LEAVE_TYPE: u32 = 10;

/// The import section for manifest `m` — function imports in [`layout_of`]'s
/// order, then the frame memory (and Ruffle's own memory where required).
fn emit_imports(m: &Manifest) -> ImportSection {
    let mut imports = ImportSection::new();
    // Helper imports occupy function indices 0..num_helpers (before `run`).
    for i in 0..m.num_helpers {
        imports.import("env", &format!("h{i}"), EntityType::Function(1));
    }
    // The arity-2 helpers (if used) follow the arity-1 ones, `gp` then `gs`.
    if m.has_getprop {
        imports.import("env", "gp", EntityType::Function(2));
    }
    if m.has_getslot {
        imports.import("env", "gs", EntityType::Function(2));
    }
    // `gpi` (property inline-cache miss) is `(receiver, k, site) -> value` = type 8,
    // following the other property-read imports.
    if m.has_ic {
        imports.import("env", "gpi", EntityType::Function(8));
    }
    // `gpf` (get_property_fast) is arity-3 `(receiver, name, k) -> result` = type 3.
    if m.has_getprop_fast {
        imports.import("env", "gpf", EntityType::Function(3));
    }
    // `fp` (find_prop) is arity-2 `(k, strict) -> object` = type 2, after the reads.
    if m.has_findprop {
        imports.import("env", "fp", EntityType::Function(2));
    }
    // Then the arity-2 two-stack helpers `t0..t{N}` (compares).
    for i in 0..m.num_helpers2 {
        imports.import("env", &format!("t{i}"), EntityType::Function(2));
    }
    // Then the used ternary (arity-3) set helpers, in kind order, as `s{k}`.
    for k in 0..HELPER3_KINDS {
        if m.set3_mask & (1 << k) != 0 {
            imports.import("env", &format!("s{k}"), EntityType::Function(3));
        }
    }
    // Then (if the method calls) the call imports: `cm` (call_method, type 3),
    // `pca` (push_call_arg, type 4). `perr` (pending_error, type 5) follows and is
    // also needed by `has_dm` methods (dm ops throw #1506 out of band). `has_call_direct`
    // imports it too, for the SENTINEL short-circuit (see the `call_index` note).
    if m.has_call || m.has_call_direct {
        imports.import("env", "cm", EntityType::Function(3));
    }
    // `cmi` (direct-call miss + refill) is `(receiver, disp, argc, cell_addr) -> value`
    // = type 9, following `cm`.
    if m.has_call_direct {
        imports.import("env", "cmi", EntityType::Function(9));
        // `sen`/`sle`: direct-call scope enter/leave (`emit_call_direct` calls them around
        // a `call_indirect` to a SCOPE-using callee — the cell's `ic_base` high bit — to
        // retarget/restore the shared activation's `scope_depth`). `sen()->i32` = type 5
        // (like `perr`); `sle(i32)->()` = type 10.
        imports.import("env", "sen", EntityType::Function(5));
        imports.import("env", "sle", EntityType::Function(SCOPE_LEAVE_TYPE));
    }
    if m.has_callprop {
        imports.import("env", "cp", EntityType::Function(3));
    }
    // `csup` (construct_super) is arity-2 `(receiver, argc) -> dummy` = type 2.
    if m.has_construct_super {
        imports.import("env", "csup", EntityType::Function(2));
    }
    // `callv` (call_value) is arity-3 `(function, receiver, argc) -> result` = type 3.
    if m.has_call_value {
        imports.import("env", "callv", EntityType::Function(3));
    }
    let any_call =
        m.has_call || m.has_call_direct || m.has_callprop || m.has_construct_super || m.has_call_value || m.has_vcall;
    if any_call {
        imports.import("env", "pca", EntityType::Function(4));
    }
    if m.needs_perr() {
        imports.import("env", "perr", EntityType::Function(5));
    }
    // `coerce` (arity-2 `(value, class_idx) -> result`) follows `perr`.
    if m.has_coerce {
        imports.import("env", "coerce", EntityType::Function(2));
    }
    // `vc` (arity-4 `(a, imm, spill, kind) -> result`, type 7) follows `coerce`.
    if m.has_vcall {
        imports.import("env", "vc", EntityType::Function(7));
    }
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
    // Memory 1 = `dm` (Ruffle's own linear memory) for inline `li*`/`si*`. On web
    // it's Ruffle's *shared* (SharedArrayBuffer) memory, so `shared: true` there to
    // match; native (wasmi mock) is non-shared. A shared memory needs a `maximum`.
    // On web it is imported ALWAYS — the prologue reads the register snapshot from
    // it (see the `memory.copy` at the body's start); native imports it only for
    // dm ops (wasmi writes the frame directly, no prologue).
    // The inline `getslot` fast path chases object pointers on memory 1 too —
    // relevant natively only under the test-forced layout (web always imports it).
    if m.has_dm
        || cfg!(target_arch = "wasm32")
        || (m.has_getslot && slot_layout().is_some())
    {
        let shared = cfg!(target_arch = "wasm32");
        imports.import(
            "env",
            "dm",
            EntityType::Memory(MemoryType {
                minimum: 1,
                maximum: shared.then_some(65536),
                memory64: false,
                shared,
                page_size_log2: None,
            }),
        );
    }
    // Shared `__indirect_function_table` — the JIT→JIT direct call `call_indirect`s a
    // callee's `run` (or its amalgam dispatcher) slot in it (see `emit_call_direct`).
    // Imported ⇒ table 0; any LOCAL table (a generation's member table) then shifts to
    // table 1. `minimum: 0` — the host table (`wasm_bindgen::function_table`, holding
    // the reserved run slots) is always larger.
    if m.has_call_direct {
        imports.import(
            "env",
            "__indirect_function_table",
            EntityType::Table(TableType {
                element_type: RefType::FUNCREF,
                table64: false,
                minimum: 0,
                maximum: None,
                shared: false,
            }),
        );
        // `dp` (global 0): the memory-1 byte address of the runner's `DEPTH` frame
        // counter (`&raw mut runner::web::DEPTH`). A direct call to a NON-LEAF callee
        // must bump `DEPTH` around the `call_indirect` so the callee's own re-entrant
        // helper calls (`cm` → `run`) place their frames ABOVE the callee's, not
        // aliasing it (see `emit_call_direct`). `emit_call_direct` reads/writes `DEPTH`
        // inline via memory 1 at this address — no Rust round-trip on the hot path.
        // Imported iff a direct call is emitted (same gate as the table above); the web
        // runner binds it in `instantiate`, native prod never sets `has_call_direct`.
        imports.import(
            "env",
            "dp",
            EntityType::Global(GlobalType {
                val_type: ValType::I32,
                mutable: false,
                shared: false,
            }),
        );
    }
    imports
}

/// The imported-global index of `dp` (the `DEPTH` cell address). Only global in the
/// module, so index 0 (globals are a separate index space from functions/memories, so
/// adding it perturbs no function/`run` index). Valid only when `has_call_direct`.
const DEPTH_PTR_GLOBAL: u32 = 0;

/// The table index of a generation module's LOCAL member table: `1` when the shared
/// `__indirect_function_table` is imported (some member makes a direct call, so it is
/// table 0), else `0`.
fn member_table_index(union: &Manifest) -> u32 {
    union.has_call_direct as u32
}



/// Emits one method's WASM function body against `lay` (whose import indices the
/// emitted `call`s reference). Shared by [`compile_full`] (a single-method module
/// laid out from its own manifest) and [`compile_generation`] (many methods laid
/// out against the generation's UNION manifest). `None` = the lowering declines.
fn emit_body(
    ops: &[JitOp],
    switches: &[SwitchTable],
    exceptions: &[ExcRange],
    lay: &Layout,
    // Per-local "is numeric on entry" (from the method's param types). Seeds the
    // repr analysis so a `:Number`/`:int`/`:uint` parameter is `Numeric` at block
    // 0. Empty → all `Boxed` (numeric-ness then only flows from in-method producers).
    numeric_seed: &[bool],
    // Resolved multiname pointers baked at `getproperty`/IC sites (see `compile_full`).
    mn_list: &[*const ()],
    // Resolved coerce-class pointers baked at `Coerce` sites (see `compile_full`).
    coerce_classes: &[*const ()],
    // Pre-coerced default `Value` bits per param (see `compile_method`); the direct-call
    // prologue fills the missing optional slots `[required_count, param_count)` from these.
    param_defaults: &[u64],
    // Index of the first optional (defaulted) param — the fill boundary.
    required_count: u32,
) -> Option<Function> {
    let has_handlers = !exceptions.is_empty();
    // Shared cache buffer layout: `[property-IC cells | call-cache cells]`. The call
    // region starts after the IC cells, so its byte offset is `num_ic_sites *
    // IC_CELL_SIZE` (sites are dense post-`renumber_ic_sites`).
    let call_region_offset = ops
        .iter()
        .filter(|o| matches!(o, JitOp::GetPropertyIc(..)))
        .count() as u32
        * IC_CELL_SIZE;
    // `ReturnValueCoerced` inside a handler range isn't wired for exception
    // dispatch: its lowering emits a bare `Return` right after `coerce_return`, so
    // a failed return-type coercion (`#1034`, stashed in `PENDING_ERROR`) would
    // propagate OUT of the method instead of being caught by the method's OWN
    // try/catch — the interpreter catches it. Decline such methods (the throwing
    // `returnvalue` sits literally inside a `try` of a typed-return function; rare).
    // Every other throwing op in a range dispatches via the compile loop (see
    // `is_throwing_call_or_dm`); `ReturnValueCoerced` can't, as it's a terminator.
    if has_handlers
        && ops.iter().enumerate().any(|(i, op)| {
            matches!(op, JitOp::ReturnValueCoerced)
                && exceptions.iter().any(|e| i >= e.from && i < e.to)
        })
    {
        return None;
    }
    // Distinct catch-target op indices — each is a block leader, and its block
    // materializes the caught exception on entry.
    let catch_targets: std::collections::BTreeSet<usize> =
        exceptions.iter().map(|e| e.target).collect();

    let mut leaders = basic_block_leaders_sw(ops, switches)?;
    // Catch targets are jump destinations too.
    for &t in &catch_targets {
        if t >= ops.len() {
            return None;
        }
        if let Err(pos) = leaders.binary_search(&t) {
            leaders.insert(pos, t);
        }
    }
    let leaders = leaders;
    // Map an op index (that is a leader) to its basic-block index.
    let bb_of = |op_idx: usize| -> Option<usize> { leaders.iter().position(|&l| l == op_idx) };
    let num_bbs = leaders.len();
    // Whether the `perr` import is present (so a post-op bail can be emitted).
    let perr_present = manifest(ops).needs_perr();
    // Catch-target op indices → their block indices, for the dispatch if-chain.
    let mut catch_bbs: Vec<(i64, i32)> = Vec::with_capacity(catch_targets.len());
    for &t in &catch_targets {
        catch_bbs.push((t as i64, bb_of(t)? as i32));
    }

    // Locals: SCRATCH, SCRATCH2, BLOCK (i32); SCRATCH64, SCRATCH64_2 (i64);
    // SCRATCH_F64 (f64); then SPILL_POOL i64 operand-stack spill slots.
    let mut body = Function::new([
        (3, ValType::I32),
        (2, ValType::I64),
        (1, ValType::F64),
        (SPILL_POOL, ValType::I64),
    ]);

    // Web prologue: copy the register snapshot from Ruffle's own memory
    // (memory 1, at `regs_ptr`, `regs_len` bytes) into this frame's slice of the
    // frame memory (memory 0, at `state_ptr`) — wasm→wasm, replacing the runner's
    // per-call JS `Uint8Array.set` copy. Native (wasmi) writes the frame memory
    // directly, so it emits no prologue and imports memory 1 only for dm ops.
    if cfg!(target_arch = "wasm32") {
        // `regs_ptr != 0` — NORMAL entry: copy the whole register snapshot (`num_locals`
        // slots) from Ruffle's memory (memory 1) into this frame (memory 0).
        // `regs_ptr == 0` — a JIT→JIT DIRECT call: the caller pre-wrote `regs_len / 8`
        // slots (this + args) straight into the frame and the rest of the locals
        // (`regs_len/8 .. num_locals`) are stale scratch, so set them to `undefined` —
        // exactly what `init_from_method` does for locals beyond the arguments. This is
        // what lets a callee with extra locals be a direct-call target. `regs_ptr` is a
        // real heap address on the normal path, so it's never 0 there.
        let num_locals_bytes = (numeric_seed.len() as u32) * 8;
        body.instruction(&Instruction::LocalGet(REGS_PTR));
        body.instruction(&Instruction::If(BlockType::Empty));
        body.instruction(&Instruction::LocalGet(STATE_PTR)); // dst (memory 0)
        body.instruction(&Instruction::LocalGet(REGS_PTR)); // src (memory 1)
        body.instruction(&Instruction::LocalGet(REGS_LEN)); // bytes
        body.instruction(&Instruction::MemoryCopy { src_mem: 1, dst_mem: 0 });
        body.instruction(&Instruction::Else);
        // ptr = state_ptr + regs_len (first stale local); end = state_ptr + num_locals*8.
        body.instruction(&Instruction::LocalGet(STATE_PTR));
        body.instruction(&Instruction::LocalGet(REGS_LEN));
        body.instruction(&Instruction::I32Add);
        body.instruction(&Instruction::LocalSet(SCRATCH));
        body.instruction(&Instruction::Block(BlockType::Empty));
        body.instruction(&Instruction::Loop(BlockType::Empty));
        body.instruction(&Instruction::LocalGet(SCRATCH));
        body.instruction(&Instruction::LocalGet(STATE_PTR));
        body.instruction(&Instruction::I32Const(num_locals_bytes as i32));
        body.instruction(&Instruction::I32Add);
        body.instruction(&Instruction::I32GeU);
        body.instruction(&Instruction::BrIf(1)); // ptr >= end → break the block
        body.instruction(&Instruction::LocalGet(SCRATCH));
        body.instruction(&Instruction::I64Const(UNDEFINED_BITS as i64));
        body.instruction(&Instruction::I64Store(MemArg { offset: 0, align: 3, memory_index: 0 }));
        body.instruction(&Instruction::LocalGet(SCRATCH));
        body.instruction(&Instruction::I32Const(8));
        body.instruction(&Instruction::I32Add);
        body.instruction(&Instruction::LocalSet(SCRATCH));
        body.instruction(&Instruction::Br(0)); // continue loop
        body.instruction(&Instruction::End); // loop
        body.instruction(&Instruction::End); // block
        body.instruction(&Instruction::End); // if
    }

    // Optional-param default fill (direct-call frames). On a JIT→JIT direct call the caller
    // writes only `{this, argc args}`; params in `[argc, param_count)` went unwritten, so the
    // zero-loop above left them `undefined`. Overwrite each optional param slot the caller did
    // NOT provide (`slot_byte >= regs_len`) with its baked, pre-coerced default (bit-identical
    // to `resolve_parameters`). On a NORMAL entry `regs_len` spans the full frame, so every
    // param slot is `< regs_len` and NOTHING is written (the snapshot/runner already holds the
    // resolved defaults) — hence it's sound to run unconditionally on BOTH the wasm
    // copy-prologue path and the native (no-prologue) path. Bounded to `MAX_DIRECT_ARGC`: a
    // method with more params can't be a direct target (`all_params_directable`), so higher
    // slots are never provided-short — the extra fills would be dead code.
    let fill_end = param_defaults.len().min(MAX_DIRECT_ARGC as usize);
    for i in (required_count as usize)..fill_end {
        let slot_byte = ((i + 1) * 8) as u64;
        // slot NOT provided by the caller ⟺ (i+1)*8 >= regs_len.
        body.instruction(&Instruction::I32Const(slot_byte as i32));
        body.instruction(&Instruction::LocalGet(REGS_LEN));
        body.instruction(&Instruction::I32GeU);
        body.instruction(&Instruction::If(BlockType::Empty));
        body.instruction(&Instruction::LocalGet(STATE_PTR));
        body.instruction(&Instruction::I64Const(param_defaults[i] as i64));
        body.instruction(&Instruction::I64Store(MemArg { offset: slot_byte, align: 3, memory_index: 0 }));
        body.instruction(&Instruction::End);
    }

    // Whether the boxed path is in use (its operand stack is i64, so live values can
    // be spilled to the i64 pool across a branch). The int path's stack is i32 and
    // the double path is branch-free, so neither spills — a non-empty stack across a
    // branch there still declines.
    let spill_ok = !uses_int_model(ops);
    // Operand-stack height on entry to each basic block (`-1` = not yet reached).
    // Propagated forward as branches/fall-throughs are emitted; verified bytecode
    // makes every predecessor agree, so an inconsistency is a bail.
    let mut entry_depth = vec![-1i32; num_bbs];
    entry_depth[0] = 0;

    // Type-specialization value-representation inference (see `crate::typed`).
    // Per-block-entry local reprs, seeded all-`Boxed` (numeric-ness flows from
    // in-method producers — `pushdouble`, `pushint`, numeric arith, and locals
    // assigned those). Only computed when the block partition matches `typed`'s
    // (no switches, no catch targets — else the `bb` indices wouldn't align), and
    // under debug for the arity safety net regardless. `None` → no elision.
    let reprs_ok = catch_targets.is_empty() && switches.is_empty();
    let ce_entry: Option<Vec<Vec<crate::typed::Repr>>> =
        if (JIT_CHECK_ELISION && reprs_ok) || (cfg!(debug_assertions) && reprs_ok) {
            // Entry local reprs: a numeric-typed parameter is `Numeric`, all else
            // `Boxed`. Sized to the frame so any local index resolves.
            let entry_locals: Vec<crate::typed::Repr> = (0..crate::MAX_FRAME_SLOTS as usize)
                .map(|i| {
                    if numeric_seed.get(i).copied().unwrap_or(false) {
                        crate::typed::Repr::Numeric
                    } else {
                        crate::typed::Repr::Boxed
                    }
                })
                .collect();
            crate::typed::infer_local_reprs(ops, &entry_locals)
        } else {
            None
        };

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
        // Reload any operands live on entry (spilled by predecessors). `-1` = an
        // unreachable block (dead code after a return); treat as empty.
        let mut depth: i32 = entry_depth[bb].max(0);
        if depth as u32 > SPILL_POOL {
            return None;
        }
        emit_reload(&mut body, depth);

        // A catch-target block is entered only via exception dispatch, which reset
        // the operand stack; materialize the caught exception value here (the
        // handler code — `newcatch`, stores, …—consumes it). Entry depth is 0.
        if catch_targets.contains(&start) {
            body.instruction(&Instruction::I64Const(0));
            body.instruction(&Instruction::Call(POP_CAUGHT));
            depth += 1;
        }

        // Physical repr of each live operand-stack slot for phase-2 unboxing:
        // `true` = the slot physically holds a raw `f64` (not a boxed `Value`).
        // Entry slots are reloaded/materialized as boxed `i64`, so all `false`.
        // Inert unless `JIT_UNBOX` (nothing is ever marked `f64`).
        let mut phys: Vec<bool> = vec![false; depth.max(0) as usize];

        // Safety net for the type-specialization stack model: when this block is
        // entered with an empty operand stack (the common case), cross-check that
        // `typed::stack_arity`'s predicted depth matches `emit_body`'s authoritative
        // `depth` at every op. An arity mistake trips the JIT test suite in debug
        // builds — it can never reach the codegen that would rely on it. No effect
        // on release builds or lowering.
        // The per-op operand-stack reprs for this block (empty-entry blocks only,
        // so `simulate_block`'s empty-start stack aligns). Drives both the arity
        // safety net (debug) and check-elision (release).
        let block_pre: Option<Vec<Vec<crate::typed::Repr>>> = if depth == 0 {
            ce_entry.as_ref().and_then(|e| {
                crate::typed::simulate_block(&ops[start..end], e[bb].clone()).map(|(_, pre)| pre)
            })
        } else {
            None
        };

        for (pos, &op) in ops[start..end].iter().enumerate() {
            let op_idx = start + pos;
            #[cfg(debug_assertions)]
            if let Some(pre) = &block_pre {
                if depth as usize != pre[pos].len() {
                    panic!(
                        "stack_arity depth mismatch at op {pos} ({op:?}) in block {bb}: \
                         real={} predicted={}\n  block ops: {:?}",
                        depth,
                        pre[pos].len(),
                        &ops[start..end],
                    );
                }
            }

            // Phase-2 physical unboxing. `is_f64_arith`: this op is a proven-both-
            // `Number` arith that will CONSUME f64-aware operands and LEAVE its
            // result unboxed. Every other op expects boxed operands, so box any
            // live `f64` it would consume first — all of them before a terminator/
            // branch (which spills the operand stack as `i64`), else just its pop.
            let is_f64_arith = JIT_UNBOX
                && matches!(op, JitOp::ArithInt(_))
                && block_pre.as_ref().is_some_and(|p| {
                    let s = &p[pos];
                    s.len() >= 2
                        && s[s.len() - 1].is_f64_unboxable()
                        && s[s.len() - 2].is_f64_unboxable()
                });
            if JIT_UNBOX && !is_f64_arith {
                let k = if op.is_terminator() {
                    depth
                } else {
                    crate::typed::stack_arity(&op).0 as i32
                };
                box_top(&mut body, &mut phys, depth, k as u32)?;
            }

            match op {
                JitOp::ReturnValue => {
                    if depth < 1 {
                        return None;
                    }
                    emit_box_int(&mut body);
                    body.instruction(&Instruction::Return);
                    depth = 0;
                }
                JitOp::ReturnValueBoxed => {
                    if depth < 1 {
                        return None;
                    }
                    // Top of stack is already a raw `Value` (i64) — return as-is.
                    body.instruction(&Instruction::Return);
                    depth = 0;
                }
                JitOp::ReturnValueCoerced => {
                    if depth < 1 {
                        return None;
                    }
                    // Coerce the top `Value` to the method's declared return type; on
                    // a `#1034` failure `coerce_return` sets `PENDING_ERROR` + returns
                    // `undefined`, and `try_run` propagates the error after the run.
                    body.instruction(&Instruction::Call(COERCE_RETURN));
                    body.instruction(&Instruction::Return);
                    depth = 0;
                }
                JitOp::ReturnValueCoerceBaked(rtype_ptr) => {
                    if depth < 1 {
                        return None;
                    }
                    // Coerce the top `Value` to the baked return-type `Class` via the
                    // `coerce` helper (`(value, class_ptr)`), then return — no per-method
                    // `RUN_CTX.return_type`, so directable. Same `#1034`/`PENDING_ERROR`
                    // behaviour as `coerce_return` (the caller's `perr` check bails).
                    body.instruction(&Instruction::I64Const(rtype_ptr as i64));
                    body.instruction(&Instruction::Call(lay.coerce_index));
                    body.instruction(&Instruction::Return);
                    depth = 0;
                }
                JitOp::ReturnVoidBoxed(bits) => {
                    // `returnvoid` ignores the operand stack and returns the value its
                    // declared `return_type` coerces to (undefined/0/false/null).
                    // `Return` is stack-polymorphic, so any leftover values are fine.
                    body.instruction(&Instruction::I64Const(bits as i64));
                    body.instruction(&Instruction::Return);
                    depth = 0;
                }
                JitOp::ReturnDouble => {
                    if depth < 1 {
                        return None;
                    }
                    emit_box_double(&mut body); // f64 -> Number `Value` bits
                    body.instruction(&Instruction::Return);
                    depth = 0;
                }
                JitOp::Jump(t) => {
                    // Carry any live operands across the branch: spill them (boxed
                    // path only), the target block reloads them on entry.
                    if depth > 0 && (!spill_ok || depth as u32 > SPILL_POOL) {
                        return None;
                    }
                    emit_spill(&mut body, depth);
                    let target = bb_of(t)?;
                    record_entry(&mut entry_depth, target, depth)?;
                    body.instruction(&Instruction::I32Const(target as i32));
                    body.instruction(&Instruction::LocalSet(BLOCK));
                    body.instruction(&Instruction::Br(loop_depth));
                    depth = 0;
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
                JitOp::IfTrueBoxed(t) | JitOp::IfFalseBoxed(t) => {
                    if depth < 1 {
                        return None;
                    }
                    // Operands below the condition are carried across the branch.
                    let carried = depth - 1;
                    if carried > 0 && (!spill_ok || carried as u32 > SPILL_POOL) {
                        return None;
                    }
                    // Coerce the top `Value` (i64) to a 0/1 condition — inline for
                    // Boolean/int boxes, `to_boolean` helper otherwise.
                    // `IfFalseBoxed` negates it. The carried operands sit below
                    // and are untouched.
                    emit_to_boolean_i32(&mut body);
                    if matches!(op, JitOp::IfFalseBoxed(_)) {
                        body.instruction(&Instruction::I32Eqz);
                    }
                    let taken = bb_of(t)?;
                    let fallthrough = bb + 1;
                    body.instruction(&Instruction::If(BlockType::Empty));
                    body.instruction(&Instruction::I32Const(taken as i32));
                    body.instruction(&Instruction::LocalSet(BLOCK));
                    body.instruction(&Instruction::Else);
                    body.instruction(&Instruction::I32Const(fallthrough as i32));
                    body.instruction(&Instruction::LocalSet(BLOCK));
                    body.instruction(&Instruction::End);
                    // Condition consumed; spill the carried operands (both successors
                    // reload the same `carried` slots on entry).
                    emit_spill(&mut body, carried);
                    record_entry(&mut entry_depth, taken, carried)?;
                    record_entry(&mut entry_depth, fallthrough, carried)?;
                    body.instruction(&Instruction::Br(loop_depth));
                    depth = 0;
                }
                JitOp::Throw => {
                    if depth < 1 {
                        return None;
                    }
                    // `Throw` is a terminator, so it's the last op of the block.
                    let op_idx = end - 1;
                    if has_handlers {
                        // Stash the thrown value as the pending error, discard the
                        // operand stack, then dispatch: on a catch, jump to the handler
                        // block (which pops the caught value); else return (propagate).
                        body.instruction(&Instruction::Call(THROW));
                        for _ in 0..depth {
                            body.instruction(&Instruction::Drop);
                        }
                        // `br_to_loop = loop_depth + 1`: the core's own caught `If`.
                        emit_dispatch_core(&mut body, op_idx, &catch_bbs, loop_depth + 1);
                    } else {
                        // No handler: stash the error and return; `try_run` propagates.
                        body.instruction(&Instruction::Call(THROW));
                        body.instruction(&Instruction::Return);
                    }
                    depth = 0;
                }
                JitOp::LookupSwitch(idx) => {
                    if depth != 1 {
                        return None;
                    }
                    let sw = switches.get(idx as usize)?;
                    let n = sw.cases.len();
                    // Selector `Value` (i64) → ToInt32 (`coerce_i`) → raw i32 (the
                    // low 32 bits of the int box) into SCRATCH. Matches the interp's
                    // `pop_stack().as_i32()`; a `br_table` clamps an out-of-range
                    // index to the default, exactly like `as_i32 as usize` there.
                    body.instruction(&Instruction::Call(COERCE_I));
                    body.instruction(&Instruction::I32WrapI64);
                    body.instruction(&Instruction::LocalSet(SCRATCH));
                    // A nested-block `br_table` that maps the selector to a target
                    // BB: default wrapper outermost, case 0 innermost. Each landing
                    // sets BLOCK and branches to the dispatch loop.
                    for _ in 0..=n {
                        body.instruction(&Instruction::Block(BlockType::Empty));
                    }
                    body.instruction(&Instruction::LocalGet(SCRATCH));
                    let case_depths: Vec<u32> = (0..n as u32).collect();
                    body.instruction(&Instruction::BrTable(case_depths.into(), n as u32));
                    // Close case block k → its landing: set BLOCK = bb(case k), br loop.
                    // The switch consumed the whole stack (selector), so every target
                    // has entry depth 0.
                    for k in 0..n {
                        body.instruction(&Instruction::End);
                        let cbb = bb_of(sw.cases[k])?;
                        record_entry(&mut entry_depth, cbb, 0)?;
                        body.instruction(&Instruction::I32Const(cbb as i32));
                        body.instruction(&Instruction::LocalSet(BLOCK));
                        // (n + 1) - (k + 1) new blocks still open above the loop.
                        body.instruction(&Instruction::Br(loop_depth + (n - k) as u32));
                    }
                    // Close the default wrapper → default landing.
                    body.instruction(&Instruction::End);
                    let dbb = bb_of(sw.default)?;
                    record_entry(&mut entry_depth, dbb, 0)?;
                    body.instruction(&Instruction::I32Const(dbb as i32));
                    body.instruction(&Instruction::LocalSet(BLOCK));
                    body.instruction(&Instruction::Br(loop_depth));
                    depth = 0;
                }
                // Check-elision: a `multiply`/`subtract`/`add` whose BOTH operands
                // the repr analysis proved numeric — drop the `is_numeric` guard +
                // helper fallback. Falls through to the general `emit_linear`
                // lowering when unproven.
                JitOp::ArithInt(index)
                    if (JIT_CHECK_ELISION || JIT_UNBOX)
                        && block_pre.as_ref().is_some_and(|p| {
                            let s = &p[pos];
                            s.len() >= 2
                                && s[s.len() - 1].is_numeric()
                                && s[s.len() - 2].is_numeric()
                        }) =>
                {
                    if depth < 2 {
                        return None;
                    }
                    let s = &block_pre.as_ref().expect("guard checked Some")[pos];
                    let a_repr = s[s.len() - 2];
                    let b_repr = s[s.len() - 1];
                    let both_number =
                        a_repr == crate::typed::Repr::Number && b_repr == crate::typed::Repr::Number;
                    // A possibly-heap-boxed operand needs the box-safe unbox. (A
                    // physically-`f64` operand — always a canonical arith result,
                    // repr `Number` — never sets this, so `phys` and `boxsafe` can't
                    // both be true.)
                    let a_boxsafe = a_repr.needs_boxsafe_unbox();
                    let b_boxsafe = b_repr.needs_boxsafe_unbox();
                    if is_f64_arith {
                        // Phase 2: consume operands per their PHYSICAL repr, leave
                        // the result unboxed as f64 (marked below).
                        let d = depth as usize;
                        emit_arith_number_phys(
                            &mut body,
                            index,
                            phys[d - 2],
                            phys[d - 1],
                            a_boxsafe,
                            b_boxsafe,
                        );
                    } else if both_number {
                        // Both `Number` → pure f64 path (skip `is_int` too), boxed.
                        emit_arith_number(&mut body, index);
                    } else {
                        // Numeric (maybe int / heap-boxed) → drop `is_numeric` guard
                        // + helper; box-safe unbox any `NumberBoxed` operand.
                        emit_arith_int_elided(&mut body, index, a_boxsafe, b_boxsafe);
                    }
                    depth -= 1;
                }
                other => emit_linear(&mut body, other, &mut depth, &lay, call_region_offset, mn_list, coerce_classes)?,
            }

            // Phase-2: keep `phys` aligned with the authoritative `depth` after the
            // op. Popped slots drop off the top; new pushes are boxed (`false`); an
            // `f64`-arith's result is the exception — mark its top slot `f64`.
            if JIT_UNBOX {
                phys.resize(depth.max(0) as usize, false);
                if is_f64_arith {
                    if let Some(t) = phys.last_mut() {
                        *t = true;
                    }
                }
            }
            // A call/dm op can throw out of band (`PENDING_ERROR`). Handle it right
            // after the op so a thrown error doesn't run on with a swallowed result.
            if perr_present && is_throwing_call_or_dm(op) {
                if has_handlers {
                    let in_handler = exceptions.iter().any(|e| op_idx >= e.from && op_idx < e.to);
                    if in_handler {
                        // Spill live operands (discarded on a throw, reloaded on no
                        // throw), check for a pending error, and dispatch to the catch
                        // block. `br_to_loop = loop_depth + 2`: the pending `If` plus
                        // the dispatch core's own caught `If`.
                        if depth as u32 > SPILL_POOL {
                            return None;
                        }
                        emit_spill(&mut body, depth);
                        body.instruction(&Instruction::Call(lay.perr_index));
                        body.instruction(&Instruction::If(BlockType::Empty));
                        emit_dispatch_core(&mut body, op_idx, &catch_bbs, loop_depth + 2);
                        body.instruction(&Instruction::End);
                        emit_reload(&mut body, depth);
                    } else {
                        // Outside every handler range: a thrown error propagates out.
                        emit_perr_bail(&mut body, lay.perr_index);
                    }
                } else if !is_self_bailing_call(op) {
                    // No-handler method: calls already bailed inline (in `emit_linear`);
                    // every other throwing op (dm, `coerce`/`coerces`, `astypelate`/
                    // `istypelate`, property/slot access) emits its bail here so a
                    // thrown error stops the method promptly rather than running on
                    // with a swallowed result.
                    emit_perr_bail(&mut body, lay.perr_index);
                }
            }
        }

        // Fell off the block end without a terminator: continue to the next BB,
        // carrying any live operands (spilled here, reloaded on that block's entry).
        if end == ops.len() || !ops[end - 1].is_terminator() {
            if depth > 0 && (!spill_ok || depth as u32 > SPILL_POOL) {
                return None;
            }
            // Box any live `f64` before the fall-through spill (which stores i64).
            if JIT_UNBOX {
                box_top(&mut body, &mut phys, depth, depth as u32)?;
            }
            if bb + 1 < num_bbs {
                emit_spill(&mut body, depth);
                record_entry(&mut entry_depth, bb + 1, depth)?;
                body.instruction(&Instruction::I32Const((bb + 1) as i32));
                body.instruction(&Instruction::LocalSet(BLOCK));
                body.instruction(&Instruction::Br(loop_depth));
            } else if depth != 0 {
                // Falling off the last block with a live stack is malformed.
                return None;
            }
        }
    }

    body.instruction(&Instruction::End); // loop
    body.instruction(&Instruction::Unreachable);
    body.instruction(&Instruction::End); // function
    Some(body)
}


// Native-only: these execute the emitted module through wasmi.
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use wasmi::{Engine, Func, Instance, Memory, MemoryType as WMemoryType, Module, Store};

    // A GENERATION module containing the newer op lowerings (`hasnext2`,
    // `inclocal`/`declocal`, the `newfunction` vcall kind) must VALIDATE — a
    // member emitted against the union layout with a helper/import mismatch
    // produces an invalid module, which on web skips the whole batch install
    // ("generation module INVALID — install skipped").
    #[test]
    fn generation_with_new_ops_validates() {
        let m1 = [JitOp::HasNext2(1, 2), JitOp::ReturnValueBoxed];
        let m2 = [
            JitOp::IncDecLocalNum(1, false),
            JitOp::IncDecLocalNum(2, true),
            JitOp::ReturnVoidBoxed(UNDEFINED_BITS),
        ];
        let m3 = [JitOp::VCall(vc::NEW_FUNCTION, 0, 0, true), JitOp::ReturnValueBoxed];
        let m4 = [JitOp::GetLocalValue(1), JitOp::CoerceInt(false), JitOp::ReturnValueBoxed];
        let members = [
            GenMember { ops: &m1, switches: &[], exceptions: &[], name: "", numeric_seed: &[], mn_list: &[], coerce_classes: &[], param_defaults: &[], required_count: 0 },
            GenMember { ops: &m2, switches: &[], exceptions: &[], name: "", numeric_seed: &[], mn_list: &[], coerce_classes: &[], param_defaults: &[], required_count: 0 },
            GenMember { ops: &m3, switches: &[], exceptions: &[], name: "", numeric_seed: &[], mn_list: &[], coerce_classes: &[], param_defaults: &[], required_count: 0 },
            GenMember { ops: &m4, switches: &[], exceptions: &[], name: "", numeric_seed: &[], mn_list: &[], coerce_classes: &[], param_defaults: &[], required_count: 0 },
        ];
        let (bytes, _union) = compile_generation(&members).expect("generation compiles");
        let engine = Engine::default();
        Module::new(&engine, &bytes).expect("generation module validates");
    }

    #[test]
    fn helper_dominated_declines_only_crossing_heavy() {
        // Mostly getproperty crossings, no inline compute → dominated.
        let heavy = [
            JitOp::GetLocalValue(0),
            JitOp::GetProperty(0),
            JitOp::GetProperty(1),
            JitOp::GetSlot(2, false),
            JitOp::GetProperty(3),
            JitOp::ReturnValueBoxed,
        ];
        assert!(helper_dominated(&heavy)); // 4 crossings, 0 wins

        // The same 4 crossings but with enough inline compute (2 wins) to justify the
        // JIT → NOT dominated (`4 > 2*2` is false).
        let mixed = [
            JitOp::GetLocalValue(0),
            JitOp::GetProperty(0),
            JitOp::GetProperty(1),
            JitOp::GetSlot(2, false),
            JitOp::GetProperty(3),
            JitOp::CmpNum(1),
            JitOp::ArithInt(11),
            JitOp::ReturnValueBoxed,
        ];
        assert!(!helper_dominated(&mixed));

        // Few crossings (< 4) never decline, even with zero inline compute — a small
        // method like `getlocal; getslot; returnvalue` still compiles.
        let small = [JitOp::GetLocalValue(0), JitOp::GetSlot(0, false), JitOp::ReturnValueBoxed];
        assert!(!helper_dominated(&small));

        // A FlasCC-shaped domainMemory hot path (dm ops are inline wins) compiles even
        // with a couple of property crossings.
        let flascc = [
            JitOp::GetLocalValue(0),
            JitOp::DmLoad(4),
            JitOp::DmStore(4),
            JitOp::DmLoad(1),
            JitOp::GetSlot(0, false),
            JitOp::AddIBoxed,
            JitOp::ReturnValueBoxed,
        ];
        assert!(!helper_dominated(&flascc)); // 1 crossing, 4 wins
    }

    #[test]
    fn int_mark_matches_core() {
        let bits: u64 = unsafe { std::mem::transmute(ruffle_core::avm2::Value::from(0i32)) };
        assert_eq!(bits, VALUE_INT_MARK, "int box mark drifted from core");
    }

    #[test]
    fn undefined_bits_matches_core() {
        let bits: u64 = unsafe { std::mem::transmute(ruffle_core::avm2::Value::Undefined) };
        assert_eq!(bits, UNDEFINED_BITS, "undefined bits drifted from core");
    }

    // Boxed `returnvoid` returns `undefined`; boxed `dup` duplicates a `Value`.
    // `return dup(local0)` ⇒ two copies pushed, one returned = local0's bits.
    #[test]
    fn lowers_return_void_and_dup_value() {
        let ret_void = compile(&[JitOp::ReturnVoidBoxed(UNDEFINED_BITS)]).expect("compiles");
        assert_eq!(run(&ret_void, &[]), UNDEFINED_BITS);

        let dup = [JitOp::GetLocalValue(0), JitOp::DupValue, JitOp::ReturnValueBoxed];
        let bytes = compile(&dup).expect("compiles");
        assert_eq!(run(&bytes, &[0xDEAD_BEEF_u64]), 0xDEAD_BEEF);
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
            .get_typed_func::<(i32, i32, i32, i32, i32, i32), i64>(&store, "run")
            .expect("run export");
        run.call(&mut store, (0, 0, 0, 0, 0, 0)).expect("runs") as u64
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

    // Proves the helper-call ABI end-to-end on wasmi: the emitted module imports
    // a host function, calls it with a raw `Value`, the host reaches a
    // thread-local context (which will carry the `Activation`), and the result
    // flows back. `return h0(local0)`.
    #[test]
    fn lowers_call_helper() {
        use std::cell::Cell;
        thread_local!(static CTX: Cell<i64> = const { Cell::new(0) });

        let ops = [
            JitOp::GetLocalValue(0),
            JitOp::CallHelper(0),
            JitOp::ReturnValueBoxed,
        ];
        let bytes = compile(&ops).expect("compiles");

        let engine = Engine::default();
        let module = Module::new(&engine, &bytes).expect("valid wasm");
        let mut store = Store::new(&engine, ());
        let memory = Memory::new(&mut store, WMemoryType::new(1, None).unwrap()).unwrap();
        // local 0 = an arbitrary raw `Value` bit pattern.
        memory.write(&mut store, 0, &1000u64.to_le_bytes()).unwrap();

        CTX.with(|c| c.set(7));
        let h0 = Func::wrap(&mut store, |arg: i64| -> i64 {
            CTX.with(|c| arg.wrapping_add(c.get()))
        });

        // Imports in declaration order: h0 (function), then memory.
        let instance =
            Instance::new(&mut store, &module, &[h0.into(), memory.into()]).expect("instantiates");
        let run = instance
            .get_typed_func::<(i32, i32, i32, i32, i32, i32), i64>(&store, "run")
            .expect("run export");
        assert_eq!(run.call(&mut store, (0, 0, 0, 0, 0, 0)).expect("runs") as u64, 1007);
    }

    // Boxed loop with a `CallHelper2` compare condition — the exact shape the new
    // compares enable (`while (i < n) i = incr(i); return i`). Pins the
    // compare→IfFalseBoxed→Jump control flow. Fake raw-i64 helpers.
    #[test]
    fn lowers_boxed_compare_loop() {
        use wasmi::{Engine, Func, Instance, Memory, MemoryType as WMemoryType, Module, Store};
        let ops = [
            JitOp::GetLocalValue(1), // 0: head — i
            JitOp::GetLocalValue(2), // 1: n
            JitOp::CallHelper2(1),   // 2: cmp_lt(i, n) (HELPERS2[1])
            JitOp::IfFalseBoxed(8),  // 3: if !(i<n) -> exit
            JitOp::GetLocalValue(1), // 4: body
            JitOp::CallHelper(0),    // 5: increment
            JitOp::SetLocalValue(1), // 6: i = i+1
            JitOp::Jump(0),          // 7
            JitOp::GetLocalValue(1), // 8: exit
            JitOp::ReturnValueBoxed, // 9
        ];
        let m = manifest(&ops);
        assert_eq!((m.num_helpers, m.num_helpers2), (6, 2)); // h0..h5 + t0,t1
        let bytes = compile(&ops).expect("compiles");

        let engine = Engine::default();
        let module = Module::new(&engine, &bytes).expect("valid wasm");
        let mut store = Store::new(&engine, ());
        let memory = Memory::new(&mut store, WMemoryType::new(1, None).unwrap()).unwrap();
        memory.write(&mut store, 8, &0u64.to_le_bytes()).unwrap(); // local1 = i = 0
        memory.write(&mut store, 16, &5u64.to_le_bytes()).unwrap(); // local2 = n = 5

        let h0 = Func::wrap(&mut store, |a: i64| -> i64 { a + 1 }); // increment
        let hid = || {};
        let _ = hid;
        let h1 = Func::wrap(&mut store, |a: i64| -> i64 { a });
        let h2 = Func::wrap(&mut store, |a: i64| -> i64 { a });
        let h3 = Func::wrap(&mut store, |a: i64| -> i64 { a });
        let h4 = Func::wrap(&mut store, |a: i64| -> i64 { a });
        let h5 = Func::wrap(&mut store, |a: i64| -> i64 { (a != 0) as i64 }); // to_boolean
        let t0 = Func::wrap(&mut store, |a: i64, _b: i64| -> i64 { a }); // unused
        let t1 = Func::wrap(&mut store, |a: i64, b: i64| -> i64 { (a < b) as i64 }); // cmp_lt

        let instance = Instance::new(
            &mut store,
            &module,
            &[
                h0.into(), h1.into(), h2.into(), h3.into(), h4.into(), h5.into(),
                t0.into(), t1.into(), memory.into(),
            ],
        )
        .expect("instantiates");
        let run = instance
            .get_typed_func::<(i32, i32, i32, i32, i32, i32), i64>(&store, "run")
            .expect("run export");
        // i: 0 → loops while i<5 → 5.
        assert_eq!(run.call(&mut store, (0, 0, 0, 0, 0, 0)).expect("runs") as u64, 5);
    }

    // `lookupswitch`: a computed multi-way branch lowered to a nested-block
    // `br_table` over the BB dispatch. `switch (local0) { case 0: 100; case 1: 200;
    // default: 999 }`. h12 (coerce_i) is the identity here, so the selector's low
    // 32 bits are the index; each arm returns a boxed int whose low 32 bits we check.
    #[test]
    fn lowers_lookup_switch() {
        use wasmi::{Engine, Func, Instance, Memory, MemoryType as WMemoryType, Module, Store};
        let ops = [
            JitOp::GetLocalValue(0), // 0: push selector
            JitOp::LookupSwitch(0),  // 1: cases=[2,4], default=6
            JitOp::PushIntValue(100), // 2: case 0
            JitOp::ReturnValueBoxed, // 3
            JitOp::PushIntValue(200), // 4: case 1
            JitOp::ReturnValueBoxed, // 5
            JitOp::PushIntValue(999), // 6: default
            JitOp::ReturnValueBoxed, // 7
        ];
        let switches = [SwitchTable {
            default: 6,
            cases: Box::new([2usize, 4]),
        }];
        // `lookupswitch` imports h0..=h12 (coerce_i selector conversion).
        assert_eq!(manifest(&ops).num_helpers, COERCE_I + 1);

        let bytes = compile_with_switches(&ops, &switches).expect("compiles");
        let engine = Engine::default();
        let module = Module::new(&engine, &bytes).expect("valid wasm");
        let mut store = Store::new(&engine, ());
        let memory = Memory::new(&mut store, WMemoryType::new(1, None).unwrap()).unwrap();

        let mut imports: Vec<wasmi::Extern> = Vec::new();
        for _ in 0..=COERCE_I {
            // Every arity-1 helper is the identity; h12 (coerce_i) identity means the
            // selector's low 32 bits are used directly as the switch index.
            let h = Func::wrap(&mut store, |a: i64| -> i64 { a });
            imports.push(h.into());
        }
        imports.push(memory.into());

        let instance = Instance::new(&mut store, &module, &imports).expect("instantiates");
        let run = instance
            .get_typed_func::<(i32, i32, i32, i32, i32, i32), i64>(&store, "run")
            .expect("run export");

        for (selector, expected) in [(0u64, 100u32), (1, 200), (5, 999), (99, 999)] {
            memory.write(&mut store, 0, &selector.to_le_bytes()).unwrap();
            let got = run.call(&mut store, (0, 0, 0, 0, 0, 0)).expect("runs") as u32;
            assert_eq!(got, expected, "selector {selector} → {expected}");
        }
    }

    // Inline numeric compare (`CmpNum`): the numeric fast path (int/number/mixed,
    // incl. NaN) is computed in-module with no helper call; a non-numeric operand
    // falls back to the two-stack helper. `return local1 < local2`.
    #[test]
    fn lowers_inline_numeric_compare() {
        use wasmi::{Engine, Func, Instance, Memory, MemoryType as WMemoryType, Module, Store};
        const TRUE_BITS: u64 = VALUE_BOOL_MARK | 1;
        const FALSE_BITS: u64 = VALUE_BOOL_MARK;
        const FALLBACK: u64 = 0xABCD; // sentinel returned by the fake helper
        let num = |n: f64| n.to_bits();
        let int = |n: i32| VALUE_INT_MARK | (n as u32 as u64);
        let obj: u64 = 0xFFFD_0000_0000_0001; // TAG_OBJECT box — not numeric

        let ops = [
            JitOp::GetLocalValue(1),
            JitOp::GetLocalValue(2),
            JitOp::CmpNum(1), // lessthan
            JitOp::ReturnValueBoxed,
        ];
        let m = manifest(&ops);
        assert_eq!((m.num_helpers, m.num_helpers2), (0, 2)); // t0,t1 for the fallback
        let bytes = compile(&ops).expect("compiles");

        let cases: &[(u64, u64, u64)] = &[
            (int(3), int(5), TRUE_BITS),        // int < int (fast)
            (int(5), int(5), FALSE_BITS),       // not less
            (num(5.0), num(3.0), FALSE_BITS),   // number < number (fast)
            (int(3), num(5.5), TRUE_BITS),      // mixed int/number (fast)
            (num(-2.0), int(1), TRUE_BITS),     // mixed number/int (fast)
            (num(f64::NAN), num(1.0), FALSE_BITS), // NaN < x → false
            (obj, int(5), FALLBACK),            // non-numeric → helper fallback
        ];
        for &(a, b, want) in cases {
            let engine = Engine::default();
            let module = Module::new(&engine, &bytes).expect("valid wasm");
            let mut store = Store::new(&engine, ());
            let memory = Memory::new(&mut store, WMemoryType::new(1, None).unwrap()).unwrap();
            memory.write(&mut store, 8, &a.to_le_bytes()).unwrap(); // local1
            memory.write(&mut store, 16, &b.to_le_bytes()).unwrap(); // local2
            let t0 = Func::wrap(&mut store, |a: i64, _b: i64| -> i64 { a });
            let t1 = Func::wrap(&mut store, |_a: i64, _b: i64| -> i64 { FALLBACK as i64 });
            let instance =
                Instance::new(&mut store, &module, &[t0.into(), t1.into(), memory.into()])
                    .expect("instantiates");
            let run = instance
                .get_typed_func::<(i32, i32, i32, i32, i32, i32), i64>(&store, "run")
                .expect("run export");
            assert_eq!(
                run.call(&mut store, (0, 0, 0, 0, 0, 0)).expect("runs") as u64,
                want,
                "a={a:#018x} b={b:#018x}"
            );
        }
    }

    // Inline bitwise (`BitOpInt`): both int-boxed → the `i32` op inline; a non-int
    // operand → the two-stack helper fallback. `return local1 & local2`.
    #[test]
    fn lowers_inline_bitop_int() {
        use wasmi::{Engine, Func, Instance, Memory, MemoryType as WMemoryType, Module, Store};
        const FALLBACK: u64 = 0xBEEF;
        let int = |n: i32| VALUE_INT_MARK | (n as u32 as u64);
        let obj: u64 = 0xFFFD_0000_0000_0001; // TAG_OBJECT — not int

        let ops = [
            JitOp::GetLocalValue(1),
            JitOp::GetLocalValue(2),
            JitOp::BitOpInt(5), // bitand (HELPERS2[5])
            JitOp::ReturnValueBoxed,
        ];
        let m = manifest(&ops);
        assert_eq!((m.num_helpers, m.num_helpers2), (0, 6)); // t0..t5, fallback at t5
        let bytes = compile(&ops).expect("compiles");

        let cases: &[(u64, u64, u64)] = &[
            (int(12), int(10), int(12 & 10)),     // 8, inline
            (int(-1), int(0x0F), int(-1 & 0x0F)), // 15, inline (negative int ok for &)
            (obj, int(5), FALLBACK),              // non-int → helper
        ];
        for &(a, b, want) in cases {
            let engine = Engine::default();
            let module = Module::new(&engine, &bytes).expect("valid wasm");
            let mut store = Store::new(&engine, ());
            let memory = Memory::new(&mut store, WMemoryType::new(1, None).unwrap()).unwrap();
            memory.write(&mut store, 8, &a.to_le_bytes()).unwrap();
            memory.write(&mut store, 16, &b.to_le_bytes()).unwrap();
            let id = |s: &mut Store<()>| Func::wrap(s, |a: i64, _b: i64| -> i64 { a });
            let (t0, t1, t2, t3, t4) = (
                id(&mut store), id(&mut store), id(&mut store), id(&mut store), id(&mut store),
            );
            let t5 = Func::wrap(&mut store, |_a: i64, _b: i64| -> i64 { FALLBACK as i64 });
            let instance = Instance::new(
                &mut store,
                &module,
                &[t0.into(), t1.into(), t2.into(), t3.into(), t4.into(), t5.into(), memory.into()],
            )
            .expect("instantiates");
            let run = instance
                .get_typed_func::<(i32, i32, i32, i32, i32, i32), i64>(&store, "run")
                .expect("run export");
            assert_eq!(run.call(&mut store, (0, 0, 0, 0, 0, 0)).unwrap() as u64, want, "a={a:#018x} b={b:#018x}");
        }
    }

    // Inline generic arithmetic (`ArithInt`): both int-boxed → `i64` op, boxed as
    // `int` if it fits else `Number` (matching the helper's checked-int-else-Number);
    // a non-int operand → the two-stack helper fallback. `return local1 * local2`.
    #[test]
    fn lowers_inline_arith_int() {
        use wasmi::{Engine, Func, Instance, Memory, MemoryType as WMemoryType, Module, Store};
        const FALLBACK: u64 = 0xCAFE;
        let int = |n: i32| VALUE_INT_MARK | (n as u32 as u64);
        let num = |n: f64| n.to_bits();
        let obj: u64 = 0xFFFD_0000_0000_0001;

        let ops = [
            JitOp::GetLocalValue(1),
            JitOp::GetLocalValue(2),
            JitOp::ArithInt(11), // multiply (HELPERS2[11])
            JitOp::ReturnValueBoxed,
        ];
        let m = manifest(&ops);
        assert_eq!((m.num_helpers, m.num_helpers2), (0, 12)); // t0..t11, fallback at t11
        let bytes = compile(&ops).expect("compiles");

        let cases: &[(u64, u64, u64)] = &[
            (int(6), int(7), int(42)),                     // fits → int
            (int(-3), int(4), int(-12)),                   // negative fits → int
            (int(100000), int(100000), num(1e10)),         // overflow → Number(1e10)
            (int(-100000), int(100000), num(-1e10)),       // negative overflow → Number
            (num(3.5), int(2), num(7.0)),                  // numeric middle path → Number
            (int(2), num(3.5), num(7.0)),                  // mixed the other way
            (num(2.0), num(4.0), num(8.0)),                // Number × Number stays Number
            (num(f64::INFINITY), num(0.0), CANON_NAN),     // inf·0 → canonicalized NaN
            (obj, int(2), FALLBACK),                       // non-numeric → helper
        ];
        for &(a, b, want) in cases {
            let engine = Engine::default();
            let module = Module::new(&engine, &bytes).expect("valid wasm");
            let mut store = Store::new(&engine, ());
            let memory = Memory::new(&mut store, WMemoryType::new(1, None).unwrap()).unwrap();
            memory.write(&mut store, 8, &a.to_le_bytes()).unwrap();
            memory.write(&mut store, 16, &b.to_le_bytes()).unwrap();
            let mut externs: Vec<wasmi::Extern> = Vec::new();
            for i in 0..12u32 {
                let f = if i == 11 {
                    Func::wrap(&mut store, |_a: i64, _b: i64| -> i64 { FALLBACK as i64 })
                } else {
                    Func::wrap(&mut store, |a: i64, _b: i64| -> i64 { a })
                };
                externs.push(f.into());
            }
            externs.push(memory.into());
            let instance = Instance::new(&mut store, &module, &externs).expect("instantiates");
            let run = instance
                .get_typed_func::<(i32, i32, i32, i32, i32, i32), i64>(&store, "run")
                .expect("run export");
            assert_eq!(run.call(&mut store, (0, 0, 0, 0, 0, 0)).unwrap() as u64, want, "a={a:#018x} b={b:#018x}");
        }
    }

    // Inline `divide` (`ArithNum`): numeric operands → the f64 division inline
    // (ALWAYS a `Number`, even for exact int quotients — matching the helper);
    // non-numeric → the two-stack helper fallback.
    #[test]
    fn lowers_inline_arith_num_divide() {
        use wasmi::{Engine, Func, Instance, Memory, MemoryType as WMemoryType, Module, Store};
        const FALLBACK: u64 = 0xD1DE;
        let int = |n: i32| VALUE_INT_MARK | (n as u32 as u64);
        let num = |n: f64| n.to_bits();
        let obj: u64 = 0xFFFD_0000_0000_0001;

        let ops = [
            JitOp::GetLocalValue(1),
            JitOp::GetLocalValue(2),
            JitOp::ArithNum(13), // divide (HELPERS2[13])
            JitOp::ReturnValueBoxed,
        ];
        let m = manifest(&ops);
        assert_eq!((m.num_helpers, m.num_helpers2), (0, 14)); // t0..t13, fallback at t13
        let bytes = compile(&ops).expect("compiles");

        let cases: &[(u64, u64, u64)] = &[
            (int(7), int(2), num(3.5)),           // int/int → Number
            (int(6), int(2), num(3.0)),           // exact quotient STAYS a Number
            (num(1.0), num(0.0), num(f64::INFINITY)), // 1/0 → inf
            (num(0.0), num(0.0), CANON_NAN),      // 0/0 → canonicalized NaN
            (num(-4.5), int(3), num(-1.5)),       // mixed
            (obj, int(2), FALLBACK),              // non-numeric → helper
        ];
        for &(a, b, want) in cases {
            let engine = Engine::default();
            let module = Module::new(&engine, &bytes).expect("valid wasm");
            let mut store = Store::new(&engine, ());
            let memory = Memory::new(&mut store, WMemoryType::new(1, None).unwrap()).unwrap();
            memory.write(&mut store, 8, &a.to_le_bytes()).unwrap();
            memory.write(&mut store, 16, &b.to_le_bytes()).unwrap();
            let mut externs: Vec<wasmi::Extern> = Vec::new();
            for i in 0..14u32 {
                let f = if i == 13 {
                    Func::wrap(&mut store, |_a: i64, _b: i64| -> i64 { FALLBACK as i64 })
                } else {
                    Func::wrap(&mut store, |a: i64, _b: i64| -> i64 { a })
                };
                externs.push(f.into());
            }
            externs.push(memory.into());
            let instance = Instance::new(&mut store, &module, &externs).expect("instantiates");
            let run = instance
                .get_typed_func::<(i32, i32, i32, i32, i32, i32), i64>(&store, "run")
                .expect("run export");
            assert_eq!(run.call(&mut store, (0, 0, 0, 0, 0, 0)).unwrap() as u64, want, "a={a:#018x} b={b:#018x}");
        }
    }

    // Inline `coerce_i`/`coerce_u` (`CoerceInt`): an int passes through (`coerce_i`
    // always; `coerce_u` only when non-negative), else the arity-1 helper fallback.
    #[test]
    fn lowers_inline_coerce_int() {
        use wasmi::{Engine, Func, Instance, Memory, MemoryType as WMemoryType, Module, Store};
        const FALLBACK: u64 = 0xF00D;
        let int = |n: i32| VALUE_INT_MARK | (n as u32 as u64);
        let num = |n: f64| n.to_bits();

        // (signed, fallback_helper_index, cases)
        let variants: &[(bool, u32, &[(u64, u64)])] = &[
            // coerce_i: int passes through; a finite Number takes the inline
            // `ToInt32` (trunc toward zero, wrapping mod 2^32); NaN/±Inf hit the
            // helper (h12).
            (
                true,
                12,
                &[
                    (int(5), int(5)),
                    (int(-7), int(-7)),
                    (num(3.14), int(3)),
                    (num(-3.99), int(-3)),
                    (num(4294967296.0), int(0)), // 2^32 wraps to 0
                    (num(f64::NAN), FALLBACK),
                    (num(f64::INFINITY), FALLBACK),
                ],
            ),
            // coerce_u: non-negative int passes through; a negative int / finite
            // Number takes the inline `ToUint32` (int box when ≤ i32::MAX, else a
            // `Number` box); NaN hits the helper (h11).
            (
                false,
                11,
                &[
                    (int(5), int(5)),
                    (int(-1), num(4294967295.0)),
                    (num(1.0), int(1)),
                    (num(2147483648.0), num(2147483648.0)),
                    (num(f64::NAN), FALLBACK),
                ],
            ),
        ];
        for &(signed, fb_idx, cases) in variants {
            let ops = [JitOp::GetLocalValue(1), JitOp::CoerceInt(signed), JitOp::ReturnValueBoxed];
            let m = manifest(&ops);
            assert_eq!(m.num_helpers, fb_idx + 1); // h0..h{fb_idx}
            assert!(m.needs_perr()); // CoerceInt is a throwing op (helper can #1050)
            let bytes = compile(&ops).expect("compiles");
            for &(a, want) in cases {
                let engine = Engine::default();
                let module = Module::new(&engine, &bytes).expect("valid wasm");
                let mut store = Store::new(&engine, ());
                let memory = Memory::new(&mut store, WMemoryType::new(1, None).unwrap()).unwrap();
                memory.write(&mut store, 8, &a.to_le_bytes()).unwrap();
                let mut externs: Vec<wasmi::Extern> = Vec::new();
                for i in 0..(fb_idx + 1) {
                    let f = if i == fb_idx {
                        Func::wrap(&mut store, |_a: i64| -> i64 { FALLBACK as i64 })
                    } else {
                        Func::wrap(&mut store, |a: i64| -> i64 { a })
                    };
                    externs.push(f.into());
                }
                // `perr` (no pending error) — imported right after the h helpers.
                externs.push(Func::wrap(&mut store, || -> i32 { 0 }).into());
                externs.push(memory.into());
                let instance = Instance::new(&mut store, &module, &externs).expect("instantiates");
                let run = instance
                    .get_typed_func::<(i32, i32, i32, i32, i32, i32), i64>(&store, "run")
                    .expect("run export");
                assert_eq!(
                    run.call(&mut store, (0, 0, 0, 0, 0, 0)).unwrap() as u64,
                    want,
                    "signed={signed} a={a:#018x}"
                );
            }
        }
    }

    // Inline domainMemory: a `si32; li32` round-trip through **memory 1** (the
    // fast path — no helper call), plus a reservation MISS (`dm_len` too small)
    // that must route both ops through the imported dm-helper FALLBACKS (which in
    // production reach the real storage — e.g. an unshared domainMemory). Uses a
    // wasmi engine with multi-memory and a controlled memory 1.
    #[test]
    fn lowers_dm_inline() {
        use wasmi::{
            Caller, Config, Engine, Func, Instance, Memory, MemoryType as WMemoryType, Module,
            Store,
        };
        let box_int = |v: i32| VALUE_INT_MARK | (v as u32 as u64);

        // store local1 (value) at local2 (addr), read it back: → value.
        let ops = [
            JitOp::GetLocalValue(1), // value
            JitOp::GetLocalValue(2), // addr
            JitOp::DmStore(4),
            JitOp::GetLocalValue(2), // addr
            JitOp::DmLoad(4),
            JitOp::ReturnValueBoxed,
        ];
        let m = manifest(&ops);
        assert!(m.has_dm);
        // The miss-branch fallbacks are imported: `h0..=h10` (li32) + the `s3`
        // ternary (si32).
        assert_eq!(m.num_helpers, DM_LOAD32 + 1);
        assert_ne!(m.set3_mask & (1 << DM_STORE_KIND), 0);
        let bytes = compile(&ops).expect("compiles");

        let mut config = Config::default();
        config.wasm_multi_memory(true);
        let engine = Engine::new(&config);
        let module = Module::new(&engine, &bytes).expect("valid multi-memory wasm");
        // Store data: (store-fallback calls `(value, addr, width)`, load-fallback
        // call args).
        type DmLog = (Vec<(i64, i64, i64)>, Vec<i64>);
        let mut store = Store::new(&engine, DmLog::default());
        let frame = Memory::new(&mut store, WMemoryType::new(1, None).unwrap()).unwrap();
        let dm = Memory::new(&mut store, WMemoryType::new(1, None).unwrap()).unwrap();
        // local1 = value = 0x12345678, local2 = addr = 8.
        frame.write(&mut store, 8, &box_int(0x12345678).to_le_bytes()).unwrap();
        frame.write(&mut store, 16, &box_int(8).to_le_bytes()).unwrap();
        // The descriptor cell at memory1[32]: [base=64, cap=64] — the emitted
        // code re-reads it per access (`dm_base` param = the CELL's address).
        dm.write(&mut store, 32, &64u32.to_le_bytes()).unwrap();
        dm.write(&mut store, 36, &64u32.to_le_bytes()).unwrap();

        const LOAD_SENTINEL: i64 = 0x7777;
        let mut externs: Vec<wasmi::Extern> = Vec::new();
        for i in 0..=DM_LOAD32 {
            externs.push(if i == DM_LOAD32 {
                // The li32 fallback: record the boxed addr, return a sentinel.
                Func::wrap(&mut store, |mut c: Caller<'_, DmLog>, a: i64| -> i64 {
                    c.data_mut().1.push(a);
                    LOAD_SENTINEL
                })
                .into()
            } else {
                Func::wrap(&mut store, |_: Caller<'_, DmLog>, a: i64| -> i64 { a }).into()
            });
        }
        // `s3` — the si32 fallback: record `(value, addr, width)`.
        externs.push(
            Func::wrap(
                &mut store,
                |mut c: Caller<'_, DmLog>, v: i64, a: i64, n: i64| -> i64 {
                    c.data_mut().0.push((v, a, n));
                    0
                },
            )
            .into(),
        );
        externs.push(frame.into());
        externs.push(dm.into());
        let instance = Instance::new(&mut store, &module, &externs).expect("instantiates");
        let run = instance
            .get_typed_func::<(i32, i32, i32, i32, i32, i32), i64>(&store, "run")
            .expect("run export");
        // run(state_ptr=0, dm_base=DESC@32): fast path — store+load at
        // memory1[base=64 + addr=8], no fallback calls.
        assert_eq!(
            run.call(&mut store, (0, 32, 0, 0, 0, 0)).expect("runs") as u64,
            box_int(0x12345678)
        );
        assert_eq!(dm.data(&store)[72..76], 0x12345678u32.to_le_bytes());
        assert!(store.data().0.is_empty() && store.data().1.is_empty());

        // Shrink the cell's cap to 8 → addr 8 (+4) misses → BOTH ops take the
        // helper fallback: the store logs `(value, addr, 4)`, the load logs the
        // addr and its sentinel is the result.
        dm.write(&mut store, 36, &8u32.to_le_bytes()).unwrap();
        assert_eq!(
            run.call(&mut store, (0, 32, 0, 0, 0, 0)).expect("runs"),
            LOAD_SENTINEL
        );
        assert_eq!(
            store.data().0,
            vec![(box_int(0x12345678) as i64, box_int(8) as i64, 4)]
        );
        assert_eq!(store.data().1, vec![box_int(8) as i64]);

        // desc == 0 (domainMemory unavailable) → fallback too.
        assert_eq!(
            run.call(&mut store, (0, 0, 0, 0, 0, 0)).expect("runs"),
            LOAD_SENTINEL
        );
        assert_eq!(store.data().0.len(), 2);
        assert_eq!(store.data().1.len(), 2);
    }

    // Inline domainMemory *float* store/load (`sf32`/`sf64`/`lf32`/`lf64`): the value
    // round-trips through memory 1 as an `f32`/`f64` and comes back a `Number` Value.
    #[test]
    fn lowers_dm_float_inline() {
        use wasmi::{
            Config, Engine, Func, Instance, Memory, MemoryType as WMemoryType, Module, Store,
        };
        let int_val = |v: i32| VALUE_INT_MARK | (v as u32 as u64);

        // store local1 (Number) at local2 (addr), read it back → the Number.
        let round_trip = |width: u32, value_bits: u64, addr: i32| -> (u64, Vec<u8>) {
            let ops = [
                JitOp::GetLocalValue(1), // value
                JitOp::GetLocalValue(2), // addr
                JitOp::DmStoreF(width),
                JitOp::GetLocalValue(2), // addr
                JitOp::DmLoadF(width),
                JitOp::ReturnValueBoxed,
            ];
            let m = manifest(&ops);
            assert!(m.has_dm);
            // Miss-branch fallbacks imported: `h0..=h{lf32|lf64}` + the `s4` ternary.
            assert_eq!(m.num_helpers, if width == 4 { DM_LOADF32 } else { DM_LOADF64 } + 1);
            assert_ne!(m.set3_mask & (1 << DM_STORE_F_KIND), 0);
            let bytes = compile(&ops).expect("compiles");
            let mut config = Config::default();
            config.wasm_multi_memory(true);
            let engine = Engine::new(&config);
            let module = Module::new(&engine, &bytes).expect("valid multi-memory wasm");
            let mut store = Store::new(&engine, ());
            let frame = Memory::new(&mut store, WMemoryType::new(1, None).unwrap()).unwrap();
            let dm = Memory::new(&mut store, WMemoryType::new(1, None).unwrap()).unwrap();
            frame.write(&mut store, 8, &value_bits.to_le_bytes()).unwrap(); // local1
            frame.write(&mut store, 16, &int_val(addr).to_le_bytes()).unwrap(); // local2
            // Descriptor cell at memory1[32]: [base=64, cap=64].
            dm.write(&mut store, 32, &64u32.to_le_bytes()).unwrap();
            dm.write(&mut store, 36, &64u32.to_le_bytes()).unwrap();
            let mut externs: Vec<wasmi::Extern> = Vec::new();
            for _ in 0..m.num_helpers {
                externs.push(Func::wrap(&mut store, |a: i64| -> i64 { a }).into());
            }
            // `s4` — the sf* fallback (unused here: the accesses stay in bounds).
            externs
                .push(Func::wrap(&mut store, |_: i64, _: i64, _: i64| -> i64 { 0 }).into());
            externs.push(frame.into());
            externs.push(dm.into());
            let instance =
                Instance::new(&mut store, &module, &externs).expect("instantiates");
            let run = instance
                .get_typed_func::<(i32, i32, i32, i32, i32, i32), i64>(&store, "run")
                .expect("run export");
            let got = run.call(&mut store, (0, 32, 0, 0, 0, 0)).expect("runs") as u64;
            let at = 64 + addr as usize; // base=64 + addr
            let mem = dm.data(&store)[at..at + width as usize].to_vec();
            (got, mem)
        };

        // f64: a `Number` Value's bits are the raw `f64`, so it round-trips exactly.
        let (got, mem) = round_trip(8, 3.14f64.to_bits(), 8);
        assert_eq!(got, 3.14f64.to_bits());
        assert_eq!(mem, 3.14f64.to_le_bytes());

        // f32: 1.5 is exact in `f32`, so demote→promote is lossless.
        let (got, mem) = round_trip(4, 1.5f64.to_bits(), 8);
        assert_eq!(got, 1.5f64.to_bits());
        assert_eq!(mem, 1.5f32.to_le_bytes());
    }

    // The heap-box-safe unbox primitive: `emit_numval_boxsafe` must yield the
    // correct `f64` for every numeric input shape, INCLUDING a `TAG_BOXED_DOUBLE`
    // (a colliding NaN `number_lossless` stashed in a `Gc<f64>`), which it derefs
    // via `f64.load` on memory 1. The stubbed helpers in the other dm tests can't
    // produce a real boxed double, so exercise the primitive directly here.
    #[test]
    fn numval_boxsafe_covers_all_cases() {
        use wasm_encoder::{
            CodeSection, ExportSection, FunctionSection, MemorySection, Module as EncModule,
            TypeSection,
        };
        // A standalone module `t(i64) -> f64` = emit_numval_boxsafe(param0), with a
        // second memory (index 1) it can `f64.load` a boxed double from.
        let mut types = TypeSection::new();
        types.ty().function([ValType::I64], [ValType::F64]);
        let mut funcs = FunctionSection::new();
        funcs.function(0);
        let mut mems = MemorySection::new();
        let mt = MemoryType {
            minimum: 1,
            maximum: None,
            memory64: false,
            shared: false,
            page_size_log2: None,
        };
        mems.memory(mt); // memory 0 (unused, keeps indices honest)
        mems.memory(mt); // memory 1 (the heap the boxed double lives in)
        let mut exports = ExportSection::new();
        exports.export("t", ExportKind::Func, 0);
        exports.export("mem1", ExportKind::Memory, 1);
        let mut code = CodeSection::new();
        let mut f = Function::new([]); // param0 is the only local
        emit_numval_boxsafe(&mut f, 0);
        f.instruction(&Instruction::End);
        code.function(&f);
        let mut module = EncModule::new();
        module.section(&types);
        module.section(&funcs);
        module.section(&mems);
        module.section(&exports);
        module.section(&code);
        let bytes = module.finish();

        let mut config = wasmi::Config::default();
        config.wasm_multi_memory(true);
        let engine = Engine::new(&config);
        let module = Module::new(&engine, &bytes).expect("valid multi-memory wasm");
        let mut store = Store::new(&engine, ());
        let instance = Instance::new(&mut store, &module, &[]).expect("instantiates");
        let mem1 = instance.get_memory(&store, "mem1").expect("mem1 export");
        mem1.grow(&mut store, 1).ok();
        let run = instance
            .get_typed_func::<i64, f64>(&store, "t")
            .expect("t export");

        // (a) inline `Number`: the bits ARE the f64.
        assert_eq!(run.call(&mut store, 3.14f64.to_bits() as i64).unwrap(), 3.14);
        // (b) int box (positive and negative low 32 bits).
        assert_eq!(run.call(&mut store, (VALUE_INT_MARK | 42) as i64).unwrap(), 42.0);
        assert_eq!(
            run.call(&mut store, (VALUE_INT_MARK | (-7i32 as u32 as u64)) as i64).unwrap(),
            -7.0
        );
        // (c) heap-boxed double: a colliding negative-quiet-NaN whose EXACT bits are
        // stashed at memory1[128]; the Value carries `TAG_BOXED_DOUBLE` + that
        // pointer. `emit_numval_boxsafe` must `f64.load` the exact bits back.
        let colliding = 0xFFF8_0000_0000_0001u64; // bits & BOX_MARK == BOX_MARK
        assert_eq!(colliding & BOX_MARK, BOX_MARK);
        mem1.write(&mut store, 128, &colliding.to_le_bytes()).unwrap();
        let boxed = BOX_MARK | (TAG_BOXED_DOUBLE << TAG_SHIFT) | 128u64;
        assert_eq!(
            run.call(&mut store, boxed as i64).unwrap().to_bits(),
            colliding,
            "boxed double must deref to its exact bits"
        );
        // (d) swallowed-OOB `Undefined` → a pure `0.0` (no trap, no memory access).
        assert_eq!(run.call(&mut store, UNDEFINED_BITS as i64).unwrap(), 0.0);
    }

    // End-to-end phase-2 over a `NumberBoxed` (`DmLoadF` result): the copyTo-shaped
    // `lf64(addr) * lf64(addr)` must physically unbox both loads (heap-box-safely)
    // and compute the product. Exercises the whole pipeline (infer → is_f64_arith →
    // emit_arith_number_phys w/ box-safe unbox → box_top before return).
    #[test]
    fn dm_loadf_feeds_physical_arith() {
        use wasmi::{
            Config, Engine, Func, Instance, Memory, MemoryType as WMemoryType, Module, Store,
        };
        let int_val = |v: i32| VALUE_INT_MARK | (v as u32 as u64);
        let ops = [
            JitOp::GetLocalValue(1), // addr
            JitOp::DmLoadF(8),       // x (NumberBoxed)
            JitOp::GetLocalValue(1), // addr
            JitOp::DmLoadF(8),       // x (NumberBoxed)
            JitOp::ArithInt(11),     // multiply -> x*x (physical f64)
            JitOp::ReturnValueBoxed,
        ];
        let m = manifest(&ops);
        assert!(m.has_dm);
        let bytes = compile(&ops).expect("compiles");
        let mut config = Config::default();
        config.wasm_multi_memory(true);
        let engine = Engine::new(&config);
        let module = Module::new(&engine, &bytes).expect("valid multi-memory wasm");
        let mut store = Store::new(&engine, ());
        let frame = Memory::new(&mut store, WMemoryType::new(1, None).unwrap()).unwrap();
        let dm = Memory::new(&mut store, WMemoryType::new(1, None).unwrap()).unwrap();
        frame.write(&mut store, 8, &int_val(0).to_le_bytes()).unwrap(); // local1 = addr 0
        dm.write(&mut store, 32, &64u32.to_le_bytes()).unwrap(); // base=64
        dm.write(&mut store, 36, &64u32.to_le_bytes()).unwrap(); // cap=64
        dm.write(&mut store, 64, &3.0f64.to_le_bytes()).unwrap(); // value at base+0
        let mut externs: Vec<wasmi::Extern> = Vec::new();
        for _ in 0..m.num_helpers {
            externs.push(Func::wrap(&mut store, |a: i64| -> i64 { a }).into());
        }
        // The arity-2 `t{i}` arith-helper fallbacks (unused: the fast path elides
        // them, but the module still imports them).
        for _ in 0..m.num_helpers2 {
            externs.push(Func::wrap(&mut store, |a: i64, _: i64| -> i64 { a }).into());
        }
        externs.push(frame.into());
        externs.push(dm.into());
        let instance = Instance::new(&mut store, &module, &externs).expect("instantiates");
        let run = instance
            .get_typed_func::<(i32, i32, i32, i32, i32, i32), i64>(&store, "run")
            .expect("run export");
        let got = run.call(&mut store, (0, 32, 0, 0, 0, 0)).expect("runs") as u64;
        // 3.0 * 3.0 = 9.0, boxed as a canonical `Number` (its bits ARE the f64).
        assert_eq!(got, 9.0f64.to_bits());
    }

    // `coerceb`: `ToBoolean(v)` (via `to_boolean` = h5) boxed as a `Boolean` Value.
    #[test]
    fn lowers_coerce_bool() {
        use wasmi::{Engine, Func, Instance, Memory, MemoryType as WMemoryType, Module, Store};
        let ops = [JitOp::GetLocalValue(0), JitOp::CoerceBool, JitOp::ReturnValueBoxed];
        assert_eq!(manifest(&ops).num_helpers, TO_BOOLEAN + 1); // h0..h5
        let bytes = compile(&ops).expect("compiles");
        let engine = Engine::default();
        let module = Module::new(&engine, &bytes).expect("valid wasm");
        let mut store = Store::new(&engine, ());
        let memory = Memory::new(&mut store, WMemoryType::new(1, None).unwrap()).unwrap();
        let mut externs: Vec<wasmi::Extern> = Vec::new();
        for i in 0..6u32 {
            let f = if i == 5 {
                Func::wrap(&mut store, |v: i64| -> i64 { (v != 0) as i64 }) // to_boolean
            } else {
                Func::wrap(&mut store, |a: i64| -> i64 { a })
            };
            externs.push(f.into());
        }
        externs.push(memory.into());
        let instance = Instance::new(&mut store, &module, &externs).expect("instantiates");
        let run = instance
            .get_typed_func::<(i32, i32, i32, i32, i32, i32), i64>(&store, "run")
            .expect("run export");
        for (v, expect) in [(0u64, VALUE_BOOL_MARK), (7, VALUE_BOOL_MARK | 1)] {
            memory.write(&mut store, 0, &v.to_le_bytes()).unwrap();
            assert_eq!(run.call(&mut store, (0, 0, 0, 0, 0, 0)).unwrap() as u64, expect, "coerceb {v}");
        }
    }

    // `coerce <class>` lowering: pushes the class-table index `k`, calls
    // `coerce(value, k)`, and (as a throwing op) bails the whole method with
    // `undefined` when `perr` reports a pending error. No calls/dm, so the imports
    // are just `perr` then `coerce` then memory. Fake hosts stand in for the real
    // `coerce_to_type`/error helpers.
    #[test]
    fn lowers_coerce() {
        use wasmi::{Engine, Func, Instance, Memory, MemoryType as WMemoryType, Module, Store};
        // value=local0; `coerce k=0`.
        let ops = [JitOp::GetLocalValue(0), JitOp::Coerce(0), JitOp::ReturnValueBoxed];
        let m = manifest(&ops);
        assert!(m.has_coerce && !m.has_call && !m.dm_throws && m.num_helpers == 0);
        // Layout: perr at 0, coerce at 1, run at 2.
        let lay = layout(&ops);
        assert_eq!((lay.perr_index, lay.coerce_index, lay.run_index), (0, 1, 2));
        // `Coerce(0)` bakes `coerce_classes[0]`; a null pointer here bakes `0`, so the
        // mock `coerce` below still sees `k == 0` (it's never dereferenced).
        let bytes =
            compile_full(&ops, &[], &[], &[], &[], &[std::ptr::null::<()>()], &[], 0).expect("compiles");

        let go = |err: i32| -> u64 {
            let engine = Engine::default();
            let module = Module::new(&engine, &bytes).expect("valid wasm");
            let mut store = Store::new(&engine, ());
            let memory = Memory::new(&mut store, WMemoryType::new(1, None).unwrap()).unwrap();
            memory.write(&mut store, 0, &42u64.to_le_bytes()).unwrap(); // value=local0
            // `coerce(value, k)`: encode both so a wrong value/index fails the assert.
            let coerce = Func::wrap(&mut store, |value: i64, k: i64| -> i64 {
                assert_eq!(k, 0);
                value * 2 + k // 42*2 = 84
            });
            let perr = Func::wrap(&mut store, move || -> i32 { err });
            // Imports in declaration order: perr, coerce, memory.
            let instance = Instance::new(
                &mut store,
                &module,
                &[perr.into(), coerce.into(), memory.into()],
            )
            .expect("instantiates");
            let run = instance
                .get_typed_func::<(i32, i32, i32, i32, i32, i32), i64>(&store, "run")
                .expect("run export");
            run.call(&mut store, (0, 0, 0, 0, 0, 0)).expect("runs") as u64
        };

        assert_eq!(go(0), 84); // no error → the coerced value flows through
        assert_eq!(go(1), UNDEFINED_BITS); // pending error → method bails with undefined
    }

    // `callmethod` lowering: spills two args, calls `cm(receiver, id, argc)`, and
    // returns the result — and the emitted post-call error check bails the whole
    // method (returning `undefined`) when `perr` reports a pending error. Fake
    // wasmi imports stand in for the real spill/call/error helpers.
    #[test]
    fn lowers_call_method() {
        use std::cell::RefCell;
        use wasmi::{Engine, Func, Instance, Memory, MemoryType as WMemoryType, Module, Store};
        thread_local!(static ARGS: RefCell<Vec<i64>> = const { RefCell::new(Vec::new()) });

        // receiver=local0, args=(local1, local2); `callmethod id=7 argc=2`.
        let ops = [
            JitOp::GetLocalValue(0),
            JitOp::GetLocalValue(1),
            JitOp::GetLocalValue(2),
            JitOp::CallMethod(7, 2, true),
            JitOp::ReturnValueBoxed,
        ];
        let m = manifest(&ops);
        assert!(m.has_call);
        let bytes = compile(&ops).expect("compiles");

        // Run once with `perr` returning `err`, asserting the result.
        let go = |err: i32| -> u64 {
            ARGS.with(|a| a.borrow_mut().clear());
            let engine = Engine::default();
            let module = Module::new(&engine, &bytes).expect("valid wasm");
            let mut store = Store::new(&engine, ());
            let memory = Memory::new(&mut store, WMemoryType::new(1, None).unwrap()).unwrap();
            memory.write(&mut store, 0, &100u64.to_le_bytes()).unwrap(); // receiver
            memory.write(&mut store, 8, &20u64.to_le_bytes()).unwrap(); // arg0
            memory.write(&mut store, 16, &3u64.to_le_bytes()).unwrap(); // arg1

            // `pca` spills top-first: [arg1, arg0].
            let pca = Func::wrap(&mut store, |v: i64| ARGS.with(|a| a.borrow_mut().push(v)));
            // `cm(receiver, id, argc)`: un-reverse the spilled args, encode
            // everything so a wrong receiver/id/argc/order fails the assert.
            let cm = Func::wrap(&mut store, |receiver: i64, id: i64, argc: i64| -> i64 {
                let mut raw = ARGS.with(|a| a.borrow_mut().clone());
                raw.reverse(); // → [arg0, arg1]
                assert_eq!((id, argc, raw.len()), (7, 2, 2));
                receiver + id + raw[0] * 10 + raw[1] // 100 + 7 + 200 + 3 = 310
            });
            let perr = Func::wrap(&mut store, move || -> i32 { err });
            // Imports in declaration order: cm, pca, perr, memory.
            let instance = Instance::new(
                &mut store,
                &module,
                &[cm.into(), pca.into(), perr.into(), memory.into()],
            )
            .expect("instantiates");
            let run = instance
                .get_typed_func::<(i32, i32, i32, i32, i32, i32), i64>(&store, "run")
                .expect("run export");
            run.call(&mut store, (0, 0, 0, 0, 0, 0)).expect("runs") as u64
        };

        assert_eq!(go(0), 310); // no error → the call result flows through
        assert_eq!(go(1), UNDEFINED_BITS); // pending error → method bails with undefined
    }

    // `callproperty` lowering: spills two args, calls `cp(receiver, k, argc)` (the
    // `k`-th multiname), returns the result, and bails on a pending error. Mirrors
    // `lowers_call_method` but through the `cp` import (import order cp, pca, perr).
    #[test]
    fn lowers_call_property() {
        use std::cell::RefCell;
        use wasmi::{Engine, Func, Instance, Memory, MemoryType as WMemoryType, Module, Store};
        thread_local!(static ARGS: RefCell<Vec<i64>> = const { RefCell::new(Vec::new()) });

        // receiver=local0, args=(local1, local2); `callproperty k=5 argc=2`.
        let ops = [
            JitOp::GetLocalValue(0),
            JitOp::GetLocalValue(1),
            JitOp::GetLocalValue(2),
            JitOp::CallProperty(5, 2, true),
            JitOp::ReturnValueBoxed,
        ];
        let m = manifest(&ops);
        assert!(m.has_callprop && !m.has_call);
        let bytes = compile(&ops).expect("compiles");

        let go = |err: i32| -> u64 {
            ARGS.with(|a| a.borrow_mut().clear());
            let engine = Engine::default();
            let module = Module::new(&engine, &bytes).expect("valid wasm");
            let mut store = Store::new(&engine, ());
            let memory = Memory::new(&mut store, WMemoryType::new(1, None).unwrap()).unwrap();
            memory.write(&mut store, 0, &100u64.to_le_bytes()).unwrap(); // receiver
            memory.write(&mut store, 8, &20u64.to_le_bytes()).unwrap(); // arg0
            memory.write(&mut store, 16, &3u64.to_le_bytes()).unwrap(); // arg1

            let pca = Func::wrap(&mut store, |v: i64| ARGS.with(|a| a.borrow_mut().push(v)));
            // `cp(receiver, k, argc)`: verify the multiname index `k`, un-reverse args.
            let cp = Func::wrap(&mut store, |receiver: i64, k: i64, argc: i64| -> i64 {
                let mut raw = ARGS.with(|a| a.borrow_mut().clone());
                raw.reverse(); // → [arg0, arg1]
                assert_eq!((k, argc, raw.len()), (5, 2, 2));
                receiver + k + raw[0] * 10 + raw[1] // 100 + 5 + 200 + 3 = 308
            });
            let perr = Func::wrap(&mut store, move || -> i32 { err });
            // Imports in declaration order: cp, pca, perr, memory.
            let instance = Instance::new(
                &mut store,
                &module,
                &[cp.into(), pca.into(), perr.into(), memory.into()],
            )
            .expect("instantiates");
            let run = instance
                .get_typed_func::<(i32, i32, i32, i32, i32, i32), i64>(&store, "run")
                .expect("run export");
            run.call(&mut store, (0, 0, 0, 0, 0, 0)).expect("runs") as u64
        };

        assert_eq!(go(0), 308); // no error → the call result flows through
        assert_eq!(go(1), UNDEFINED_BITS); // pending error → method bails with undefined
    }

    // `getpropertyfast`: `[receiver, name]` → `gpf(receiver, name, k)` (arity-3),
    // pushing the result. Net -1.
    #[test]
    fn lowers_get_property_fast() {
        use wasmi::{Engine, Func, Instance, Memory, MemoryType as WMemoryType, Module, Store};
        let ops = [
            JitOp::GetLocalValue(1), // receiver
            JitOp::GetLocalValue(2), // name
            JitOp::GetPropertyFast(5, false),
            JitOp::ReturnValueBoxed,
        ];
        let m = manifest(&ops);
        assert!(m.has_getprop_fast);
        let bytes = compile(&ops).expect("compiles");
        let engine = Engine::default();
        let module = Module::new(&engine, &bytes).expect("valid wasm");
        let mut store = Store::new(&engine, ());
        let memory = Memory::new(&mut store, WMemoryType::new(1, None).unwrap()).unwrap();
        memory.write(&mut store, 8, &100u64.to_le_bytes()).unwrap(); // local1 = receiver
        memory.write(&mut store, 16, &7u64.to_le_bytes()).unwrap(); // local2 = name
        // gpf(receiver, name, k) encodes all three so a wrong order/index fails.
        let gpf = Func::wrap(&mut store, |receiver: i64, name: i64, k: i64| -> i64 {
            receiver * 1000 + name * 10 + k
        });
        // A throwing property read propagates via `PENDING_ERROR`, so the module
        // imports `perr` and checks it after the read (no error here → 0).
        let perr = Func::wrap(&mut store, || -> i32 { 0 });
        // Imports: gpf, perr, memory.
        let instance = Instance::new(&mut store, &module, &[gpf.into(), perr.into(), memory.into()])
            .expect("instantiates");
        let run = instance
            .get_typed_func::<(i32, i32, i32, i32, i32, i32), i64>(&store, "run")
            .expect("run export");
        assert_eq!(run.call(&mut store, (0, 0, 0, 0, 0, 0)).unwrap() as u64, 100 * 1000 + 7 * 10 + 5);
    }

    // `getscriptglobals`: pushes the `k`-th script's global (pre-resolved bits) by
    // calling helper 17 (`get_script_globals`) with the immediate `k`.
    #[test]
    fn lowers_get_script_globals() {
        use wasmi::{Engine, Func, Instance, Memory, MemoryType as WMemoryType, Module, Store};
        let ops = [JitOp::GetScriptGlobals(3), JitOp::ReturnValueBoxed];
        let m = manifest(&ops);
        assert_eq!(m.num_helpers, 18); // h0..h17 (get_script_globals = 17)
        let bytes = compile(&ops).expect("compiles");
        let engine = Engine::default();
        let module = Module::new(&engine, &bytes).expect("valid wasm");
        let mut store = Store::new(&engine, ());
        let memory = Memory::new(&mut store, WMemoryType::new(1, None).unwrap()).unwrap();
        // Bind h0..h17; h17 echoes its arg so the immediate `k` flows through as the result.
        let mut externs: Vec<wasmi::Extern> = Vec::new();
        for i in 0..18u32 {
            let f = if i == 17 {
                Func::wrap(&mut store, |k: i64| -> i64 { k + 0xF00 })
            } else {
                Func::wrap(&mut store, |a: i64| -> i64 { a })
            };
            externs.push(f.into());
        }
        externs.push(memory.into());
        let instance = Instance::new(&mut store, &module, &externs).expect("instantiates");
        let run = instance
            .get_typed_func::<(i32, i32, i32, i32, i32, i32), i64>(&store, "run")
            .expect("run export");
        assert_eq!(run.call(&mut store, (0, 0, 0, 0, 0, 0)).unwrap() as u64, 3 + 0xF00);
    }

    // Exception dispatch: `throw` inside a handler range routes through `dispatch_exc`
    // (h20), which here returns the catch target op index; control jumps to the catch
    // block, whose entry pulls the caught value via `pop_caught` (h22). `new_catch`
    // (h21) builds the scope. Pins the emitted dispatch control flow end-to-end.
    #[test]
    fn lowers_exception_dispatch() {
        use wasmi::{Engine, Func, Instance, Memory, MemoryType as WMemoryType, Module, Store};
        let thrown = VALUE_INT_MARK | 55;
        let caught_marker = VALUE_INT_MARK | 999;
        let ops = [
            JitOp::PushConst(thrown), // 0: value to throw
            JitOp::Throw,             // 1: throw — in handler [0,2), dispatches
            JitOp::NewCatch(0),       // 2: catch target — caught value already materialized
            JitOp::Pop,               // 3: drop the newcatch scope
            JitOp::ReturnValueBoxed,  // 4: return the caught value
        ];
        let exceptions = [ExcRange { from: 0, to: 2, target: 2 }];
        // Handlers present ⇒ imports h0..=h22 (dispatch/new_catch/pop_caught).
        assert_eq!(manifest(&ops).num_helpers, POP_CAUGHT + 1);

        let bytes = compile_full(&ops, &[], &exceptions, &[], &[], &[], &[], 0).expect("compiles");
        let engine = Engine::default();
        let module = Module::new(&engine, &bytes).expect("valid wasm");
        let mut store = Store::new(&engine, ());
        let memory = Memory::new(&mut store, WMemoryType::new(1, None).unwrap()).unwrap();
        let mut externs: Vec<wasmi::Extern> = Vec::new();
        for i in 0..=POP_CAUGHT {
            let f = match i {
                20 => Func::wrap(&mut store, |_op: i64| -> i64 { 2 }), // dispatch → catch target op 2
                21 => Func::wrap(&mut store, |_idx: i64| -> i64 { 0 }), // new_catch → dummy scope
                22 => Func::wrap(&mut store, move |_: i64| -> i64 { caught_marker as i64 }), // pop_caught
                _ => Func::wrap(&mut store, |a: i64| -> i64 { a }),
            };
            externs.push(f.into());
        }
        externs.push(memory.into());
        let instance = Instance::new(&mut store, &module, &externs).expect("instantiates");
        let run = instance
            .get_typed_func::<(i32, i32, i32, i32, i32, i32), i64>(&store, "run")
            .expect("run export");
        assert_eq!(run.call(&mut store, (0, 0, 0, 0, 0, 0)).unwrap() as u64, caught_marker);
    }

    // Call-in-try dispatch: a `callmethod` inside a handler range. On no throw
    // (`perr`=0) the call result flows through to the normal return; on a throw
    // (`perr`=1) the pending error is dispatched to the catch block, whose entry
    // pulls the caught value. Exercises the spill/perr/dispatch/reload path.
    #[test]
    fn lowers_call_in_try_dispatch() {
        use wasmi::{Engine, Func, Instance, Memory, MemoryType as WMemoryType, Module, Store};
        let call_result = VALUE_INT_MARK | 310;
        let caught_marker = VALUE_INT_MARK | 777;
        let ops = [
            JitOp::GetLocalValue(0),    // 0: receiver
            JitOp::CallMethod(7, 0, true), // 1: call (in handler [1,2)), may throw
            JitOp::ReturnValueBoxed,    // 2: normal return (no throw) — the call result
            JitOp::NewCatch(0),         // 3: catch target — caught value materialized
            JitOp::Pop,                 // 4: drop the newcatch scope
            JitOp::ReturnValueBoxed,    // 5: return the caught value
        ];
        let exceptions = [ExcRange { from: 1, to: 2, target: 3 }];
        assert_eq!(manifest(&ops).num_helpers, POP_CAUGHT + 1); // h0..=h22

        let bytes = compile_full(&ops, &[], &exceptions, &[], &[], &[], &[], 0).expect("compiles");
        let go = |err: i32| -> u64 {
            let engine = Engine::default();
            let module = Module::new(&engine, &bytes).expect("valid wasm");
            let mut store = Store::new(&engine, ());
            let memory = Memory::new(&mut store, WMemoryType::new(1, None).unwrap()).unwrap();
            let mut externs: Vec<wasmi::Extern> = Vec::new();
            for i in 0..=POP_CAUGHT {
                let f = match i {
                    20 => Func::wrap(&mut store, |_op: i64| -> i64 { 3 }), // dispatch → target op 3
                    21 => Func::wrap(&mut store, |_idx: i64| -> i64 { 0 }), // new_catch → dummy
                    22 => Func::wrap(&mut store, move |_: i64| -> i64 { caught_marker as i64 }),
                    _ => Func::wrap(&mut store, |a: i64| -> i64 { a }),
                };
                externs.push(f.into());
            }
            // Then the call imports: cm, pca, perr, memory.
            let cm = Func::wrap(&mut store, move |_r: i64, _id: i64, _argc: i64| -> i64 {
                call_result as i64
            });
            let pca = Func::wrap(&mut store, |_v: i64| {});
            let perr = Func::wrap(&mut store, move || -> i32 { err });
            externs.push(cm.into());
            externs.push(pca.into());
            externs.push(perr.into());
            externs.push(memory.into());
            let instance = Instance::new(&mut store, &module, &externs).expect("instantiates");
            let run = instance
                .get_typed_func::<(i32, i32, i32, i32, i32, i32), i64>(&store, "run")
                .expect("run export");
            run.call(&mut store, (0, 0, 0, 0, 0, 0)).expect("runs") as u64
        };
        assert_eq!(go(0), call_result, "no throw: call result flows to the normal return");
        assert_eq!(go(1), caught_marker, "throw: dispatched to the catch block");
    }

    // Spill-across-branch: a ternary `local0 ? 111 : 222` leaves the chosen value
    // live on the operand stack across the merge block. The two predecessors spill
    // it to the pool; the merge block reloads it. Pins the spill/reload plumbing.
    #[test]
    fn lowers_spill_across_branch() {
        use wasmi::{Engine, Func, Instance, Memory, MemoryType as WMemoryType, Module, Store};
        let a = VALUE_INT_MARK | 111;
        let b = VALUE_INT_MARK | 222;
        let ops = [
            JitOp::GetLocalValue(0),  // 0: cond
            JitOp::IfFalseBoxed(4),   // 1: if !cond -> op4 (else branch)
            JitOp::PushConst(a),      // 2: then value (live across merge)
            JitOp::Jump(5),           // 3: -> merge, carrying the value
            JitOp::PushConst(b),      // 4: else value (falls through to merge)
            JitOp::ReturnValueBoxed,  // 5: merge — return the carried value
        ];
        let bytes = compile(&ops).expect("compiles"); // must NOT bail on the live stack
        let engine = Engine::default();
        let module = Module::new(&engine, &bytes).expect("valid wasm");
        let mut store = Store::new(&engine, ());
        let memory = Memory::new(&mut store, WMemoryType::new(1, None).unwrap()).unwrap();
        // h0..h5, h5 = to_boolean (used by IfFalseBoxed).
        let mut externs: Vec<wasmi::Extern> = Vec::new();
        for i in 0..6u32 {
            let f = if i == 5 {
                Func::wrap(&mut store, |v: i64| -> i64 { (v != 0) as i64 })
            } else {
                Func::wrap(&mut store, |a: i64| -> i64 { a })
            };
            externs.push(f.into());
        }
        externs.push(memory.into());
        let instance = Instance::new(&mut store, &module, &externs).expect("instantiates");
        let run = instance
            .get_typed_func::<(i32, i32, i32, i32, i32, i32), i64>(&store, "run")
            .expect("run export");
        for (cond, expected) in [(1u64, a), (0, b), (7, a)] {
            memory.write(&mut store, 0, &cond.to_le_bytes()).unwrap();
            assert_eq!(run.call(&mut store, (0, 0, 0, 0, 0, 0)).unwrap() as u64, expected, "cond={cond}");
        }
    }

    // `throw`: pops the thrown value, calls helper 19 (`throw_value`, which stashes
    // the error), and returns. Here h19 records the thrown bits and returns a marker.
    #[test]
    fn lowers_throw() {
        use std::cell::Cell;
        use wasmi::{Engine, Func, Instance, Memory, MemoryType as WMemoryType, Module, Store};
        thread_local!(static THROWN: Cell<i64> = const { Cell::new(0) });
        let ops = [JitOp::GetLocalValue(0), JitOp::Throw];
        assert_eq!(manifest(&ops).num_helpers, 20); // h0..h19 (throw_value = 19)
        let bytes = compile(&ops).expect("compiles");
        let engine = Engine::default();
        let module = Module::new(&engine, &bytes).expect("valid wasm");
        let mut store = Store::new(&engine, ());
        let memory = Memory::new(&mut store, WMemoryType::new(1, None).unwrap()).unwrap();
        memory.write(&mut store, 0, &0x1234u64.to_le_bytes()).unwrap(); // local0 = thrown value
        let mut externs: Vec<wasmi::Extern> = Vec::new();
        for i in 0..20u32 {
            let f = if i == 19 {
                Func::wrap(&mut store, |v: i64| -> i64 {
                    THROWN.with(|c| c.set(v)); // record what was thrown
                    0xE770 // stand-in for the returned `undefined` bits
                })
            } else {
                Func::wrap(&mut store, |a: i64| -> i64 { a })
            };
            externs.push(f.into());
        }
        externs.push(memory.into());
        let instance = Instance::new(&mut store, &module, &externs).expect("instantiates");
        let run = instance
            .get_typed_func::<(i32, i32, i32, i32, i32, i32), i64>(&store, "run")
            .expect("run export");
        assert_eq!(run.call(&mut store, (0, 0, 0, 0, 0, 0)).unwrap() as u64, 0xE770);
        assert_eq!(THROWN.with(|c| c.get()) as u64, 0x1234, "thrown value reached the helper");
    }

    // `pushstring`: pushes the `k`-th pre-resolved string `Value` bits by calling
    // helper 18 (`get_push_string`) with the immediate `k` (like getscriptglobals).
    #[test]
    fn lowers_push_string() {
        use wasmi::{Engine, Func, Instance, Memory, MemoryType as WMemoryType, Module, Store};
        let ops = [JitOp::PushString(2), JitOp::ReturnValueBoxed];
        let m = manifest(&ops);
        assert_eq!(m.num_helpers, 19); // h0..h18 (get_push_string = 18)
        assert!(m.has_push_strings);
        let bytes = compile(&ops).expect("compiles");
        let engine = Engine::default();
        let module = Module::new(&engine, &bytes).expect("valid wasm");
        let mut store = Store::new(&engine, ());
        let memory = Memory::new(&mut store, WMemoryType::new(1, None).unwrap()).unwrap();
        // Bind h0..h18; h18 echoes `k` (tagged) so the immediate index flows through.
        let mut externs: Vec<wasmi::Extern> = Vec::new();
        for i in 0..19u32 {
            let f = if i == 18 {
                Func::wrap(&mut store, |k: i64| -> i64 { k | 0xABC0 })
            } else {
                Func::wrap(&mut store, |a: i64| -> i64 { a })
            };
            externs.push(f.into());
        }
        externs.push(memory.into());
        let instance = Instance::new(&mut store, &module, &externs).expect("instantiates");
        let run = instance
            .get_typed_func::<(i32, i32, i32, i32, i32, i32), i64>(&store, "run")
            .expect("run export");
        assert_eq!(run.call(&mut store, (0, 0, 0, 0, 0, 0)).unwrap() as u64, 2 | 0xABC0);
    }

    // Boxed in-place int local inc/dec: `local0 = 5; ++;++;-- → 6`. The non-int
    // fallback pulls in the `increment_i`/`decrement_i` helper imports (h28/h29),
    // stubbed as identity here — the int-boxed input must never reach them.
    #[test]
    fn lowers_inc_dec_local_i() {
        use wasmi::{Engine, Func, Instance, Memory, MemoryType as WMemoryType, Module, Store};
        let ops = [
            JitOp::IncDecLocalIValue(0, true),
            JitOp::IncDecLocalIValue(0, true),
            JitOp::IncDecLocalIValue(0, false),
            JitOp::GetLocalValue(0),
            JitOp::ReturnValueBoxed,
        ];
        let m = manifest(&ops);
        assert_eq!(m.num_helpers, DECREMENT_I + 1); // h0..h29 (inc/dec_i fallbacks)
        let bytes = compile(&ops).expect("compiles");

        let engine = Engine::default();
        let module = Module::new(&engine, &bytes).expect("valid wasm");
        let mut store = Store::new(&engine, ());
        let memory = Memory::new(&mut store, WMemoryType::new(1, None).unwrap()).unwrap();
        memory.write(&mut store, 0, &(VALUE_INT_MARK | 5).to_le_bytes()).unwrap();
        let mut externs: Vec<wasmi::Extern> = Vec::new();
        for _ in 0..m.num_helpers {
            externs.push(Func::wrap(&mut store, |a: i64| -> i64 { a }).into());
        }
        externs.push(memory.into());
        let instance = Instance::new(&mut store, &module, &externs).expect("instantiates");
        let run = instance
            .get_typed_func::<(i32, i32, i32, i32, i32, i32), i64>(&store, "run")
            .expect("run export");
        assert_eq!(run.call(&mut store, (0, 0, 0, 0, 0, 0)).unwrap() as u64, VALUE_INT_MARK | 6);
    }

    // Boxed primitive constant push: `PushConst(bits)` puts exactly `bits` on the
    // stack (e.g. a `uint`/`Number`/`null`/`Boolean` `Value`), returned unchanged.
    #[test]
    fn lowers_push_const() {
        for bits in [
            UNDEFINED_BITS,
            VALUE_BOOL_MARK | 1,
            0x7FF8_0000_0000_0000,           // a Number (NaN-canonical)
            2.5f64.to_bits(),                // a Number
            VALUE_INT_MARK | (42u32 as u64), // an int
        ] {
            let ops = [JitOp::PushConst(bits), JitOp::ReturnValueBoxed];
            let bytes = compile(&ops).expect("compiles");
            assert_eq!(run(&bytes, &[]), bits, "PushConst({bits:#018x})");
        }
    }

    // `constructsuper` lowering: spills the args, calls `csup(receiver, argc)`, drops
    // the void result, and bails on a pending error. It must consume exactly
    // receiver+args, leaving a sentinel pushed *below* them untouched.
    #[test]
    fn lowers_construct_super() {
        use std::cell::RefCell;
        use wasmi::{Engine, Func, Instance, Memory, MemoryType as WMemoryType, Module, Store};
        thread_local!(static ARGS: RefCell<Vec<i64>> = const { RefCell::new(Vec::new()) });
        const SENTINEL: u64 = 0xFEED;

        // sentinel; receiver=local0; arg0=local1; constructsuper argc=1; return sentinel.
        let ops = [
            JitOp::PushConst(SENTINEL),
            JitOp::GetLocalValue(0),
            JitOp::GetLocalValue(1),
            JitOp::ConstructSuper(1),
            JitOp::ReturnValueBoxed,
        ];
        let m = manifest(&ops);
        assert!(m.has_construct_super);
        let bytes = compile(&ops).expect("compiles");

        let go = |err: i32| -> u64 {
            ARGS.with(|a| a.borrow_mut().clear());
            let engine = Engine::default();
            let module = Module::new(&engine, &bytes).expect("valid wasm");
            let mut store = Store::new(&engine, ());
            let memory = Memory::new(&mut store, WMemoryType::new(1, None).unwrap()).unwrap();
            memory.write(&mut store, 0, &100u64.to_le_bytes()).unwrap(); // receiver
            memory.write(&mut store, 8, &7u64.to_le_bytes()).unwrap(); // arg0
            let pca = Func::wrap(&mut store, |v: i64| ARGS.with(|a| a.borrow_mut().push(v)));
            let csup = Func::wrap(&mut store, |receiver: i64, argc: i64| -> i64 {
                let raw = ARGS.with(|a| a.borrow_mut().clone());
                assert_eq!((receiver, argc, raw.as_slice()), (100, 1, [7].as_slice()));
                0 // void dummy
            });
            let perr = Func::wrap(&mut store, move || -> i32 { err });
            // Imports in declaration order: csup, pca, perr, memory.
            let instance = Instance::new(
                &mut store,
                &module,
                &[csup.into(), pca.into(), perr.into(), memory.into()],
            )
            .expect("instantiates");
            let run = instance
                .get_typed_func::<(i32, i32, i32, i32, i32, i32), i64>(&store, "run")
                .expect("run export");
            run.call(&mut store, (0, 0, 0, 0, 0, 0)).expect("runs") as u64
        };

        assert_eq!(go(0), SENTINEL); // constructsuper consumed only receiver+arg → sentinel returned
        assert_eq!(go(1), UNDEFINED_BITS); // pending error → method bails
    }

    // Boxed `swap` reverses the top two `Value`s (bit-exact via reinterpret).
    #[test]
    fn lowers_boxed_swap() {
        let ops = [
            JitOp::GetLocalValue(1),
            JitOp::GetLocalValue(2),
            JitOp::SwapValue, // [l1, l2] -> [l2, l1]
            JitOp::ReturnValueBoxed, // returns the new top = l1
        ];
        let bytes = compile(&ops).expect("compiles");
        // Arbitrary bit patterns (incl. a NaN-space one) must round-trip exactly.
        assert_eq!(run(&bytes, &[0, 0xAAAA_AAAA_AAAA_AAAA, 0xFFF8_1234_5678_9ABC]), 0xAAAA_AAAA_AAAA_AAAA);
    }

    // Boxed control flow: a countdown loop through the dispatch-loop lowering with
    // a boxed `IfFalseBoxed` condition and a `Jump` back-edge.
    // `i = local1; while (to_boolean(i)) { i = decrement(i) } return increment(i)`.
    // Helpers are faked as raw-i64 arithmetic (increment/+1, decrement/-1,
    // to_boolean/!=0), so this pins the *control flow*, not the boxing: starting
    // i=5 loops to 0, then increments to 1.
    // Boxed int arithmetic (`storelocal` + unbox/op/re-box): the boxed path reads
    // int `Value`s, so `(local1 - local2) * local1`, with `storelocal` keeping the
    // difference on the stack, must round-trip through the box.
    #[test]
    fn lowers_boxed_int_arithmetic() {
        use wasmi::{Engine, Func, Instance, Memory, MemoryType as WMemoryType, Module, Store};
        let ops = [
            JitOp::GetLocalValue(1),
            JitOp::GetLocalValue(2),
            JitOp::SubtractIBoxed,   // a - b
            JitOp::StoreLocalValue(3), // local3 = a-b, keep it on the stack
            JitOp::GetLocalValue(1),
            JitOp::MultiplyIBoxed, // (a-b) * a
            JitOp::IncrementIBoxed,
            JitOp::ReturnValueBoxed,
        ];
        let m = manifest(&ops);
        assert_eq!(m.num_helpers, INCREMENT_I + 1); // h0..h28 (increment_i fallback)
        assert_eq!(m.num_helpers2, H2_MULTIPLY_I + 1); // t0..t21 (_i helper fallbacks)
        let bytes = compile(&ops).expect("compiles");

        // The h stubs are identity, the t stubs a sentinel: int-boxed and finite
        // `Number` operands must stay on the inline paths and never reach either.
        let run = |slots: &[u64]| -> u64 {
            let engine = Engine::default();
            let module = Module::new(&engine, &bytes).expect("valid wasm");
            let mut store = Store::new(&engine, ());
            let memory = Memory::new(&mut store, WMemoryType::new(1, None).unwrap()).unwrap();
            let mut buf = Vec::new();
            for s in slots {
                buf.extend_from_slice(&s.to_le_bytes());
            }
            memory.write(&mut store, 0, &buf).unwrap();
            let mut externs: Vec<wasmi::Extern> = Vec::new();
            for _ in 0..m.num_helpers {
                externs.push(Func::wrap(&mut store, |a: i64| -> i64 { a }).into());
            }
            for _ in 0..m.num_helpers2 {
                externs.push(Func::wrap(&mut store, |_a: i64, _b: i64| -> i64 { 0xDEAD }).into());
            }
            externs.push(memory.into());
            let instance = Instance::new(&mut store, &module, &externs).expect("instantiates");
            let run = instance
                .get_typed_func::<(i32, i32, i32, i32, i32, i32), i64>(&store, "run")
                .expect("run export");
            run.call(&mut store, (0, 0, 0, 0, 0, 0)).expect("runs") as u64
        };

        // a=10, b=3: (10-3)*10 + 1 = 71.
        let slots = [int_value_bits(0), int_value_bits(10), int_value_bits(3), int_value_bits(0)];
        assert_eq!(run(&slots), int_value_bits(71));
        // And local3 got the difference (7), stored mid-expression.
        let neg = [int_value_bits(0), int_value_bits(-2), int_value_bits(5), int_value_bits(0)];
        // (-2 - 5) * -2 + 1 = 15.
        assert_eq!(run(&neg), int_value_bits(15));
        // A `uint`-in-`Number`-box operand (4294967295 → ToInt32 = -1) takes the
        // inline numeric middle path: (-1 - 3) * -1 + 1 = 5.
        let unsigned = [
            int_value_bits(0),
            4294967295.0f64.to_bits(),
            int_value_bits(3),
            int_value_bits(0),
        ];
        assert_eq!(run(&unsigned), int_value_bits(5));
    }

    #[test]
    fn lowers_boxed_countdown_loop() {
        let ops = [
            JitOp::GetLocalValue(1), // 0: head — load counter
            JitOp::IfFalseBoxed(6),  // 1: if !counter -> exit(6)
            JitOp::GetLocalValue(1), // 2: body
            JitOp::CallHelper(1),    // 3: decrement
            JitOp::SetLocalValue(1), // 4: counter = counter - 1
            JitOp::Jump(0),          // 5: back-edge
            JitOp::GetLocalValue(1), // 6: exit — counter is 0
            JitOp::CallHelper(0),    // 7: increment -> 1
            JitOp::ReturnValueBoxed, // 8
        ];
        // Boxed branches pull in `to_boolean` at h5, so the module imports h0..h5.
        assert_eq!(manifest(&ops).num_helpers, 6);
        let bytes = compile(&ops).expect("compiles");

        let engine = Engine::default();
        let module = Module::new(&engine, &bytes).expect("valid wasm");
        let mut store = Store::new(&engine, ());
        let memory = Memory::new(&mut store, WMemoryType::new(1, None).unwrap()).unwrap();
        // local1 = 5 (raw i64 counter); local0 unused.
        memory
            .write(&mut store, 8, &5u64.to_le_bytes())
            .unwrap();

        // h0=increment(+1), h1=decrement(-1), h2..h4=identity, h5=to_boolean(!=0).
        let h0 = Func::wrap(&mut store, |a: i64| -> i64 { a + 1 });
        let h1 = Func::wrap(&mut store, |a: i64| -> i64 { a - 1 });
        let hid2 = Func::wrap(&mut store, |a: i64| -> i64 { a });
        let hid3 = Func::wrap(&mut store, |a: i64| -> i64 { a });
        let hid4 = Func::wrap(&mut store, |a: i64| -> i64 { a });
        let h5 = Func::wrap(&mut store, |a: i64| -> i64 { (a != 0) as i64 });

        let instance = Instance::new(
            &mut store,
            &module,
            &[
                h0.into(),
                h1.into(),
                hid2.into(),
                hid3.into(),
                hid4.into(),
                h5.into(),
                memory.into(),
            ],
        )
        .expect("instantiates");
        let run = instance
            .get_typed_func::<(i32, i32, i32, i32, i32, i32), i64>(&store, "run")
            .expect("run export");
        // 5 → (loop) → 0 → increment → 1.
        assert_eq!(run.call(&mut store, (0, 0, 0, 0, 0, 0)).expect("runs") as u64, 1);
    }

    #[test]
    fn lowers_starling_scale_setter_shape() {
        // The exact boxed shape of Starling's `DisplayObject.set scale(value)`
        // (`scaleX = value; scaleY = value` via two setter calls):
        //   getlocal1; dup; getlocal0; swap; callmethod 58,1(void);
        //   getlocal0; swap; callmethod 56,1(void); returnvoid
        // Verifies the dup/swap + arg-spill (`pca`) + `cm` machinery passes the
        // right (receiver, disp-id, arg) to each call, in order.
        use wasmi::{Caller, Engine, Func, Instance, Memory, MemoryType as WMemoryType, Module, Store};
        let ops = [
            JitOp::GetLocalValue(1),        // [v]
            JitOp::DupValue,                // [v, v]
            JitOp::GetLocalValue(0),        // [v, v, this]
            JitOp::SwapValue,               // [v, this, v]
            JitOp::CallMethod(58, 1, false), // scaleX = v → [v]
            JitOp::GetLocalValue(0),        // [v, this]
            JitOp::SwapValue,               // [this, v]
            JitOp::CallMethod(56, 1, false), // scaleY = v → [this] (dropped result)
            JitOp::ReturnVoidBoxed(UNDEFINED_BITS),
        ];
        let m = manifest(&ops);
        assert!(m.has_call);
        let bytes = compile(&ops).expect("compiles");
        let engine = Engine::default();
        let module = Module::new(&engine, &bytes).expect("valid wasm");
        // Store data: the spilled-args stack + the (receiver, id, argc, arg) call log.
        type CallLog = (Vec<i64>, Vec<(i64, i64, i64, i64)>);
        let mut store = Store::new(&engine, CallLog::default());
        let memory = Memory::new(&mut store, WMemoryType::new(1, None).unwrap()).unwrap();
        let this_bits = 0x1111_2222_3333_4444u64;
        let v_bits = 0x5555_6666_7777_8888u64;
        memory.write(&mut store, 0, &this_bits.to_le_bytes()).unwrap(); // local0
        memory.write(&mut store, 8, &v_bits.to_le_bytes()).unwrap(); // local1
        let cm = Func::wrap(
            &mut store,
            |mut c: Caller<'_, CallLog>, recv: i64, id: i64, argc: i64| -> i64 {
                let arg = c.data_mut().0.pop().expect("one spilled arg");
                c.data_mut().1.push((recv, id, argc, arg));
                0
            },
        );
        let pca = Func::wrap(&mut store, |mut c: Caller<'_, CallLog>, v: i64| {
            c.data_mut().0.push(v);
        });
        let perr = Func::wrap(&mut store, |_: Caller<'_, CallLog>| -> i32 { 0 });
        // Imports in declaration order: cm, pca, perr, memory.
        let instance = Instance::new(
            &mut store,
            &module,
            &[cm.into(), pca.into(), perr.into(), memory.into()],
        )
        .expect("instantiates");
        let run = instance
            .get_typed_func::<(i32, i32, i32, i32, i32, i32), i64>(&store, "run")
            .expect("run export");
        assert_eq!(run.call(&mut store, (0, 0, 0, 0, 0, 0)).expect("runs") as u64, UNDEFINED_BITS);
        let calls = &store.data().1;
        assert_eq!(
            calls,
            &vec![
                (this_bits as i64, 58, 1, v_bits as i64),
                (this_bits as i64, 56, 1, v_bits as i64),
            ],
            "cm must receive (this, disp-id, argc) with the value spilled as the arg"
        );
        assert!(store.data().0.is_empty(), "no leaked spilled args");
    }

    #[test]
    fn safe_vcall_kinds_are_directable() {
        // A method whose only "advanced" op is a context-only VCall kind reads no
        // RUN_CTX table, superclass, or scope → it stays a direct-call target.
        for kind in [vc::NEW_ARRAY, vc::NEW_OBJECT, vc::CONVERT_S, vc::TYPE_OF, vc::CONSTRUCT] {
            let m = manifest(&[JitOp::VCall(kind, 0, 0, true), JitOp::ReturnValueBoxed]);
            assert!(m.directable(true), "context-only vcall kind {kind} → directable");
            assert!(m.directable_decline_reason(true).is_none());
        }
        // The table / super / scope kinds still block, each with a named reason.
        for (kind, reason) in [
            (vc::CONSTRUCT_PROP, "vcall_table"),
            (vc::CALL_SUPER, "vcall_super"),
            (vc::NEW_FUNCTION, "vcall_scope"),
        ] {
            let m = manifest(&[JitOp::VCall(kind, 0, 0, true), JitOp::ReturnValueBoxed]);
            assert!(!m.directable(true), "kind {kind} must stay non-directable");
            assert_eq!(m.directable_decline_reason(true), Some(reason));
        }
    }

    #[test]
    fn lowers_vcall_kinds() {
        // Three `VCall` shapes in one method:
        //   `setproperty` (static mn): receiver + one spilled value, result dropped;
        //   `constructslot`: receiver + one spilled ctor arg, result pushed;
        //   `newarray`: NO receiver (dummy 0), one spilled element, result pushed.
        // Verifies spill order, the dummy-receiver push, imm/spill/kind marshaling,
        // and the void form's Drop.
        use wasmi::{Caller, Engine, Func, Instance, Memory, MemoryType as WMemoryType, Module, Store};
        let ops = [
            JitOp::GetLocalValue(0),                        // [this]
            JitOp::PushIntValue(7),                         // [this, 7]
            JitOp::VCall(vc::SET_PROP_STATIC, 3, 1, false), // this.mn3 = 7 → []
            JitOp::GetLocalValue(0),                        // [this]
            JitOp::PushIntValue(9),                         // [this, 9]
            JitOp::VCall(vc::CONSTRUCT_SLOT, 5, 1, true),   // new this.slot5(9) → [obj]
            JitOp::VCall(vc::NEW_ARRAY, 0, 1, true),        // [obj] → [[obj]]
            JitOp::ReturnValueBoxed,
        ];
        let m = manifest(&ops);
        assert!(m.has_vcall && m.needs_perr());
        let bytes = compile(&ops).expect("compiles");
        let engine = Engine::default();
        let module = Module::new(&engine, &bytes).expect("valid wasm");
        // Store data: the spilled-args stack + the (a, imm, spill, kind, arg) log.
        type VcLog = (Vec<i64>, Vec<(i64, i64, i64, i64, i64)>);
        let mut store = Store::new(&engine, VcLog::default());
        let memory = Memory::new(&mut store, WMemoryType::new(1, None).unwrap()).unwrap();
        let this_bits = 0x1111_2222_3333_4444u64;
        memory.write(&mut store, 0, &this_bits.to_le_bytes()).unwrap(); // local0
        let vc_mock = Func::wrap(
            &mut store,
            |mut c: Caller<'_, VcLog>, a: i64, imm: i64, spill: i64, kind: i64| -> i64 {
                let arg = c.data_mut().0.pop().expect("one spilled operand");
                c.data_mut().1.push((a, imm, spill, kind, arg));
                0x4242 // the pushed result (dropped for the void form)
            },
        );
        let pca = Func::wrap(&mut store, |mut c: Caller<'_, VcLog>, v: i64| {
            c.data_mut().0.push(v);
        });
        let perr = Func::wrap(&mut store, |_: Caller<'_, VcLog>| -> i32 { 0 });
        // Imports in declaration order: pca, perr, vc, memory.
        let instance = Instance::new(
            &mut store,
            &module,
            &[pca.into(), perr.into(), vc_mock.into(), memory.into()],
        )
        .expect("instantiates");
        let run = instance
            .get_typed_func::<(i32, i32, i32, i32, i32, i32), i64>(&store, "run")
            .expect("run export");
        assert_eq!(run.call(&mut store, (0, 0, 0, 0, 0, 0)).expect("runs"), 0x4242);
        let int7 = (VALUE_INT_MARK | 7) as i64;
        let int9 = (VALUE_INT_MARK | 9) as i64;
        let calls = &store.data().1;
        assert_eq!(
            calls,
            &vec![
                (this_bits as i64, 3, 1, vc::SET_PROP_STATIC as i64, int7),
                (this_bits as i64, 5, 1, vc::CONSTRUCT_SLOT as i64, int9),
                (0, 0, 1, vc::NEW_ARRAY as i64, 0x4242), // dummy receiver, constructslot's result spilled
            ],
            "vc must receive (a, imm, spill, kind) with operands spilled via pca"
        );
        assert!(store.data().0.is_empty(), "no leaked spilled operands");
    }

    #[test]
    fn lowers_getslot_inline() {
        // Exercises the (wasm32-gated) inline `getslot` codegen natively by forcing
        // a fake slots layout and building a mock object in memory 1:
        //   object @64: slots_ptr (u32 @ 64+8) → 128, slots_len (u32 @ 64+12) = 4
        //   slots  @128: slot[2] = 0xABCD (raw Value bits)
        // Local 0 = object-boxed 64 → the fast path loads slot 2 directly (the `gs`
        // mock must NOT run); local 1 = int-boxed → falls back to `gs`.
        use wasmi::{Caller, Engine, Func, Instance, Memory, MemoryType as WMemoryType, Module, Store};
        tests_slot_layout::force(Some((8, 12)));
        let ops = [
            JitOp::GetLocalValue(1),
            JitOp::GetSlot(3, false), // int receiver → gs fallback (result dropped)
            JitOp::Pop,
            JitOp::GetLocalValue(0),
            JitOp::GetSlot(2, false), // object receiver → inline fast path
            JitOp::ReturnValueBoxed,
        ];
        let m = manifest(&ops);
        let bytes = compile(&ops).expect("compiles");
        tests_slot_layout::force(None);
        assert!(m.has_getslot);
        let engine = Engine::default();
        let module = Module::new(&engine, &bytes).expect("valid wasm");
        // Store data: the (receiver, slot_id) pairs the `gs` fallback received.
        let mut store = Store::new(&engine, Vec::<(i64, i64)>::new());
        let memory = Memory::new(&mut store, WMemoryType::new(1, None).unwrap()).unwrap();
        let dm = Memory::new(&mut store, WMemoryType::new(1, None).unwrap()).unwrap();
        // The mock object + slots in memory 1.
        dm.write(&mut store, 64 + 8, &128u32.to_le_bytes()).unwrap(); // slots_ptr
        dm.write(&mut store, 64 + 12, &4u32.to_le_bytes()).unwrap(); // slots_len
        dm.write(&mut store, 128 + 2 * 8, &0xABCDu64.to_le_bytes()).unwrap(); // slot 2
        let obj_bits = (VALUE_OBJECT_TAG16 as u64) << 48 | 64;
        let int_bits = VALUE_INT_MARK | 7;
        memory.write(&mut store, 0, &obj_bits.to_le_bytes()).unwrap(); // local0
        memory.write(&mut store, 8, &int_bits.to_le_bytes()).unwrap(); // local1
        let gs = Func::wrap(
            &mut store,
            |mut c: Caller<'_, Vec<(i64, i64)>>, recv: i64, slot_id: i64| -> i64 {
                c.data_mut().push((recv, slot_id));
                0x7777
            },
        );
        let perr = Func::wrap(&mut store, |_: Caller<'_, Vec<(i64, i64)>>| -> i32 { 0 });
        // Imports in declaration order: gs, perr, memory, dm.
        let instance = Instance::new(
            &mut store,
            &module,
            &[gs.into(), perr.into(), memory.into(), dm.into()],
        )
        .expect("instantiates");
        let run = instance
            .get_typed_func::<(i32, i32, i32, i32, i32, i32), i64>(&store, "run")
            .expect("run export");
        // The returned value is the object receiver's slot 2, read INLINE.
        assert_eq!(run.call(&mut store, (0, 0, 0, 0, 0, 0)).expect("runs"), 0xABCD);
        assert_eq!(
            store.data().as_slice(),
            &[(int_bits as i64, 3)],
            "only the non-object receiver may reach the gs fallback"
        );
    }

    #[test]
    fn call_direct_guard_hit_writes_frame_and_call_indirects() {
        // Isolated test of the JIT→JIT direct-call codegen (argc=0). Hand-build a
        // module whose `run` inlines `emit_call_direct`; a mock callee sits in the
        // funcref table (reached by `call_indirect`) and returns the receiver bits it
        // reads back out of the frame we wrote. Drives hit / vtable-miss / sentinel.
        use wasm_encoder::{
            CodeSection, ConstExpr as CE, ElementSection, Elements, EntityType, ExportKind,
            ExportSection, Function as EncFn, FunctionSection, ImportSection, Instruction as Ins,
            MemoryType as EncMem, Module as EncModule, RefType as ERef, TableSection,
            TableType as ETable, TypeSection, ValType as VT, GlobalType as EGlobal,
        };
        use wasmi::{
            Caller, Config, Engine, Func, Global, Instance, Memory, MemoryType as WMemoryType,
            Module, Mutability, Store, Val as WVal,
        };

        const VTABLE_OFF: u32 = 16;
        const CELL: u32 = 256;
        const STRIDE_T: u32 = 64;
        const STATE_PTR_V: i32 = 0; // caller frame base → callee at +STRIDE_T = 64
        const DISP: u32 = 5;
        const CM_RET: i64 = 0xDEAD;

        // Types: 0=callee run(6×i32)->i64, 1=caller run(i64,i32,i32,i32,i32)->i64,
        //        2=cm(i64,i64,i64)->i64, 3=pca(i64)->().
        let mut types = TypeSection::new();
        types.ty().function([VT::I32; 6], [VT::I64]); // 0: standalone run
        types.ty().function([VT::I64, VT::I32, VT::I32, VT::I32, VT::I32], [VT::I64]); // 1: caller
        types.ty().function([VT::I64, VT::I64, VT::I64, VT::I32], [VT::I64]); // 2: cmi(recv,disp,argc,cell_addr)
        types.ty().function([VT::I64], []); // 3: pca
        types.ty().function([VT::I32; 7], [VT::I64]); // 4: dispatcher (validates the dead member branch)
        types.ty().function([VT::I64, VT::I64, VT::I64], [VT::I64]); // 5: cm(recv,disp,argc)
        let mut imports = ImportSection::new();
        imports.import("env", "cmi", EntityType::Function(2)); // func 0 (miss/refill)
        imports.import("env", "cm", EntityType::Function(5)); // func 1 (sentinel → plain, no refill)
        imports.import("env", "pca", EntityType::Function(3)); // func 2
        let mem = || EncMem { minimum: 1, maximum: None, memory64: false, shared: false, page_size_log2: None };
        imports.import("env", "mem0", EntityType::Memory(mem())); // memory 0 (frame)
        imports.import("env", "mem1", EntityType::Memory(mem())); // memory 1 (heap)
        // `dp` (global 0): DEPTH-cell address for the direct-call frame bump.
        imports.import("env", "dp", EntityType::Global(EGlobal { val_type: VT::I32, mutable: false, shared: false }));
        let mut functions = FunctionSection::new();
        functions.function(0); // callee = func 3
        functions.function(1); // caller = func 4
        // Table 0: one funcref slot holding the callee (table index 0).
        let mut tables = TableSection::new();
        tables.table(ETable { element_type: ERef::FUNCREF, table64: false, minimum: 1, maximum: Some(1), shared: false });
        let mut exports = ExportSection::new();
        exports.export("run", ExportKind::Func, 4);
        let mut elements = ElementSection::new();
        elements.active(Some(0), &CE::i32_const(0), Elements::Functions([3u32][..].into()));
        // Callee: `return i64.load(memory0, state_ptr + 0)` — the receiver we wrote.
        let mut callee = EncFn::new([]);
        callee.instruction(&Ins::LocalGet(0)); // state_ptr
        callee.instruction(&Ins::I64Load(MemArg { offset: 0, align: 3, memory_index: 0 }));
        callee.instruction(&Ins::End);
        // Caller: params recv(0), state_ptr(1), cell_base(2), dm_base(3), dm_len(4);
        // scratch locals 5,6 (i32), 7 (i64).
        let mut caller = EncFn::new([(2, VT::I32), (1, VT::I64)]);
        caller.instruction(&Ins::LocalGet(0)); // receiver on stack
        emit_call_direct(
            &mut caller, /*argc*/ 0, DISP, VTABLE_OFF, /*instance_class_off*/ VTABLE_OFF,
            /*cell_base_local*/ 2, /*cell_offset*/ 0,
            /*state_ptr_local*/ 1, /*dm_base*/ 3, /*dm_len*/ 4, /*run_type*/ 0,
            /*dispatch_type*/ 4, /*cmi*/ 0, /*cm*/ 1, /*pca*/ 2,
            /*sen*/ NO_SCOPE_HELPER, /*sle*/ NO_SCOPE_HELPER,
            /*sp*/ 5, /*ptr*/ 6, /*bits*/ 7, /*scope*/ 6,
            STRIDE_T, /*mem_bytes*/ 65536, /*push*/ true,
        );
        caller.instruction(&Ins::End);
        let mut code = CodeSection::new();
        code.function(&callee);
        code.function(&caller);
        let mut m = EncModule::new();
        m.section(&types);
        m.section(&imports);
        m.section(&functions);
        m.section(&tables);
        m.section(&exports);
        m.section(&elements);
        m.section(&code);
        let bytes = m.finish();

        let mut config = Config::default();
        config.wasm_multi_memory(true);
        let engine = Engine::new(&config);
        let module = Module::new(&engine, &bytes).expect("valid wasm");
        let mut store = Store::new(&engine, Vec::<(i64, i64, i64)>::new());
        let mem0 = Memory::new(&mut store, WMemoryType::new(1, None).unwrap()).unwrap();
        let mem1 = Memory::new(&mut store, WMemoryType::new(1, None).unwrap()).unwrap();
        // `cmi` (miss/refill) and `cm` (sentinel/plain) both record the generic call so
        // the assertions can tell each miss variant fired; the cell_addr arg is ignored.
        let cmi = Func::wrap(&mut store, |mut c: Caller<'_, Vec<(i64, i64, i64)>>, r: i64, d: i64, a: i64, _cell: i32| -> i64 {
            c.data_mut().push((r, d, a));
            CM_RET
        });
        let cm = Func::wrap(&mut store, |mut c: Caller<'_, Vec<(i64, i64, i64)>>, r: i64, d: i64, a: i64| -> i64 {
            c.data_mut().push((r, d, a));
            CM_RET
        });
        let pca = Func::wrap(&mut store, |_: Caller<'_, Vec<(i64, i64, i64)>>, _v: i64| {});
        // `dp` global: the memory-1 address of the DEPTH cell the direct-call
        // save/bump/restore reads/writes. A scratch offset (2048) clear of the frame
        // (0..) / object (64) / cell (256) the test uses; the bump there is inert.
        let dp = Global::new(&mut store, WVal::I32(2048), Mutability::Const);
        let instance = Instance::new(&mut store, &module, &[cmi.into(), cm.into(), pca.into(), mem0.into(), mem1.into(), dp.into()]).expect("instantiates");
        let run = instance.get_typed_func::<(i64, i32, i32, i32, i32), i64>(&store, "run").expect("run");

        // Object @64 in memory 1: vtable@80 = 0xAAAA.
        mem1.write(&mut store, 64 + VTABLE_OFF as usize, &0xAAAAu32.to_le_bytes()).unwrap();
        let obj = ((VALUE_OBJECT_TAG16 as u64) << 48 | 64) as i64;

        // 1) HIT (standalone): cell = {vtable:0xAAAA, run_index:0 (table slot),
        // ic_base:0, member_idx:NO_MEMBER}.
        mem1.write(&mut store, CELL as usize, &0xAAAAu32.to_le_bytes()).unwrap();
        mem1.write(&mut store, CELL as usize + 4, &0u32.to_le_bytes()).unwrap();
        mem1.write(&mut store, CELL as usize + 8, &0u32.to_le_bytes()).unwrap();
        mem1.write(&mut store, CELL as usize + 12, &CALL_NO_MEMBER.to_le_bytes()).unwrap();
        let got = run.call(&mut store, (obj, STATE_PTR_V, CELL as i32, 0, 0)).unwrap();
        assert_eq!(got, obj, "hit: call_indirect ran the callee, which read the receiver from the frame we wrote");
        assert!(store.data().is_empty(), "hit must NOT call cm");
        // The receiver was written into the callee frame slot 0 at state_ptr+STRIDE.
        let mut buf = [0u8; 8];
        mem0.read(&store, STATE_PTR_V as usize + STRIDE_T as usize, &mut buf).unwrap();
        assert_eq!(i64::from_le_bytes(buf), obj, "receiver written to callee frame slot 0");

        // 2) MISS (vtable mismatch) → cmi(receiver, disp, argc=0, cell_addr): call + refill.
        mem1.write(&mut store, CELL as usize, &0xBBBBu32.to_le_bytes()).unwrap();
        assert_eq!(run.call(&mut store, (obj, STATE_PTR_V, CELL as i32, 0, 0)).unwrap(), CM_RET);
        assert_eq!(store.data().as_slice(), &[(obj, DISP as i64, 0)]);

        // 3) SENTINEL (vtable matches but run_index == CALL_SENTINEL) → plain cm, NO refill.
        store.data_mut().clear();
        mem1.write(&mut store, CELL as usize, &0xAAAAu32.to_le_bytes()).unwrap();
        mem1.write(&mut store, CELL as usize + 4, &CALL_SENTINEL.to_le_bytes()).unwrap();
        assert_eq!(run.call(&mut store, (obj, STATE_PTR_V, CELL as i32, 0, 0)).unwrap(), CM_RET);
        assert_eq!(store.data().as_slice(), &[(obj, DISP as i64, 0)]);
    }

    #[test]
    fn call_direct_bumps_and_restores_depth() {
        // The DEPTH-management contract for NON-LEAF callees: around the HIT
        // `call_indirect`, `emit_call_direct` sets the `DEPTH` cell (memory 1 @ the `dp`
        // global) to `DEPTH + 1` so the callee's own re-entrant `run`s frame above it,
        // then restores it. Callee here observes the live `DEPTH` and returns it; the
        // caller checks it was `base + 1` during the call and `base` again after.
        use wasm_encoder::{
            CodeSection, ConstExpr as CE, ElementSection, Elements, EntityType, ExportKind,
            ExportSection, Function as EncFn, FunctionSection, GlobalType as EGlobal, ImportSection,
            Instruction as Ins, MemoryType as EncMem, Module as EncModule, RefType as ERef,
            TableSection, TableType as ETable, TypeSection, ValType as VT,
        };
        use wasmi::{
            Config, Engine, Func, Global, Instance, Memory, MemoryType as WMemoryType, Module,
            Mutability, Store, Val as WVal,
        };

        const VTABLE_OFF: u32 = 16;
        const CELL: u32 = 256;
        const STRIDE_T: u32 = 64;
        const STATE_PTR_V: i32 = 0;
        const DISP: u32 = 5;
        const DP_CELL: u32 = 2048; // memory-1 offset of the DEPTH cell (clear of frame/obj/cell)
        const BASE_DEPTH: i32 = 7;

        let mut types = TypeSection::new();
        types.ty().function([VT::I32; 6], [VT::I64]); // 0: standalone run
        types.ty().function([VT::I64, VT::I32, VT::I32, VT::I32, VT::I32], [VT::I64]); // 1: caller
        types.ty().function([VT::I64, VT::I64, VT::I64, VT::I32], [VT::I64]); // 2: cmi
        types.ty().function([VT::I64], []); // 3: pca
        types.ty().function([VT::I32; 7], [VT::I64]); // 4: dispatcher
        types.ty().function([VT::I64, VT::I64, VT::I64], [VT::I64]); // 5: cm
        let mut imports = ImportSection::new();
        imports.import("env", "cmi", EntityType::Function(2));
        imports.import("env", "cm", EntityType::Function(5));
        imports.import("env", "pca", EntityType::Function(3));
        let mem = || EncMem { minimum: 1, maximum: None, memory64: false, shared: false, page_size_log2: None };
        imports.import("env", "mem0", EntityType::Memory(mem()));
        imports.import("env", "mem1", EntityType::Memory(mem()));
        imports.import("env", "dp", EntityType::Global(EGlobal { val_type: VT::I32, mutable: false, shared: false }));
        let mut functions = FunctionSection::new();
        functions.function(0); // callee = func 3
        functions.function(1); // caller = func 4
        let mut tables = TableSection::new();
        tables.table(ETable { element_type: ERef::FUNCREF, table64: false, minimum: 1, maximum: Some(1), shared: false });
        let mut exports = ExportSection::new();
        exports.export("run", ExportKind::Func, 4);
        let mut elements = ElementSection::new();
        elements.active(Some(0), &CE::i32_const(0), Elements::Functions([3u32][..].into()));
        // Callee: returns the live DEPTH it observes — `i64(load mem1[dp])`.
        let mut callee = EncFn::new([]);
        callee.instruction(&Ins::GlobalGet(0)); // dp = DEPTH-cell address
        callee.instruction(&Ins::I32Load(MemArg { offset: 0, align: 2, memory_index: 1 }));
        callee.instruction(&Ins::I64ExtendI32U);
        callee.instruction(&Ins::End);
        let mut caller = EncFn::new([(2, VT::I32), (1, VT::I64)]);
        caller.instruction(&Ins::LocalGet(0)); // receiver
        emit_call_direct(
            &mut caller, /*argc*/ 0, DISP, VTABLE_OFF, /*instance_class_off*/ VTABLE_OFF,
            /*cell_base_local*/ 2, /*cell_offset*/ 0,
            /*state_ptr_local*/ 1, /*dm_base*/ 3, /*dm_len*/ 4, /*run_type*/ 0,
            /*dispatch_type*/ 4, /*cmi*/ 0, /*cm*/ 1, /*pca*/ 2,
            /*sen*/ NO_SCOPE_HELPER, /*sle*/ NO_SCOPE_HELPER,
            /*sp*/ 5, /*ptr*/ 6, /*bits*/ 7, /*scope*/ 6,
            STRIDE_T, /*mem_bytes*/ 65536, /*push*/ true,
        );
        caller.instruction(&Ins::End);
        let mut code = CodeSection::new();
        code.function(&callee);
        code.function(&caller);
        let mut m = EncModule::new();
        m.section(&types);
        m.section(&imports);
        m.section(&functions);
        m.section(&tables);
        m.section(&exports);
        m.section(&elements);
        m.section(&code);
        let bytes = m.finish();

        let mut config = Config::default();
        config.wasm_multi_memory(true);
        let engine = Engine::new(&config);
        let module = Module::new(&engine, &bytes).expect("valid wasm");
        let mut store = Store::new(&engine, ());
        let mem0 = Memory::new(&mut store, WMemoryType::new(1, None).unwrap()).unwrap();
        let mem1 = Memory::new(&mut store, WMemoryType::new(1, None).unwrap()).unwrap();
        let cmi = Func::wrap(&mut store, |_r: i64, _d: i64, _a: i64, _c: i32| -> i64 { 0 });
        let cm = Func::wrap(&mut store, |_r: i64, _d: i64, _a: i64| -> i64 { 0 });
        let pca = Func::wrap(&mut store, |_v: i64| {});
        let dp = Global::new(&mut store, WVal::I32(DP_CELL as i32), Mutability::Const);
        let instance = Instance::new(&mut store, &module, &[cmi.into(), cm.into(), pca.into(), mem0.into(), mem1.into(), dp.into()]).expect("instantiates");
        let run = instance.get_typed_func::<(i64, i32, i32, i32, i32), i64>(&store, "run").expect("run");

        // Object @64: vtable@80 = 0xAAAA; a HIT cell to the standalone callee (slot 0).
        mem1.write(&mut store, 64 + VTABLE_OFF as usize, &0xAAAAu32.to_le_bytes()).unwrap();
        let obj = ((VALUE_OBJECT_TAG16 as u64) << 48 | 64) as i64;
        mem1.write(&mut store, CELL as usize, &0xAAAAu32.to_le_bytes()).unwrap();
        mem1.write(&mut store, CELL as usize + 4, &0u32.to_le_bytes()).unwrap();
        mem1.write(&mut store, CELL as usize + 8, &0u32.to_le_bytes()).unwrap();
        mem1.write(&mut store, CELL as usize + 12, &CALL_NO_MEMBER.to_le_bytes()).unwrap();
        // Pre-seed DEPTH = BASE_DEPTH.
        mem1.write(&mut store, DP_CELL as usize, &BASE_DEPTH.to_le_bytes()).unwrap();

        let observed = run.call(&mut store, (obj, STATE_PTR_V, CELL as i32, 0, 0)).unwrap();
        assert_eq!(observed, (BASE_DEPTH + 1) as i64, "callee must observe DEPTH bumped to base+1");
        let mut buf = [0u8; 4];
        mem1.read(&store, DP_CELL as usize, &mut buf).unwrap();
        assert_eq!(i32::from_le_bytes(buf), BASE_DEPTH, "DEPTH restored to base after the call");
    }

    #[test]
    fn call_direct_guard_typed_arg_hits_only_on_matching_tag() {
        // argc=1 with an `int` param: the cell holds an int check-word pair, so the HIT
        // guard `call_indirect`s only when the arg is int-tagged; a Number arg (tag
        // mismatch) misses to the coercing `cm` (the cell stays for the next call).
        use wasm_encoder::{
            CodeSection, ConstExpr as CE, ElementSection, Elements, EntityType, ExportKind,
            ExportSection, Function as EncFn, FunctionSection, ImportSection, Instruction as Ins,
            MemoryType as EncMem, Module as EncModule, RefType as ERef, TableSection,
            TableType as ETable, TypeSection, ValType as VT, GlobalType as EGlobal,
        };
        use wasmi::{
            Caller, Config, Engine, Func, Global, Instance, Memory, MemoryType as WMemoryType,
            Module, Mutability, Store, Val as WVal,
        };

        const VTABLE_OFF: u32 = 16;
        const CELL: u32 = 256;
        const STRIDE_T: u32 = 128; // > 16-byte header + 1 arg pair (fits the callee frame)
        const DISP: u32 = 5;
        const CM_RET: i64 = 0xDEAD;

        // Types: 0=callee run(6×i32)->i64, 1=caller run(i64,i64,i32,i32)->i64,
        //        2=cmi(i64,i64,i64,i32)->i64, 3=pca(i64)->(), 4=dispatcher(7×i32)->i64,
        //        5=cm(i64,i64,i64)->i64.
        let mut types = TypeSection::new();
        types.ty().function([VT::I32; 6], [VT::I64]); // 0
        types.ty().function([VT::I64, VT::I64, VT::I32, VT::I32], [VT::I64]); // 1: caller(recv,arg0,state_ptr,cell)
        types.ty().function([VT::I64, VT::I64, VT::I64, VT::I32], [VT::I64]); // 2: cmi
        types.ty().function([VT::I64], []); // 3: pca
        types.ty().function([VT::I32; 7], [VT::I64]); // 4: dispatcher
        types.ty().function([VT::I64, VT::I64, VT::I64], [VT::I64]); // 5: cm
        let mut imports = ImportSection::new();
        imports.import("env", "cmi", EntityType::Function(2)); // func 0
        imports.import("env", "cm", EntityType::Function(5)); // func 1
        imports.import("env", "pca", EntityType::Function(3)); // func 2
        let mem = || EncMem { minimum: 1, maximum: None, memory64: false, shared: false, page_size_log2: None };
        imports.import("env", "mem0", EntityType::Memory(mem()));
        imports.import("env", "mem1", EntityType::Memory(mem()));
        // `dp` (global 0): DEPTH-cell address for the direct-call frame bump.
        imports.import("env", "dp", EntityType::Global(EGlobal { val_type: VT::I32, mutable: false, shared: false }));
        let mut functions = FunctionSection::new();
        functions.function(0); // callee = func 3
        functions.function(1); // caller = func 4
        let mut tables = TableSection::new();
        tables.table(ETable { element_type: ERef::FUNCREF, table64: false, minimum: 1, maximum: Some(1), shared: false });
        let mut exports = ExportSection::new();
        exports.export("run", ExportKind::Func, 4);
        let mut elements = ElementSection::new();
        elements.active(Some(0), &CE::i32_const(0), Elements::Functions([3u32][..].into()));
        // Callee: return `i64.load(memory0, state_ptr + 8)` — the ARG we wrote (slot 1),
        // proving the HIT ran with the arg in place.
        let mut callee = EncFn::new([]);
        callee.instruction(&Ins::LocalGet(0));
        callee.instruction(&Ins::I64Load(MemArg { offset: 8, align: 3, memory_index: 0 }));
        callee.instruction(&Ins::End);
        // Caller(recv=0, arg0=1, state_ptr=2, cell_base=3); scratch 4,5 (i32), 6 (i64).
        let mut caller = EncFn::new([(2, VT::I32), (1, VT::I64)]);
        caller.instruction(&Ins::LocalGet(0)); // receiver
        caller.instruction(&Ins::LocalGet(1)); // arg0 (on top)
        emit_call_direct(
            &mut caller, /*argc*/ 1, DISP, VTABLE_OFF, /*instance_class_off*/ VTABLE_OFF,
            /*cell_base_local*/ 3, /*cell_offset*/ 0,
            /*state_ptr_local*/ 2, /*dm_base*/ 4, /*dm_len*/ 4, /*run_type*/ 0,
            /*dispatch_type*/ 4, /*cmi*/ 0, /*cm*/ 1, /*pca*/ 2,
            /*sen*/ NO_SCOPE_HELPER, /*sle*/ NO_SCOPE_HELPER,
            /*sp*/ 4, /*ptr*/ 5, /*bits*/ 6, /*scope*/ 5,
            STRIDE_T, /*mem_bytes*/ 65536, /*push*/ true,
        );
        caller.instruction(&Ins::End);
        let mut code = CodeSection::new();
        code.function(&callee);
        code.function(&caller);
        let mut m = EncModule::new();
        m.section(&types);
        m.section(&imports);
        m.section(&functions);
        m.section(&tables);
        m.section(&exports);
        m.section(&elements);
        m.section(&code);
        let bytes = m.finish();

        let mut config = Config::default();
        config.wasm_multi_memory(true);
        let engine = Engine::new(&config);
        let module = Module::new(&engine, &bytes).expect("valid wasm");
        let mut store = Store::new(&engine, Vec::<(i64, i64, i64)>::new());
        let mem0 = Memory::new(&mut store, WMemoryType::new(1, None).unwrap()).unwrap();
        let mem1 = Memory::new(&mut store, WMemoryType::new(1, None).unwrap()).unwrap();
        let cmi = Func::wrap(&mut store, |mut c: Caller<'_, Vec<(i64, i64, i64)>>, r: i64, d: i64, a: i64, _cell: i32| -> i64 {
            c.data_mut().push((r, d, a));
            CM_RET
        });
        let cm = Func::wrap(&mut store, |mut c: Caller<'_, Vec<(i64, i64, i64)>>, r: i64, d: i64, a: i64| -> i64 {
            c.data_mut().push((r, d, a));
            CM_RET
        });
        // `pca` records the arg it spilled (so the miss path is observable).
        let pca = Func::wrap(&mut store, |_: Caller<'_, Vec<(i64, i64, i64)>>, _v: i64| {});
        // `dp` global: the memory-1 address of the DEPTH cell the direct-call
        // save/bump/restore reads/writes. A scratch offset (2048) clear of the frame
        // (0..) / object (64) / cell (256) the test uses; the bump there is inert.
        let dp = Global::new(&mut store, WVal::I32(2048), Mutability::Const);
        let instance = Instance::new(&mut store, &module, &[cmi.into(), cm.into(), pca.into(), mem0.into(), mem1.into(), dp.into()]).expect("instantiates");
        let run = instance.get_typed_func::<(i64, i64, i32, i32), i64>(&store, "run").expect("run");

        // Object @64 in memory 1, vtable@80 = 0xAAAA.
        mem1.write(&mut store, 64 + VTABLE_OFF as usize, &0xAAAAu32.to_le_bytes()).unwrap();
        let obj = ((VALUE_OBJECT_TAG16 as u64) << 48 | 64) as i64;

        // Cell: {vtable:0xAAAA, run_index:0, ic_base:0, member:NO_MEMBER} + arg0 word pair
        // for an `int` param: mask_and_want = VALUE_TAG_MASK | 1, expected = INT tag.
        mem1.write(&mut store, CELL as usize, &0xAAAAu32.to_le_bytes()).unwrap();
        mem1.write(&mut store, CELL as usize + 4, &0u32.to_le_bytes()).unwrap();
        mem1.write(&mut store, CELL as usize + 8, &0u32.to_le_bytes()).unwrap();
        mem1.write(&mut store, CELL as usize + 12, &CALL_NO_MEMBER.to_le_bytes()).unwrap();
        mem1.write(&mut store, CELL as usize + 16, &(VALUE_TAG_MASK | 1).to_le_bytes()).unwrap();
        mem1.write(&mut store, CELL as usize + 24, &VALUE_INT_MARK.to_le_bytes()).unwrap();

        // 1) HIT: int-tagged arg matches the check-word → call_indirect; callee returns
        // the arg from the frame. cm/cmi NOT called.
        let int_arg = (VALUE_INT_MARK | 7) as i64;
        assert_eq!(run.call(&mut store, (obj, int_arg, 0, CELL as i32)).unwrap(), int_arg);
        assert!(store.data().is_empty(), "matching-typed arg must HIT, not call the helper");

        // 2) MISS: a Number (f64) arg fails the int check → coercing `cm` (cell kept).
        let num_arg = 3.0f64.to_bits() as i64;
        assert_eq!(run.call(&mut store, (obj, num_arg, 0, CELL as i32)).unwrap(), CM_RET);
        assert_eq!(store.data().as_slice(), &[(obj, DISP as i64, 1)], "tag mismatch → cm");
        // The cell is untouched (no refill), so a later matching arg HITs again.
        store.data_mut().clear();
        assert_eq!(run.call(&mut store, (obj, int_arg, 0, CELL as i32)).unwrap(), int_arg);
        assert!(store.data().is_empty(), "int arg HITs again after a prior mismatch");
    }

    #[test]
    fn call_direct_guard_class_arg_exact_class_and_null() {
        // argc=1 with a concrete-CLASS param: the cell holds the exact-class marker + the
        // expected `instance_class` pointer. The HIT guard `call_indirect`s only when the
        // arg is `null` or an object whose `instance_class` == expected; a wrong-class
        // object misses to the coercing `cm`.
        use wasm_encoder::{
            CodeSection, ConstExpr as CE, ElementSection, Elements, EntityType, ExportKind,
            ExportSection, Function as EncFn, FunctionSection, ImportSection, Instruction as Ins,
            MemoryType as EncMem, Module as EncModule, RefType as ERef, TableSection,
            TableType as ETable, TypeSection, ValType as VT, GlobalType as EGlobal,
        };
        use wasmi::{
            Caller, Config, Engine, Func, Global, Instance, Memory, MemoryType as WMemoryType,
            Module, Mutability, Store, Val as WVal,
        };

        const VTABLE_OFF: u32 = 16;
        const CLASS_OFF: u32 = 20;
        const CELL: u32 = 256;
        const STRIDE_T: u32 = 128;
        const DISP: u32 = 5;
        const CM_RET: i64 = 0xDEAD;
        const WANT_CLASS: u32 = 0xC0DE; // expected instance_class pointer

        let mut types = TypeSection::new();
        types.ty().function([VT::I32; 6], [VT::I64]); // 0: callee run
        types.ty().function([VT::I64, VT::I64, VT::I32, VT::I32], [VT::I64]); // 1: caller
        types.ty().function([VT::I64, VT::I64, VT::I64, VT::I32], [VT::I64]); // 2: cmi
        types.ty().function([VT::I64], []); // 3: pca
        types.ty().function([VT::I32; 7], [VT::I64]); // 4: dispatcher
        types.ty().function([VT::I64, VT::I64, VT::I64], [VT::I64]); // 5: cm
        let mut imports = ImportSection::new();
        imports.import("env", "cmi", EntityType::Function(2)); // func 0
        imports.import("env", "cm", EntityType::Function(5)); // func 1
        imports.import("env", "pca", EntityType::Function(3)); // func 2
        let mem = || EncMem { minimum: 1, maximum: None, memory64: false, shared: false, page_size_log2: None };
        imports.import("env", "mem0", EntityType::Memory(mem()));
        imports.import("env", "mem1", EntityType::Memory(mem()));
        // `dp` (global 0): DEPTH-cell address for the direct-call frame bump.
        imports.import("env", "dp", EntityType::Global(EGlobal { val_type: VT::I32, mutable: false, shared: false }));
        let mut functions = FunctionSection::new();
        functions.function(0); // callee = func 3
        functions.function(1); // caller = func 4
        let mut tables = TableSection::new();
        tables.table(ETable { element_type: ERef::FUNCREF, table64: false, minimum: 1, maximum: Some(1), shared: false });
        let mut exports = ExportSection::new();
        exports.export("run", ExportKind::Func, 4);
        let mut elements = ElementSection::new();
        elements.active(Some(0), &CE::i32_const(0), Elements::Functions([3u32][..].into()));
        let mut callee = EncFn::new([]);
        callee.instruction(&Ins::LocalGet(0));
        callee.instruction(&Ins::I64Load(MemArg { offset: 8, align: 3, memory_index: 0 })); // arg0
        callee.instruction(&Ins::End);
        let mut caller = EncFn::new([(2, VT::I32), (1, VT::I64)]);
        caller.instruction(&Ins::LocalGet(0)); // receiver
        caller.instruction(&Ins::LocalGet(1)); // arg0
        emit_call_direct(
            &mut caller, /*argc*/ 1, DISP, VTABLE_OFF, /*instance_class_off*/ CLASS_OFF,
            /*cell_base_local*/ 3, /*cell_offset*/ 0,
            /*state_ptr_local*/ 2, /*dm_base*/ 4, /*dm_len*/ 4, /*run_type*/ 0,
            /*dispatch_type*/ 4, /*cmi*/ 0, /*cm*/ 1, /*pca*/ 2,
            /*sen*/ NO_SCOPE_HELPER, /*sle*/ NO_SCOPE_HELPER,
            /*sp*/ 4, /*ptr*/ 5, /*bits*/ 6, /*scope*/ 5,
            STRIDE_T, /*mem_bytes*/ 65536, /*push*/ true,
        );
        caller.instruction(&Ins::End);
        let mut code = CodeSection::new();
        code.function(&callee);
        code.function(&caller);
        let mut m = EncModule::new();
        m.section(&types);
        m.section(&imports);
        m.section(&functions);
        m.section(&tables);
        m.section(&exports);
        m.section(&elements);
        m.section(&code);
        let bytes = m.finish();

        let mut config = Config::default();
        config.wasm_multi_memory(true);
        let engine = Engine::new(&config);
        let module = Module::new(&engine, &bytes).expect("valid wasm");
        let mut store = Store::new(&engine, Vec::<(i64, i64, i64)>::new());
        let mem0 = Memory::new(&mut store, WMemoryType::new(1, None).unwrap()).unwrap();
        let mem1 = Memory::new(&mut store, WMemoryType::new(1, None).unwrap()).unwrap();
        let cmi = Func::wrap(&mut store, |mut c: Caller<'_, Vec<(i64, i64, i64)>>, r: i64, d: i64, a: i64, _cell: i32| -> i64 {
            c.data_mut().push((r, d, a));
            CM_RET
        });
        let cm = Func::wrap(&mut store, |mut c: Caller<'_, Vec<(i64, i64, i64)>>, r: i64, d: i64, a: i64| -> i64 {
            c.data_mut().push((r, d, a));
            CM_RET
        });
        let pca = Func::wrap(&mut store, |_: Caller<'_, Vec<(i64, i64, i64)>>, _v: i64| {});
        // `dp` global: the memory-1 address of the DEPTH cell the direct-call
        // save/bump/restore reads/writes. A scratch offset (2048) clear of the frame
        // (0..) / object (64) / cell (256) the test uses; the bump there is inert.
        let dp = Global::new(&mut store, WVal::I32(2048), Mutability::Const);
        let instance = Instance::new(&mut store, &module, &[cmi.into(), cm.into(), pca.into(), mem0.into(), mem1.into(), dp.into()]).expect("instantiates");
        let run = instance.get_typed_func::<(i64, i64, i32, i32), i64>(&store, "run").expect("run");

        // Receiver obj @64: vtable@80 = 0xAAAA (guards the receiver), instance_class@84 =
        // WANT_CLASS. A same-class arg object obj_ok @160 also has instance_class WANT_CLASS;
        // a wrong-class arg obj_bad @288 has a different instance_class.
        mem1.write(&mut store, 64 + VTABLE_OFF as usize, &0xAAAAu32.to_le_bytes()).unwrap();
        mem1.write(&mut store, 64 + CLASS_OFF as usize, &WANT_CLASS.to_le_bytes()).unwrap();
        mem1.write(&mut store, 160 + CLASS_OFF as usize, &WANT_CLASS.to_le_bytes()).unwrap();
        mem1.write(&mut store, 288 + CLASS_OFF as usize, &0xBEEFu32.to_le_bytes()).unwrap();
        let obj = ((VALUE_OBJECT_TAG16 as u64) << 48 | 64) as i64;
        let obj_ok = ((VALUE_OBJECT_TAG16 as u64) << 48 | 160) as i64;
        let obj_bad = ((VALUE_OBJECT_TAG16 as u64) << 48 | 288) as i64;

        // Cell: header + arg0 = {CALL_CLASS_MARK, expected class ptr = WANT_CLASS}.
        mem1.write(&mut store, CELL as usize, &0xAAAAu32.to_le_bytes()).unwrap();
        mem1.write(&mut store, CELL as usize + 4, &0u32.to_le_bytes()).unwrap();
        mem1.write(&mut store, CELL as usize + 8, &0u32.to_le_bytes()).unwrap();
        mem1.write(&mut store, CELL as usize + 12, &CALL_NO_MEMBER.to_le_bytes()).unwrap();
        mem1.write(&mut store, CELL as usize + 16, &CALL_CLASS_MARK.to_le_bytes()).unwrap();
        mem1.write(&mut store, CELL as usize + 24, &(WANT_CLASS as u64).to_le_bytes()).unwrap();

        // 1) HIT: exact-class arg → call_indirect; callee returns the arg.
        assert_eq!(run.call(&mut store, (obj, obj_ok, 0, CELL as i32)).unwrap(), obj_ok);
        assert!(store.data().is_empty(), "exact-class arg must HIT");

        // 2) HIT: null arg coerces identically to any class param.
        let null_arg = 0xFFF9_0000_0000_0000u64 as i64;
        assert_eq!(run.call(&mut store, (obj, null_arg, 0, CELL as i32)).unwrap(), null_arg);
        assert!(store.data().is_empty(), "null arg must HIT a class param");

        // 3) MISS: a wrong-class object → coercing cm.
        assert_eq!(run.call(&mut store, (obj, obj_bad, 0, CELL as i32)).unwrap(), CM_RET);
        assert_eq!(store.data().as_slice(), &[(obj, DISP as i64, 1)], "wrong class → cm");

        // 4) MISS: a non-object, non-null (int) arg → cm (cell kept, so 1) still HITs).
        store.data_mut().clear();
        let int_arg = (VALUE_INT_MARK | 7) as i64;
        assert_eq!(run.call(&mut store, (obj, int_arg, 0, CELL as i32)).unwrap(), CM_RET);
        assert_eq!(store.data().as_slice(), &[(obj, DISP as i64, 1)], "int for class param → cm");
        store.data_mut().clear();
        assert_eq!(run.call(&mut store, (obj, obj_ok, 0, CELL as i32)).unwrap(), obj_ok);
        assert!(store.data().is_empty(), "exact-class arg HITs again after mismatches");
    }

    #[test]
    fn directable_predicate_leaves_calls_slots_but_not_dm_or_props() {
        // Pure numeric leaf: `local1 + local2; return` → directable.
        let pure = [JitOp::GetLocal(1), JitOp::GetLocal(2), JitOp::AddI, JitOp::ReturnValue];
        assert!(manifest(&pure).directable(false), "pure numeric leaf is a direct-call target");

        // Inline domainMemory leaf (li8) IS directable now: dm reads the current
        // activation's domainMemory (context, not a per-method table) and the
        // caller-forwarded `dm_base` is the callee's under single-domain content.
        let dm = [JitOp::GetLocal(1), JitOp::DmLoad(1), JitOp::ReturnValue];
        assert!(manifest(&dm).directable(true), "dm leaf IS directable (single-domain)");

        // A call IS directable now (the non-leaf case): `emit_call_direct` bumps/restores
        // DEPTH around the direct `call_indirect`, so a callee's own re-entrant `run`s
        // frame above it. Both the helper `cm` (`CallMethod`) and a nested direct call
        // (`CallMethodDirect`) qualify.
        let calls = [JitOp::GetLocalValue(0), JitOp::CallMethod(1, 0, true), JitOp::ReturnValueBoxed];
        assert!(manifest(&calls).directable(true), "a non-leaf method that calls IS directable");
        let direct = [JitOp::GetLocalValue(0), JitOp::CallMethodDirect(1, 0, true, 0), JitOp::ReturnValueBoxed];
        assert!(manifest(&direct).directable(true), "a method with a nested direct call IS directable");

        // A compare/arithmetic helper (`t{i}`, HELPERS2) IS directable: context-only, and
        // a valueOf re-entry is frame-safe under the direct-call DEPTH bump.
        let cmp = [JitOp::GetLocalValue(0), JitOp::GetLocalValue(1), JitOp::CallHelper2(0), JitOp::ReturnValueBoxed];
        assert!(manifest(&cmp).directable(true), "an arity-2 compare helper IS directable");

        // Inline numeric coercions (context-only `coerce_i`/`coerce_s` helpers) ARE
        // directable; a safe arity-1 helper (`increment` = h0) too.
        let ci = [JitOp::GetLocalValue(0), JitOp::CoerceInt(true), JitOp::ReturnValueBoxed];
        assert!(manifest(&ci).directable(true), "inline CoerceInt IS directable");
        let cs = [JitOp::GetLocalValue(0), JitOp::CoerceString, JitOp::ReturnValueBoxed];
        assert!(manifest(&cs).directable(true), "CoerceString IS directable");
        let inc = [JitOp::GetLocalValue(0), JitOp::CallHelper(0), JitOp::ReturnValueBoxed];
        assert!(manifest(&inc).directable(true), "a context-only arity-1 helper (increment) IS directable");

        // A scope-stack helper (`getscopeobject`/`pushscope`/`popscope`) IS directable now:
        // the caller `scope_enter`/`scope_leave`s the shared activation's `scope_depth`
        // around the direct call (flagged via the cell's `ic_base` high bit) — see
        // `uses_scope`. But `getouterscope` (the bound outer chain) stays excluded.
        let scope = [JitOp::GetScopeObject(0), JitOp::ReturnValueBoxed];
        assert!(manifest(&scope).directable(true), "scope-stack op IS directable (caller retargets scope_depth)");
        assert!(manifest(&scope).uses_scope(), "getscopeobject flags uses_scope for the cell");
        let outer = [JitOp::GetOuterScope(0), JitOp::ReturnValueBoxed];
        assert!(!manifest(&outer).directable(true), "getouterscope reads the caller's bound scope → not directable");
        // `Op::Coerce{class}` IS directable now: the emitter bakes the class pointer, so
        // `coerce` reads no per-method table (only the activation as context).
        let coerce = [JitOp::GetLocalValue(0), JitOp::Coerce(0), JitOp::ReturnValueBoxed];
        assert!(manifest(&coerce).directable(true), "baked-class Coerce IS directable");

        // A `getproperty` IS directable now: the emitter bakes the multiname pointer, so
        // `gp` reads no per-method table (only the activation as context). (`getpropfast`
        // / `findproperty` stay excluded — activation-stack / scope-chain dependent.)
        let prop = [JitOp::GetLocalValue(0), JitOp::GetProperty(0), JitOp::ReturnValueBoxed];
        assert!(manifest(&prop).directable(true), "baked-multiname getproperty IS directable");

        // A getslot leaf: directable IFF all getslots are null-safe (then the
        // activation-reading slow path is dead; the fast path is activation-free).
        let slot = [JitOp::GetLocalValue(0), JitOp::GetSlot(0, false), JitOp::ReturnValueBoxed];
        assert!(manifest(&slot).directable(true), "null-safe getslot leaf IS directable");
        assert!(!manifest(&slot).directable(false), "a non-null-safe getslot may hit the throwing slow path");
    }

    #[test]
    fn getprop_ic_guard_hit_miss() {
        // Exercises the property-IC codegen in ISOLATION (before it is wired into
        // the manifest/runner machinery): hand-build a tiny module whose `run`
        // inlines `emit_getprop_ic`, drive it under wasmi with a mock `gpi` + a
        // mock object/cell in memory 1, and check all three arms.
        use wasm_encoder::{
            CodeSection, EntityType, ExportKind, ExportSection, Function as EncFn,
            FunctionSection, ImportSection, Instruction as Ins, MemoryType as EncMem,
            Module as EncModule, TypeSection, ValType as VT,
        };
        use wasmi::{
            Caller, Config, Engine, Func, Instance, Memory, MemoryType as WMemoryType, Module,
            Store,
        };

        // Mock layout (arbitrary, 4-byte aligned). Object @64, slots @128, cell @256.
        const VTABLE_OFF: u32 = 16;
        const SLOTS_PTR_OFF: u32 = 8;
        const CELL: u32 = 256;
        const K: u64 = 5;
        const SITE: u32 = 0;
        const GPI_RET: i64 = 0x9999;

        // run(recv: i64, ic_base: i32) -> i64 { <IC guard> }. gpi = func 0, run = func 1.
        let mut types = TypeSection::new();
        types.ty().function([VT::I64, VT::I64, VT::I32], [VT::I64]); // 0: gpi
        types.ty().function([VT::I64, VT::I32], [VT::I64]); // 1: run
        let mut imports = ImportSection::new();
        imports.import("env", "gpi", EntityType::Function(0));
        let mem = || EncMem {
            minimum: 1,
            maximum: None,
            memory64: false,
            shared: false,
            page_size_log2: None,
        };
        imports.import("env", "mem0", EntityType::Memory(mem()));
        imports.import("env", "mem1", EntityType::Memory(mem()));
        let mut functions = FunctionSection::new();
        functions.function(1);
        let mut exports = ExportSection::new();
        exports.export("run", ExportKind::Func, 1);
        // Params: 0 = recv (i64), 1 = ic_base (i32). Locals: 2 = i32 scratch, 3 = i64 scratch.
        let mut body = EncFn::new([(1, VT::I32), (1, VT::I64)]);
        body.instruction(&Ins::LocalGet(0));
        emit_getprop_ic(&mut body, SLOTS_PTR_OFF, VTABLE_OFF, /*ic_base_local*/ 1, K, SITE, 0, 2, 3);
        body.instruction(&Ins::End);
        let mut code = CodeSection::new();
        code.function(&body);
        let mut m = EncModule::new();
        m.section(&types);
        m.section(&imports);
        m.section(&functions);
        m.section(&exports);
        m.section(&code);
        let bytes = m.finish();

        let mut config = Config::default();
        config.wasm_multi_memory(true);
        let engine = Engine::new(&config);
        let module = Module::new(&engine, &bytes).expect("valid wasm");
        // Store data records the (recv, k, site) triples the `gpi` mock received.
        let mut store = Store::new(&engine, Vec::<(i64, i64, i32)>::new());
        let mem0 = Memory::new(&mut store, WMemoryType::new(1, None).unwrap()).unwrap();
        let mem1 = Memory::new(&mut store, WMemoryType::new(1, None).unwrap()).unwrap();
        let gpi = Func::wrap(
            &mut store,
            |mut c: Caller<'_, Vec<(i64, i64, i32)>>, recv: i64, k: i64, site: i32| -> i64 {
                c.data_mut().push((recv, k, site));
                GPI_RET
            },
        );
        let instance = Instance::new(&mut store, &module, &[gpi.into(), mem0.into(), mem1.into()])
            .expect("instantiates");
        let run = instance
            .get_typed_func::<(i64, i32), i64>(&store, "run")
            .expect("run export");

        // Mock object @64: vtable@80 = 0xAAAA, slots_ptr@72 = 128; slots@128: slot[2]@144 = 0xDEAD.
        mem1.write(&mut store, 64 + VTABLE_OFF as usize, &0xAAAAu32.to_le_bytes()).unwrap();
        mem1.write(&mut store, 64 + SLOTS_PTR_OFF as usize, &128u32.to_le_bytes()).unwrap();
        mem1.write(&mut store, 128 + 2 * 8, &0xDEADu64.to_le_bytes()).unwrap();
        let obj = ((VALUE_OBJECT_TAG16 as u64) << 48 | 64) as i64;

        // 1) HIT: cell.class == obj.vtable, cell.slot_id = 2 → inline load; gpi untouched.
        mem1.write(&mut store, CELL as usize, &0xAAAAu32.to_le_bytes()).unwrap();
        mem1.write(&mut store, CELL as usize + 4, &2u32.to_le_bytes()).unwrap();
        assert_eq!(run.call(&mut store, (obj, CELL as i32)).unwrap(), 0xDEAD, "guard hit loads the cached slot inline");
        assert!(store.data().is_empty(), "a guard hit must NOT call gpi");

        // 2) MISS (class mismatch): cell.class != obj.vtable → gpi with the cell
        // address (`ic_base + site*8` = CELL, since site 0).
        mem1.write(&mut store, CELL as usize, &0xBBBBu32.to_le_bytes()).unwrap();
        assert_eq!(run.call(&mut store, (obj, CELL as i32)).unwrap(), GPI_RET, "class miss routes to gpi");
        assert_eq!(store.data().as_slice(), &[(obj, K as i64, CELL as i32)]);

        // 3) Non-object receiver (int-boxed) → gpi (guard never loads an object).
        store.data_mut().clear();
        let int = (VALUE_INT_MARK | 7) as i64;
        assert_eq!(run.call(&mut store, (int, CELL as i32)).unwrap(), GPI_RET, "non-object routes to gpi");
        assert_eq!(store.data().as_slice(), &[(int, K as i64, CELL as i32)]);
    }

    #[test]
    fn compiles_generation_with_dispatcher() {
        // Two methods with DIFFERENT import needs in one generation module:
        //   #0: `local1 + local2` (int path — no imports)
        //   #1: `helper0(local1)` (arity-1 helper import)
        // Both dispatch through the exported 6-param `run(method_idx, ...)`.
        use wasmi::{Config, Engine, Func, Instance, Memory, MemoryType as WMemoryType, Module, Store};
        let m0 = [
            JitOp::GetLocal(1),
            JitOp::GetLocal(2),
            JitOp::AddI,
            JitOp::ReturnValue,
        ];
        let m1 = [JitOp::GetLocalValue(1), JitOp::CallHelper(0), JitOp::ReturnValueBoxed];
        let members = [
            GenMember { ops: &m0, switches: &[], exceptions: &[], name: "", numeric_seed: &[], mn_list: &[], coerce_classes: &[], param_defaults: &[], required_count: 0 },
            GenMember { ops: &m1, switches: &[], exceptions: &[], name: "", numeric_seed: &[], mn_list: &[], coerce_classes: &[], param_defaults: &[], required_count: 0 },
        ];
        let (bytes, union) = compile_generation(&members).expect("generation compiles");
        assert_eq!(union.num_helpers, 1); // the union carries m1's h0

        let mut config = Config::default();
        config.wasm_multi_memory(true);
        let engine = Engine::new(&config);
        let module = Module::new(&engine, &bytes).expect("valid wasm");
        let mut store = Store::new(&engine, ());
        let memory = Memory::new(&mut store, WMemoryType::new(1, None).unwrap()).unwrap();
        // locals: 1 = int 3, 2 = int 4 (int path reads raw low bits; boxed reads bits).
        let int_bits = |v: i32| VALUE_INT_MARK | (v as u32 as u64);
        memory.write(&mut store, 8, &int_bits(3).to_le_bytes()).unwrap();
        memory.write(&mut store, 16, &int_bits(4).to_le_bytes()).unwrap();
        let h0 = Func::wrap(&mut store, |a: i64| -> i64 { a ^ 0xFF });
        let instance = Instance::new(&mut store, &module, &[h0.into(), memory.into()])
            .expect("instantiates");
        let run = instance
            .get_typed_func::<(i32, i32, i32, i32, i32, i32, i32), i64>(&store, "run")
            .expect("dispatcher export");
        // Method 0: 3 + 4 → int-boxed 7.
        assert_eq!(run.call(&mut store, (0, 0, 0, 0, 0, 0, 0)).expect("runs") as u64, int_bits(7));
        // Method 1: h0(local1 bits) = bits ^ 0xFF.
        assert_eq!(
            run.call(&mut store, (1, 0, 0, 0, 0, 0, 0)).expect("runs") as u64,
            int_bits(3) ^ 0xFF
        );
    }

    #[test]
    fn generation_union_covers_every_import() {
        // REGRESSION: a generation whose members collectively use EVERY import
        // family must compile to a VALID module — a `Manifest` field missing from
        // the union fold once emitted member bodies calling an undeclared import
        // (`vc`), so every generation install failed, the entry-slot pool
        // exhausted, and methods silently degraded to the slow JS entry
        // (12.6s of `js-to-wasm` in a Starling profile).
        use wasmi::{Config, Engine, Module};
        let m0 = [
            JitOp::GetLocalValue(0),
            JitOp::VCall(vc::TYPE_OF, 0, 0, true), // vc + pca/perr
            JitOp::ReturnValueBoxed,
        ];
        let m1 = [
            JitOp::GetLocalValue(0),
            JitOp::GetSlot(1, false), // gs (+ perr)
            JitOp::GetLocalValue(1),
            JitOp::ArithNum(13), // t0..t13
            JitOp::Coerce(0),    // coerce
            JitOp::ReturnValueCoerced, // h16
        ];
        let m2 = [
            JitOp::GetLocalValue(0),
            JitOp::GetLocalValue(1),
            JitOp::CallMethod(3, 1, true), // cm
            JitOp::GetProperty(0),         // gp
            JitOp::ReturnValueBoxed,
        ];
        // m2's `GetProperty(0)` bakes `mn_list[0]`, m1's `Coerce(0)` bakes
        // `coerce_classes[0]` — dummy non-null pointers here (the module is only
        // VALIDATED, never run, so they're never dereferenced).
        let m2_mn = [1usize as *const ()];
        let m1_cc = [1usize as *const ()];
        let members = [
            GenMember { ops: &m0, switches: &[], exceptions: &[], name: "", numeric_seed: &[], mn_list: &[], coerce_classes: &[], param_defaults: &[], required_count: 0 },
            GenMember { ops: &m1, switches: &[], exceptions: &[], name: "", numeric_seed: &[], mn_list: &[], coerce_classes: &m1_cc, param_defaults: &[], required_count: 0 },
            GenMember { ops: &m2, switches: &[], exceptions: &[], name: "", numeric_seed: &[], mn_list: &m2_mn, coerce_classes: &[], param_defaults: &[], required_count: 0 },
        ];
        let (bytes, union) = compile_generation(&members).expect("generation compiles");
        assert!(union.has_vcall && union.has_getslot && union.has_call && union.has_coerce);
        assert_eq!(union.num_helpers2, 14);
        // wasmi's Module::new VALIDATES — an undeclared-import call fails here.
        let mut config = Config::default();
        config.wasm_multi_memory(true);
        let engine = Engine::new(&config);
        Module::new(&engine, &bytes).expect("union layout must yield a valid module");
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
