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

use ruffle_core::avm2::{Op, Value};

use crate::lower::{vc, JitOp, SwitchTable};

/// The raw NaN-boxed bits of a primitive constant-push op's `Value`, or `None` if
/// `op` isn't one of the bakeable constant pushes. All are primitives (no `Gc`), so
/// the bits are self-contained and stable — computable without an `Activation`.
fn const_push_bits(op: &Op) -> Option<u64> {
    let v: Value<'static> = match *op {
        Op::PushUint { value } => Value::from(value),
        Op::PushDouble { value } => Value::from(value),
        Op::PushNull => Value::Null,
        Op::PushTrue => Value::from(true),
        Op::PushFalse => Value::from(false),
        Op::PushUndefined => Value::Undefined,
        _ => return None,
    };
    // SAFETY: `Value` is a `#[repr(transparent)]` NaN-boxed `u64`; these variants
    // hold no `Gc` pointer, so their bits are a self-contained constant.
    Some(unsafe { std::mem::transmute::<Value<'static>, u64>(v) })
}

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
            // The raw-i32 model returns an `int`; that's only sound when the declared
            // return type is `int` (or none/`*`). A `:uint`/`:Number`/… return needs
            // coercion (`:uint` of `-10` → `4294967286`) — decline so the boxed path's
            // `ReturnValueCoerced` handles it.
            Op::ReturnValue { return_type } => match return_type {
                None => JitOp::ReturnValue,
                Some(c) if c.is_builtin_int() => JitOp::ReturnValue,
                Some(_) => return None,
            },
            // Structurally-neutral ops (in the int model): keep them 1:1 so that
            // branch targets (op indices) stay aligned.
            Op::Pop | Op::PushScope => JitOp::Pop,
            Op::Dup => JitOp::Dup,
            // Debug metadata is a no-op (see the boxed path).
            Op::Nop
            | Op::CoerceI
            | Op::Kill { .. }
            | Op::Debug { .. }
            | Op::DebugFile { .. }
            | Op::DebugLine { .. } => JitOp::Nop,
            _ => return None,
        });
    }
    Some(out)
}

/// Whether `callmethod` JITs. **OFF** — JIT-compiling call-bearing FlasCC methods
/// deterministically corrupts the C heap (`#1506` in `F_idalloc`, seen with CallMethod
/// on, gone with it off). The CallMethod *mechanism* verifies correct by inspection
/// (dispatch matches `op_call_method`, args/receiver/depth/perr all fine) and the
/// call-aware differential verify finds no per-method divergence, so the root is
/// still open — likely a runtime effect the verify can't see (e.g. an ESP-restore
/// leak around the call). Call-bearing methods decline → interpreter (game works).
/// **Diagnostic: `true`** (+ web `shared_verified()` + `INLINE_DM=false`) to run the
/// call-only call-aware verify and catch the corruptor. Ship value is `false`.
const JIT_CALLMETHOD: bool = true;

/// Whether `callmethod` is compiled behind the JIT→JIT direct-call cache
/// ([`crate::lower::emit_call_direct`]). **`true`, hardened to a provably-sound
/// subset.** The two known HIT-path corruptors are now closed by construction:
///
/// 1. **Uncoerced args.** The HIT path writes the caller's args into the callee frame
///    without the coercion `init_from_method`/`resolve_parameters` performs, so an
///    `int`→`Number` (etc.) mismatch used to hand the callee garbage bits (an early
///    `#1034`, then a `Method should exist` panic from a receiver a prior HIT had
///    corrupted). `WasmJit::direct_target` now additionally requires
///    [`Compiled::all_params_untyped`] — every param `*` (untyped), for which
///    `resolve_parameters` is a no-op, so the uncoerced write is **bit-identical** to
///    the interpreter's. Typed-param callees stay on the coercing `cm`/`cmi` helper
///    (no regression). (The earlier `argc == param_count && num_locals == argc+1`
///    default/stale-locals gate remains.)
/// 2. **One-frame overflow.** A direct call writes the callee frame at
///    `state_ptr + STRIDE` with no bounds check; from the deepest runner-checked
///    frame that lands one STRIDE past the memory. The runner now commits a one-STRIDE
///    **guard band** (`runner::web::ALLOC_PAGES`) beyond the depth-decline limit;
///    directable callees are leaves (≤1 extra frame), so this is exact.
///
/// Coverage is deliberately narrow (untyped params only) — widening to typed params
/// needs an inline per-arg tag guard in the HIT path (miss to the coercing helper on
/// mismatch) + per-callsite param-signature info; that is the next step, gated by
/// `shared_verified` diff mode. Flip to `false` to fall every call back to the helper.
const JIT_DIRECT_CALL: bool = true;

/// Whether `callproperty`/`callpropvoid` (the multiname method-call form — the hot
/// FlasCC C-call path) JIT. Only **non-lazy** multinames compile (a lazy multiname
/// pops its runtime name/namespace off the operand stack in `fill_with_runtime_params`,
/// which the spill-based arg handling doesn't model — so it declines). Mirrors the
/// interpreter's `op_call_property`/`op_call_prop_void`; a throw propagates via the
/// emitted pending-error bail. Shares the arg-spill + `perr` machinery with `callmethod`.
const JIT_CALLPROP: bool = true;

/// Whether `constructsuper` (invoke the superclass constructor — every class instance
/// initializer chains to its super ctor, so this is huge in class-heavy content like
/// Starling) JITs. Mirrors `op_construct_super` (`Activation::super_init`); shares the
/// arg-spill + `perr` machinery with the call ops. Void (pushes nothing).
const JIT_CONSTRUCT_SUPER: bool = true;

/// Whether `Op::Call` (call a function *value* on the stack — no receiver/disp-id)
/// JITs. Mirrors `op_call` (`Value::call`); shares the arg-spill + `perr` machinery.
const JIT_CALL_VALUE: bool = true;

/// Whether `findpropstrict`/`findproperty` JIT. This is the FlasCC C-call idiom
/// (`findpropstrict F_fn; …; callpropvoid F_fn`) — the single biggest blocker for
/// the hot CrossBridge/Lua methods, whose dispatch loops are saturated with it (and
/// with `getlex`, which the verifier splits into `findpropstrict` + `getproperty`).
/// Mirrors `op_find_prop_strict`/`op_find_property` via the `fp` helper; only
/// non-lazy multinames compile (a lazy one consumes runtime name/ns off the stack).
const JIT_FINDPROP: bool = true;

/// Whether `getpropertyfast` (dynamic-name property read, `arr[i]`/dictionaries) JITs.
/// Mirrors `op_get_property_fast` (index/dictionary fast path + a `fill_with_runtime_
/// params` slow fallback) via the arity-3 `gpf` helper. The lazy multiname's runtime
/// name is on the stack.
const JIT_GETPROP_FAST: bool = true;

/// Whether `lookupswitch` (a computed multi-way branch — FlasCC's C `switch`) JITs.
/// Lowered to a nested-block `br_table` over the basic-block dispatch (see
/// [`crate::lower::compile_with_switches`]); its targets live in a [`SwitchTable`]
/// side-table because [`JitOp`] is `Copy`.
const JIT_LOOKUP_SWITCH: bool = true;

/// Whether `pushstring` JITs. A string `Value` holds a `Gc` pointer (not bakeable
/// at compile time), so — like `getscriptglobals` — the bits are pre-resolved per
/// run into a side-table (see [`crate::push_string_table`]) and pushed by index.
const JIT_PUSH_STRING: bool = true;

/// Whether `throw` JITs. Stashes the thrown value as a pending error and returns;
/// `compile_method` gates it to methods with **no exception handlers** (a local
/// `catch`/`finally` would intercept the throw, which this simple form can't model).
const JIT_THROW: bool = true;

/// Whether `newcatch` + full exception dispatch JITs (try/catch/finally). A method
/// with handlers routes each throwable op (throw, throwing calls, dm-OOB) through
/// core `Activation::handle_err`; see `lower::compile_full`.
const JIT_NEWCATCH: bool = true;

/// Whether domainMemory `li*`/`si*` are **inlined** (memory-1 `i32.load`/`store`)
/// rather than helper calls. **Off** so `si*` funnels through the logging `dm_store`
/// (core write-log) for the call-only verify. Restore to `cfg!(target_arch =
/// "wasm32")` once the CallMethod corruptor is found.
const INLINE_DM: bool = cfg!(target_arch = "wasm32");

/// Confirmed **not** the 1 MB divergence root, so on.
const JIT_GENERIC_ARITH: bool = true;

/// Whether `coerce <class>` (`ToType`) is JIT'd via the `coerce` import + a per-run
/// class table. A failing coercion (`#1034`) propagates via `PENDING_ERROR` like a
/// throwing call, so it joins the `perr` bail/dispatch machinery.
const JIT_COERCE: bool = true;

/// Whether `astypelate`/`istypelate` (`value as/is Type`) JIT via the arity-2
/// helpers (a non-class type operand throws `#1041` et al. via `PENDING_ERROR`).
/// (Bisected off during the Starling-benchmark OOM hunt — proven innocent; the
/// culprit was the default-domainMemory 1 GiB `SHARED_RESERVE` promotion.)
const JIT_TYPE_LATE: bool = true;

/// Whether dynamic compares (`lessthan`/`equals`/…) are **inlined** as an `f64`
/// comparison with a numeric fast path (`CmpNum`) rather than always crossing to
/// the two-stack helper (`CallHelper2`). The inline path removes the JS-boundary
/// crossing per compare for the common numeric case (both operands `int` or
/// `Number`), falling back to the helper only for mixed/object operands.
const INLINE_CMP: bool = true;

/// Whether bitwise ops (`bitand`/`bitor`/`bitxor`/`lshift`/`rshift`/`urshift`) are
/// **inlined** as an `i32` op (`BitOpInt`) rather than always crossing to the
/// two-stack helper. Two inline tiers: int operands (their low 32 bits are already
/// `ToInt32`) and numeric `Number` operands with `|v| < 2^63` (pure-wasm wrapping
/// `ToInt32` — the hot CrossBridge case, where these ops hit `Number`-boxed
/// pointers/ints). Only strings/objects and out-of-range `Number`s reach the
/// helper. `urshift` inlines too, boxing its `uint` result as `int` (`< 2^31`) or
/// `Number` (`>= 2^31`) to match the helper.
const INLINE_BITWISE: bool = true;

/// Whether `coerce_i`/`coerce_u` are **inlined** with an int fast path (`CoerceInt`)
/// — an int passes straight through (`coerce_i`: `ToInt32` of an int is itself;
/// `coerce_u`: `ToUint32` == the int when non-negative), else the arity-1 helper.
const INLINE_COERCE: bool = true;

/// Whether generic `multiply`/`subtract`/`divide` are **inlined**. `multiply`/
/// `subtract` (`ArithInt`) take an int fast path (`i64` op, box `int` if it fits
/// else `Number`) plus a numeric middle path (f64 op → `Number`); `divide`
/// (`ArithNum`) inlines the f64 op for numeric operands (always a `Number`).
/// String/object operands (side-effecting `valueOf` coercion) fall back to the
/// two-stack helper. `modulo` stays a helper — wasm has no f64 remainder.
const INLINE_ARITH: bool = true;

/// Whether the [`JitOp::VCall`] family JITs — the generic variadic helper covering
/// the ops the decline histograms surfaced (`constructslot`/`newclass`/
/// `setproperty`/`applytype`/`callnative`/`callsuper`/`nextvalue`/`newactivation`/
/// `construct`/`newarray`/…). Each kind mirrors its interpreter `op_*` in
/// `helpers::vcall`; a throw propagates via `PENDING_ERROR` like a call.
const JIT_VCALL: bool = true;

/// Whether `add` uses the inline `ArithInt` form (int fast path + f64 middle
/// path; helper only for string/object operands). Re-enabled after the
/// vertex-corruption bisect: the culprit was the domainMemory promotion in
/// `helpers::dm_base_len` (mirror coherence), not this.
const INLINE_ADD: bool = true;

// Indices into [`crate::helpers::HELPERS`] (keep in sync with that table).
const HELPER_INCREMENT: u32 = 0;
const HELPER_DECREMENT: u32 = 1;
const HELPER_NEGATE: u32 = 2;
const HELPER_BITNOT: u32 = 3;
const HELPER_NOT: u32 = 4;
// 5 = to_boolean, 6 = push_scope, 7 = get_scope_object (called implicitly by the
// branch/scope ops in `lower`, not via `CallHelper`).
const HELPER_DM_LOAD8: u32 = 8;
const HELPER_DM_LOAD16: u32 = 9;
const HELPER_DM_LOAD32: u32 = 10;
const HELPER_COERCE_U: u32 = 11;
const HELPER_COERCE_I: u32 = 12;
const HELPER_SXI1: u32 = 13;
const HELPER_SXI8: u32 = 14;
const HELPER_SXI16: u32 = 15;

// Indices into [`crate::helpers::HELPERS2`] — the arity-2 two-stack compares,
// bitwise, and generic numeric arithmetic.
const CMP_EQ: u32 = 0;
const CMP_LT: u32 = 1;
const CMP_LE: u32 = 2;
const CMP_GT: u32 = 3;
const CMP_GE: u32 = 4;
const BIT_AND: u32 = 5;
const BIT_OR: u32 = 6;
const BIT_XOR: u32 = 7;
const LSHIFT: u32 = 8;
const RSHIFT: u32 = 9;
const URSHIFT: u32 = 10;
const MULTIPLY: u32 = 11;
const SUBTRACT: u32 = 12;
const DIVIDE: u32 = 13;
const MODULO: u32 = 14;
const STRICT_EQ: u32 = 15;
const AS_TYPE_LATE: u32 = 16;
const IS_TYPE_LATE: u32 = 17;
const ADD: u32 = 18;

// Indices into [`crate::helpers::HELPERS3`] — the arity-3 helpers (keep in sync
// with that table and `lower::HELPER3_KINDS`).
const SET_SLOT: u32 = 0;
const SET_SLOT_NO_COERCE: u32 = 1;
const SET_SLOT_COERCE_I: u32 = 2;
const DM_STORE: u32 = 3;

/// Translates to the **boxed** (GC-aware) path: raw `Value`s on the WASM stack,
/// with ops the int fast path can't do lowered to imported-helper `call`s. Used
/// when a method isn't int-sound. Handles local moves, boxed constants, unary
/// helpers, property reads/writes, and **control flow** (branches/loops via the
/// shared dispatch-loop lowering); bails on anything else (arithmetic, calls, …).
///
/// Every value it puts on the stack is a genuine `Value`, so returning any of
/// them is sound (see the `ReturnValue` arm).
pub fn translate_boxed(
    ops: &[Op],
    // Op indices whose `getslot` returns a `Number` (from the optimizer) — marks
    // those `GetSlot`s for phase-2 unboxing. Sorted; empty is fine.
    number_slots: &[u32],
) -> Option<(Vec<JitOp>, Vec<SwitchTable>)> {
    let mut out: Vec<JitOp> = Vec::with_capacity(ops.len());
    let mut switches: Vec<SwitchTable> = Vec::new();
    // Running index into the method's multiname table (one entry per getproperty
    // op, in op order). The runner rebuilds the table in the same order, so these
    // indices line up (see `WasmJit::try_run`).
    // The standard method prologue `getlocal0; pushscope` with reads only of
    // scope 0 means `getscopeobject 0` IS `getlocal0` — compile it as a plain
    // local read and skip the real scope stack entirely (both the per-pushscope
    // AND per-read helper crossings; `get_scope_object` was the hottest helper
    // in an OpenTTD gameplay profile).
    let scope_this = scope_is_this(ops);
    // Whether the method reads OR captures scopes: only then does `pushscope`
    // need to touch the real scope stack (otherwise it just discards, as before).
    // `newclass` *captures* the current scope chain (`ClassObject::from_class` →
    // `create_scopechain`) as the class's outer scope — every method of the new
    // class resolves outer scopes through it. Discarding the pushes there compiled
    // a wrong (shorter) class scope into being: the class's methods then read the
    // wrong outer scope and e.g. `getglobalslot` hit the class object instead of
    // the global (`#1026 Slot N exceeds slotCount=0 of Test$`). `NewClass` and
    // `NewFunction` are the scope-capturing ops that compile (`callstatic`/
    // `findprop*` decline); the capture needs the real stack even when
    // `scope_this` holds.
    let captures_scope = ops
        .iter()
        .any(|op| matches!(op, Op::NewClass { .. } | Op::NewFunction { .. }));
    let needs_scope = captures_scope
        || (!scope_this && ops.iter().any(|op| matches!(op, Op::GetScopeObject { .. })));
    let mut next_mn: u32 = 0;
    let mut next_script: u32 = 0;
    let mut next_string: u32 = 0;
    let mut next_coerce: u32 = 0;
    let mut next_native: u32 = 0;
    let mut next_ns: u32 = 0;
    let mut next_ic_site: u32 = 0;
    let mut next_call_site: u32 = 0;
    for (i, op) in ops.iter().enumerate() {
        let returns_number = number_slots.binary_search(&(i as u32)).is_ok();
        out.push(boxed_op(
            op,
            &mut next_mn,
            &mut next_script,
            &mut next_string,
            &mut next_coerce,
            &mut next_native,
            &mut next_ns,
            &mut next_ic_site,
            &mut next_call_site,
            needs_scope,
            scope_this,
            &mut switches,
            returns_number,
        )?);
    }
    Some((out, switches))
}

/// Whether the method's whole scope usage is the standard prologue
/// `getlocal0; pushscope` — the single push, never popped, only scope index 0
/// ever read, and local 0 never reassigned. Then every `getscopeobject 0`
/// provably yields `local0` and the real scope stack can be skipped.
fn scope_is_this(ops: &[Op]) -> bool {
    if !matches!(ops, [Op::GetLocal { index: 0 }, Op::PushScope, ..]) {
        return false;
    }
    let single_push = ops.iter().filter(|o| matches!(o, Op::PushScope)).count() == 1;
    single_push
        && !ops.iter().any(|o| {
            matches!(
                o,
                Op::PushWith
                    | Op::PopScope
                    | Op::GetScopeObject { index: 1.. }
                    | Op::SetLocal { index: 0 }
                    | Op::StoreLocal { index: 0 }
                    | Op::Kill { index: 0 }
            )
        })
}

/// Maps a single core `Op` to its boxed [`JitOp`], or `None` if unsupported.
/// `next_mn` is the running multiname-table index (bumped by property reads);
/// `needs_scope` makes `pushscope` push the real scope stack (vs discard). The
/// single source of truth for what the boxed path supports — [`translate_boxed`]
/// and the [`first_unsupported_boxed`] diagnostic both go through it.
#[allow(clippy::too_many_arguments)]
fn boxed_op(
    op: &Op,
    next_mn: &mut u32,
    next_script: &mut u32,
    next_string: &mut u32,
    next_coerce: &mut u32,
    next_native: &mut u32,
    next_ns: &mut u32,
    next_ic_site: &mut u32,
    next_call_site: &mut u32,
    needs_scope: bool,
    scope_this: bool,
    switches: &mut Vec<SwitchTable>,
    // Whether THIS op is a `getslot` the optimizer proved returns a `Number`.
    returns_number: bool,
) -> Option<JitOp> {
    Some(match *op {
        // See [`scope_is_this`]: the only scope entry is the prologue-pushed
        // `this`, so the read is a plain local-0 load — no helper crossing.
        Op::GetScopeObject { index: 0 } if scope_this => JitOp::GetLocalValue(0),
        Op::GetLocal { index } => JitOp::GetLocalValue(index),
        Op::SetLocal { index } => JitOp::SetLocalValue(index),
        Op::Increment => JitOp::CallHelper(HELPER_INCREMENT),
        Op::Decrement => JitOp::CallHelper(HELPER_DECREMENT),
        Op::Negate => JitOp::CallHelper(HELPER_NEGATE),
        Op::BitNot => JitOp::CallHelper(HELPER_BITNOT),
        Op::Not => JitOp::CallHelper(HELPER_NOT),
        // Verifier `setlocal i; getlocal i` fusion: store but keep on the stack.
        Op::StoreLocal { index } => JitOp::StoreLocalValue(index),
        // Int arithmetic (verifier-proven int operands): unbox/op/re-box inline.
        Op::AddI => JitOp::AddIBoxed,
        Op::SubtractI => JitOp::SubtractIBoxed,
        Op::MultiplyI => JitOp::MultiplyIBoxed,
        Op::IncrementI => JitOp::IncrementIBoxed,
        Op::DecrementI => JitOp::DecrementIBoxed,
        // domainMemory (FlasCC "RAM", the hottest FlasCC op). On **web** it's
        // inlined (`i32.load`/`store` on memory 1 = Ruffle's own linear memory — no
        // JS-boundary helper crossing). On **native** it stays a helper: memory 1
        // would be a wasmi mock disconnected from the real (native-RAM) domain
        // memory, so the real-interpreter differential test can't compare — the
        // inline codegen is covered by `lower::tests::lowers_dm_inline` instead.
        Op::Li8 if INLINE_DM => JitOp::DmLoad(1),
        Op::Li16 if INLINE_DM => JitOp::DmLoad(2),
        Op::Li32 if INLINE_DM => JitOp::DmLoad(4),
        Op::Si8 if INLINE_DM => JitOp::DmStore(1),
        Op::Si16 if INLINE_DM => JitOp::DmStore(2),
        Op::Si32 if INLINE_DM => JitOp::DmStore(4),
        // domainMemory float load/store (`lf32`/`lf64`/`sf32`/`sf64`) — inline
        // `f32`/`f64` access on memory 1 (web only; declines on native, like the int
        // dm's inline path, covered by `lower::tests::lowers_dm_float_inline`).
        Op::Lf32 if INLINE_DM => JitOp::DmLoadF(4),
        Op::Lf64 if INLINE_DM => JitOp::DmLoadF(8),
        Op::Sf32 if INLINE_DM => JitOp::DmStoreF(4),
        Op::Sf64 if INLINE_DM => JitOp::DmStoreF(8),
        Op::Li8 => JitOp::CallHelper(HELPER_DM_LOAD8),
        Op::Li16 => JitOp::CallHelper(HELPER_DM_LOAD16),
        Op::Li32 => JitOp::CallHelper(HELPER_DM_LOAD32),
        Op::Si8 => JitOp::CallHelper3(DM_STORE, 1),
        Op::Si16 => JitOp::CallHelper3(DM_STORE, 2),
        Op::Si32 => JitOp::CallHelper3(DM_STORE, 4),
        // Stack swap + int/uint coercion + sign-extend.
        Op::Swap => JitOp::SwapValue,
        Op::CoerceU if INLINE_COERCE => JitOp::CoerceInt(false),
        Op::CoerceI if INLINE_COERCE => JitOp::CoerceInt(true),
        Op::CoerceU => JitOp::CallHelper(HELPER_COERCE_U),
        Op::CoerceI => JitOp::CallHelper(HELPER_COERCE_I),
        // `coerceb` — `ToBoolean` boxed as a `Boolean` `Value` (inline `to_boolean`).
        Op::CoerceB => JitOp::CoerceBool,
        // `coerces` — `ToString` (a throwing `toString` bails via `PENDING_ERROR`).
        Op::CoerceS => JitOp::CoerceString,
        // `coerce <class>` — `ToType`. `k` indexes the per-run class table (op order,
        // matching `coerce_class_table`); a failing coercion bails via `PENDING_ERROR`.
        Op::Coerce { .. } if JIT_COERCE => {
            let k = *next_coerce;
            *next_coerce += 1;
            JitOp::Coerce(k)
        }
        // `strictequals` / `astypelate` / `istypelate` — arity-2 two-stack helpers
        // (`===`, `value as Type`, `value is Type`). The type ops **throw** on a
        // non-class type operand (`#1041` et al.) via `PENDING_ERROR` + a post-op
        // `perr` bail/dispatch, exactly like the interpreter — games do hit this
        // (e.g. `x is someVar` with a non-class `someVar`) and catch the error.
        Op::StrictEquals => JitOp::CallHelper2(STRICT_EQ),
        Op::AsTypeLate if JIT_TYPE_LATE => JitOp::CallHelper2(AS_TYPE_LATE),
        Op::IsTypeLate if JIT_TYPE_LATE => JitOp::CallHelper2(IS_TYPE_LATE),
        Op::Sxi1 => JitOp::CallHelper(HELPER_SXI1),
        Op::Sxi8 => JitOp::CallHelper(HELPER_SXI8),
        Op::Sxi16 => JitOp::CallHelper(HELPER_SXI16),
        // Generic (untyped) numeric arithmetic → arity-2 two-stack helpers that
        // mirror the interpreter's `op_*` (int fast paths + coercion order). `Add`
        // is deliberately excluded — it may be string concat, not numeric.
        Op::Multiply if INLINE_ARITH => JitOp::ArithInt(MULTIPLY),
        Op::Subtract { .. } if INLINE_ARITH => JitOp::ArithInt(SUBTRACT),
        Op::Divide if INLINE_ARITH => JitOp::ArithNum(DIVIDE),
        Op::Multiply if JIT_GENERIC_ARITH => JitOp::CallHelper2(MULTIPLY),
        Op::Subtract { .. } if JIT_GENERIC_ARITH => JitOp::CallHelper2(SUBTRACT),
        Op::Divide if JIT_GENERIC_ARITH => JitOp::CallHelper2(DIVIDE),
        Op::Modulo if JIT_GENERIC_ARITH => JitOp::CallHelper2(MODULO),
        // `add` — numeric add or string concat (`Value::add`). The inline paths
        // fire only for NUMERIC operands (int+int → checked int, else f64 add →
        // Number — exactly `Value::add`'s numeric arms); a string/object operand
        // falls back to the generic helper, which handles concat/`valueOf`.
        // NOTE: the ArithInt(ADD) inline form is gated separately (`INLINE_ADD`,
        // OFF) while bisecting a Starling vertex-corruption regression.
        Op::Add { .. } if INLINE_ADD => JitOp::ArithInt(ADD),
        Op::Add { .. } if JIT_GENERIC_ARITH => JitOp::CallHelper2(ADD),
        // Dynamic comparisons → a Boolean `Value`. Inlined as an `f64` compare with
        // a numeric fast path (`CmpNum`) when enabled, else the two-stack helper.
        Op::Equals if INLINE_CMP => JitOp::CmpNum(CMP_EQ),
        Op::LessThan if INLINE_CMP => JitOp::CmpNum(CMP_LT),
        Op::LessEquals if INLINE_CMP => JitOp::CmpNum(CMP_LE),
        Op::GreaterThan if INLINE_CMP => JitOp::CmpNum(CMP_GT),
        Op::GreaterEquals if INLINE_CMP => JitOp::CmpNum(CMP_GE),
        Op::Equals => JitOp::CallHelper2(CMP_EQ),
        Op::LessThan => JitOp::CallHelper2(CMP_LT),
        Op::LessEquals => JitOp::CallHelper2(CMP_LE),
        Op::GreaterThan => JitOp::CallHelper2(CMP_GT),
        Op::GreaterEquals => JitOp::CallHelper2(CMP_GE),
        // Bitwise (operands coerced to i32/u32; `<<`/`>>`/`>>>` masked by 0x1F).
        // Inlined as an `i32` op with int and numeric-`Number` fast paths when
        // enabled; `urshift` inlines too, boxing its `uint` result as `int` or
        // `Number` (see `BitOpInt` lowering).
        Op::BitAnd if INLINE_BITWISE => JitOp::BitOpInt(BIT_AND),
        Op::BitOr if INLINE_BITWISE => JitOp::BitOpInt(BIT_OR),
        Op::BitXor if INLINE_BITWISE => JitOp::BitOpInt(BIT_XOR),
        Op::LShift if INLINE_BITWISE => JitOp::BitOpInt(LSHIFT),
        Op::RShift if INLINE_BITWISE => JitOp::BitOpInt(RSHIFT),
        Op::URShift if INLINE_BITWISE => JitOp::BitOpInt(URSHIFT),
        Op::BitAnd => JitOp::CallHelper2(BIT_AND),
        Op::BitOr => JitOp::CallHelper2(BIT_OR),
        Op::BitXor => JitOp::CallHelper2(BIT_XOR),
        Op::LShift => JitOp::CallHelper2(LSHIFT),
        Op::RShift => JitOp::CallHelper2(RSHIFT),
        Op::URShift => JitOp::CallHelper2(URSHIFT),
        // Static property reads carry a resolved (non-lazy) multiname — the
        // verifier only emits `GetPropertyStatic` for those. `Slow` reads are the
        // opposite: *always* lazy (a runtime namespace, or a runtime name that
        // isn't a valid dynamic name), so the interpreter pops the runtime
        // params off the operand stack in `fill_with_runtime_params` — which the
        // `GetProperty(k)` helper doesn't model. They decline (below).
        Op::GetPropertyStatic { .. } => {
            let k = *next_mn;
            *next_mn += 1;
            // Behind a monomorphic inline cache where the object layout is known
            // (web); otherwise the generic `gp` helper path. Same semantics; the IC
            // only adds an inline vtable guard + cached slot load on hits.
            if crate::lower::ic_available() {
                let site = *next_ic_site;
                *next_ic_site += 1;
                JitOp::GetPropertyIc(k, site)
            } else {
                JitOp::GetProperty(k)
            }
        }
        Op::GetPropertySlow { .. } => return None,
        // `findpropstrict`/`findproperty` — resolve a multiname through the scope
        // chain (the FlasCC C-call idiom). `k` indexes the run's multiname table,
        // in op order (matching `lib::multiname_table`). Only non-lazy multinames
        // compile; a lazy one pops runtime name/ns off the stack, which the `fp`
        // helper doesn't model. `strict` → throws `#1065` on a miss.
        Op::FindPropStrict { multiname } if JIT_FINDPROP => {
            if multiname.has_lazy_component() {
                return None;
            }
            let k = *next_mn;
            *next_mn += 1;
            JitOp::FindProp(k, true)
        }
        Op::FindProperty { multiname } if JIT_FINDPROP => {
            if multiname.has_lazy_component() {
                return None;
            }
            let k = *next_mn;
            *next_mn += 1;
            JitOp::FindProp(k, false)
        }
        // Dynamic-name read (`arr[i]`): the lazy multiname `k` (name off the stack) —
        // the `get_property_fast` helper does the fast index/dictionary path or fills
        // the multiname. Bumps the same multiname counter as the static reads.
        Op::GetPropertyFast { .. } if JIT_GETPROP_FAST => {
            let k = *next_mn;
            *next_mn += 1;
            JitOp::GetPropertyFast(k, returns_number)
        }
        // The verifier's resolved form of a typed property read: a direct slot
        // fetch (no multiname). One of the hottest AVM2 ops.
        Op::GetSlot { index } => JitOp::GetSlot(index as u32, returns_number),
        // Boxed constant push (verifier normalizes pushbyte/pushshort → PushInt).
        Op::PushInt { value } => JitOp::PushIntValue(value),
        // Other primitive constant pushes — bake the `Value` bits at translate time
        // (no `Gc`, so it's self-contained). `pushstring`/`pushnamespace` carry a
        // `Gc`/`Namespace` and are excluded (would need a runtime object).
        Op::PushUint { .. }
        | Op::PushDouble { .. }
        | Op::PushNull
        | Op::PushTrue
        | Op::PushFalse
        | Op::PushUndefined => JitOp::PushConst(const_push_bits(op)?),
        // `pushstring` — the string `Value` holds a `Gc`, so it can't be baked like
        // the constants above. `k` indexes the per-run string table (op order,
        // matching `push_string_table`); the `get_push_string` helper reads it.
        Op::PushString { .. } if JIT_PUSH_STRING => {
            let k = *next_string;
            *next_string += 1;
            JitOp::PushString(k)
        }
        // `throw` — pop a value, raise it as an error. Only sound when the method
        // has no exception handler (checked in `compile_method`, which has the
        // method's `exceptions` table); a handler-free throw just propagates out.
        Op::Throw if JIT_THROW => JitOp::Throw,
        // `newcatch` — build the catch/finally scope object for handler `index`.
        Op::NewCatch { index } if JIT_NEWCATCH => JitOp::NewCatch(index as u32),
        // Slot writes (the resolved form of a typed property write): pop
        // receiver+value, store into the slot via an arity-3 helper.
        Op::SetSlot { index } => JitOp::CallHelper3(SET_SLOT, index as u32),
        Op::SetSlotNoCoerce { index } => JitOp::CallHelper3(SET_SLOT_NO_COERCE, index as u32),
        Op::SetSlotCoerceI { index } => JitOp::CallHelper3(SET_SLOT_COERCE_I, index as u32),
        // Control flow. Branch offsets are op indices (verifier-resolved), so
        // they map straight across. `IfTrue`/`IfFalse` coerce the popped
        // `Value` to a condition via the `to_boolean` helper (in `lower`).
        Op::Jump { offset } => JitOp::Jump(offset),
        Op::IfTrue { offset } => JitOp::IfTrueBoxed(offset),
        Op::IfFalse { offset } => JitOp::IfFalseBoxed(offset),
        // `lookupswitch` — pop an int selector, branch to `cases[selector]` (op
        // index) or `default`. Offsets are verifier-resolved op indices, so they
        // map straight across. The targets can't fit a `Copy` `JitOp`, so they go
        // into the `switches` side-table and the op carries its index.
        Op::LookupSwitch(ls) if JIT_LOOKUP_SWITCH => {
            let idx = switches.len() as u32;
            switches.push(SwitchTable {
                default: ls.default_offset.get(),
                cases: ls.case_offsets.iter().map(|c| c.get()).collect(),
            });
            JitOp::LookupSwitch(idx)
        }
        // Every value on the boxed stack is a genuine `Value` (a local read, a
        // constant, or a helper result), so returning any of them is sound:
        // `value_from_bits` reconstructs a real (never fabricated) pointer, no
        // collection happens mid-run, and a by-value return stores into no
        // `Gc` (needs no write barrier). See `helpers::get_property`.
        // A declared (non-`*`) `return_type` needs runtime coercion (`:uint` of a
        // negative int, `:Vector.<T>` of a generic vector, …); `None`/`*` returns raw.
        Op::ReturnValue {
            return_type: Some(_),
        } => JitOp::ReturnValueCoerced,
        Op::ReturnValue { .. } => JitOp::ReturnValueBoxed,
        // `returnvoid`: return the value the declared `return_type` coerces to (void/
        // none → `undefined`, numeric → `0`, boolean → `false`, else `null`).
        Op::ReturnVoid { return_type } => {
            JitOp::ReturnVoidBoxed(crate::return_void_bits(return_type))
        }
        Op::Dup => JitOp::DupValue,
        // Scope access. `getscopeobject` reads the method's own local scopes, so
        // `pushscope` must push the real scope stack when scopes are read; else it
        // just discards (the scope object is unused).
        Op::GetScopeObject { index } => JitOp::GetScopeObject(index as u32),
        // `getouterscope` — a captured/outer scope by index (closures read these).
        Op::GetOuterScope { index } => JitOp::GetOuterScope(index as u32),
        // Boxed in-place int local increment/decrement (verifier-proven int local).
        Op::IncLocalI { index } => JitOp::IncDecLocalIValue(index, true),
        Op::DecLocalI { index } => JitOp::IncDecLocalIValue(index, false),
        // `inclocal`/`declocal` — the *Number* forms: `local = ToNumber(local) ± 1`
        // via the `increment`/`decrement` helper (their throwing-coercion caveat
        // applies — swallowed, like the stack `increment`/`decrement` ops).
        Op::IncLocal { index } => JitOp::IncDecLocalNum(index, true),
        Op::DecLocal { index } => JitOp::IncDecLocalNum(index, false),
        // `hasnext2` — the `for..in`/`for each` enumeration step: reads AND
        // writes two register operands via the `hasnext2` helper + fetchers,
        // pushes the loop-continue Boolean. Throwing (perr).
        Op::HasNext2 { object_register, index_register } => {
            JitOp::HasNext2(object_register, index_register)
        }
        // `getscriptglobals` — the script's global object. `k` indexes the per-run
        // pre-resolved script-globals table (in op order), built in `try_run`.
        Op::GetScriptGlobals { .. } => {
            let k = *next_script;
            *next_script += 1;
            JitOp::GetScriptGlobals(k)
        }
        // `callmethod` — a resolved (disp-id) method call. Variadic args are
        // spilled; a thrown error propagates via the emitted pending-error check.
        // `JIT_CALLMETHOD` gates it OFF for a bisect: does JIT'ing the C allocator's
        // call-bearing methods cause the `#1506` divergence? (Flip to `true` to
        // re-enable once that's resolved.)
        Op::CallMethod {
            index,
            num_args,
            push_return_value,
        } if JIT_CALLMETHOD => {
            // Behind the JIT→JIT direct-call cache where the inline object layout +
            // shared table are available (web); else the generic `cm` helper. Same
            // semantics; the cache adds an inline vtable guard + `call_indirect` on a
            // hit to a directable callee.
            if JIT_DIRECT_CALL && crate::lower::ic_available() {
                let site = *next_call_site;
                *next_call_site += 1;
                JitOp::CallMethodDirect(index as u32, num_args, push_return_value, site)
            } else {
                JitOp::CallMethod(index as u32, num_args, push_return_value)
            }
        }
        // `callproperty`/`callpropvoid` — a multiname method call (the hot FlasCC
        // C-call path). Only non-lazy multinames compile (a lazy one consumes runtime
        // name/ns off the stack, which the spill doesn't model). `k` indexes the run's
        // multiname table, in op order (matching `lib::multiname_table`); pushes the
        // result for `callproperty`, discards it for `callpropvoid`.
        Op::CallProperty { multiname, num_args } if JIT_CALLPROP => {
            if multiname.has_lazy_component() {
                return None;
            }
            let k = *next_mn;
            *next_mn += 1;
            JitOp::CallProperty(k, num_args, true)
        }
        Op::CallPropVoid { multiname, num_args } if JIT_CALLPROP => {
            if multiname.has_lazy_component() {
                return None;
            }
            let k = *next_mn;
            *next_mn += 1;
            JitOp::CallProperty(k, num_args, false)
        }
        // `constructsuper` — invoke the superclass constructor on the receiver (the
        // super_init helper reads the activation's bound superclass). Void.
        Op::ConstructSuper { num_args } if JIT_CONSTRUCT_SUPER => JitOp::ConstructSuper(num_args),
        // `call` — invoke a function value: stack `[.., function, receiver, args…]`.
        Op::Call { num_args } if JIT_CALL_VALUE => JitOp::CallValue(num_args),
        // --- The generic variadic-helper family (`vc` import; see `lower::vc`).
        // Each kind mirrors its interpreter `op_*` in `helpers::vcall`; extra stack
        // operands spill through `pca` like call args, throws propagate via
        // `PENDING_ERROR` + the shared perr bail/dispatch. Multiname forms compile
        // only non-lazy — except the `*Fast` forms, whose lazy runtime *name* is a
        // stack operand handled explicitly (like `getpropertyfast`).
        // `constructslot` — construct the class held in the receiver's slot.
        Op::ConstructSlot { index, num_args } if JIT_VCALL => {
            JitOp::VCall(vc::CONSTRUCT_SLOT, index as u32, num_args, true)
        }
        // `construct` — construct a constructor value on the stack.
        Op::Construct { num_args } if JIT_VCALL => JitOp::VCall(vc::CONSTRUCT, 0, num_args, true),
        // `constructprop` — construct `source.mn(args…)`.
        Op::ConstructProp { multiname, num_args } if JIT_VCALL => {
            if multiname.has_lazy_component() {
                return None;
            }
            let k = *next_mn;
            *next_mn += 1;
            JitOp::VCall(vc::CONSTRUCT_PROP, k, num_args, true)
        }
        // `callsuper` — invoke the superclass's `mn` on the receiver.
        Op::CallSuper { multiname, num_args } if JIT_VCALL => {
            if multiname.has_lazy_component() {
                return None;
            }
            let k = *next_mn;
            *next_mn += 1;
            JitOp::VCall(vc::CALL_SUPER, k, num_args, true)
        }
        // `callnative` — the verifier's direct native-method fast call. The fn
        // pointer lives in the per-run natives table (`k`, in op order).
        Op::CallNative { num_args, push_return_value, .. } if JIT_VCALL => {
            let k = *next_native;
            *next_native += 1;
            JitOp::VCall(vc::CALL_NATIVE, k, num_args, push_return_value)
        }
        // `applytype` — parameterize a generic class (`Vector.<T>`).
        Op::ApplyType { num_types } if JIT_VCALL => {
            JitOp::VCall(vc::APPLY_TYPE, 0, num_types, true)
        }
        // `newarray` / `newobject` — literal construction from stack operands
        // (`newobject` pops name/value *pairs* → spill 2·argc).
        Op::NewArray { num_args } if JIT_VCALL => JitOp::VCall(vc::NEW_ARRAY, 0, num_args, true),
        Op::NewObject { num_args } if JIT_VCALL => {
            JitOp::VCall(vc::NEW_OBJECT, 0, num_args * 2, true)
        }
        // `getsuper`/`setsuper` — super-qualified property access.
        Op::GetSuper { multiname } if JIT_VCALL => {
            if multiname.has_lazy_component() {
                return None;
            }
            let k = *next_mn;
            *next_mn += 1;
            JitOp::VCall(vc::GET_SUPER, k, 0, true)
        }
        Op::SetSuper { multiname } if JIT_VCALL => {
            if multiname.has_lazy_component() {
                return None;
            }
            let k = *next_mn;
            *next_mn += 1;
            JitOp::VCall(vc::SET_SUPER, k, 1, false)
        }
        // `deleteproperty` — pushes whether the delete succeeded.
        Op::DeleteProperty { multiname } if JIT_VCALL => {
            if multiname.has_lazy_component() {
                return None;
            }
            let k = *next_mn;
            *next_mn += 1;
            JitOp::VCall(vc::DELETE_PROPERTY, k, 0, true)
        }
        // `nextvalue` — enumeration value at the popped index (`for each` loops).
        Op::NextValue if JIT_VCALL => JitOp::VCall(vc::NEXT_VALUE, 0, 1, true),
        // `in` — property-existence test (`name in obj`).
        Op::In if JIT_VCALL => JitOp::VCall(vc::IN, 0, 1, true),
        // `setproperty` (static multiname) — the write counterpart of
        // `GetPropertyStatic`. (`SetPropertySlow` — a lazy *namespace* — declines.)
        Op::SetPropertyStatic { .. } if JIT_VCALL => {
            let k = *next_mn;
            *next_mn += 1;
            JitOp::VCall(vc::SET_PROP_STATIC, k, 1, false)
        }
        // `setproperty` (fast/lazy-name) — `arr[i] = v` / dictionary writes; the
        // runtime name is a stack operand, `k` is the lazy multiname template.
        Op::SetPropertyFast { .. } if JIT_VCALL => {
            let k = *next_mn;
            *next_mn += 1;
            JitOp::VCall(vc::SET_PROP_FAST, k, 2, false)
        }
        // `newclass` — construct the ClassObject for `class` on the popped base.
        // The class rides the coerce-class table (`next_coerce`, op order), like
        // `coerce`/`newactivation`.
        Op::NewClass { .. } if JIT_VCALL => {
            let k = *next_coerce;
            *next_coerce += 1;
            JitOp::VCall(vc::NEW_CLASS, k, 0, true)
        }
        // `newfunction` — build a closure capturing the current scope chain (the
        // erased `Method` rides the coerce-class table; `captures_scope` above
        // forces the real scope stack on, exactly like `newclass`).
        Op::NewFunction { .. } if JIT_VCALL => {
            let k = *next_coerce;
            *next_coerce += 1;
            JitOp::VCall(vc::NEW_FUNCTION, k, 0, true)
        }
        // `newactivation` — the method's activation object (closure locals).
        Op::NewActivation { .. } if JIT_VCALL => {
            let k = *next_coerce;
            *next_coerce += 1;
            JitOp::VCall(vc::NEW_ACTIVATION, k, 0, true)
        }
        // `typeof` — the ECMAScript type-name string.
        Op::TypeOf if JIT_VCALL => JitOp::VCall(vc::TYPE_OF, 0, 0, true),
        // `pushnamespace` — a NamespaceObject for the `k`-th namespace (per-run table).
        Op::PushNamespace { .. } if JIT_VCALL => {
            let k = *next_ns;
            *next_ns += 1;
            JitOp::VCall(vc::PUSH_NAMESPACE, k, 0, true)
        }
        // `coerce_d` (`ToNumber`) / `convert_s` (`ToString`) — can throw via a
        // `valueOf`/`toString`, hence the vcall (perr) form, not a plain helper.
        Op::CoerceD if JIT_VCALL => JitOp::VCall(vc::COERCE_D, 0, 0, true),
        Op::ConvertS if JIT_VCALL => JitOp::VCall(vc::CONVERT_S, 0, 0, true),
        Op::PushScope if needs_scope => JitOp::PushScopeReal,
        // `popscope` pops the real scope stack only when scopes are read; otherwise
        // (nothing was pushed there) it's a no-op.
        Op::PopScope if needs_scope => JitOp::PopScopeReal,
        Op::PopScope => JitOp::Nop,
        Op::Pop | Op::PushScope => JitOp::Pop,
        // Debug metadata (`debug`/`debugfile`/`debugline`) is semantically a no-op:
        // it only feeds the AVM2 debugger's source/line mapping, which the JIT
        // doesn't surface. Emitting a `Nop` (rather than declining) keeps the 1:1
        // op mapping branch offsets rely on, and no per-run side table counts these
        // ops, so indices stay in sync. This was the single largest decline reason
        // (nearly every Sparkworkz/AS3 method opens with `debugfile`).
        Op::Nop
        | Op::Kill { .. }
        | Op::Debug { .. }
        | Op::DebugFile { .. }
        | Op::DebugLine { .. } => JitOp::Nop,
        _ => return None,
    })
}

/// The variant name of the first op [`translate_boxed`] can't lower (for
/// decline-reason diagnostics), or `None` if every op is supported (a decline
/// then came from `lower::compile`, e.g. a non-empty stack across a branch).
pub fn first_unsupported_boxed(ops: &[Op]) -> Option<String> {
    // `needs_scope`/`scope_this` don't affect *whether* an op is supported (only
    // how `pushscope`/`getscopeobject` lower), so any values give the same answer.
    let mut next_mn = 0;
    let mut next_script = 0;
    let mut next_string = 0;
    let mut next_coerce = 0;
    let mut next_native = 0;
    let mut next_ns = 0;
    let mut next_ic_site = 0;
    let mut next_call_site = 0;
    let mut switches = Vec::new();
    ops.iter()
        .find(|op| {
            boxed_op(
                op,
                &mut next_mn,
                &mut next_script,
                &mut next_string,
                &mut next_coerce,
                &mut next_native,
                &mut next_ns,
                &mut next_ic_site,
                &mut next_call_site,
                true,
                false,
                &mut switches,
                false,
            )
            .is_none()
        })
        .map(op_variant_name)
}

/// The bare variant name of an `Op` (e.g. `"CallProperty"`), extracted from its
/// `Debug` form — used to bucket decline reasons.
fn op_variant_name(op: &Op) -> String {
    let dbg = format!("{op:?}");
    dbg.split([' ', '{', '('])
        .next()
        .unwrap_or("?")
        .to_string()
}

/// Translates to the **double fast path**: unboxed `f64` on the WASM stack,
/// numeric ops emitted *inline* (no helper calls). Used when a method is
/// `Number`-typed (see [`crate::analysis::double_sound`]). Straight-line only for
/// now (no branches); bails otherwise.
pub fn translate_double(ops: &[Op]) -> Option<Vec<JitOp>> {
    let mut out = Vec::with_capacity(ops.len());
    for op in ops {
        out.push(match *op {
            Op::GetLocal { index } => JitOp::GetLocalDouble(index),
            Op::SetLocal { index } => JitOp::SetLocalDouble(index),
            // Verifier peephole for `setlocal i; getlocal i`: store but keep the
            // value on the stack.
            Op::StoreLocal { index } => JitOp::StoreLocalDouble(index),
            Op::PushDouble { value } => JitOp::PushDouble(value.to_bits()),
            // General arithmetic — valid as `f64` ops only because `double_sound`
            // proves both operands are `Number` (so `add` is `f64.add`, never
            // string concat).
            Op::Add { .. } => JitOp::AddD,
            Op::Subtract { .. } => JitOp::SubtractD,
            Op::Multiply => JitOp::MultiplyD,
            Op::Divide => JitOp::DivideD,
            Op::Increment => JitOp::IncrementD,
            Op::Decrement => JitOp::DecrementD,
            Op::Negate => JitOp::NegateD,
            Op::ReturnValue { .. } => JitOp::ReturnDouble,
            Op::Pop | Op::PushScope => JitOp::Pop,
            // Debug metadata is a no-op (see the boxed path).
            Op::Nop
            | Op::Kill { .. }
            | Op::Debug { .. }
            | Op::DebugFile { .. }
            | Op::DebugLine { .. } => JitOp::Nop,
            _ => return None,
        });
    }
    Some(out)
}

/// A tiny reference interpreter over [`JitOp`], mirroring the raw-`i32` model the
/// WASM lowering compiles to. It is the "interpreter" side of the differential
/// tests: `compile(ops)` executed under a WASM runtime must agree with this for
/// every input. Locals are `Value` bit patterns; the operand stack is raw `i32`.
#[cfg(all(test, not(target_arch = "wasm32")))]
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
            // Boxed-`Value` / double / helper ops aren't part of the int model.
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
            | JitOp::VCall(..)
            | JitOp::DmLoad(_)
            | JitOp::DmStore(_)
            | JitOp::DmLoadF(_)
            | JitOp::DmStoreF(_)
            | JitOp::LookupSwitch(_)
            | JitOp::PushString(_)
            | JitOp::Throw
            | JitOp::NewCatch(_)
            | JitOp::Coerce(_)
            | JitOp::FindProp(_, _)
            | JitOp::PopScopeReal => {
                unreachable!("non-int ops not in the int reference interpreter")
            }
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
    fn decline_reason_names_the_first_unsupported_op() {
        // Supported prefix, then an unsupported op → its variant name.
        // (`CheckFilter` — E4X — isn't in the boxed subset.)
        let ops = [
            Op::GetLocal { index: 0 },
            Op::CheckFilter,
            Op::ReturnValue { return_type: None },
        ];
        assert_eq!(first_unsupported_boxed(&ops).as_deref(), Some("CheckFilter"));
        // Fully-supported boxed method → no reason (a decline would be `<compile>`).
        let ok = [Op::GetLocal { index: 0 }, Op::ReturnValue { return_type: None }];
        assert_eq!(first_unsupported_boxed(&ok), None);
    }

    #[test]
    fn vcall_ops_translate_boxed() {
        // Gc-free vcall kinds map 1:1 (op indices stay aligned for branches).
        let ops = [
            Op::GetLocal { index: 0 },
            Op::PushScope,
            Op::GetLocal { index: 1 },
            Op::GetLocal { index: 2 },
            Op::NewArray { num_args: 2 },
            Op::TypeOf,
            Op::ReturnValue { return_type: None },
        ];
        let (out, switches) = translate_boxed(&ops, &[]).expect("boxed supported");
        assert!(switches.is_empty());
        assert_eq!(out[4], JitOp::VCall(vc::NEW_ARRAY, 0, 2, true));
        assert_eq!(out[5], JitOp::VCall(vc::TYPE_OF, 0, 0, true));
        // `newobject` spills PAIRS (2·argc); `construct` spills argc.
        let ops = [
            Op::NewObject { num_args: 3 },
            Op::Construct { num_args: 1 },
            Op::ReturnValue { return_type: None },
        ];
        let (out, _) = translate_boxed(&ops, &[]).expect("boxed supported");
        assert_eq!(out[0], JitOp::VCall(vc::NEW_OBJECT, 0, 6, true));
        assert_eq!(out[1], JitOp::VCall(vc::CONSTRUCT, 0, 1, true));
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
        let jit = run(&bytes, regs, &crate::lower::manifest(ops), 0).expect("native runner");
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
            let got = run(&bytes, &regs, &crate::lower::manifest(&ops), 0).unwrap();
            let expected: i32 = (0..n).sum();
            assert_eq!(got, int_bits(expected), "sum({n})");
        }
    }

    #[test]
    fn boxed_render_shaped_loop_terminates_and_sums() {
        // Mirrors the loop skeleton of Starling's `DynamicBatcher.render` as the
        // verifier emits it, through the **boxed** path (the real method is
        // call-bearing → boxed): the bound arrives boxed and is `coerce_i`'d, the
        // counter steps via `inclocal_i`, entry jumps past the body to the
        // condition, and the back-edge is `lessthan; iftrue`.
        use ruffle_core::avm2::Op;
        let core_ops = [
            Op::GetLocal { index: 0 },             // 0
            Op::PushScope,                         // 1
            Op::GetLocal { index: 1 },             // 2: n (boxed)
            Op::CoerceI,                           // 3
            Op::SetLocal { index: 4 },             // 4: n = int(n)
            Op::PushInt { value: 0 },              // 5
            Op::SetLocal { index: 5 },             // 6: i = 0
            Op::PushInt { value: 0 },              // 7
            Op::SetLocal { index: 2 },             // 8: s = 0
            Op::Jump { offset: 15 },               // 9: -> condition
            Op::GetLocal { index: 2 },             // 10: body: s
            Op::GetLocal { index: 5 },             // 11: i
            Op::AddI,                              // 12
            Op::SetLocal { index: 2 },             // 13: s += i
            Op::IncLocalI { index: 5 },            // 14: i++
            Op::GetLocal { index: 5 },             // 15: condition: i
            Op::GetLocal { index: 4 },             // 16: n
            Op::LessThan,                          // 17
            Op::IfTrue { offset: 10 },             // 18: -> body
            Op::GetLocal { index: 2 },             // 19
            Op::ReturnValue { return_type: None }, // 20
        ];
        let (ops, switches) = translate_boxed(&core_ops, &[]).expect("boxed supported");
        assert!(switches.is_empty());
        let bytes = compile(&ops).expect("compiles");
        for n in [0, 1, 2, 16, 100] {
            let regs =
                [int_bits(0), int_bits(n), int_bits(0), int_bits(0), int_bits(0), int_bits(0)];
            let got = run(&bytes, &regs, &crate::lower::manifest(&ops), 0).expect("runs");
            let expected: i32 = (0..n).sum();
            assert_eq!(got, int_bits(expected), "sum({n})");
        }
    }

    #[test]
    fn boxed_countdown_loop_terminates_and_sums() {
        // Mirrors `DisplayObjectContainer.hitTest`'s countdown: `i = n - 1` (with
        // the verifier's `coerce_i; storelocal; pushint 1; subtract_i` prologue),
        // entry jumps past the body AND the `declocal_i` step to the condition,
        // back-edge `greaterequals; iftrue`.
        use ruffle_core::avm2::Op;
        let core_ops = [
            Op::GetLocal { index: 0 },             // 0
            Op::PushScope,                         // 1
            Op::GetLocal { index: 1 },             // 2: n (boxed)
            Op::CoerceI,                           // 3
            Op::StoreLocal { index: 3 },           // 4: local3 = n (kept on stack)
            Op::PushInt { value: 1 },              // 5
            Op::SubtractI,                         // 6: n - 1
            Op::SetLocal { index: 4 },             // 7: i = n - 1
            Op::PushInt { value: 0 },              // 8
            Op::SetLocal { index: 2 },             // 9: s = 0
            Op::Jump { offset: 16 },               // 10: -> condition (skip the step)
            Op::GetLocal { index: 2 },             // 11: body: s
            Op::GetLocal { index: 4 },             // 12: i
            Op::AddI,                              // 13
            Op::SetLocal { index: 2 },             // 14: s += i
            Op::DecLocalI { index: 4 },            // 15: step: i--
            Op::GetLocal { index: 4 },             // 16: condition: i
            Op::PushInt { value: 0 },              // 17
            Op::GreaterEquals,                     // 18: i >= 0
            Op::IfTrue { offset: 11 },             // 19: -> body
            Op::GetLocal { index: 2 },             // 20
            Op::ReturnValue { return_type: None }, // 21
        ];
        let (ops, switches) = translate_boxed(&core_ops, &[]).expect("boxed supported");
        assert!(switches.is_empty());
        let bytes = compile(&ops).expect("compiles");
        for n in [0, 1, 2, 16, 100] {
            let regs =
                [int_bits(0), int_bits(n), int_bits(0), int_bits(0), int_bits(0), int_bits(0)];
            let got = run(&bytes, &regs, &crate::lower::manifest(&ops), 0).expect("runs");
            let expected: i32 = (0..n).sum();
            assert_eq!(got, int_bits(expected), "countdown sum({n})");
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
