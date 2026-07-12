//! Rust helpers imported by compiled type-0 modules.
//!
//! On native the WASM is sandboxed, so any op reading a GC object calls back into one
//! of these (the helper runs in Ruffle's process with heap access). Phase 2's only
//! helper is [`get_slot`], which is **Activation-free and GC-mutation-free** — a pure
//! slot read. Helpers that need the callee's own context (`getproperty` getters,
//! coercions that re-enter, calls) come with the Phase 3 reification ABI.
//!
//! Values cross the boundary as raw `i64` bits (see [`crate::value`]). Within a frame
//! the GC neither moves nor frees, so an object `Value`'s pointer bits stay valid for
//! the duration of the helper call.

use ruffle_core::avm2::error::{
    make_error_1041, make_error_1127, make_error_1506, make_null_or_undefined_error,
};
use ruffle_core::avm2::object::{ArrayObject, FunctionObject, Object, ScriptObject};
use ruffle_core::avm2::property::Property;
use ruffle_core::avm2::script::Script;
use ruffle_core::avm2::{
    Activation, ArrayStorage, Class, Error, FunctionArgs, Method, Multiname, NativeMethodImpl,
    Scope, TObject, Value, ValueEnum,
};

const SUPER_ON_PRIMITIVE: &str = "Super ops should not appear in primitive functions";

/// Shared super-op receiver prep: coerce the receiver to the bound superclass's type and
/// null-check it into an `Object` (mirrors the interpreter). Returns `None` (→ `undefined`
/// / no-op) if there is no bound superclass — which can't happen for a valid super op.
fn super_receiver<'gc>(
    receiver: Value<'gc>,
    mn: &Multiname<'gc>,
    act: &mut Activation<'_, 'gc>,
) -> Result<Option<Object<'gc>>, Error<'gc>> {
    let Some(sup) = act.bound_superclass_object() else {
        return Ok(None);
    };
    let coerced = receiver.coerce_to_type(act, sup.inner_class_definition())?;
    let obj = coerced.as_object_null_check(act, Some(mn), SUPER_ON_PRIMITIVE)?;
    Ok(Some(obj))
}

/// `getsuper`: `super.<mn>` on `receiver` via the bound superclass's vtable. Reifies;
/// stash-on-throw (coercion / null-check / getter).
pub fn get_super(receiver_bits: i64, mn_ptr: i64) -> i64 {
    // SAFETY: live `Value` this frame; `mn_ptr` a live baked `*const Multiname`.
    let receiver: Value<'_> = unsafe { from_bits(receiver_bits as u64) };
    let mn: &Multiname<'_> = unsafe { &*(mn_ptr as usize as *const Multiname) };
    // SAFETY: called from JIT wasm inside `with_run_ctx`; `act` does not escape.
    let mut act = unsafe { context::reify() };
    let result = super_receiver(receiver, mn, &mut act).and_then(|obj| match obj {
        Some(o) => act.bound_superclass_object().unwrap().get_super(mn, o, &mut act),
        None => Ok(Value::Undefined),
    });
    match result {
        Ok(v) => to_bits(v) as i64,
        Err(e) => {
            context::stash_error(e);
            SENTINEL_BITS as i64
        }
    }
}

/// `setsuper`: `super.<mn> = value` on `receiver`. Stack: `receiver` deeper, `value` top.
pub fn set_super(receiver_bits: i64, mn_ptr: i64, value_bits: i64) {
    // SAFETY: live `Value`s this frame; `mn_ptr` a live baked `*const Multiname`.
    let receiver: Value<'_> = unsafe { from_bits(receiver_bits as u64) };
    let value: Value<'_> = unsafe { from_bits(value_bits as u64) };
    let mn: &Multiname<'_> = unsafe { &*(mn_ptr as usize as *const Multiname) };
    // SAFETY: called from JIT wasm inside `with_run_ctx`; `act` does not escape.
    let mut act = unsafe { context::reify() };
    let result = super_receiver(receiver, mn, &mut act).and_then(|obj| match obj {
        Some(o) => act.bound_superclass_object().unwrap().set_super(mn, value, o, &mut act),
        None => Ok(()),
    });
    if let Err(e) = result {
        context::stash_error(e);
    }
}

/// `callsuper` core: `super.<mn>(args)` on `receiver`. Args read from the outgoing scratch.
///
/// # Safety
/// As [`call_property_bits`].
pub unsafe fn call_super_bits(receiver_bits: i64, mn_ptr: i64, arg_bits: &[i64]) -> i64 {
    let args = unsafe { crate::value::bits_as_values(arg_bits) };
    let receiver: Value<'_> = unsafe { from_bits(receiver_bits as u64) };
    let mn: &Multiname<'_> = unsafe { &*(mn_ptr as usize as *const Multiname) };
    // SAFETY: called from JIT wasm inside `with_run_ctx`; `act` does not escape.
    let mut act = unsafe { context::reify() };
    let result = super_receiver(receiver, mn, &mut act).and_then(|obj| match obj {
        Some(o) => act.bound_superclass_object().unwrap().call_super(
            mn,
            o,
            FunctionArgs::from_slice(args),
            &mut act,
        ),
        None => Ok(Value::Undefined),
    });
    match result {
        Ok(v) => to_bits(v) as i64,
        Err(e) => {
            context::stash_error(e);
            SENTINEL_BITS as i64
        }
    }
}

/// `constructsuper` core: `super(args)` — `super_init` on the null-checked receiver. Void
/// (returns `undefined`); stash-on-throw.
///
/// # Safety
/// As [`call_property_bits`].
pub unsafe fn construct_super_bits(receiver_bits: i64, arg_bits: &[i64]) -> i64 {
    let args = unsafe { crate::value::bits_as_values(arg_bits) };
    let receiver: Value<'_> = unsafe { from_bits(receiver_bits as u64) };
    // SAFETY: called from JIT wasm inside `with_run_ctx`; `act` does not escape.
    let mut act = unsafe { context::reify() };
    let result = match receiver.null_check(&mut act, None) {
        Ok(r) => act.super_init(r, FunctionArgs::from_slice(args)).map(|_| ()),
        Err(e) => Err(e),
    };
    if let Err(e) = result {
        context::stash_error(e);
        return SENTINEL_BITS as i64;
    }
    UNDEFINED_BITS as i64
}

/// `callsuper` (web funcref): reads `argc` args from frame memory, delegates.
#[cfg(target_arch = "wasm32")]
pub fn call_super(receiver_bits: i64, mn_ptr: i64, args_off: i64, argc: i64) -> i64 {
    // SAFETY: `args_off` is where the compiled body `i64.store`d `argc` `Value`s this frame.
    let arg_bits: &[i64] =
        unsafe { core::slice::from_raw_parts(args_off as usize as *const i64, argc as usize) };
    unsafe { call_super_bits(receiver_bits, mn_ptr, arg_bits) }
}

/// `constructsuper` (web funcref): reads `argc` args from frame memory, delegates.
#[cfg(target_arch = "wasm32")]
pub fn construct_super(receiver_bits: i64, args_off: i64, argc: i32) -> i64 {
    // SAFETY: `args_off` is where the compiled body `i64.store`d `argc` `Value`s this frame.
    let arg_bits: &[i64] =
        unsafe { core::slice::from_raw_parts(args_off as usize as *const i64, argc as usize) };
    unsafe { construct_super_bits(receiver_bits, arg_bits) }
}

use crate::context;
use crate::emit::{SENTINEL_BITS, UNDEFINED_BITS};
use crate::value::{from_bits, to_bits};

/// Binary-operator codes shared by [`binop`] and `translate` (kept together so they can't
/// drift). The numbering is arbitrary but must match on both sides.
pub mod binop_code {
    pub const ADD: i32 = 0;
    pub const SUBTRACT: i32 = 1;
    pub const MULTIPLY: i32 = 2;
    pub const DIVIDE: i32 = 3;
    pub const MODULO: i32 = 4;
    pub const BITAND: i32 = 5;
    pub const BITOR: i32 = 6;
    pub const BITXOR: i32 = 7;
    pub const LSHIFT: i32 = 8;
    pub const RSHIFT: i32 = 9;
    pub const URSHIFT: i32 = 10;
    pub const EQUALS: i32 = 11;
    pub const STRICT_EQUALS: i32 = 12;
    pub const LESS_THAN: i32 = 13;
    pub const LESS_EQUALS: i32 = 14;
    pub const GREATER_THAN: i32 = 15;
    pub const GREATER_EQUALS: i32 = 16;
    pub const ADD_I: i32 = 17; // int add (wrapping)
    pub const SUBTRACT_I: i32 = 18; // int subtract (wrapping)
    pub const MULTIPLY_I: i32 = 19; // int multiply (wrapping)
}

/// Unary-operator codes shared by [`unop`] and `translate` (includes the primitive
/// coerce/convert ops — all one-in-one-out with the same throwing shape).
pub mod unop_code {
    pub const NEGATE: i32 = 0;
    pub const INCREMENT: i32 = 1;
    pub const DECREMENT: i32 = 2;
    pub const NOT: i32 = 3;
    pub const BITNOT: i32 = 4;
    pub const COERCE_B: i32 = 5; // to Boolean
    pub const COERCE_D: i32 = 6; // to Number
    pub const COERCE_I: i32 = 7; // to int
    pub const COERCE_U: i32 = 8; // to uint
    pub const COERCE_S: i32 = 9; // to String (null/undefined→null, String→self)
    pub const CONVERT_S: i32 = 10; // to String (always)
    pub const COERCE_O: i32 = 11; // to Object (null/undefined→null else self)
    pub const INCREMENT_I: i32 = 12; // int increment (wrapping)
    pub const DECREMENT_I: i32 = 13; // int decrement (wrapping)
    pub const NEGATE_I: i32 = 14; // int negate (wrapping)
}

/// `getslot`: `receiver.get_slot(slot_id)`. Emitted only for verifier-proven
/// **null-safe** sites, so the receiver is always an object and this never throws.
/// A pure read — no `Activation`, no write barrier.
pub fn get_slot(obj_bits: i64, slot_id: i64) -> i64 {
    // SAFETY: `obj_bits` is a live object `Value` produced by the JIT this frame
    // (null-safe site → never null/primitive).
    let value: Value<'_> = unsafe { from_bits(obj_bits as u64) };
    let object = value
        .as_object()
        .expect("null-safe getslot receiver must be an object");
    to_bits(object.get_slot(slot_id as usize)) as i64
}

/// `getslot` for a receiver the verifier did NOT prove non-null: null-checks the receiver
/// (throws #1009 / "Cannot get_slot on primitive", like `op_get_slot`) before the slot read.
/// Reifies (needs an Activation to build the error); stash-on-throw → `BailIfError`. Lets a
/// non-null-safe `getslot` compile instead of declining (the dominant hot-path blocker).
pub fn get_slot_checked(obj_bits: i64, slot_id: i64) -> i64 {
    let value: Value<'_> = unsafe { from_bits(obj_bits as u64) };
    // Fast path: a non-null object receiver (the overwhelmingly common case) needs NO
    // Activation — just a tag test + the slot read. Only reify to build the #1009 error when
    // the receiver is actually null/primitive.
    if let Some(object) = value.as_object() {
        return to_bits(object.get_slot(slot_id as usize)) as i64;
    }
    // SAFETY: called from JIT wasm inside `with_run_ctx`; `act` does not escape.
    let mut act = unsafe { context::reify() };
    match value.as_object_null_check(&mut act, None, "Cannot get_slot on primitive") {
        Ok(object) => to_bits(object.get_slot(slot_id as usize)) as i64,
        Err(e) => {
            context::stash_error(e);
            SENTINEL_BITS as i64
        }
    }
}

/// Property inline-cache backing store: `IC_ENTRIES` pairs of `[vtable_ptr: u32, slot_id:
/// u32]`, one per compiled `GetProperty` site (its address is baked into the module). The
/// compiled fast path reads an entry INLINE (web only — GC memory) and, on a `vtable` hit,
/// reads the object's slot directly; [`get_property_ic`] fills it on a miss. `vtable == 0` is
/// empty. Single-thread (see `context`): touched only from the one AVM2 thread on web; never
/// on native (the JIT uses the plain `gp` helper there).
const IC_ENTRIES: usize = 16384;
static mut IC_CACHE: [u32; IC_ENTRIES * 2] = [0u32; IC_ENTRIES * 2];

/// Base address of [`IC_CACHE`] — the JIT bakes `base + site*8` per `GetProperty` site.
pub fn ic_cache_base() -> usize {
    core::ptr::addr_of!(IC_CACHE) as usize
}

/// Number of inline-cache sites available (the JIT falls back to the plain `gp` helper past
/// this, so a huge content can't overflow the cache).
pub fn ic_cache_capacity() -> usize {
    IC_ENTRIES
}

/// `getproperty` inline-cache MISS handler. Resolves `mn` against the receiver's vtable; if it
/// is a plain slot, records `(vtable_ptr, slot_id)` at `cache_addr` so the compiled fast path
/// reads it inline next time, and returns the slot value. Otherwise (getter/method/dynamic/
/// primitive) runs the full `get_property` (a getter runs AS3, may throw → SENTINEL). Reifies.
pub fn get_property_ic(receiver_bits: i64, mn_ptr: i64, cache_addr: i64) -> i64 {
    let value: Value<'_> = unsafe { from_bits(receiver_bits as u64) };
    let mn: &Multiname<'_> = unsafe { &*(mn_ptr as usize as *const Multiname) };
    // SAFETY: called from JIT wasm inside `with_run_ctx`; `act` does not escape.
    let mut act = unsafe { context::reify() };
    if let Some(obj) = value.as_object() {
        let vt = obj.vtable();
        if let Some(Property::Slot { slot_id } | Property::ConstSlot { slot_id }) = vt.get_trait(mn)
        {
            // Cache the monomorphic (vtable → slot) resolution for the inline fast path.
            let cell = cache_addr as usize as *mut u32;
            unsafe {
                *cell = vt.as_ptr() as usize as u32; // wasm32: 32-bit vtable pointer
                *cell.add(1) = slot_id as u32;
            }
            return to_bits(obj.get_slot(slot_id)) as i64;
        }
    }
    // Not a plain slot — full get_property (null-check + getter/dynamic dispatch).
    match value.null_check(&mut act, Some(mn)).and_then(|v| v.get_property(mn, &mut act)) {
        Ok(v) => to_bits(v) as i64,
        Err(e) => {
            context::stash_error(e);
            SENTINEL_BITS as i64
        }
    }
}

/// `coerce_return`: coerce `value` to the method's return type `class` (a baked,
/// erased `Class` pointer), reifying a fresh callee-owned `Activation` (§reification).
/// Emitted only at a typed `returnvalue` whose value's class the translator could NOT
/// prove already matches — so a real coercion is needed. On a throwing coercion
/// (`#1034`), stashes the error and returns `undefined`; since this op is always
/// immediately followed by `Return`, `try_enter` takes the error after the run.
/// `getproperty` (polymorphic, static multiname): `receiver.get_property(mn)` — `mn` is
/// a baked `*const Multiname` (the multiname is alive for the method's run). Reifies a
/// callee-owned Activation (a triggered getter runs AS3, may re-enter/throw). On a throw,
/// stashes the error and returns `undefined`; the compiled body emits a `BailIfError`
/// (perr) right after, so it returns at once and `try_enter` propagates the error — safe
/// mid-body, not only when it feeds the return.
pub fn get_property(receiver_bits: i64, mn_ptr: i64) -> i64 {
    // SAFETY: `receiver_bits` is a live `Value` this frame; `mn_ptr` is a baked
    // `*const Multiname` (its target alive for the method's run — non-moving GC).
    let value: Value<'_> = unsafe { from_bits(receiver_bits as u64) };
    let mn: &Multiname<'_> = unsafe { &*(mn_ptr as usize as *const Multiname) };
    // SAFETY: called from JIT wasm inside `with_run_ctx`; `act` does not escape.
    let mut act = unsafe { context::reify() };
    // Null-check the receiver first (like the interpreter) — `get_property` calls `vtable()`,
    // which PANICS on null/undefined. The throw is stashed and surfaced by `BailIfError`.
    match value.null_check(&mut act, Some(mn)).and_then(|v| v.get_property(mn, &mut act)) {
        Ok(v) => to_bits(v) as i64,
        Err(e) => {
            context::stash_error(e);
            SENTINEL_BITS as i64
        }
    }
}

/// `perr` (pending-error): `1` if a helper stashed a thrown error this run, else `0`. A
/// compiled body calls this after a mid-body throwing op and returns early when set, so
/// the error surfaces before any later op runs on a bogus result (the in-wasm error bail
/// that lets getproperty/callproperty appear mid-body). Non-consuming — `try_enter` takes
/// the error after the run.
pub fn pending_error() -> i32 {
    context::has_pending() as i32
}

/// `getouterscope`: the `index`-th captured (outer) scope's values object. Reifies — the
/// reified Activation's `outer` IS the callee's captured `ScopeChain` (from the RunCtx), so
/// `jit_outer_scope` reads exactly what the interpreter would. No throw.
pub fn outer_scope(index: i32) -> i64 {
    // SAFETY: called from JIT wasm inside `with_run_ctx`; `act` does not escape.
    let act = unsafe { context::reify() };
    to_bits(act.jit_outer_scope(index as usize)) as i64
}

/// `getscriptglobals`: `script.globals(context)` — the script's globals object (lazy script
/// init may run → can throw). `script_ptr` is a baked `Script` handle. Stash-on-throw.
pub fn script_globals(script_ptr: i64) -> i64 {
    // SAFETY: `script_ptr` is a baked live `Script` (single-`Gc` newtype) for the method's run.
    let script: Script<'_> =
        unsafe { core::mem::transmute::<*const (), Script>(script_ptr as usize as *const ()) };
    // SAFETY: called from JIT wasm inside `with_run_ctx`; `act` does not escape.
    let mut act = unsafe { context::reify() };
    match script.globals(act.context) {
        Ok(o) => {
            let v: Value<'_> = o.into();
            to_bits(v) as i64
        }
        Err(e) => {
            context::stash_error(e);
            SENTINEL_BITS as i64
        }
    }
}

/// `newactivation`: a fresh activation object carrying `activation_class`'s vtable (no
/// prototype — activation classes only have slots). `class_ptr` baked. No throw.
pub fn new_activation(class_ptr: i64) -> i64 {
    // SAFETY: `class_ptr` is a baked live `Class` handle for the method's run.
    let class: Class<'_> =
        unsafe { core::mem::transmute::<*const (), Class>(class_ptr as usize as *const ()) };
    // SAFETY: called from JIT wasm inside `with_run_ctx`; `act` does not escape.
    let act = unsafe { context::reify() };
    let obj = ScriptObject::custom_object(act.gc(), class, None, class.vtable());
    let v: Value<'_> = obj.into();
    to_bits(v) as i64
}

/// `pushscope`: null-checks the object (like the interpreter) and pushes it onto the SHARED
/// `avm2.scope_stack` (visible to reified findproperty/getters — a separate stack would be
/// invisible to them). Throws on null/undefined → stashed, surfaced by `BailIfError`.
pub fn push_scope(bits: i64) {
    // SAFETY: a live `Value` this frame.
    let v: Value<'_> = unsafe { from_bits(bits as u64) };
    // SAFETY: called from JIT wasm inside `with_run_ctx`; `act` does not escape.
    let mut act = unsafe { context::reify() };
    match v.null_check(&mut act, None) {
        Ok(obj) => act.push_scope(Scope::new(obj)),
        Err(e) => context::stash_error(e),
    }
}

/// `popscope`: pops the top of the shared scope stack. No throw.
pub fn pop_scope() {
    // SAFETY: called from JIT wasm inside `with_run_ctx`; `act` does not escape.
    let mut act = unsafe { context::reify() };
    act.pop_scope();
}

/// `getscopeobject`: the `index`-th LOCAL scope's object (relative to this run's scope base
/// on the shared stack). No throw.
pub fn get_scope(index: i32) -> i64 {
    // SAFETY: called from JIT wasm inside `with_run_ctx`; `act` does not escape.
    let act = unsafe { context::reify() };
    to_bits(act.jit_local_scope(context::scope_base(), index as usize)) as i64
}

/// `istypelate`: `value is Type` — `type` (top) must be a class object (else #1041), then
/// push `value.is_of_type(type)` as a Boolean. Reifies for the error path; `is_of_type` is
/// pure. Stack order: `value` deeper, `type` on top.
pub fn is_type_late(value_bits: i64, type_bits: i64) -> i64 {
    // SAFETY: live `Value`s this frame.
    let value: Value<'_> = unsafe { from_bits(value_bits as u64) };
    let type_val: Value<'_> = unsafe { from_bits(type_bits as u64) };
    // SAFETY: called from JIT wasm inside `with_run_ctx`; `act` does not escape.
    let mut act = unsafe { context::reify() };
    match type_val.as_object().and_then(|o| o.as_class_object()) {
        Some(co) => {
            let is = value.is_of_type(co.inner_class_definition());
            to_bits(Value::from(is)) as i64
        }
        None => {
            context::stash_error(make_error_1041(&mut act));
            SENTINEL_BITS as i64
        }
    }
}

/// `astypelate`: `value as Type` — `class` (top) undefined → #1007-ish; non-object → error;
/// non-class-object → #1041; else `value` if it is-of-type, otherwise `null`. Reifies.
pub fn as_type_late(value_bits: i64, class_bits: i64) -> i64 {
    // SAFETY: live `Value`s this frame.
    let value: Value<'_> = unsafe { from_bits(value_bits as u64) };
    let class: Value<'_> = unsafe { from_bits(class_bits as u64) };
    // SAFETY: called from JIT wasm inside `with_run_ctx`; `act` does not escape.
    let mut act = unsafe { context::reify() };
    if matches!(class.unpack(), ValueEnum::Undefined) {
        context::stash_error(make_null_or_undefined_error(&mut act, class, None));
        return SENTINEL_BITS as i64;
    }
    match class.as_object() {
        Some(o) => match o.as_class_object() {
            Some(co) => {
                let out = if value.is_of_type(co.inner_class_definition()) {
                    value
                } else {
                    Value::Null
                };
                to_bits(out) as i64
            }
            None => {
                context::stash_error(make_error_1041(&mut act));
                SENTINEL_BITS as i64
            }
        },
        None => {
            context::stash_error(make_null_or_undefined_error(&mut act, Value::Null, None));
            SENTINEL_BITS as i64
        }
    }
}

/// `truthy`: `Value::coerce_to_boolean()` as an i32 (for `iftrue`/`iffalse` dispatch). Pure
/// — no Activation, no throw (ECMA ToBoolean never calls `valueOf`).
pub fn truthy(bits: i64) -> i32 {
    // SAFETY: a live `Value` this frame.
    let v: Value<'_> = unsafe { from_bits(bits as u64) };
    v.coerce_to_boolean() as i32
}

/// `cp` core: `receiver.call_property(mn, args)` where `mn` is a baked `*const Multiname`
/// and `arg_bits` are the `argc` argument `Value`s the caller `i64.store`d into the frame's
/// outgoing-arg scratch (read from that memory by the platform runner — one crossing, not
/// one per arg). Reifies a callee-owned Activation (the callee runs AS3, may re-enter/throw).
/// On a throw, stashes the error and returns `undefined`; the compiled body emits a
/// `BailIfError` (perr) right after, so it returns at once and `try_enter` propagates it.
///
/// # Safety
/// `arg_bits` must be live `Value`s the JIT stored this frame; `mn_ptr` a live `*const
/// Multiname` (non-moving GC keeps its target alive for the run).
pub unsafe fn call_property_bits(receiver_bits: i64, mn_ptr: i64, arg_bits: &[i64]) -> i64 {
    let args = unsafe { crate::value::bits_as_values(arg_bits) };
    let value: Value<'_> = unsafe { from_bits(receiver_bits as u64) };
    let mn: &Multiname<'_> = unsafe { &*(mn_ptr as usize as *const Multiname) };
    // SAFETY: called from JIT wasm inside `with_run_ctx`; `act` does not escape.
    let mut act = unsafe { context::reify() };
    // Null-check first (like the interpreter) — `call_property` calls `vtable()`, which
    // PANICS on null/undefined.
    match value
        .null_check(&mut act, Some(mn))
        .and_then(|v| v.call_property(mn, FunctionArgs::from_slice(args), &mut act))
    {
        Ok(v) => to_bits(v) as i64,
        Err(e) => {
            context::stash_error(e);
            SENTINEL_BITS as i64
        }
    }
}

/// `call_method` core: `receiver.call_method_with_args(disp_id, args)` after a null-check —
/// the by-disp-id call the optimizer emits (`CallMethod`). Args read from the outgoing-arg
/// scratch like [`call_property_bits`]. Reifies; stash-on-throw.
///
/// # Safety
/// As [`call_property_bits`].
pub unsafe fn call_method_bits(receiver_bits: i64, disp_id: i32, arg_bits: &[i64]) -> i64 {
    let args = unsafe { crate::value::bits_as_values(arg_bits) };
    let receiver: Value<'_> = unsafe { from_bits(receiver_bits as u64) };
    // SAFETY: called from JIT wasm inside `with_run_ctx`; `act` does not escape.
    let mut act = unsafe { context::reify() };
    let result = receiver
        .null_check(&mut act, None)
        .and_then(|r| r.call_method_with_args(disp_id as usize, FunctionArgs::from_slice(args), &mut act));
    match result {
        Ok(v) => to_bits(v) as i64,
        Err(e) => {
            context::stash_error(e);
            SENTINEL_BITS as i64
        }
    }
}

/// `construct` core: `ctor.construct(args)` (`new ctor(args)`). Reifies; `Value::construct`
/// returns `Err(#1007)` for a non-constructable value (no panic), so no null-check needed.
/// Stash-on-throw.
///
/// # Safety
/// As [`call_property_bits`].
pub unsafe fn construct_bits(ctor_bits: i64, arg_bits: &[i64]) -> i64 {
    let args = unsafe { crate::value::bits_as_values(arg_bits) };
    let ctor: Value<'_> = unsafe { from_bits(ctor_bits as u64) };
    // SAFETY: called from JIT wasm inside `with_run_ctx`; `act` does not escape.
    let mut act = unsafe { context::reify() };
    match ctor.construct(&mut act, FunctionArgs::from_slice(args)) {
        Ok(v) => to_bits(v) as i64,
        Err(e) => {
            context::stash_error(e);
            SENTINEL_BITS as i64
        }
    }
}

/// `construct` (web funcref): reads `argc` args from frame memory at `args_off`, delegates.
#[cfg(target_arch = "wasm32")]
pub fn construct(ctor_bits: i64, args_off: i64, argc: i32) -> i64 {
    // SAFETY: `args_off` is where the compiled body `i64.store`d `argc` `Value`s this frame.
    let arg_bits: &[i64] =
        unsafe { core::slice::from_raw_parts(args_off as usize as *const i64, argc as usize) };
    unsafe { construct_bits(ctor_bits, arg_bits) }
}

/// `in`: `name in value` (Boolean). Mirrors `op_in`. Reifies; stash-on-throw (coercion).
pub fn op_in(name_bits: i64, value_bits: i64) -> i64 {
    let name_value: Value<'_> = unsafe { from_bits(name_bits as u64) };
    let value: Value<'_> = unsafe { from_bits(value_bits as u64) };
    // SAFETY: called from JIT wasm inside `with_run_ctx`; `act` does not escape.
    let mut act = unsafe { context::reify() };
    match in_inner(name_value, value, &mut act) {
        Ok(b) => to_bits(Value::from(b)) as i64,
        Err(e) => {
            context::stash_error(e);
            SENTINEL_BITS as i64
        }
    }
}

fn in_inner<'gc>(
    name_value: Value<'gc>,
    value: Value<'gc>,
    act: &mut Activation<'_, 'gc>,
) -> Result<bool, Error<'gc>> {
    let value = value.null_check(act, None)?;
    let has_prop = match value.unpack() {
        ValueEnum::Object(obj) => {
            if let Some(dictionary) = obj.as_dictionary_object()
                && let Some(name_object) = name_value.as_object()
            {
                return Ok(dictionary.has_property_by_object(name_object));
            }
            let name = name_value.coerce_to_string(act)?;
            let multiname = Multiname::new(act.avm2().find_public_namespace(), name);
            obj.has_property_via_in(act, &multiname)?
        }
        _ => {
            let name = name_value.coerce_to_string(act)?;
            let multiname = Multiname::new(act.avm2().find_public_namespace(), name);
            if value.has_trait(act, &multiname) {
                true
            } else if let Some(proto) = value.proto(act) {
                proto.has_property(&multiname)
            } else {
                false
            }
        }
    };
    Ok(has_prop)
}

/// `nextvalue`/`nextname`: the `for..in` enumerant accessor. Mirrors `op_next_value` /
/// `op_next_name`. Stack `[value, index]`. Reifies; stash-on-throw.
pub fn next_value(value_bits: i64, index_bits: i64) -> i64 {
    next_enumerant(value_bits, index_bits, false)
}

/// `nextname`: see [`next_value`].
pub fn next_name(value_bits: i64, index_bits: i64) -> i64 {
    next_enumerant(value_bits, index_bits, true)
}

fn next_enumerant(value_bits: i64, index_bits: i64, want_name: bool) -> i64 {
    let value: Value<'_> = unsafe { from_bits(value_bits as u64) };
    let index_v: Value<'_> = unsafe { from_bits(index_bits as u64) };
    // SAFETY: called from JIT wasm inside `with_run_ctx`; `act` does not escape.
    let mut act = unsafe { context::reify() };
    match next_enumerant_inner(value, index_v, &mut act, want_name) {
        Ok(v) => to_bits(v) as i64,
        Err(e) => {
            context::stash_error(e);
            SENTINEL_BITS as i64
        }
    }
}

fn next_enumerant_inner<'gc>(
    value: Value<'gc>,
    index_v: Value<'gc>,
    act: &mut Activation<'_, 'gc>,
    want_name: bool,
) -> Result<Value<'gc>, Error<'gc>> {
    let cur_index = index_v.coerce_to_i32(act)?;
    if cur_index <= 0 {
        return Ok(if want_name { Value::Null } else { Value::Undefined });
    }
    let value = value.null_check(act, None)?;
    let object = match value.unpack() {
        ValueEnum::Object(obj) => obj,
        _ => value.proto(act).expect("Primitives always have a prototype"),
    };
    if want_name {
        object.get_enumerant_name(cur_index as u32, act)
    } else {
        object.get_enumerant_value(cur_index as u32, act)
    }
}

/// `hasnext`: the `for..in` cursor advance. Mirrors `op_has_next`. Stack `[value, index]`;
/// pushes the next enumerant index (0 = done). Reifies; stash-on-throw (index coercion).
pub fn has_next(value_bits: i64, index_bits: i64) -> i64 {
    let value: Value<'_> = unsafe { from_bits(value_bits as u64) };
    let index_v: Value<'_> = unsafe { from_bits(index_bits as u64) };
    // SAFETY: called from JIT wasm inside `with_run_ctx`; `act` does not escape.
    let mut act = unsafe { context::reify() };
    match has_next_inner(value, index_v, &mut act) {
        Ok(n) => to_bits(Value::from(n)) as i64,
        Err(e) => {
            context::stash_error(e);
            SENTINEL_BITS as i64
        }
    }
}

fn has_next_inner<'gc>(
    value: Value<'gc>,
    index_v: Value<'gc>,
    act: &mut Activation<'_, 'gc>,
) -> Result<u32, Error<'gc>> {
    let cur_index = index_v.coerce_to_i32(act)?;
    if cur_index < 0 {
        return Ok(0);
    }
    let object = match value.unpack() {
        ValueEnum::Undefined | ValueEnum::Null => None,
        ValueEnum::Object(obj) => Some(obj),
        _ => value.proto(act),
    };
    match object {
        Some(object) => object.get_next_enumerant(cur_index as u32, act),
        None => Ok(0),
    }
}

/// `newfunction`: build a closure from the baked method + the current scope chain. Mirrors
/// `op_new_function`. `method_ptr` is a baked [`Method::as_ptr`]. Reifies; no throw.
pub fn new_function(method_ptr: i64) -> i64 {
    // SAFETY: `method_ptr` is a baked `Method::as_ptr()`, live this GC-quiescent run.
    let method = unsafe { Method::from_ptr(method_ptr as usize as *const ()) };
    // SAFETY: called from JIT wasm inside `with_run_ctx`; `act` does not escape.
    let mut act = unsafe { context::reify() };
    // `reify` retargets `scope_depth` to the method's base, so this captures the method's
    // own local scope frame (mirrors the interpreter's `create_scopechain`).
    let scope = act.create_scopechain();
    let new_fn = FunctionObject::from_method(act.context, method, scope, None, None);
    to_bits(new_fn.into()) as i64
}

/// `applytype` core: `base.<T…>` (e.g. `Vector.<int>`). Mirrors `op_apply_type`. Reifies;
/// stash-on-throw (#1127 non-applicable base, or `apply` itself).
///
/// # Safety
/// As [`construct_bits`].
pub unsafe fn apply_type_bits(base_bits: i64, arg_bits: &[i64]) -> i64 {
    let args = unsafe { crate::value::bits_as_values(arg_bits) };
    let base_v: Value<'_> = unsafe { from_bits(base_bits as u64) };
    // SAFETY: called from JIT wasm inside `with_run_ctx`; `act` does not escape.
    let mut act = unsafe { context::reify() };
    let Some(base) = base_v.as_object() else {
        context::stash_error(make_error_1127(&mut act));
        return SENTINEL_BITS as i64;
    };
    match base.apply(&mut act, args) {
        Ok(v) => to_bits(v.into()) as i64,
        Err(e) => {
            context::stash_error(e);
            SENTINEL_BITS as i64
        }
    }
}

/// `applytype` (web funcref): reads `argc` types from frame memory, delegates.
#[cfg(target_arch = "wasm32")]
pub fn apply_type(base_bits: i64, args_off: i64, argc: i32) -> i64 {
    // SAFETY: `args_off` is where the compiled body `i64.store`d `argc` `Value`s this frame.
    let arg_bits: &[i64] =
        unsafe { core::slice::from_raw_parts(args_off as usize as *const i64, argc as usize) };
    unsafe { apply_type_bits(base_bits, arg_bits) }
}

/// `constructslot` core: `new (source.slot[index])(args)`. Mirrors `op_construct_slot`.
/// Reifies; stash-on-throw (null source / non-constructable slot).
///
/// # Safety
/// As [`construct_bits`].
pub unsafe fn construct_slot_bits(source_bits: i64, index: i32, arg_bits: &[i64]) -> i64 {
    let args = unsafe { crate::value::bits_as_values(arg_bits) };
    let source_v: Value<'_> = unsafe { from_bits(source_bits as u64) };
    // SAFETY: called from JIT wasm inside `with_run_ctx`; `act` does not escape.
    let mut act = unsafe { context::reify() };
    let source =
        match source_v.as_object_null_check(&mut act, None, "Cannot get_slot on primitive") {
            Ok(o) => o,
            Err(e) => {
                context::stash_error(e);
                return SENTINEL_BITS as i64;
            }
        };
    let ctor = source.get_slot(index as usize);
    match ctor.construct(&mut act, FunctionArgs::from_slice(args)) {
        Ok(v) => to_bits(v) as i64,
        Err(e) => {
            context::stash_error(e);
            SENTINEL_BITS as i64
        }
    }
}

/// `constructslot` (web funcref): reads `argc` args from frame memory, delegates.
#[cfg(target_arch = "wasm32")]
pub fn construct_slot(source_bits: i64, index: i32, args_off: i64, argc: i32) -> i64 {
    // SAFETY: `args_off` is where the compiled body `i64.store`d `argc` `Value`s this frame.
    let arg_bits: &[i64] =
        unsafe { core::slice::from_raw_parts(args_off as usize as *const i64, argc as usize) };
    unsafe { construct_slot_bits(source_bits, index, arg_bits) }
}

/// `constructprop` core: `new source.<mn>(args)`. Mirrors `op_construct_prop`. `mn` is a
/// baked static `Multiname` (lazy names decline). Reifies; stash-on-throw.
///
/// # Safety
/// As [`construct_bits`]; `mn_ptr` is a baked `*const Multiname` live this run.
pub unsafe fn construct_prop_bits(source_bits: i64, mn_ptr: i64, arg_bits: &[i64]) -> i64 {
    let args = unsafe { crate::value::bits_as_values(arg_bits) };
    let source_v: Value<'_> = unsafe { from_bits(source_bits as u64) };
    let mn: &Multiname<'_> = unsafe { &*(mn_ptr as usize as *const Multiname) };
    // SAFETY: called from JIT wasm inside `with_run_ctx`; `act` does not escape.
    let mut act = unsafe { context::reify() };
    match source_v
        .null_check(&mut act, Some(mn))
        .and_then(|source| source.construct_prop(&mut act, mn, FunctionArgs::from_slice(args)))
    {
        Ok(v) => to_bits(v) as i64,
        Err(e) => {
            context::stash_error(e);
            SENTINEL_BITS as i64
        }
    }
}

/// `constructprop` (web funcref): reads `argc` args from frame memory, delegates.
#[cfg(target_arch = "wasm32")]
pub fn construct_prop(source_bits: i64, mn_ptr: i64, args_off: i64, argc: i64) -> i64 {
    // SAFETY: `args_off` is where the compiled body `i64.store`d `argc` `Value`s this frame.
    let arg_bits: &[i64] =
        unsafe { core::slice::from_raw_parts(args_off as usize as *const i64, argc as usize) };
    unsafe { construct_prop_bits(source_bits, mn_ptr, arg_bits) }
}

/// `call` core: `function.call(receiver, args)`. Mirrors `op_call`. Reifies; stash-on-throw.
///
/// # Safety
/// As [`construct_bits`].
pub unsafe fn call_fn_bits(function_bits: i64, receiver_bits: i64, arg_bits: &[i64]) -> i64 {
    let args = unsafe { crate::value::bits_as_values(arg_bits) };
    let function: Value<'_> = unsafe { from_bits(function_bits as u64) };
    let receiver: Value<'_> = unsafe { from_bits(receiver_bits as u64) };
    // SAFETY: called from JIT wasm inside `with_run_ctx`; `act` does not escape.
    let mut act = unsafe { context::reify() };
    match function.call(&mut act, receiver, FunctionArgs::from_slice(args)) {
        Ok(v) => to_bits(v) as i64,
        Err(e) => {
            context::stash_error(e);
            SENTINEL_BITS as i64
        }
    }
}

/// `call` (web funcref): reads `argc` args from frame memory, delegates.
#[cfg(target_arch = "wasm32")]
pub fn call_fn(function_bits: i64, receiver_bits: i64, args_off: i64, argc: i64) -> i64 {
    // SAFETY: `args_off` is where the compiled body `i64.store`d `argc` `Value`s this frame.
    let arg_bits: &[i64] =
        unsafe { core::slice::from_raw_parts(args_off as usize as *const i64, argc as usize) };
    unsafe { call_fn_bits(function_bits, receiver_bits, arg_bits) }
}

/// `newobject` core: build a dynamic object from `[name, value]` pairs. Mirrors
/// `op_new_object`. `pair_bits` = `[name0, value0, …]`; consumed top-down (last pair first)
/// to match the interpreter's insertion order. Reifies; stash-on-throw (name coercion).
///
/// # Safety
/// As [`construct_bits`]; `pair_bits.len()` is even (`2 × num_pairs`).
pub unsafe fn new_object_bits(pair_bits: &[i64]) -> i64 {
    // SAFETY: called from JIT wasm inside `with_run_ctx`; `act` does not escape.
    let mut act = unsafe { context::reify() };
    let object = ScriptObject::new_object(act.context);
    // The interpreter pops (value, name) from the top, so it inserts the LAST pushed pair
    // first — replicate that order for enumeration parity.
    let num_pairs = pair_bits.len() / 2;
    for k in (0..num_pairs).rev() {
        let name: Value<'_> = unsafe { from_bits(pair_bits[2 * k] as u64) };
        let value: Value<'_> = unsafe { from_bits(pair_bits[2 * k + 1] as u64) };
        match name.coerce_to_string(&mut act) {
            Ok(s) => object.set_dynamic_property(s, value, act.gc()),
            Err(e) => {
                context::stash_error(e);
                return SENTINEL_BITS as i64;
            }
        }
    }
    to_bits(object.into()) as i64
}

/// `newobject` (web funcref): reads `2 × num_pairs` slots from frame memory, delegates.
#[cfg(target_arch = "wasm32")]
pub fn new_object(args_off: i64, num_pairs: i32) -> i64 {
    // SAFETY: `args_off` is where the compiled body `i64.store`d the pair `Value`s this frame.
    let pair_bits: &[i64] = unsafe {
        core::slice::from_raw_parts(args_off as usize as *const i64, num_pairs as usize * 2)
    };
    unsafe { new_object_bits(pair_bits) }
}

/// `hasnext2` core: advances the for..in cursor. Mirrors `op_has_next_2`. Takes the current
/// `[object, index]` register bits, returns `(more?, new_index_bits, new_object_bits)` — the
/// wrapper writes the two updated values back into the frame locals. Reifies; stash-on-throw
/// (returns unchanged locals + `undefined` so the `BailIfError` surfaces the error).
///
/// # Safety
/// As [`construct_bits`]; called inside `with_run_ctx`.
pub unsafe fn has_next_2_bits(obj_bits: i64, idx_bits: i64) -> (i64, i64, i64) {
    // SAFETY: called from JIT wasm inside `with_run_ctx`; `act` does not escape.
    let mut act = unsafe { context::reify() };
    match has_next_2_inner(obj_bits, idx_bits, &mut act) {
        Ok((more, new_idx, new_obj)) => (to_bits(Value::from(more)) as i64, new_idx, new_obj),
        Err(e) => {
            context::stash_error(e);
            (SENTINEL_BITS as i64, idx_bits, obj_bits)
        }
    }
}

fn has_next_2_inner<'gc>(
    obj_bits: i64,
    idx_bits: i64,
    act: &mut Activation<'_, 'gc>,
) -> Result<(bool, i64, i64), Error<'gc>> {
    let idx_v: Value<'gc> = unsafe { from_bits(idx_bits as u64) };
    let object_reg: Value<'gc> = unsafe { from_bits(obj_bits as u64) };
    let cur_index = idx_v.coerce_to_i32(act)?;
    if cur_index < 0 {
        // Interpreter pushes `false` and leaves the locals untouched — write them back as-is.
        return Ok((false, idx_bits, obj_bits));
    }
    let mut cur_index = cur_index as u32;
    let mut result_value = object_reg;
    let mut object = None;
    match object_reg.unpack() {
        ValueEnum::Undefined | ValueEnum::Null => {
            cur_index = 0;
        }
        ValueEnum::Object(obj) => {
            object = obj.proto();
            cur_index = obj.get_next_enumerant(cur_index, act)?;
        }
        _ => {
            let proto = object_reg.proto(act);
            if let Some(proto) = proto {
                object = proto.proto();
                cur_index = proto.get_next_enumerant(cur_index, act)?;
            }
        }
    };
    while let (Some(cur_object), 0) = (object, cur_index) {
        cur_index = cur_object.get_next_enumerant(cur_index, act)?;
        result_value = cur_object.into();
        object = cur_object.proto();
    }
    if cur_index == 0 {
        result_value = Value::Null;
    }
    Ok((cur_index != 0, to_bits(Value::from(cur_index)) as i64, to_bits(result_value) as i64))
}

/// MOP (domain-memory / FlasCC "alchemy") op codes, shared with `translate`.
pub mod mop_code {
    // `mop_load` (1 input → 1 output): memory loads + sign-extends.
    pub const LI8: i32 = 0;
    pub const LI16: i32 = 1;
    pub const LI32: i32 = 2;
    pub const LF32: i32 = 3;
    pub const LF64: i32 = 4;
    pub const SXI1: i32 = 5;
    pub const SXI8: i32 = 6;
    pub const SXI16: i32 = 7;
    // `mop_store` (2 inputs → void): memory stores.
    pub const SI8: i32 = 0;
    pub const SI16: i32 = 1;
    pub const SI32: i32 = 2;
    pub const SF32: i32 = 3;
    pub const SF64: i32 = 4;
}

/// `li8/li16/li32/lf32/lf64` + `sxi1/sxi8/sxi16`: a domain-memory load or a sign-extend,
/// selected by `code`. Loads read `activation.domain_memory()` at the popped address (throws
/// #1506 out-of-bounds); sign-extends don't touch memory. Reifies; stash-on-throw.
pub fn mop_load(addr_bits: i64, code: i32) -> i64 {
    let addr_v: Value<'_> = unsafe { from_bits(addr_bits as u64) };
    // SAFETY: called from JIT wasm inside `with_run_ctx`; `act` does not escape.
    let mut act = unsafe { context::reify() };
    match mop_load_inner(addr_v, code, &mut act) {
        Ok(v) => to_bits(v) as i64,
        Err(e) => {
            context::stash_error(e);
            SENTINEL_BITS as i64
        }
    }
}

fn mop_load_inner<'gc>(
    addr_v: Value<'gc>,
    code: i32,
    act: &mut Activation<'_, 'gc>,
) -> Result<Value<'gc>, Error<'gc>> {
    use mop_code::*;
    // Sign-extend ops operate on the value directly — no memory access.
    match code {
        SXI1 => {
            let v = addr_v.coerce_to_i32(act)?;
            return Ok(Value::Integer(v.wrapping_shl(31).wrapping_shr(31)));
        }
        SXI8 => {
            let v = addr_v.coerce_to_i32(act)?;
            return Ok(Value::Integer((v.wrapping_shl(23).wrapping_shr(23) & 0xFF) as i8 as i32));
        }
        SXI16 => {
            let v = addr_v.coerce_to_i32(act)?;
            return Ok(Value::Integer((v.wrapping_shl(15).wrapping_shr(15) & 0xFFFF) as i16 as i32));
        }
        _ => {}
    }
    let address = addr_v.coerce_to_i32(act)? as usize;
    let dm = act.domain_memory().storage();
    let v = match code {
        LI8 => match dm.dm_get(address) {
            Some(val) => Value::Integer(val as i32),
            None => return Err(make_error_1506(act)),
        },
        LI16 => {
            if address > dm.dm_len() - 2 {
                return Err(make_error_1506(act));
            }
            let val = dm.dm_read::<2>(address).ok_or_else(|| make_error_1506(act))?;
            Value::Integer(u16::from_le_bytes(val) as i32)
        }
        LI32 => {
            if address > dm.dm_len() - 4 {
                return Err(make_error_1506(act));
            }
            let val = dm.dm_read::<4>(address).ok_or_else(|| make_error_1506(act))?;
            Value::Integer(i32::from_le_bytes(val))
        }
        LF32 => {
            if address > dm.dm_len() - 4 {
                return Err(make_error_1506(act));
            }
            let val = dm.dm_read::<4>(address).ok_or_else(|| make_error_1506(act))?;
            // Preserve exact bits (like the interpreter): promote f32→f64 losslessly.
            Value::number_lossless(act.gc(), f32::from_le_bytes(val) as f64)
        }
        LF64 => {
            if address > dm.dm_len() - 8 {
                return Err(make_error_1506(act));
            }
            let val = dm.dm_read::<8>(address).ok_or_else(|| make_error_1506(act))?;
            Value::number_lossless(act.gc(), f64::from_le_bytes(val))
        }
        _ => return Err(make_error_1506(act)), // unknown code (shouldn't happen)
    };
    Ok(v)
}

/// For the JIT's INLINE `li*`/`si*` fast path: the address of the current domainMemory's
/// stable `[base, cap, len]` descriptor cell, or `0` if it isn't shareable (then every access
/// takes the helper). Called ONCE at a compiled method's entry (its result parked in a local
/// for the run) — the inline code then reads base+len from the cell on every access, so growth
/// moves are observed without re-reifying. Mirrors jit1's `dm_base_len`: only ALREADY-shareable
/// buffers get the inline path (promoting content-driven ByteArrays mid-frame desynced Starling
/// rendering), so we never `make_shareable` here. Reifies.
pub fn dm_desc_ptr() -> i64 {
    // SAFETY: called from JIT wasm inside `with_run_ctx`; `act` does not escape.
    let mut act = unsafe { context::reify() };
    let mut storage = act.domain_memory().storage_mut();
    if !storage.is_shareable() {
        return 0;
    }
    match storage.dm_base_len() {
        Some((desc, _)) => desc as i64,
        None => 0,
    }
}

/// `si8/si16/si32/sf32/sf64`: a domain-memory store selected by `code`. Pops the value and
/// the address (address is coerced FIRST, mirroring the interpreter's operand order); throws
/// #1506 out-of-bounds. Returns `undefined` (discarded). Reifies; stash-on-throw.
pub fn mop_store(value_bits: i64, addr_bits: i64, code: i32) -> i64 {
    let value_v: Value<'_> = unsafe { from_bits(value_bits as u64) };
    let addr_v: Value<'_> = unsafe { from_bits(addr_bits as u64) };
    // SAFETY: called from JIT wasm inside `with_run_ctx`; `act` does not escape.
    let mut act = unsafe { context::reify() };
    match mop_store_inner(value_v, addr_v, code, &mut act) {
        Ok(()) => UNDEFINED_BITS as i64,
        Err(e) => {
            context::stash_error(e);
            SENTINEL_BITS as i64
        }
    }
}

fn mop_store_inner<'gc>(
    value_v: Value<'gc>,
    addr_v: Value<'gc>,
    code: i32,
    act: &mut Activation<'_, 'gc>,
) -> Result<(), Error<'gc>> {
    use mop_code::*;
    let address = addr_v.coerce_to_i32(act)? as usize;
    match code {
        SI8 => {
            let val = value_v.coerce_to_i32(act)? as i8;
            let mut dm = act.domain_memory().storage_mut();
            if address >= dm.dm_len() {
                return Err(make_error_1506(act));
            }
            dm.dm_set(address, val as u8);
        }
        SI16 => {
            let val = value_v.coerce_to_i32(act)? as i16;
            let mut dm = act.domain_memory().storage_mut();
            if address > dm.dm_len() - 2 {
                return Err(make_error_1506(act));
            }
            dm.dm_write(address, &val.to_le_bytes()).map_err(|e| e.to_avm(act))?;
        }
        SI32 => {
            let val = value_v.coerce_to_i32(act)?;
            let mut dm = act.domain_memory().storage_mut();
            if address > dm.dm_len() - 4 {
                return Err(make_error_1506(act));
            }
            dm.dm_write(address, &val.to_le_bytes()).map_err(|e| e.to_avm(act))?;
        }
        SF32 => {
            let val = value_v.coerce_to_number(act)? as f32;
            let mut dm = act.domain_memory().storage_mut();
            if address > dm.dm_len() - 4 {
                return Err(make_error_1506(act));
            }
            dm.dm_write(address, &val.to_le_bytes()).map_err(|e| e.to_avm(act))?;
        }
        SF64 => {
            let val = value_v.coerce_to_number(act)?;
            let mut dm = act.domain_memory().storage_mut();
            if address > dm.dm_len() - 8 {
                return Err(make_error_1506(act));
            }
            dm.dm_write(address, &val.to_le_bytes()).map_err(|e| e.to_avm(act))?;
        }
        _ => return Err(make_error_1506(act)),
    }
    Ok(())
}

/// `throw`: `throw value`. Mirrors `op_throw` — converts the operand to an `Error` and
/// stashes it; the compiled body then `Return`s and `try_enter` propagates the stashed
/// error (a throw in a method with NO exception handler always unwinds out of the method,
/// so no local catch is modeled). Reifies. Returns `undefined` (discarded — the error wins).
pub fn throw_value(value_bits: i64) -> i64 {
    let value: Value<'_> = unsafe { from_bits(value_bits as u64) };
    // SAFETY: called from JIT wasm inside `with_run_ctx`; `act` does not escape.
    let mut act = unsafe { context::reify() };
    context::stash_error(Error::from_value(&mut act, value));
    SENTINEL_BITS as i64
}

/// `hasnext2` (web funcref): reads the two frame locals, delegates, writes the updated pair
/// back, returns the Boolean.
#[cfg(target_arch = "wasm32")]
pub fn has_next_2(obj_reg: i32, idx_reg: i32, frame_off: i64) -> i64 {
    let obj_ptr = (frame_off as usize + obj_reg as usize * 8) as *mut i64;
    let idx_ptr = (frame_off as usize + idx_reg as usize * 8) as *mut i64;
    // SAFETY: the frame locals live at `frame_off + reg*8` for this run (see `emit`'s `slot`).
    let (obj_bits, idx_bits) = unsafe { (*obj_ptr, *idx_ptr) };
    let (result, new_idx, new_obj) = unsafe { has_next_2_bits(obj_bits, idx_bits) };
    unsafe {
        *idx_ptr = new_idx;
        *obj_ptr = new_obj;
    }
    result
}

/// `call_native` core: calls the baked native method fn-pointer directly (`CallNative` — the
/// optimizer's fast path for known-native calls). `method_ptr` is a `NativeMethodImpl` fn
/// pointer (on wasm32 its integer value is its `__indirect_function_table` index). Null-checks
/// the receiver (can throw); the native method itself "should not return Err" but is handled.
///
/// # Safety
/// `method_ptr` is a live `NativeMethodImpl` baked by `translate`; `arg_bits` are `Value`s
/// stored this frame.
pub unsafe fn call_native_bits(receiver_bits: i64, method_ptr: i64, arg_bits: &[i64]) -> i64 {
    let method: NativeMethodImpl =
        unsafe { core::mem::transmute::<usize, NativeMethodImpl>(method_ptr as usize) };
    let args = unsafe { crate::value::bits_as_values(arg_bits) };
    let receiver: Value<'_> = unsafe { from_bits(receiver_bits as u64) };
    // SAFETY: called from JIT wasm inside `with_run_ctx`; `act` does not escape.
    let mut act = unsafe { context::reify() };
    let result = match receiver.null_check(&mut act, None) {
        Ok(r) => method(&mut act, r, args),
        Err(e) => Err(e),
    };
    match result {
        Ok(v) => to_bits(v) as i64,
        Err(e) => {
            context::stash_error(e);
            SENTINEL_BITS as i64
        }
    }
}

/// `call_native` (web funcref): reads `argc` args from frame memory at `args_off`, delegates.
#[cfg(target_arch = "wasm32")]
pub fn call_native(receiver_bits: i64, method_ptr: i64, args_off: i64, argc: i64) -> i64 {
    // SAFETY: `args_off` is where the compiled body `i64.store`d `argc` `Value`s this frame.
    let arg_bits: &[i64] =
        unsafe { core::slice::from_raw_parts(args_off as usize as *const i64, argc as usize) };
    unsafe { call_native_bits(receiver_bits, method_ptr, arg_bits) }
}

/// `call_method` (web funcref): reads `argc` args from frame memory at `args_off`, delegates.
#[cfg(target_arch = "wasm32")]
pub fn call_method(receiver_bits: i64, disp_id: i32, args_off: i64, argc: i32) -> i64 {
    // SAFETY: `args_off` is where the compiled body `i64.store`d `argc` `Value`s this frame.
    let arg_bits: &[i64] =
        unsafe { core::slice::from_raw_parts(args_off as usize as *const i64, argc as usize) };
    unsafe { call_method_bits(receiver_bits, disp_id, arg_bits) }
}

/// `new_array` core: `newarray` — builds an `Array` from the `argc` elements stored in the
/// outgoing-arg scratch. Reifies for the GC context (array allocation can't throw). Returns
/// the array `Value`.
///
/// # Safety
/// As [`call_property_bits`].
pub unsafe fn new_array_bits(arg_bits: &[i64]) -> i64 {
    let elems = unsafe { crate::value::bits_as_values(arg_bits) };
    // SAFETY: called from JIT wasm inside `with_run_ctx`; `act` does not escape.
    let mut act = unsafe { context::reify() };
    let storage: ArrayStorage<'_> = elems.iter().copied().collect();
    let array = ArrayObject::from_storage(act.context, storage);
    let value: Value<'_> = array.into();
    to_bits(value) as i64
}

/// `new_array` (web funcref): reads `argc` elements from frame memory at `args_off`.
#[cfg(target_arch = "wasm32")]
pub fn new_array(args_off: i64, argc: i32) -> i64 {
    // SAFETY: `args_off` is where the compiled body `i64.store`d `argc` `Value`s this frame.
    let arg_bits: &[i64] =
        unsafe { core::slice::from_raw_parts(args_off as usize as *const i64, argc as usize) };
    unsafe { new_array_bits(arg_bits) }
}

/// `cp` (web funcref): reads the `argc` outgoing args straight from the frame memory at
/// `args_off` — on wasm32 the frame lives in the main module's linear memory, so `args_off`
/// IS a valid pointer — then delegates to [`call_property_bits`]. No per-arg boundary.
#[cfg(target_arch = "wasm32")]
pub fn call_property(receiver_bits: i64, mn_ptr: i64, args_off: i64, argc: i64) -> i64 {
    // SAFETY: `args_off` is where the compiled body `i64.store`d `argc` `Value`s this frame,
    // in the same (main) linear memory this code runs in.
    let arg_bits: &[i64] =
        unsafe { core::slice::from_raw_parts(args_off as usize as *const i64, argc as usize) };
    unsafe { call_property_bits(receiver_bits, mn_ptr, arg_bits) }
}

/// `delete_property` (static multiname): `object.delete_property(mn)` after a null-check,
/// pushing the `Boolean` result. Reifies; stash-on-throw. Only emitted for non-lazy names.
pub fn delete_property(receiver_bits: i64, mn_ptr: i64) -> i64 {
    // SAFETY: live `Value` this frame; `mn_ptr` a live baked `*const Multiname`.
    let value: Value<'_> = unsafe { from_bits(receiver_bits as u64) };
    let mn: &Multiname<'_> = unsafe { &*(mn_ptr as usize as *const Multiname) };
    // SAFETY: called from JIT wasm inside `with_run_ctx`; `act` does not escape.
    let mut act = unsafe { context::reify() };
    match value
        .null_check(&mut act, Some(mn))
        .and_then(|obj| obj.delete_property(&mut act, mn))
    {
        Ok(b) => to_bits(Value::from(b)) as i64,
        Err(e) => {
            context::stash_error(e);
            SENTINEL_BITS as i64
        }
    }
}

/// `getproperty` FAST path (`obj[name]`, `name` a runtime value): the integer-index /
/// dictionary fast paths, else the slow fallback (fill the lazy `MultinameL` with `name` and
/// `get_property`). Mirrors `op_get_property_fast`. `mn_ptr` = the lazy multiname template.
pub fn get_prop_index(object_bits: i64, name_bits: i64, mn_ptr: i64) -> i64 {
    // SAFETY: live `Value`s this frame; `mn_ptr` a live baked `*const Multiname` (lazy).
    let object: Value<'_> = unsafe { from_bits(object_bits as u64) };
    let name: Value<'_> = unsafe { from_bits(name_bits as u64) };
    let mn: &Multiname<'_> = unsafe { &*(mn_ptr as usize as *const Multiname) };
    // SAFETY: called from JIT wasm inside `with_run_ctx`; `act` does not escape.
    let mut act = unsafe { context::reify() };
    // Fast path.
    if let ValueEnum::Object(obj) = object.unpack() {
        match name.unpack() {
            ValueEnum::Integer(_) | ValueEnum::Number(_) => {
                if let Some(index) = name.try_as_index() {
                    if let Some(value) = obj.get_index_property(index) {
                        return to_bits(value) as i64;
                    }
                }
            }
            ValueEnum::Object(name_object) => {
                if let Some(dict) = obj.as_dictionary_object() {
                    return to_bits(dict.get_property_by_object(name_object)) as i64;
                }
            }
            _ => {}
        }
    }
    // Slow fallback: fill the multiname with `name` directly (the reified builtin Activation
    // has no operand stack, so `fill_with_runtime_params` can't be used), then get_property.
    let filled = match mn.fill_with_runtime_name(name, &mut act) {
        Ok(f) => f,
        Err(e) => {
            context::stash_error(e);
            return SENTINEL_BITS as i64;
        }
    };
    match object.null_check(&mut act, Some(&filled)).and_then(|o| o.get_property(&filled, &mut act)) {
        Ok(v) => to_bits(v) as i64,
        Err(e) => {
            context::stash_error(e);
            SENTINEL_BITS as i64
        }
    }
}

/// `setproperty` FAST path (`obj[name] = value`). Fast index/dictionary paths, else slow
/// fallback. Void (returns `undefined`). Mirrors `op_set_property_fast`.
pub fn set_prop_index(object_bits: i64, name_bits: i64, value_bits: i64, mn_ptr: i64) -> i64 {
    // SAFETY: live `Value`s this frame; `mn_ptr` a live baked `*const Multiname` (lazy).
    let object: Value<'_> = unsafe { from_bits(object_bits as u64) };
    let name: Value<'_> = unsafe { from_bits(name_bits as u64) };
    let value: Value<'_> = unsafe { from_bits(value_bits as u64) };
    let mn: &Multiname<'_> = unsafe { &*(mn_ptr as usize as *const Multiname) };
    // SAFETY: called from JIT wasm inside `with_run_ctx`; `act` does not escape.
    let mut act = unsafe { context::reify() };
    // Fast path.
    if let ValueEnum::Object(obj) = object.unpack() {
        match name.unpack() {
            ValueEnum::Integer(_) | ValueEnum::Number(_) => {
                if let Some(index) = name.try_as_index() {
                    if let Some(result) = obj.set_index_property(&mut act, index, value) {
                        if let Err(e) = result {
                            context::stash_error(e);
                            return SENTINEL_BITS as i64;
                        }
                        return UNDEFINED_BITS as i64;
                    }
                }
            }
            ValueEnum::Object(name_object) => {
                if let Some(dict) = obj.as_dictionary_object() {
                    dict.set_property_by_object(name_object, value, act.gc());
                    return UNDEFINED_BITS as i64;
                }
            }
            _ => {}
        }
    }
    // Slow fallback (fill the multiname with `name` directly — no operand stack).
    let filled = match mn.fill_with_runtime_name(name, &mut act) {
        Ok(f) => f,
        Err(e) => {
            context::stash_error(e);
            return SENTINEL_BITS as i64;
        }
    };
    let result = object
        .null_check(&mut act, Some(&filled))
        .and_then(|o| o.set_property(&filled, value, &mut act));
    if let Err(e) = result {
        context::stash_error(e);
        return SENTINEL_BITS as i64;
    }
    UNDEFINED_BITS as i64
}

/// `set_property` (static multiname): `object.set_property(mn, value)` after a null-check —
/// `mn` a baked `*const Multiname`. Reifies (a setter runs AS3, may re-enter/throw; the
/// null-check throws #1009 on null/undefined). Stash-on-throw; the body's `BailIfError`
/// surfaces it. `object` is the deeper operand, `value` the top (matching `pop value; pop
/// object`).
pub fn set_property(receiver_bits: i64, mn_ptr: i64, value_bits: i64) {
    // SAFETY: live `Value`s this frame; `mn_ptr` a live baked `*const Multiname`.
    let receiver: Value<'_> = unsafe { from_bits(receiver_bits as u64) };
    let value: Value<'_> = unsafe { from_bits(value_bits as u64) };
    let mn: &Multiname<'_> = unsafe { &*(mn_ptr as usize as *const Multiname) };
    // SAFETY: called from JIT wasm inside `with_run_ctx`; `act` does not escape.
    let mut act = unsafe { context::reify() };
    let result = receiver
        .null_check(&mut act, Some(mn))
        .and_then(|obj| obj.set_property(mn, value, &mut act));
    if let Err(e) = result {
        context::stash_error(e);
    }
}

/// `set_slot`: `object.set_slot[_no_coerce](index, value)` after an object-null-check.
/// `mode` — 0: coerce to the slot's trait type (`set_slot`); 1: coerce to int then write
/// (`SetSlotCoerceI`); 2: write verbatim (`SetSlotNoCoerce`). Reifies (coercion may throw;
/// null-check throws on a primitive receiver). `object` deeper, `value` top.
pub fn set_slot(receiver_bits: i64, index: i32, value_bits: i64, mode: i32) {
    // SAFETY: live `Value`s this frame.
    let receiver: Value<'_> = unsafe { from_bits(receiver_bits as u64) };
    let value: Value<'_> = unsafe { from_bits(value_bits as u64) };
    // SAFETY: called from JIT wasm inside `with_run_ctx`; `act` does not escape.
    let mut act = unsafe { context::reify() };
    let id = index as usize;
    let result = (|| -> Result<(), Error<'_>> {
        let obj = receiver.as_object_null_check(&mut act, None, "Cannot set_slot on primitive")?;
        match mode {
            0 => obj.set_slot(id, value, &mut act)?,
            1 => {
                let v = value.coerce_to_i32(&mut act)?;
                obj.set_slot_no_coerce(id, v.into(), act.gc());
            }
            _ => obj.set_slot_no_coerce(id, value, act.gc()),
        }
        Ok(())
    })();
    if let Err(e) = result {
        context::stash_error(e);
    }
}

pub fn coerce_return(value_bits: i64, class_ptr: i64) -> i64 {
    // SAFETY: `value_bits` is a live `Value` this frame; `class_ptr` is a baked, live
    // `Class` (single-`Gc` handle) — reverse of translate's erasure.
    let value: Value<'_> = unsafe { from_bits(value_bits as u64) };
    let class: Class<'_> =
        unsafe { core::mem::transmute::<*const (), Class<'_>>(class_ptr as usize as *const ()) };
    // SAFETY: called from JIT wasm inside `with_run_ctx`; `act` does not escape.
    let mut act = unsafe { context::reify() };
    match value.coerce_to_type(&mut act, class) {
        Ok(v) => to_bits(v) as i64,
        Err(e) => {
            context::stash_error(e);
            SENTINEL_BITS as i64
        }
    }
}

/// `binop`: the dynamic binary operators (`add`/`subtract`/…/comparisons), dispatched by a
/// [`binop_code`]. Reproduces the interpreter's op bodies EXACTLY — including the int
/// fast-paths (so `int-int` stays `int`) and the **operand-2-first** coercion order (a
/// side-effecting `valueOf` must run in the same order the interpreter would). Reifies a
/// callee-owned Activation (a coercion may call `valueOf` → re-enter/throw). On a throw,
/// stashes the error and returns `undefined`; the body's following `BailIfError` surfaces it.
pub fn binop(a_bits: i64, b_bits: i64, op: i32) -> i64 {
    // SAFETY: both are live `Value`s the JIT produced this frame.
    let v1: Value<'_> = unsafe { from_bits(a_bits as u64) };
    let v2: Value<'_> = unsafe { from_bits(b_bits as u64) };
    // SAFETY: called from JIT wasm inside `with_run_ctx`; `act` does not escape.
    let mut act = unsafe { context::reify() };
    match binop_compute(v1, v2, op, &mut act) {
        Ok(v) => to_bits(v) as i64,
        Err(e) => {
            context::stash_error(e);
            SENTINEL_BITS as i64
        }
    }
}

fn binop_compute<'gc>(
    v1: Value<'gc>,
    v2: Value<'gc>,
    op: i32,
    act: &mut Activation<'_, 'gc>,
) -> Result<Value<'gc>, Error<'gc>> {
    use binop_code::*;
    Ok(match op {
        ADD => return v1.add(v2, act),
        SUBTRACT => match (v1.unpack(), v2.unpack()) {
            (ValueEnum::Integer(n1), ValueEnum::Integer(n2)) => match n1.checked_sub(n2) {
                Some(r) => r.into(),
                None => ((n1 as i64 - n2 as i64) as f64).into(),
            },
            (ValueEnum::Number(n1), ValueEnum::Number(n2)) => (n1 - n2).into(),
            _ => {
                let b = v2.coerce_to_number(act)?;
                let a = v1.coerce_to_number(act)?;
                (a - b).into()
            }
        },
        MULTIPLY => {
            if let (ValueEnum::Integer(n1), ValueEnum::Integer(n2)) = (v1.unpack(), v2.unpack()) {
                if let Some(r) = n1.checked_mul(n2) {
                    return Ok(r.into());
                }
            }
            let b = v2.coerce_to_number(act)?;
            let a = v1.coerce_to_number(act)?;
            (a * b).into()
        }
        DIVIDE => {
            let b = v2.coerce_to_number(act)?;
            let a = v1.coerce_to_number(act)?;
            (a / b).into()
        }
        MODULO => {
            let b = v2.coerce_to_number(act)?;
            let a = v1.coerce_to_number(act)?;
            (a % b).into()
        }
        BITAND => {
            let b = v2.coerce_to_i32(act)?;
            let a = v1.coerce_to_i32(act)?;
            (a & b).into()
        }
        BITOR => {
            let b = v2.coerce_to_i32(act)?;
            let a = v1.coerce_to_i32(act)?;
            (a | b).into()
        }
        BITXOR => {
            let b = v2.coerce_to_i32(act)?;
            let a = v1.coerce_to_i32(act)?;
            (a ^ b).into()
        }
        LSHIFT => {
            let b = v2.coerce_to_u32(act)?;
            let a = v1.coerce_to_i32(act)?;
            (a << (b & 0x1F)).into()
        }
        RSHIFT => {
            let b = v2.coerce_to_u32(act)?;
            let a = v1.coerce_to_i32(act)?;
            (a >> (b & 0x1F)).into()
        }
        URSHIFT => {
            let b = v2.coerce_to_u32(act)?;
            let a = v1.coerce_to_u32(act)?;
            (a >> (b & 0x1F)).into()
        }
        EQUALS => v1.abstract_eq(&v2, act)?.into(),
        STRICT_EQUALS => v1.strict_eq(&v2).into(),
        LESS_THAN => v1.abstract_lt(&v2, act)?.unwrap_or(false).into(),
        LESS_EQUALS => (!v2.abstract_lt(&v1, act)?.unwrap_or(true)).into(),
        GREATER_THAN => v2.abstract_lt(&v1, act)?.unwrap_or(false).into(),
        GREATER_EQUALS => (!v1.abstract_lt(&v2, act)?.unwrap_or(true)).into(),
        // Integer arithmetic (`*_i`): int fast-path, else coerce both. Add/Sub coerce op1
        // first, Multiply coerces op2 first (matches the interpreter op bodies exactly).
        ADD_I => match (v1.unpack(), v2.unpack()) {
            (ValueEnum::Integer(a), ValueEnum::Integer(b)) => a.wrapping_add(b).into(),
            _ => {
                let a = v1.coerce_to_i32(act)?;
                let b = v2.coerce_to_i32(act)?;
                a.wrapping_add(b).into()
            }
        },
        SUBTRACT_I => match (v1.unpack(), v2.unpack()) {
            (ValueEnum::Integer(a), ValueEnum::Integer(b)) => a.wrapping_sub(b).into(),
            _ => {
                let a = v1.coerce_to_i32(act)?;
                let b = v2.coerce_to_i32(act)?;
                a.wrapping_sub(b).into()
            }
        },
        MULTIPLY_I => {
            let b = v2.coerce_to_i32(act)?;
            let a = v1.coerce_to_i32(act)?;
            a.wrapping_mul(b).into()
        }
        _ => Value::Undefined, // unknown code — translate never emits this
    })
}

/// `unop`: the dynamic unary operators, dispatched by a [`unop_code`]. Mirrors the
/// interpreter bodies; reifies for the coercion. Stash-on-throw like [`binop`].
pub fn unop(a_bits: i64, op: i32) -> i64 {
    // SAFETY: a live `Value` this frame.
    let v: Value<'_> = unsafe { from_bits(a_bits as u64) };
    // SAFETY: called from JIT wasm inside `with_run_ctx`; `act` does not escape.
    let mut act = unsafe { context::reify() };
    match unop_compute(v, op, &mut act) {
        Ok(v) => to_bits(v) as i64,
        Err(e) => {
            context::stash_error(e);
            SENTINEL_BITS as i64
        }
    }
}

fn unop_compute<'gc>(
    v: Value<'gc>,
    op: i32,
    act: &mut Activation<'_, 'gc>,
) -> Result<Value<'gc>, Error<'gc>> {
    use unop_code::*;
    Ok(match op {
        NEGATE => (-v.coerce_to_number(act)?).into(),
        INCREMENT => (v.coerce_to_number(act)? + 1.0).into(),
        DECREMENT => (v.coerce_to_number(act)? - 1.0).into(),
        NOT => (!v.coerce_to_boolean()).into(),
        BITNOT => (!v.coerce_to_i32(act)?).into(),
        COERCE_B => v.coerce_to_boolean().into(),
        COERCE_D => v.coerce_to_number(act)?.into(),
        COERCE_I => v.coerce_to_i32(act)?.into(),
        COERCE_U => v.coerce_to_u32(act)?.into(),
        COERCE_S => match v.unpack() {
            ValueEnum::Undefined | ValueEnum::Null => Value::Null,
            ValueEnum::String(_) => v,
            _ => v.coerce_to_string(act)?.into(),
        },
        CONVERT_S => v.coerce_to_string(act)?.into(),
        COERCE_O => match v.unpack() {
            ValueEnum::Undefined | ValueEnum::Null => Value::Null,
            _ => v,
        },
        INCREMENT_I => v.coerce_to_i32(act)?.wrapping_add(1).into(),
        DECREMENT_I => v.coerce_to_i32(act)?.wrapping_sub(1).into(),
        NEGATE_I => v.coerce_to_i32(act)?.wrapping_neg().into(),
        _ => Value::Undefined, // unknown code — translate never emits this
    })
}
