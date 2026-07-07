//! GC-aware helper functions the JIT calls, plus the thread-local `Activation`
//! they reach.
//!
//! A JIT'd method can't do GC-aware work itself (allocation, coercion, property
//! access all need the `!Send` GC arena), so it emits `call`s to these Rust
//! helpers instead. Each takes/returns a raw NaN-boxed `Value` as an `i64`.
//!
//! The current [`Activation`] is passed *out of band* through a thread-local set
//! by [`with_activation`] around each JIT run — the standard JIT trick, since the
//! host import functions the runner binds are `'static` and can't capture a
//! borrowed activation. This is single-threaded per player, and the run is
//! synchronous within the activation's scope, so the erased pointer is valid for
//! the duration of the call.

use std::cell::{Cell, RefCell};

use ruffle_core::avm2::error::{
    make_error_1041, make_error_1108, make_error_1127, make_null_or_undefined_error,
};
use ruffle_core::avm2::{
    Activation, ArrayObject, ArrayStorage, Class, ClassObject, Error, FunctionArgs, Method,
    Multiname, Namespace, NamespaceObject, NativeMethodImpl, Scope, ScriptObject, TObject, Value,
    ValueEnum,
};

/// The whole per-run helper context — installed with **one pointer swap** per
/// JIT run by [`with_run_ctx`]. This replaced nine nested `with_*` installers
/// (one thread-local each): their ~36 thread-local operations per JIT call
/// showed up as a hefty slice of `try_run`/`with_activation` self time in
/// Starling gameplay profiles, where millions of small methods enter the JIT.
///
/// The struct lives on `try_run`'s stack for the (synchronous) run, so the
/// installed pointer never dangles; helpers read fields through it (one
/// thread-local read + a field load). Every field is lifetime-erased exactly
/// like the per-field thread-locals it replaced.
pub(crate) struct RunCtx {
    /// The current activation (`&mut Activation`, erased). See [`activation`].
    activation: *mut (),
    /// The method's declared return type (`Class`, erased; null = none/`*`),
    /// for [`coerce_return`].
    return_type: *const (),
    /// Multiname table: live `Gc<Multiname>` addresses, one per mn-bearing op
    /// in op order. See [`multiname`].
    multinames: (*const *const (), usize),
    /// Pre-resolved script-global `Value` bits per `GetScriptGlobals` op.
    script_globals: (*const u64, usize),
    /// Pre-resolved string `Value` bits per `PushString` op.
    push_strings: (*const u64, usize),
    /// Erased `Class` addresses per `Coerce`/`NewClass`/`NewActivation` op.
    coerce_classes: (*const *const (), usize),
    /// Erased `NativeMethodImpl` fn pointers per `CallNative` op.
    natives: (*const *const (), usize),
    /// Erased `Namespace`s per `PushNamespace` op.
    namespaces: (*const *const (), usize),
    /// The executing method (`Method`, erased), for [`dispatch_exc`]/[`new_catch`].
    method: *const (),
}

impl RunCtx {
    /// Builds the context from the run's parts, erasing lifetimes exactly as
    /// the old `with_*` installers did. The tables are `Compiled`'s cached
    /// slices and the activation/method are the caller's — all alive for the
    /// whole synchronous run.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new<'gc>(
        activation: &mut Activation<'_, 'gc>,
        return_type: Option<Class<'gc>>,
        multinames: &[*const ()],
        script_globals: &[u64],
        push_strings: &[u64],
        coerce_classes: &[*const ()],
        natives: &[*const ()],
        namespaces: &[*const ()],
        method: Method<'gc>,
    ) -> Self {
        RunCtx {
            activation: activation as *mut Activation<'_, 'gc> as *mut (),
            // SAFETY: `Class`/`Method` are pointer-sized `Gc` handles alive for
            // the run; erased for storage, reconstructed within the same run.
            return_type: match return_type {
                Some(c) => unsafe { std::mem::transmute::<Class<'_>, *const ()>(c) },
                None => std::ptr::null(),
            },
            multinames: (multinames.as_ptr(), multinames.len()),
            script_globals: (script_globals.as_ptr(), script_globals.len()),
            push_strings: (push_strings.as_ptr(), push_strings.len()),
            coerce_classes: (coerce_classes.as_ptr(), coerce_classes.len()),
            natives: (natives.as_ptr(), natives.len()),
            namespaces: (namespaces.as_ptr(), namespaces.len()),
            method: unsafe { std::mem::transmute::<Method<'_>, *const ()>(method) },
        }
    }
}

/// Installs `ctx` as the current run context for the duration of `f`, restoring
/// the previous one after — so nested JIT runs (re-entry via a helper call) are
/// safe. ONE thread-local swap in, one out.
pub(crate) fn with_run_ctx<R>(ctx: &RunCtx, f: impl FnOnce() -> R) -> R {
    let prev = RUN_CTX.with(|c| c.replace(std::ptr::from_ref(ctx)));
    let result = f();
    RUN_CTX.with(|c| c.set(prev));
    result
}

/// The installed [`RunCtx`].
///
/// # Safety
/// Only call from a helper invoked while inside [`with_run_ctx`]; the returned
/// reference must not escape the call.
unsafe fn run_ctx<'a>() -> &'a RunCtx {
    let ptr = RUN_CTX.with(|c| c.get());
    debug_assert!(!ptr.is_null(), "JIT helper called with no run context installed");
    // SAFETY: delegated to the caller — the context lives on `try_run`'s stack
    // for the whole synchronous run.
    unsafe { &*ptr }
}

thread_local! {
    /// The installed [`RunCtx`] (null outside a run). See [`with_run_ctx`].
    static RUN_CTX: Cell<*const RunCtx> = const { Cell::new(std::ptr::null()) };
    /// The most recently caught exception's `Value` bits, stashed by [`dispatch_exc`]
    /// for the catch block's entry [`pop_caught`] to place on the operand stack.
    static CAUGHT: Cell<i64> = const { Cell::new(0) };
    /// Args spilled for a pending `call_method`, pushed top-of-stack-first by
    /// [`push_call_arg`]; `call_method` drains the last `argc` and reverses them.
    static CALL_ARGS: RefCell<Vec<i64>> = const { RefCell::new(Vec::new()) };
    /// A thrown error captured by [`call_method`] (lifetime-erased). The emitted
    /// code checks [`pending_error`] after each call and bails; `try_run` then
    /// takes it via [`take_pending_error`] and propagates it — all within the same
    /// synchronous run, so the erased `'gc` is valid.
    static PENDING_ERROR: RefCell<Option<Error<'static>>> = const { RefCell::new(None) };
}

/// `returnvalue` return-type coercion: coerces `value` to the method's declared
/// return type (from the run context); identity when none/`*`. A failing
/// coercion (`#1034`) is stashed in `PENDING_ERROR` and `try_run` propagates it.
fn coerce_return(value: i64) -> i64 {
    let ptr = unsafe { run_ctx() }.return_type;
    if ptr.is_null() {
        return value;
    }
    let activation = unsafe { activation() };
    // SAFETY: reverse of the erasure in `RunCtx::new`, within the same run.
    let class = unsafe { std::mem::transmute::<*const (), Class<'_>>(ptr) };
    match to_value(value).coerce_to_type(activation, class) {
        Ok(v) => from_value(v),
        Err(e) => {
            // SAFETY: erase `'gc` for thread-local storage; taken this same run.
            let erased: Error<'static> = unsafe { std::mem::transmute(e) };
            PENDING_ERROR.with(|slot| *slot.borrow_mut() = Some(erased));
            from_value(Value::Undefined)
        }
    }
}

/// The current activation.
///
/// # Safety
/// Only call from a helper invoked while inside [`with_run_ctx`]; do not let
/// the reference escape the call.
unsafe fn activation<'a, 'gc>() -> &'a mut Activation<'a, 'gc> {
    let ptr = unsafe { run_ctx() }.activation;
    debug_assert!(!ptr.is_null(), "JIT helper called with no activation installed");
    // SAFETY: delegated to the caller (see above).
    unsafe { &mut *(ptr as *mut Activation<'a, 'gc>) }
}

/// `getscriptglobals`: returns the pre-resolved global-object `Value` bits for the
/// `k`-th `GetScriptGlobals` op (from the run context). A plain table read —
/// the resolution (and any `#error` on script init) happened in `try_run`.
fn get_script_globals(k: i64) -> i64 {
    let (ptr, len) = unsafe { run_ctx() }.script_globals;
    let k = k as usize;
    debug_assert!(k < len, "JIT getscriptglobals index {k} out of range (len {len})");
    if k < len {
        // SAFETY: `ptr` points at a `&[u64]` alive for the run (set by
        // the run context); `k` is in range (the emitter only produces indices
        // it populated).
        unsafe { *ptr.add(k) as i64 }
    } else {
        from_value(Value::Undefined)
    }
}

/// `coerce <class>`: `ToType(value, class[k])` where `class[k]` is the `k`-th entry
/// of the run context's coerce-class table. A failing coercion (`#1034`) is
/// stashed in `PENDING_ERROR` and the emitted code bails/dispatches, exactly
/// like [`coerce_return`]. Only call inside [`with_run_ctx`].
pub(crate) fn coerce(value: i64, k: i64) -> i64 {
    let (ptr, len) = unsafe { run_ctx() }.coerce_classes;
    let k = k as usize;
    debug_assert!(k < len, "JIT coerce class index {k} out of range (len {len})");
    if k >= len {
        return value;
    }
    // SAFETY: `ptr` points at a `&[*const ()]` alive for the run; `k` is in range
    // (the emitter only produces indices it populated). The entry is a live
    // `Class` address erased for storage; reverse the erasure within the same run.
    let class = unsafe { std::mem::transmute::<*const (), Class<'_>>(*ptr.add(k)) };
    // SAFETY: helpers run only inside `with_run_ctx`.
    let activation = unsafe { activation() };
    match to_value(value).coerce_to_type(activation, class) {
        Ok(v) => from_value(v),
        Err(e) => {
            // SAFETY: erase `'gc` for thread-local storage; taken this same run.
            let erased: Error<'static> = unsafe { std::mem::transmute(e) };
            PENDING_ERROR.with(|slot| *slot.borrow_mut() = Some(erased));
            from_value(Value::Undefined)
        }
    }
}

/// The `k`-th coerce-table `Class` (shared by `coerce`/`newclass`/`newactivation`
/// — they bump the same `next_coerce` counter in op order).
///
/// # Safety
/// Only call from a helper running inside [`with_run_ctx`], with `k` in range
/// (the emitter only produces indices it populated).
unsafe fn coerce_class_at<'gc>(k: usize) -> Class<'gc> {
    let (ptr, len) = unsafe { run_ctx() }.coerce_classes;
    debug_assert!(k < len, "JIT coerce class index {k} out of range (len {len})");
    // SAFETY: `ptr` points at a `&[*const ()]` alive for the run; entry `k` is a
    // live `Class` address erased for storage (see `RunCtx::new`).
    unsafe { std::mem::transmute::<*const (), Class<'gc>>(*ptr.add(k)) }
}

/// The current method (reverse of the erasure in [`RunCtx::new`]).
///
/// # Safety
/// Only call from a helper invoked while inside [`with_run_ctx`].
unsafe fn current_method<'gc>() -> Method<'gc> {
    let ptr = unsafe { run_ctx() }.method;
    debug_assert!(!ptr.is_null(), "JIT exception helper with no method installed");
    // SAFETY: delegated to the caller.
    unsafe { std::mem::transmute::<*const (), Method<'gc>>(ptr) }
}

/// `dispatch_exc`: routes a thrown exception (already stashed in `PENDING_ERROR`)
/// at op `op_idx` through the method's handlers. Returns the caught handler's
/// target op index (`>= 0`), or `-1` to propagate (re-stashing the error). On a
/// catch, the caught value is stashed for [`pop_caught`].
fn dispatch_exc(op_idx: i64) -> i64 {
    let activation = unsafe { activation() };
    let method = unsafe { current_method() };
    let Some(error) = take_pending_error() else {
        // No pending error — should not happen (only called on a thrown path).
        return -1;
    };
    match activation.jit_dispatch_exception(method, op_idx as usize + 1, error) {
        Ok((target, caught)) => {
            CAUGHT.with(|c| c.set(from_value(caught)));
            target as i64
        }
        Err(e) => {
            // SAFETY: erase `'gc` for thread-local storage; taken this same run.
            let erased: Error<'static> = unsafe { std::mem::transmute(e) };
            PENDING_ERROR.with(|slot| *slot.borrow_mut() = Some(erased));
            -1
        }
    }
}

/// `newcatch`: builds and returns the catch scope object for handler `index`.
fn new_catch(index: i64) -> i64 {
    let activation = unsafe { activation() };
    let method = unsafe { current_method() };
    from_value(activation.jit_new_catch(method, index as usize))
}

/// Catch-block entry: returns the caught exception value stashed by [`dispatch_exc`].
fn pop_caught(_ignored: i64) -> i64 {
    CAUGHT.with(|c| c.get())
}

/// `throw`: takes the thrown `Value` bits, stashes `Error::from_value(v)` in
/// `PENDING_ERROR` (which `try_run` propagates after the run), and returns
/// `undefined` bits. The emitted code `Return`s right after, so the method stops —
/// matching the interpreter's `Err(Error::from_value(..))`. Sound only for methods
/// with no exception handlers (a local `catch`/`finally` would intercept the throw);
/// [`crate::compile_method`] declines `throw` when the method has handlers.
fn throw_value(bits: i64) -> i64 {
    let activation = unsafe { activation() };
    let error = Error::from_value(activation, to_value(bits));
    // SAFETY: erase `'gc` for thread-local storage; `try_run` takes it this same run.
    let erased: Error<'static> = unsafe { std::mem::transmute(error) };
    PENDING_ERROR.with(|slot| *slot.borrow_mut() = Some(erased));
    from_value(Value::Undefined)
}

/// `pushstring`: returns the pre-resolved `Value` bits for the `k`-th `PushString`
/// op (from the run context). A plain table read — the string atom was
/// resolved from the method's constant pool in `try_run`.
fn get_push_string(k: i64) -> i64 {
    let (ptr, len) = unsafe { run_ctx() }.push_strings;
    let k = k as usize;
    debug_assert!(k < len, "JIT pushstring index {k} out of range (len {len})");
    if k < len {
        // SAFETY: `ptr` points at a `&[u64]` alive for the run (set by
        // the run context); `k` is in range (the emitter only produces indices
        // it populated).
        unsafe { *ptr.add(k) as i64 }
    } else {
        from_value(Value::Undefined)
    }
}

/// The `k`-th multiname of the current method as a `&Multiname<'gc>`.
///
/// # Safety
/// Only call from a helper running inside [`with_run_ctx`], with `k` in range
/// (the emitter only produces indices it populated); the returned reference must
/// not escape the call. `'gc` is unchecked — bind it to the current activation's.
unsafe fn multiname<'gc>(k: usize) -> &'gc Multiname<'gc> {
    let (ptr, len) = unsafe { run_ctx() }.multinames;
    debug_assert!(k < len, "JIT getproperty multiname index {k} out of range (len {len})");
    // SAFETY: `ptr` points at a `&[*const ()]` alive for the run; entry `k` is a
    // live `Gc<Multiname>` address (see `RunCtx::new`). Delegated to caller.
    unsafe {
        let entry = *ptr.add(k);
        &*(entry as *const Multiname<'gc>)
    }
}

fn to_value<'gc>(bits: i64) -> Value<'gc> {
    // SAFETY: `Value` is a NaN-boxed `u64`. The JIT only passes bits it received
    // from a `Value` (a local slot or a prior helper result).
    unsafe { std::mem::transmute(bits as u64) }
}

fn from_value(value: Value<'_>) -> i64 {
    // SAFETY: `Value` is a NaN-boxed `u64`.
    unsafe { std::mem::transmute::<Value<'_>, u64>(value) as i64 }
}

// Unary `Value`→`Value` helpers — the general forms of ops the int fast path
// can't do (they work on *any* `Value`, coercing as AVM2 specifies, not just on
// `int`). A coercion error falls back to the input for now (proper error
// propagation is later work); every result is a value-type (Number/int/Boolean),
// never a GC pointer, so returning it is sound.

/// Helper 0 — `increment`: `ToNumber(v) + 1`.
fn increment(v: i64) -> i64 {
    // SAFETY: helpers run only inside `with_run_ctx`.
    let activation = unsafe { activation() };
    match to_value(v).coerce_to_number(activation) {
        Ok(n) => from_value(Value::from(n + 1.0)),
        Err(_) => v,
    }
}

/// Helper 1 — `decrement`: `ToNumber(v) - 1`.
fn decrement(v: i64) -> i64 {
    let activation = unsafe { activation() };
    match to_value(v).coerce_to_number(activation) {
        Ok(n) => from_value(Value::from(n - 1.0)),
        Err(_) => v,
    }
}

/// Helper 2 — `negate`: `-ToNumber(v)`.
fn negate(v: i64) -> i64 {
    let activation = unsafe { activation() };
    match to_value(v).coerce_to_number(activation) {
        Ok(n) => from_value(Value::from(-n)),
        Err(_) => v,
    }
}

/// Helper 3 — `bitnot`: `!ToInt32(v)`.
fn bit_not(v: i64) -> i64 {
    let activation = unsafe { activation() };
    match to_value(v).coerce_to_i32(activation) {
        Ok(n) => from_value(Value::from(!n)),
        Err(_) => v,
    }
}

/// Helper 4 — `not`: `!ToBoolean(v)` (infallible, no activation needed).
fn not(v: i64) -> i64 {
    from_value(Value::from(!to_value(v).coerce_to_boolean()))
}

/// Helper 5 — `to_boolean`: `ToBoolean(v)` as a *raw* `0`/`1` `i64` (not a boxed
/// `Value`). The boxed branch ops (`IfTrueBoxed`/`IfFalseBoxed`) call this and
/// then `i32.wrap` the result into the branch condition. Infallible.
fn to_boolean(v: i64) -> i64 {
    to_value(v).coerce_to_boolean() as i64
}

/// Helper 11 — `coerce_u` (`convert_u`/`coerce_u`): `ToUint32(v)` as a `uint`.
fn coerce_u(v: i64) -> i64 {
    let activation = unsafe { activation() };
    match to_value(v).coerce_to_u32(activation) {
        Ok(u) => from_value(Value::from(u)),
        Err(_) => v,
    }
}

/// Helper 12 — `coerce_i` (`convert_i`/`coerce_i`): `ToInt32(v)` as an `int`.
fn coerce_i(v: i64) -> i64 {
    let activation = unsafe { activation() };
    match to_value(v).coerce_to_i32(activation) {
        Ok(n) => from_value(Value::from(n)),
        Err(_) => v,
    }
}

// Sign-extend ops (`sxi1`/`sxi8`/`sxi16`): `ToInt32(v)`, then sign-extend from the
// low 1/8/16 bits. Mirror `op_sxi*` exactly (int result). A throwing coercion
// falls back to the input (infallible-ABI caveat).

/// Helper 13 — `sxi1`: sign-extend from bit 0.
fn sxi1(v: i64) -> i64 {
    let activation = unsafe { activation() };
    match to_value(v).coerce_to_i32(activation) {
        Ok(n) => from_value(Value::from(n.wrapping_shl(31).wrapping_shr(31))),
        Err(_) => v,
    }
}

/// Helper 14 — `sxi8`: sign-extend from bit 7.
fn sxi8(v: i64) -> i64 {
    let activation = unsafe { activation() };
    match to_value(v).coerce_to_i32(activation) {
        Ok(n) => from_value(Value::from((n.wrapping_shl(23).wrapping_shr(23) & 0xFF) as i8 as i32)),
        Err(_) => v,
    }
}

/// Helper 15 — `sxi16`: sign-extend from bit 15.
fn sxi16(v: i64) -> i64 {
    let activation = unsafe { activation() };
    match to_value(v).coerce_to_i32(activation) {
        Ok(n) => from_value(Value::from((n.wrapping_shl(15).wrapping_shr(15) & 0xFFFF) as i16 as i32)),
        Err(_) => v,
    }
}

/// Helper 6 — `push_scope`: `pushscope`. Pops the scope object, null-checks it,
/// pushes it onto the *real* Activation scope stack (so `getscopeobject` can read
/// it). Returns an (ignored) dummy. The scopes are cleared on method exit by the
/// caller's `Activation::cleanup`, and by the differential verifier before its
/// interpreter re-run (see `WasmJit::try_run`).
fn push_scope(v: i64) -> i64 {
    // SAFETY: helpers run only inside `with_run_ctx`.
    let activation = unsafe { activation() };
    if let Ok(obj) = to_value(v).null_check(activation, None) {
        activation.push_scope(Scope::new(obj));
    }
    0
}

/// Helper 23 — `pop_scope`: `popscope`. Pops the top of the *real* Activation scope
/// stack. No operand-stack effect; returns an (ignored) dummy.
fn pop_scope(_ignored: i64) -> i64 {
    // SAFETY: helpers run only inside `with_run_ctx`.
    let activation = unsafe { activation() };
    activation.pop_scope();
    0
}

/// Helper 24 — `get_outer_scope`: `getouterscope index`. Pushes the `index`-th
/// outer (captured) scope's values object.
fn get_outer_scope(index: i64) -> i64 {
    // SAFETY: helpers run only inside `with_run_ctx`.
    let activation = unsafe { activation() };
    from_value(activation.jit_outer_scope(index as usize))
}

/// Helper 25 — `coerce_s`: `coerces`. `ToString` coercion (null/undefined → null,
/// String passthrough, else `coerce_to_string`). A throwing `toString` stashes the
/// error in `PENDING_ERROR` (like `coerce_return`); the emitted code bails after.
fn coerce_s(v: i64) -> i64 {
    let activation = unsafe { activation() };
    let value = to_value(v);
    let coerced = match value.unpack() {
        ValueEnum::Undefined | ValueEnum::Null => Value::Null,
        ValueEnum::String(_) => value,
        _ => match value.coerce_to_string(activation) {
            Ok(s) => s.into(),
            Err(e) => {
                // SAFETY: erase `'gc` for thread-local storage; taken this same run.
                let erased: Error<'static> = unsafe { std::mem::transmute(e) };
                PENDING_ERROR.with(|slot| *slot.borrow_mut() = Some(erased));
                return from_value(Value::Undefined);
            }
        },
    };
    from_value(coerced)
}

/// Helper 7 — `get_scope_object`: `getscopeobject index`. Pushes the local scope
/// at `index` (into this method's own pushed scopes). The verifier guarantees the
/// index is valid; we guard defensively.
fn get_scope_object(index: i64) -> i64 {
    let activation = unsafe { activation() };
    match activation.scope_frame().get(index as usize) {
        Some(scope) => from_value(scope.values()),
        None => from_value(Value::Undefined),
    }
}

// domainMemory loads (`li8`/`li16`/`li32`): the FlasCC "RAM". Pop an address
// `Value`, read from the current domain's `domainMemory` `ByteArray`, push the
// loaded integer. `li8`/`li16` are unsigned; `li32` is signed. Out-of-bounds (or
// a throwing address coercion) yields `undefined` — the infallible-ABI caveat
// (valid FlasCC accesses are in-bounds; a real OOB would diverge under verify).

/// The address of the current domainMemory's stable `[base, cap]` **descriptor
/// cell** (see core `SharedByteBuffer::desc_ptr`), for the JIT's **inline**
/// `li*`/`si*` fast path: the emitted code loads base+cap from the cell on
/// EVERY access, so the (reservation-free) buffer may move on growth even under
/// a live JIT frame. Returns `(0, 0)` if unavailable (the emitted code then
/// falls back to the `dm_*` helper); the second element is unused (run ABI).
#[allow(dead_code)]
pub(crate) fn dm_base_len() -> (u32, u32) {
    // SAFETY: helpers run only inside `with_run_ctx`.
    let activation = unsafe { activation() };
    // Don't promote a buffer to shareable here — content like Starling assigns
    // a DIFFERENT ordinary ByteArray as domainMemory transiently per mesh, and
    // promotion belongs to the deliberate `dm_base_len` storage path. An
    // unshared buffer here can only be a race with reassignment — (0, 0) is
    // the safe answer (the run's inline dm ops take the helper fallback; the
    // interpreter path stays correct).
    let mut storage = activation.domain_memory().storage_mut();
    if !storage.is_shareable() {
        return (0, 0);
    }
    match storage.dm_base_len() {
        Some((desc, _)) => (desc as u32, 0),
        None => (0, 0),
    }
}

/// Stash a domainMemory-out-of-bounds `RangeError` (#1506) so the emitted `perr`
/// check bails the whole method — matching the interpreter's `op_li*`/`op_si*`,
/// which **throw** on OOB (the JIT ABI is infallible, so a throw crosses out of band
/// via `PENDING_ERROR`, exactly like `call_method`). Previously the dm helpers
/// silently returned `undefined`/skipped the write, diverging from the interpreter.
fn set_pending_1506(activation: &mut Activation<'_, '_>) {
    let err = ruffle_core::avm2::error::make_error_1506(activation);
    // SAFETY: erase `'gc` for thread-local storage; the error's Gc pointers are
    // valid for this synchronous run and `take_pending_error` retrieves it in the
    // same activation scope (see `call_method`).
    let erased: Error<'static> = unsafe { std::mem::transmute(err) };
    PENDING_ERROR.with(|slot| *slot.borrow_mut() = Some(erased));
}

/// Helper 8 — `li8`: unsigned byte load; throws #1506 on OOB.
fn dm_load8(addr: i64) -> i64 {
    let activation = unsafe { activation() };
    let Ok(address) = to_value(addr).coerce_to_i32(activation) else {
        return from_value(Value::Undefined);
    };
    let val = activation.domain_memory().storage().dm_get(address as usize);
    match val {
        Some(v) => from_value(Value::from(v)),
        None => {
            set_pending_1506(activation);
            from_value(Value::Undefined)
        }
    }
}

/// Helper 9 — `li16`: unsigned 16-bit load; throws #1506 on OOB.
fn dm_load16(addr: i64) -> i64 {
    let activation = unsafe { activation() };
    let Ok(address) = to_value(addr).coerce_to_i32(activation) else {
        return from_value(Value::Undefined);
    };
    let val = activation.domain_memory().storage().dm_read::<2>(address as usize);
    match val {
        Some(bytes) => from_value(Value::from(u16::from_le_bytes(bytes))),
        None => {
            set_pending_1506(activation);
            from_value(Value::Undefined)
        }
    }
}

/// Helper 10 — `li32`: signed 32-bit load; throws #1506 on OOB.
fn dm_load32(addr: i64) -> i64 {
    let activation = unsafe { activation() };
    let Ok(address) = to_value(addr).coerce_to_i32(activation) else {
        return from_value(Value::Undefined);
    };
    let val = activation.domain_memory().storage().dm_read::<4>(address as usize);
    match val {
        Some(bytes) => from_value(Value::from(i32::from_le_bytes(bytes))),
        None => {
            set_pending_1506(activation);
            from_value(Value::Undefined)
        }
    }
}

/// Helper 26 — `lf32` fallback: `f32` load widened to `Number`; throws #1506 on
/// OOB. Called by the emitted inline dm path when the access misses the shared
/// reservation (incl. `dm_len == 0` — an unshared domainMemory, e.g. Starling's
/// transient per-mesh assignment), routing through the real storage.
fn dm_load_f32(addr: i64) -> i64 {
    let activation = unsafe { activation() };
    let Ok(address) = to_value(addr).coerce_to_i32(activation) else {
        return from_value(Value::Undefined);
    };
    let val = activation.domain_memory().storage().dm_read::<4>(address as usize);
    match val {
        Some(bytes) => from_value(Value::from(f32::from_le_bytes(bytes) as f64)),
        None => {
            set_pending_1506(activation);
            from_value(Value::Undefined)
        }
    }
}

/// Helper 27 — `lf64` fallback: `f64` load; throws #1506 on OOB. See [`dm_load_f32`].
fn dm_load_f64(addr: i64) -> i64 {
    let activation = unsafe { activation() };
    let Ok(address) = to_value(addr).coerce_to_i32(activation) else {
        return from_value(Value::Undefined);
    };
    let val = activation.domain_memory().storage().dm_read::<8>(address as usize);
    match val {
        Some(bytes) => from_value(Value::from(f64::from_le_bytes(bytes))),
        None => {
            set_pending_1506(activation);
            from_value(Value::Undefined)
        }
    }
}

/// A JIT helper: raw `Value` in, raw `Value` out (`i64` bits either way).
pub(crate) type HelperFn = fn(i64) -> i64;

/// The helper table, indexed by `CallHelper(i)`. Keep the indices in sync with
/// the constants in [`crate::translate`] (`HELPER_*`) and `lower` (`TO_BOOLEAN` /
/// `PUSH_SCOPE` / `GET_SCOPE_OBJECT`).
pub(crate) static HELPERS: &[HelperFn] = &[
    increment,
    decrement,
    negate,
    bit_not,
    not,
    to_boolean,
    push_scope,
    get_scope_object,
    dm_load8,
    dm_load16,
    dm_load32,
    coerce_u,
    coerce_i,
    sxi1,
    sxi8,
    sxi16,
    coerce_return,
    get_script_globals,
    get_push_string,
    throw_value,
    dispatch_exc,
    new_catch,
    pop_caught,
    pop_scope,
    get_outer_scope,
    coerce_s,
    dm_load_f32,
    dm_load_f64,
];

// Binary (arity-2, two-stack) comparison helpers, indexed by `CallHelper2(i)`.
// Stack order is `(v1, v2)` (v2 on top), matching the interpreter's
// `value2 = pop; value1 = pop`. Each returns a `Boolean` `Value`. A throwing
// coercion is swallowed to the interpreter's `unwrap_or` default (a real throw
// would diverge under verify, so only non-throwing compares should be JIT'd).

/// `HELPERS2[0]` — `equals`: `v1 == v2` (abstract equality).
fn cmp_eq(v1: i64, v2: i64) -> i64 {
    let activation = unsafe { activation() };
    let r = to_value(v1)
        .abstract_eq(&to_value(v2), activation)
        .unwrap_or(false);
    from_value(Value::from(r))
}

/// `HELPERS2[1]` — `lessthan`: `v1 < v2`.
fn cmp_lt(v1: i64, v2: i64) -> i64 {
    let activation = unsafe { activation() };
    let r = to_value(v1)
        .abstract_lt(&to_value(v2), activation)
        .ok()
        .flatten()
        .unwrap_or(false);
    from_value(Value::from(r))
}

/// `HELPERS2[2]` — `lessequals`: `v1 <= v2` ≡ `!(v2 < v1)`.
fn cmp_le(v1: i64, v2: i64) -> i64 {
    let activation = unsafe { activation() };
    let r = !to_value(v2)
        .abstract_lt(&to_value(v1), activation)
        .ok()
        .flatten()
        .unwrap_or(true);
    from_value(Value::from(r))
}

/// `HELPERS2[3]` — `greaterthan`: `v1 > v2` ≡ `v2 < v1`.
fn cmp_gt(v1: i64, v2: i64) -> i64 {
    let activation = unsafe { activation() };
    let r = to_value(v2)
        .abstract_lt(&to_value(v1), activation)
        .ok()
        .flatten()
        .unwrap_or(false);
    from_value(Value::from(r))
}

/// `HELPERS2[4]` — `greaterequals`: `v1 >= v2` ≡ `!(v1 < v2)`.
fn cmp_ge(v1: i64, v2: i64) -> i64 {
    let activation = unsafe { activation() };
    let r = !to_value(v1)
        .abstract_lt(&to_value(v2), activation)
        .ok()
        .flatten()
        .unwrap_or(true);
    from_value(Value::from(r))
}

// Bitwise binary ops. The interpreter pops `value2` (top) first, so we coerce
// `v2` before `v1` to match coercion order. `<<`/`>>` mask the shift by `0x1F`.
fn bit_and(v1: i64, v2: i64) -> i64 {
    let a = unsafe { activation() };
    let (Ok(b), Ok(x)) = (to_value(v2).coerce_to_i32(a), to_value(v1).coerce_to_i32(a)) else {
        return from_value(Value::from(0));
    };
    from_value(Value::from(x & b))
}
fn bit_or(v1: i64, v2: i64) -> i64 {
    let a = unsafe { activation() };
    let (Ok(b), Ok(x)) = (to_value(v2).coerce_to_i32(a), to_value(v1).coerce_to_i32(a)) else {
        return from_value(Value::from(0));
    };
    from_value(Value::from(x | b))
}
fn bit_xor(v1: i64, v2: i64) -> i64 {
    let a = unsafe { activation() };
    let (Ok(b), Ok(x)) = (to_value(v2).coerce_to_i32(a), to_value(v1).coerce_to_i32(a)) else {
        return from_value(Value::from(0));
    };
    from_value(Value::from(x ^ b))
}
fn lshift(v1: i64, v2: i64) -> i64 {
    let a = unsafe { activation() };
    let (Ok(b), Ok(x)) = (to_value(v2).coerce_to_u32(a), to_value(v1).coerce_to_i32(a)) else {
        return from_value(Value::from(0));
    };
    from_value(Value::from(x << (b & 0x1F)))
}
fn rshift(v1: i64, v2: i64) -> i64 {
    let a = unsafe { activation() };
    let (Ok(b), Ok(x)) = (to_value(v2).coerce_to_u32(a), to_value(v1).coerce_to_i32(a)) else {
        return from_value(Value::from(0));
    };
    from_value(Value::from(x >> (b & 0x1F)))
}
fn urshift(v1: i64, v2: i64) -> i64 {
    let a = unsafe { activation() };
    let (Ok(b), Ok(x)) = (to_value(v2).coerce_to_u32(a), to_value(v1).coerce_to_u32(a)) else {
        return from_value(Value::from(0));
    };
    from_value(Value::from(x >> (b & 0x1F)))
}

// Generic (untyped) numeric arithmetic — the forms the int/double fast paths
// can't prove typed. Each mirrors the interpreter's `op_*` exactly, including its
// int fast paths (which produce an `Integer` Value, not `Number` — different
// NaN-box bits) and its coercion order (`value2` on top is coerced first). A
// throwing coercion falls back to `NaN` (infallible-ABI caveat).

/// `HELPERS2[11]` — `multiply`: `int*int→int` (checked) else `ToNumber×ToNumber`.
fn multiply(v1: i64, v2: i64) -> i64 {
    let a = unsafe { activation() };
    let (x1, x2) = (to_value(v1), to_value(v2));
    if let (ValueEnum::Integer(n1), ValueEnum::Integer(n2)) = (x1.unpack(), x2.unpack())
        && let Some(r) = n1.checked_mul(n2)
    {
        return from_value(Value::from(r));
    }
    match (x2.coerce_to_number(a), x1.coerce_to_number(a)) {
        (Ok(b), Ok(av)) => from_value(Value::from(av * b)),
        _ => from_value(Value::from(f64::NAN)),
    }
}

/// `HELPERS2[12]` — `subtract`: `int-int→int` (overflow→Number) / `num-num` / coerce.
fn subtract(v1: i64, v2: i64) -> i64 {
    let a = unsafe { activation() };
    let (x1, x2) = (to_value(v1), to_value(v2));
    let r = match (x1.unpack(), x2.unpack()) {
        (ValueEnum::Integer(n1), ValueEnum::Integer(n2)) => match n1.checked_sub(n2) {
            Some(res) => Value::from(res),
            None => Value::from((n1 as i64 - n2 as i64) as f64),
        },
        (ValueEnum::Number(n1), ValueEnum::Number(n2)) => Value::from(n1 - n2),
        _ => match (x2.coerce_to_number(a), x1.coerce_to_number(a)) {
            (Ok(b), Ok(av)) => Value::from(av - b),
            _ => Value::from(f64::NAN),
        },
    };
    from_value(r)
}

/// `HELPERS2[13]` — `divide`: `ToNumber(v1) / ToNumber(v2)` (always a Number).
fn divide(v1: i64, v2: i64) -> i64 {
    let a = unsafe { activation() };
    match (to_value(v2).coerce_to_number(a), to_value(v1).coerce_to_number(a)) {
        (Ok(b), Ok(av)) => from_value(Value::from(av / b)),
        _ => from_value(Value::from(f64::NAN)),
    }
}

/// `HELPERS2[14]` — `modulo`: `ToNumber(v1) % ToNumber(v2)` (always a Number).
fn modulo(v1: i64, v2: i64) -> i64 {
    let a = unsafe { activation() };
    match (to_value(v2).coerce_to_number(a), to_value(v1).coerce_to_number(a)) {
        (Ok(b), Ok(av)) => from_value(Value::from(av % b)),
        _ => from_value(Value::from(f64::NAN)),
    }
}

/// `HELPERS2[15]` — `strictequals`: `v1 === v2` (structural; never throws, no coercion).
fn strict_equals(v1: i64, v2: i64) -> i64 {
    from_value(Value::from(to_value(v1).strict_eq(&to_value(v2))))
}

/// `HELPERS2[16]` — `astypelate`: `value as Type` — `value` if `is_of_type`, else
/// `null` (v1 = value, v2 = type). A non-class type operand **throws** exactly like
/// the interpreter's `op_as_type_late` (`#1010` on `undefined`, `#1009` on
/// null/primitives, `#1041` on a non-class object), via `PENDING_ERROR` + the
/// emitted post-op `perr` bail/dispatch. Swallowing these to `null` diverged from
/// the interpreter and broke games that catch the error (see `is_type_late`).
fn as_type_late(value: i64, type_v: i64) -> i64 {
    // SAFETY: helpers run only inside `with_run_ctx`.
    let activation = unsafe { activation() };
    let class_v = to_value(type_v);
    if matches!(class_v.unpack(), ValueEnum::Undefined) {
        return stash_pending_error(make_null_or_undefined_error(activation, class_v, None));
    }
    let Some(obj) = class_v.as_object() else {
        // Primitive values and null both throw this error (see `op_as_type_late`).
        return stash_pending_error(make_null_or_undefined_error(activation, Value::Null, None));
    };
    let Some(class) = obj.as_class_object() else {
        return stash_pending_error(make_error_1041(activation));
    };
    let v = to_value(value);
    if v.is_of_type(class.inner_class_definition()) {
        from_value(v)
    } else {
        from_value(Value::Null)
    }
}

/// `HELPERS2[17]` — `istypelate`: `value is Type` as a `Boolean` (v1 = value, v2 =
/// type). A non-class type operand **throws** `#1041` exactly like the interpreter's
/// `op_is_type_late`, via `PENDING_ERROR` + the emitted post-op `perr`
/// bail/dispatch. (This used to be swallowed to `false`, which silently diverged
/// from the interpreter: game code doing `x is someVar` with a non-class `someVar`
/// relies on catching `#1041`, and returning `false` instead sent it down the wrong
/// path — observed as runaway recursion/allocation until `handle_alloc_error`.)
fn is_type_late(value: i64, type_v: i64) -> i64 {
    // SAFETY: helpers run only inside `with_run_ctx`.
    let activation = unsafe { activation() };
    let Some(class) = to_value(type_v).as_object().and_then(|o| o.as_class_object()) else {
        return stash_pending_error(make_error_1041(activation));
    };
    from_value(Value::from(
        to_value(value).is_of_type(class.inner_class_definition()),
    ))
}

/// `HELPERS2[18]` — `add`: `v1 + v2` (numeric add or string concat; see `Value::add`).
/// A throwing coercion (a custom `valueOf`/`toString`) is swallowed to `NaN`, like
/// the other arithmetic helpers — it never happens for the numeric/string adds games
/// actually do.
fn add(v1: i64, v2: i64) -> i64 {
    let a = unsafe { activation() };
    match to_value(v1).add(to_value(v2), a) {
        Ok(v) => from_value(v),
        Err(_) => from_value(Value::from(f64::NAN)),
    }
}

/// A JIT binary helper: `(v1, v2) -> Value` (`i64` each).
pub(crate) type Helper2Fn = fn(i64, i64) -> i64;

/// The arity-2 two-stack helper table, indexed by `CallHelper2(i)`. Keep in sync
/// with the `CMP_*` / `BIT_*` / shift / arithmetic constants in [`crate::translate`].
pub(crate) static HELPERS2: &[Helper2Fn] = &[
    cmp_eq, cmp_lt, cmp_le, cmp_gt, cmp_ge, bit_and, bit_or, bit_xor, lshift, rshift, urshift,
    multiply, subtract, divide, modulo, strict_equals, as_type_late, is_type_late, add,
];

/// The arity-2 getproperty helper (`GetProperty(k)`): reads the receiver `Value`
/// off the WASM stack and the `k`-th multiname from the current method's table,
/// null-checks the receiver, and returns `receiver.get_property(mn)`'s bits.
///
/// Returning an object result is sound: gc-arena only collects *between*
/// mutations (see `Player::run_frame`), and the whole JIT run is synchronous
/// inside the frame's `mutate`, so the result can't be collected before it's
/// pushed onto the interpreter stack; and returning a `Value` by value stores
/// into no `Gc`, so it needs no write barrier.
///
/// A throwing access (`#1009`/`#1010` null receiver, `#1069` missing property on a
/// sealed object, a throwing getter, …) propagates via `PENDING_ERROR` + the emitted
/// post-op `perr` bail/dispatch, exactly like the interpreter. (This used to be
/// swallowed to `undefined`, which silently diverged: the bogus `undefined` flowed
/// on into slots/args and surfaced later as e.g. `#1041` from an interpreted
/// `istypelate` — while the interpreter would have thrown here.)
pub(crate) fn get_property(receiver: i64, k: i64) -> i64 {
    // SAFETY: helpers run only inside `with_run_ctx`.
    let activation = unsafe { activation() };
    let mn = unsafe { multiname(k as usize) };

    let result = to_value(receiver)
        .null_check(activation, Some(mn))
        .and_then(|obj| obj.get_property(mn, activation));
    match result {
        Ok(v) => from_value(v),
        Err(e) => stash_pending_error(e),
    }
}

/// `getpropertyfast`: a dynamic-name property read (`arr[i]`, dictionaries). Mirrors
/// `op_get_property_fast` + its `op_get_property_slow` fallback, but with the
/// `receiver` + runtime `name` passed as args (not on the operand stack). The fast
/// path (integer index / dictionary object) returns directly; otherwise the runtime
/// name is pushed onto the activation stack so `fill_with_runtime_params` consumes it
/// exactly as the interpreter does (`k` = the lazy multiname template). A throwing
/// access propagates via `PENDING_ERROR` + the post-op `perr` bail, like `gp`/`gs`.
pub(crate) fn get_property_fast(receiver: i64, name: i64, k: i64) -> i64 {
    // SAFETY: helpers run only inside `with_run_ctx`.
    let activation = unsafe { activation() };
    let object_value = to_value(receiver);
    let name_value = to_value(name);

    // Fast path: array-index / dictionary access (mirrors `op_get_property_fast`).
    if let ValueEnum::Object(object) = object_value.unpack() {
        match name_value.unpack() {
            ValueEnum::Integer(_) | ValueEnum::Number(_) => {
                if let Some(index) = name_value.try_as_index() {
                    if let Some(value) = object.get_index_property(index) {
                        return from_value(value);
                    }
                }
            }
            ValueEnum::Object(name_object) => {
                if let Some(dictionary) = object.as_dictionary_object() {
                    return from_value(dictionary.get_property_by_object(name_object));
                }
            }
            _ => {}
        }
    }

    // Slow fallback (mirrors `op_get_property_slow`): fill the lazy multiname with the
    // runtime name (pushed so `fill_with_runtime_params` pops it), then `get_property`.
    let mn_template = unsafe { multiname(k as usize) };
    activation.push_stack(name_value);
    let filled = match mn_template.fill_with_runtime_params(activation) {
        Ok(filled) => filled,
        Err(e) => return stash_pending_error(e),
    };
    match object_value
        .null_check(activation, Some(&filled))
        .and_then(|o| o.get_property(&filled, activation))
    {
        Ok(v) => from_value(v),
        Err(e) => stash_pending_error(e),
    }
}

// Ternary (arity-3) setslot helpers — the write counterparts of `get_slot`. Stack
// order is `[receiver, value]` (value on top), plus the slot id as an immediate.
// Return an (ignored) dummy `i64`; the emitter drops it. A throwing write (null
// receiver, failing trait-type coercion) propagates via `PENDING_ERROR` + the
// emitted post-op `perr` bail/dispatch, exactly like the interpreter — a swallowed
// throw would silently skip the write and run on with corrupt state.

/// `HELPERS3[0]` — `setslot`: coerces the value to the slot's trait type, stores it.
pub(crate) fn set_slot(receiver: i64, value: i64, slot_id: i64) -> i64 {
    // SAFETY: helpers run only inside `with_run_ctx`.
    let activation = unsafe { activation() };
    let result = to_value(receiver)
        .as_object_null_check(activation, None, "Cannot set_slot on primitive")
        .and_then(|obj| obj.set_slot(slot_id as usize, to_value(value), activation));
    if let Err(e) = result {
        return stash_pending_error(e);
    }
    0
}

/// `HELPERS3[1]` — `setslotnocoerce`: stores the value directly (no coercion).
pub(crate) fn set_slot_no_coerce(receiver: i64, value: i64, slot_id: i64) -> i64 {
    let activation = unsafe { activation() };
    match to_value(receiver).as_object_null_check(activation, None, "Cannot set_slot on primitive")
    {
        Ok(obj) => obj.set_slot_no_coerce(slot_id as usize, to_value(value), activation.gc()),
        Err(e) => return stash_pending_error(e),
    }
    0
}

/// `HELPERS3[2]` — `setslotcoercei`: coerces the value to `int`, then stores it.
pub(crate) fn set_slot_coerce_i(receiver: i64, value: i64, slot_id: i64) -> i64 {
    let activation = unsafe { activation() };
    let result = to_value(receiver)
        .as_object_null_check(activation, None, "Cannot set_slot on primitive")
        .and_then(|obj| {
            let v = to_value(value).coerce_to_i32(activation)?;
            obj.set_slot_no_coerce(slot_id as usize, v.into(), activation.gc());
            Ok(())
        });
    if let Err(e) = result {
        return stash_pending_error(e);
    }
    0
}

/// `HELPERS3[3]` — domainMemory store (`si8`/`si16`/`si32`): stack order
/// `(value, addr)`, with the byte width (`1`/`2`/`4`) as the immediate. Writes the
/// low bytes of `value` into `domainMemory[addr]`. Out-of-bounds is ignored (see
/// the load caveat).
pub(crate) fn dm_store(value: i64, addr: i64, nbytes: i64) -> i64 {
    let activation = unsafe { activation() };
    // Match the interpreter's coercion order (address first, then value).
    let (Ok(address), Ok(val)) = (
        to_value(addr).coerce_to_i32(activation),
        to_value(value).coerce_to_i32(activation),
    ) else {
        return 0;
    };
    let address = address as usize;
    let n = nbytes as usize;
    // Verify-mode write logging now lives in `ByteArrayStorage::dm_set`/`dm_write`
    // (core), so both this helper *and* the interpreter's `si*` are captured by the
    // one thread-local log — required to roll back a call-bearing method's callee
    // writes during the call-aware verify.
    let oob = {
        let mut dm = activation.domain_memory().storage_mut();
        let oob = address.checked_add(n).is_none_or(|end| end > dm.dm_len());
        if !oob {
            match nbytes {
                1 => dm.dm_set(address, val as u8),
                2 => {
                    let _ = dm.dm_write(address, &(val as i16).to_le_bytes());
                }
                _ => {
                    let _ = dm.dm_write(address, &val.to_le_bytes());
                }
            }
        }
        oob
    };
    // OOB store throws #1506 (matches `op_si*`); the `perr` check bails the method.
    if oob {
        set_pending_1506(activation);
    }
    0
}

/// `HELPERS3[4]` — domainMemory float store (`sf32`/`sf64`) fallback: stack order
/// `(value, addr)`, byte width (`4`/`8`) as the immediate. Coerces the value to
/// `Number` and stores it as a little-endian `f32`/`f64`. Called by the emitted
/// inline dm path when the access misses the shared reservation (incl. an
/// unshared domainMemory, where `dm_len == 0`). OOB throws #1506.
pub(crate) fn dm_store_f(value: i64, addr: i64, nbytes: i64) -> i64 {
    let activation = unsafe { activation() };
    // Match the interpreter's coercion order (address first, then value).
    let (Ok(address), Ok(val)) = (
        to_value(addr).coerce_to_i32(activation),
        to_value(value).coerce_to_number(activation),
    ) else {
        return 0;
    };
    let address = address as usize;
    let n = nbytes as usize;
    let oob = {
        let mut dm = activation.domain_memory().storage_mut();
        let oob = address.checked_add(n).is_none_or(|end| end > dm.dm_len());
        if !oob {
            if n == 4 {
                let _ = dm.dm_write(address, &(val as f32).to_le_bytes());
            } else {
                let _ = dm.dm_write(address, &val.to_le_bytes());
            }
        }
        oob
    };
    if oob {
        set_pending_1506(activation);
    }
    0
}

/// Arms the domain-memory write log (verify mode). Delegates to core, where both
/// the interpreter's `si*` and this crate's `dm_store` funnel through the logging
/// `dm_set`/`dm_write` — so callee writes during a `callmethod` are captured too.
pub(crate) fn dm_log_start() {
    ruffle_core::avm2::bytearray::dm_log_start();
}

/// Disarms and returns the `(addr, old_bytes)` writes since [`dm_log_start`].
pub(crate) fn dm_log_take() -> Vec<(usize, Vec<u8>)> {
    ruffle_core::avm2::bytearray::dm_log_take()
}

/// The current shared domainMemory logical length (verifier: snapshot the `sbrk`
/// break before a run so its heap growth can be rolled back).
pub(crate) fn dm_len(activation: &mut Activation<'_, '_>) -> usize {
    activation.domain_memory().storage().dm_len()
}

/// Restore the shared domainMemory length (verifier: undo a run's heap growth).
pub(crate) fn dm_restore_len(activation: &mut Activation<'_, '_>, len: usize) {
    activation.domain_memory().storage_mut().dm_set_len(len);
}

/// Reads `n` bytes of the current domain memory at `addr` for the verifier — from
/// the **shared** buffer (`dm_get`), the authoritative store both engines write, not
/// the local mirror (which isn't grown past its initial length).
pub(crate) fn dm_read_range(activation: &mut Activation<'_, '_>, addr: usize, n: usize) -> Vec<u8> {
    let ba = activation.domain_memory();
    let storage = ba.storage();
    (0..n).map(|i| storage.dm_get(addr + i).unwrap_or(0)).collect()
}

/// Writes `data` to the shared domain memory at `addr` (verifier rollback). Called
/// only while the write-log is disarmed (post-`dm_log_take`), so `dm_write` doesn't
/// re-log the rollback itself.
pub(crate) fn dm_write_range(activation: &mut Activation<'_, '_>, addr: usize, data: &[u8]) {
    let ba = activation.domain_memory();
    let mut storage = ba.storage_mut();
    let _ = storage.dm_write(addr, data);
}

/// A JIT ternary helper: `(receiver, value, immediate) -> dummy` (`i64` each).
pub(crate) type Helper3Fn = fn(i64, i64, i64) -> i64;

/// The arity-3 helper table, indexed by `CallHelper3(k, _)`. Keep in sync with
/// the `SET_*` / `DM_STORE` constants in [`crate::translate`] and
/// `lower::HELPER3_KINDS`.
pub(crate) static HELPERS3: &[Helper3Fn] =
    &[set_slot, set_slot_no_coerce, set_slot_coerce_i, dm_store, dm_store_f];

/// The arity-2 getslot helper (`GetSlot(slot_id)`): the form the verifier lowers
/// a *resolved* property read to — a direct slot fetch with no multiname lookup
/// (one of the hottest AVM2 ops). Reads the receiver off the WASM stack,
/// null-checks it, and returns `object.get_slot(slot_id)`'s bits. Object results
/// are sound to return for the same reason as [`get_property`]; the infallible
/// ABI carries the same "no throwing access" caveat.
pub(crate) fn get_slot(receiver: i64, slot_id: i64) -> i64 {
    // Fast path: an object receiver (the overwhelmingly common case for a
    // verifier-resolved slot read) — straight to the slot, no activation TLS
    // access, no error machinery. This helper is one of the hottest crossings
    // in profiles.
    if let ValueEnum::Object(obj) = to_value(receiver).unpack() {
        return from_value(obj.get_slot(slot_id as usize));
    }
    // Slow path: null/undefined/primitive receiver throws (`#1009` et al.)
    // exactly like the interpreter, via `PENDING_ERROR` + the post-op `perr`
    // bail/dispatch.
    // SAFETY: helpers run only inside `with_run_ctx`.
    let activation = unsafe { activation() };
    match to_value(receiver).as_object_null_check(activation, None, "Cannot get_slot on primitive") {
        Ok(obj) => from_value(obj.get_slot(slot_id as usize)),
        Err(e) => stash_pending_error(e),
    }
}

// --- Method calls (`callmethod` — resolved disp-id dispatch). --------------
// Variadic, so the emitter *spills* each arg to `CALL_ARGS` via `push_call_arg`
// before the call, and `call_method` drains them. Calls can throw; the ABI is
// infallible and the JIT is authoritative, so a throw is captured in
// `PENDING_ERROR` (returning `undefined`), and the emitted code — which checks
// `pending_error` right after the call — bails out of the whole method, letting
// `WasmJit::try_run` propagate it. The interpreter would unwind at that same
// point with the same prior side effects, so this matches.

/// Spills one call argument (pushed in reverse stack order — top first).
pub(crate) fn push_call_arg(v: i64) {
    CALL_ARGS.with(|a| a.borrow_mut().push(v));
}

/// Drains the last `argc` spilled args from `CALL_ARGS` and restores argument
/// order (they were pushed top-first). Shared by `call_method`/`call_property`.
fn drain_call_args<'gc>(argc: i64) -> Vec<Value<'gc>> {
    CALL_ARGS.with(|a| {
        let mut v = a.borrow_mut();
        let len = v.len();
        let n = (argc as usize).min(len);
        let mut raw = v.split_off(len - n);
        raw.reverse(); // spilled top-first → restore argument order
        raw.into_iter().map(to_value).collect()
    })
}

/// Stashes a thrown `Error` in `PENDING_ERROR` (the emitted `perr` check bails the
/// method) and returns `undefined`. Shared by the call helpers.
fn stash_pending_error(e: Error<'_>) -> i64 {
    // SAFETY: erase `'gc` for thread-local storage; the error holds Gc pointers
    // valid for this synchronous run, and `take_pending_error` retrieves it in the
    // same activation scope (see `try_run`).
    let erased: Error<'static> = unsafe { std::mem::transmute(e) };
    PENDING_ERROR.with(|slot| *slot.borrow_mut() = Some(erased));
    from_value(Value::Undefined)
}

/// `callmethod index argc`: takes `argc` spilled args + the `receiver`, invokes
/// the method at disp-id `index`, and returns the result (`undefined` on throw,
/// with the error stashed in `PENDING_ERROR`).
pub(crate) fn call_method(receiver: i64, index: i64, argc: i64) -> i64 {
    // SAFETY: helpers run only inside `with_run_ctx`.
    let activation = unsafe { activation() };
    let args = drain_call_args(argc);
    let result = to_value(receiver).null_check(activation, None).and_then(|r| {
        r.call_method_with_args(index as usize, FunctionArgs::from_slice(&args), activation)
    });
    match result {
        Ok(v) => from_value(v),
        Err(e) => stash_pending_error(e),
    }
}

/// `callproperty k argc` / `callpropvoid k argc`: takes `argc` spilled args + the
/// `receiver`, resolves the `k`-th (non-lazy) multiname of the run's table, and
/// invokes `receiver.call_property(mn, args)` — mirroring `op_call_property`. The
/// caller drops the result for the `callpropvoid` form. Throw → `PENDING_ERROR`.
pub(crate) fn call_property(receiver: i64, k: i64, argc: i64) -> i64 {
    // SAFETY: helpers run only inside `with_run_ctx`.
    let activation = unsafe { activation() };
    let mn = unsafe { multiname(k as usize) };
    let args = drain_call_args(argc);
    // Non-lazy multiname (translate declines lazy ones), so no
    // `fill_with_runtime_params` stack interaction is needed — it would be a no-op.
    let result = to_value(receiver)
        .null_check(activation, Some(mn))
        .and_then(|r| r.call_property(mn, FunctionArgs::from_slice(&args), activation));
    match result {
        Ok(v) => from_value(v),
        Err(e) => stash_pending_error(e),
    }
}

/// `call argc` (the `Op::Call` form): takes `argc` spilled args + the `receiver` +
/// the `function` value, and invokes `function.call(receiver, args)` — mirroring
/// `op_call`. Returns the result; throw → `PENDING_ERROR`.
pub(crate) fn call_value(function: i64, receiver: i64, argc: i64) -> i64 {
    // SAFETY: helpers run only inside `with_run_ctx`.
    let activation = unsafe { activation() };
    let args = drain_call_args(argc);
    let result =
        to_value(function).call(activation, to_value(receiver), FunctionArgs::from_slice(&args));
    match result {
        Ok(v) => from_value(v),
        Err(e) => stash_pending_error(e),
    }
}

/// `constructsuper argc`: takes `argc` spilled args + the `receiver`, invokes the
/// superclass constructor on it (`Activation::super_init`, mirroring
/// `op_construct_super`). Void (returns a dummy the emitter drops). Throw →
/// `PENDING_ERROR`.
pub(crate) fn construct_super(receiver: i64, argc: i64) -> i64 {
    // SAFETY: helpers run only inside `with_run_ctx`.
    let activation = unsafe { activation() };
    let args = drain_call_args(argc);
    let result = to_value(receiver)
        .null_check(activation, None)
        .and_then(|r| activation.super_init(r, FunctionArgs::from_slice(&args)));
    if let Err(e) = result {
        return stash_pending_error(e);
    }
    0
}

/// The generic variadic helper behind [`crate::lower::JitOp::VCall`] (`vc` import):
/// `(a, imm, spill, kind)` — drains `spill` spilled operands, dispatches on the
/// [`crate::lower::vc`] kind, and mirrors the interpreter's `op_*` for that op. A
/// throw is stashed in `PENDING_ERROR` (the emitted perr check bails/dispatches).
/// The kinds are cold next to `get_slot`/arithmetic (constructors, class init,
/// literals, super access), so one shared import + a kind match beats 20 separate
/// import bindings; a kind that turns hot can be promoted to its own import later.
pub(crate) fn vcall(a: i64, imm: i64, spill: i64, kind: i64) -> i64 {
    // SAFETY: helpers run only inside `with_activation` (+ the relevant tables).
    let activation = unsafe { activation() };
    let args = drain_call_args(spill);
    match vcall_inner(activation, to_value(a), imm as usize, args, kind as u32) {
        Ok(v) => from_value(v),
        Err(e) => stash_pending_error(e),
    }
}

/// [`vcall`]'s kind dispatch. `a` is the receiver/base/value operand (dummy
/// `undefined`-ish `0` for the no-receiver kinds); `args` are the drained spilled
/// operands in original stack order (bottom→top). Void kinds return `undefined`
/// (the emitter drops it).
fn vcall_inner<'gc>(
    activation: &mut Activation<'_, 'gc>,
    a: Value<'gc>,
    imm: usize,
    args: Vec<Value<'gc>>,
    kind: u32,
) -> Result<Value<'gc>, Error<'gc>> {
    use crate::lower::vc;
    match kind {
        // `constructslot`: the ctor lives in the receiver's slot (mirrors
        // `op_construct_slot`).
        vc::CONSTRUCT_SLOT => {
            let source =
                a.as_object_null_check(activation, None, "Cannot get_slot on primitive")?;
            let ctor = source.get_slot(imm);
            ctor.construct(activation, FunctionArgs::from_slice(&args))
        }
        // `construct`: `a` IS the ctor value (mirrors `op_construct`).
        vc::CONSTRUCT => a.construct(activation, FunctionArgs::from_slice(&args)),
        // `constructprop` (non-lazy mn, so no runtime-param fill; mirrors
        // `op_construct_prop`).
        vc::CONSTRUCT_PROP => {
            let mn = unsafe { multiname(imm) };
            let source = a.null_check(activation, Some(mn))?;
            source.construct_prop(activation, mn, FunctionArgs::from_slice(&args))
        }
        // `callsuper` (non-lazy; mirrors `op_call_super`).
        vc::CALL_SUPER => {
            let mn = unsafe { multiname(imm) };
            let bso = activation
                .bound_superclass_object()
                .expect("Expected a superclass when running callsuper");
            let receiver = a.coerce_to_type(activation, bso.inner_class_definition())?;
            let receiver = receiver.as_object_null_check(
                activation,
                Some(mn),
                "Super ops should not appear in primitive functions",
            )?;
            bso.call_super(mn, receiver, FunctionArgs::from_slice(&args), activation)
        }
        // `callnative`: the verifier's direct native fast call (mirrors
        // `op_call_native` — which `expect`s the native not to `Err`).
        vc::CALL_NATIVE => {
            let f = unsafe { native_fn_at(imm) };
            let receiver = a.null_check(activation, None)?;
            Ok(f(activation, receiver, &args).expect("FastCall methods should not return Err"))
        }
        // `applytype` (mirrors `op_apply_type`).
        vc::APPLY_TYPE => {
            let base = a.as_object().ok_or_else(|| make_error_1127(activation))?;
            base.apply(activation, &args).map(Into::into)
        }
        // `newarray` (mirrors `op_new_array`).
        vc::NEW_ARRAY => {
            let storage: ArrayStorage<'gc> = args.into_iter().collect();
            Ok(ArrayObject::from_storage(activation.context, storage).into())
        }
        // `newobject`: `args` = name/value pairs. The interpreter sets pairs from
        // the top of the stack down, so iterate ours in reverse — a duplicate name
        // then resolves identically (the *first* pair wins).
        vc::NEW_OBJECT => {
            let object = ScriptObject::new_object(activation.context);
            for pair in args.chunks_exact(2).rev() {
                let (name, value) = (pair[0], pair[1]);
                object.set_dynamic_property(
                    name.coerce_to_string(activation)?,
                    value,
                    activation.gc(),
                );
            }
            Ok(object.into())
        }
        // `getsuper` (non-lazy; mirrors `op_get_super`).
        vc::GET_SUPER => {
            let mn = unsafe { multiname(imm) };
            let bso = activation
                .bound_superclass_object()
                .expect("Expected a superclass when running callsuper");
            let receiver = a.coerce_to_type(activation, bso.inner_class_definition())?;
            let receiver = receiver.as_object_null_check(
                activation,
                Some(mn),
                "Super ops should not appear in primitive functions",
            )?;
            bso.get_super(mn, receiver, activation)
        }
        // `setsuper` (non-lazy; mirrors `op_set_super`). `args[0]` = the value.
        vc::SET_SUPER => {
            let mn = unsafe { multiname(imm) };
            let value = args[0];
            let bso = activation
                .bound_superclass_object()
                .expect("Expected a superclass when running callsuper");
            let receiver = a.coerce_to_type(activation, bso.inner_class_definition())?;
            let receiver = receiver.as_object_null_check(
                activation,
                Some(mn),
                "Super ops should not appear in primitive functions",
            )?;
            bso.set_super(mn, value, receiver, activation)?;
            Ok(Value::Undefined)
        }
        // `deleteproperty` (non-lazy; mirrors `op_delete_property`'s static path).
        vc::DELETE_PROPERTY => {
            let mn = unsafe { multiname(imm) };
            let object = a.null_check(activation, Some(mn))?;
            object.delete_property(activation, mn).map(Value::from)
        }
        // `nextvalue` (mirrors `op_next_value`). `a` = the object, `args[0]` = index.
        vc::NEXT_VALUE => {
            let cur_index = args[0].coerce_to_i32(activation)?;
            if cur_index <= 0 {
                return Ok(Value::Undefined);
            }
            let value = a.null_check(activation, None)?;
            let object = match value.unpack() {
                ValueEnum::Object(obj) => obj,
                _ => value
                    .proto(activation)
                    .expect("Primitives always have a prototype"),
            };
            object.get_enumerant_value(cur_index as u32, activation)
        }
        // `in` (mirrors `op_in`). `a` = the name value, `args[0]` = the object.
        vc::IN => {
            let name_value = a;
            let value = args[0].null_check(activation, None)?;
            let has_prop = match value.unpack() {
                ValueEnum::Object(obj) => {
                    if let Some(dictionary) = obj.as_dictionary_object() {
                        if let Some(name_object) = name_value.as_object() {
                            return Ok(dictionary.has_property_by_object(name_object).into());
                        }
                    }
                    let name = name_value.coerce_to_string(activation)?;
                    let mn = Multiname::new(activation.avm2().find_public_namespace(), name);
                    obj.has_property_via_in(activation, &mn)?
                }
                _ => {
                    let name = name_value.coerce_to_string(activation)?;
                    let mn = Multiname::new(activation.avm2().find_public_namespace(), name);
                    if value.has_trait(activation, &mn) {
                        true
                    } else if let Some(proto) = value.proto(activation) {
                        proto.has_property(&mn)
                    } else {
                        // `Value::proto` always returns `Some` for primitives.
                        unreachable!()
                    }
                }
            };
            Ok(has_prop.into())
        }
        // `setproperty` with a static multiname (mirrors `op_set_property_static`).
        // `args[0]` = the value.
        vc::SET_PROP_STATIC => {
            let mn = unsafe { multiname(imm) };
            let value = args[0];
            let object = a.null_check(activation, Some(mn))?;
            object.set_property(mn, value, activation)?;
            Ok(Value::Undefined)
        }
        // `setproperty` with a lazy runtime name (mirrors `op_set_property_fast` +
        // its slow fallback). `args` = `[name, value]`.
        vc::SET_PROP_FAST => {
            let name_value = args[0];
            let value = args[1];
            if let ValueEnum::Object(object) = a.unpack() {
                match name_value.unpack() {
                    ValueEnum::Integer(_) | ValueEnum::Number(_) => {
                        if let Some(index) = name_value.try_as_index() {
                            if let Some(result) =
                                object.set_index_property(activation, index, value)
                            {
                                return result.map(|_| Value::Undefined);
                            }
                        }
                    }
                    ValueEnum::Object(name_object) => {
                        if let Some(dictionary) = object.as_dictionary_object() {
                            dictionary.set_property_by_object(
                                name_object,
                                value,
                                activation.gc(),
                            );
                            return Ok(Value::Undefined);
                        }
                    }
                    _ => {}
                }
            }
            // Slow fallback (mirrors `op_set_property_slow`): push the runtime name so
            // `fill_with_runtime_params` pops it, exactly as the interpreter does.
            let mn_template = unsafe { multiname(imm) };
            activation.push_stack(name_value);
            let filled = mn_template.fill_with_runtime_params(activation)?;
            let object = a.null_check(activation, Some(&filled))?;
            object.set_property(&filled, value, activation)?;
            Ok(Value::Undefined)
        }
        // `newclass` (mirrors `op_new_class`, incl. the early `Object`/`Class` reuse).
        vc::NEW_CLASS => {
            let class = unsafe { coerce_class_at(imm) };
            if class == activation.avm2().class_defs().object {
                let object_class = activation.avm2().classes().object;
                object_class.run_class_initializer(activation)?;
                return Ok(object_class.into());
            } else if class == activation.avm2().class_defs().class {
                let class_class = activation.avm2().classes().class;
                class_class.run_class_initializer(activation)?;
                return Ok(class_class.into());
            }
            let class_class = activation.avm2().class_defs().class;
            let base_class = a.coerce_to_type(activation, class_class)?;
            let base_class = match base_class.unpack() {
                ValueEnum::Object(o) => Some(
                    o.as_class_object()
                        .expect("Coercion to Class must return Class or null"),
                ),
                ValueEnum::Null => None,
                _ => unreachable!("Coercion to Class must return Class or null"),
            };
            if base_class.is_none() && class.super_class().is_some() {
                return Err(make_null_or_undefined_error(activation, Value::Null, None));
            } else if base_class.map(|c| c.inner_class_definition()) != class.super_class() {
                return Err(make_error_1108(activation));
            }
            ClassObject::from_class(activation, class, base_class).map(Into::into)
        }
        // `newactivation` (mirrors `op_new_activation`).
        vc::NEW_ACTIVATION => {
            let class = unsafe { coerce_class_at(imm) };
            Ok(ScriptObject::custom_object(activation.gc(), class, None, class.vtable()).into())
        }
        // `typeof` (mirrors `op_type_of` via the shared `Value::type_of`).
        vc::TYPE_OF => Ok(Value::String(a.type_of(activation))),
        // `pushnamespace` (mirrors `op_push_namespace`).
        vc::PUSH_NAMESPACE => {
            let ns = unsafe { namespace_at(imm) };
            Ok(NamespaceObject::from_namespace(activation, ns).into())
        }
        // `coerce_d` (`ToNumber`) / `convert_s` (`ToString`) — a `valueOf`/`toString`
        // can throw, hence the perr-checked vcall form.
        vc::COERCE_D => a.coerce_to_number(activation).map(Value::from),
        vc::CONVERT_S => a.coerce_to_string(activation).map(Value::from),
        _ => unreachable!("unknown vcall kind {kind}"),
    }
}

/// The `imm`-th `CallNative` fn pointer of the current method.
///
/// # Safety
/// Only call from a helper running inside [`with_run_ctx`], with `imm` in range.
unsafe fn native_fn_at(imm: usize) -> NativeMethodImpl {
    let (ptr, len) = unsafe { run_ctx() }.natives;
    debug_assert!(imm < len, "JIT native fn index {imm} out of range (len {len})");
    // SAFETY: the entry is a `NativeMethodImpl` fn pointer erased in
    // `natives_table` within this same process; reversing the cast is total.
    unsafe { std::mem::transmute::<*const (), NativeMethodImpl>(*ptr.add(imm)) }
}

/// The `imm`-th `PushNamespace` namespace of the current method.
///
/// # Safety
/// Only call from a helper running inside [`with_run_ctx`], with `imm` in range.
unsafe fn namespace_at<'gc>(imm: usize) -> Namespace<'gc> {
    let (ptr, len) = unsafe { run_ctx() }.namespaces;
    debug_assert!(imm < len, "JIT namespace index {imm} out of range (len {len})");
    // SAFETY: the entry is a live `Namespace` (a niche-optimized `Option<Gc>`,
    // pointer-sized) erased in `namespaces_table`; alive for the run.
    unsafe { std::mem::transmute::<*const (), Namespace<'gc>>(*ptr.add(imm)) }
}

/// Whether a call threw during this run (the emitted code checks this after each
/// call to bail promptly). `1` = pending.
pub(crate) fn pending_error() -> i32 {
    PENDING_ERROR.with(|slot| slot.borrow().is_some()) as i32
}

/// Takes (and clears) the pending thrown error, for `try_run` to propagate.
pub(crate) fn take_pending_error<'gc>() -> Option<Error<'gc>> {
    let erased = PENDING_ERROR.with(|slot| slot.borrow_mut().take())?;
    // SAFETY: reverse of the erasure in `call_method`, within the same run.
    Some(unsafe { std::mem::transmute::<Error<'static>, Error<'gc>>(erased) })
}
