//! The reification ABI (see `AVM2_JIT_REDESIGN.md` §3, §9).
//!
//! A slow-path helper (return coercion, later `getproperty`/calls) needs a
//! `&mut Activation` to run a core op. avm2-jit3 does NOT reuse the caller's
//! Activation (the corruption class the old JIT kept hitting — a helper seeing the
//! wrong scope/super, or a compile-time coercion mutating the live caller). Instead
//! it reifies a FRESH, callee-owned Activation from `(cx, scope, bound_super)` — the
//! callee's own context — via the zero-alloc [`Activation::from_builtin`]. The frame
//! is GC-quiescent, so the erased pointers stashed here stay valid for the run.
//!
//! `try_enter` installs a [`RunCtx`] around the WASM run ([`with_run_ctx`]); helpers
//! read it to [`reify`]. A coercion/op that throws stashes the error ([`stash_error`])
//! and returns `undefined`; `try_enter` takes it after the run ([`take_error`]) and
//! propagates it (a minimal error path — full sentinel-based unwinding is Phase 5).

#[cfg(any(not(target_arch = "wasm32"), target_feature = "atomics"))]
use std::cell::{Cell, RefCell};

use gc_arena::lock::RefLock;
use gc_arena::Gc;
use ruffle_core::avm2::{Activation, ClassObject, Error, ScopeChain};
// Used only in the web-only in-WASM dispatch (`jit_push_call` / the leaked-`JitEnv` cache).
#[cfg(target_arch = "wasm32")]
use ruffle_core::avm2::Method;
use ruffle_core::context::UpdateContext;
use ruffle_core::library::MovieLibrary;

/// The callee method's owning translation-unit library — what `Activation::caller_library`
/// must return so native methods (`Font.enumerateFonts`, asset loads, security checks)
/// resolve against the JIT'd method's SWF, not `None`.
type CallerLib<'gc> = Gc<'gc, RefLock<MovieLibrary<'gc>>>;

/// The per-run context a compiled method's helpers reify their Activation from.
/// Lifetimes are erased for thread-local storage and reconstructed within the same
/// synchronous run (the frame is GC-quiescent, so this is sound).
pub(crate) struct RunCtx {
    /// The callee's captured scope chain (its `env` scope).
    scope: ScopeChain<'static>,
    /// The callee's bound superclass object (for `super` ops — its own, not the caller's).
    bound_super: Option<ClassObject<'static>>,
    /// This run's base into the shared `avm2.scope_stack` — index 0 of its LOCAL scope frame.
    /// `getscopeobject index` reads `scope_stack[scope_base + index]`; `pushscope` appends.
    scope_base: usize,
    /// The JIT'd method's owner library (its `owner_library()`) — reified as
    /// `caller_library` so native callees see the right SWF (mirrors `init_from_method`).
    caller_library: Option<CallerLib<'static>>,
}

// NB: `cx` (the ambient `*mut UpdateContext`) is deliberately NOT a `RunCtx` field — it is
// the SAME for a whole synchronous run (one player tick) and lives in the separate `AMBIENT_CX`
// slot (set once at the top-level entry). This makes `RunCtx` `cx`-free — i.e. per-callee and
// cacheable (see `JIT3_INWASM_DISPATCH_PLAN.md` §8 phase 1).

impl RunCtx {
    pub(crate) fn new<'gc>(
        scope: ScopeChain<'gc>,
        bound_super: Option<ClassObject<'gc>>,
        scope_base: usize,
        caller_library: Option<CallerLib<'gc>>,
    ) -> Self {
        // SAFETY: erase `'gc` for storage; reconstructed only within this run, where
        // the objects are alive (GC-quiescent frame).
        RunCtx {
            scope: unsafe { core::mem::transmute::<ScopeChain<'gc>, ScopeChain<'static>>(scope) },
            bound_super: unsafe {
                core::mem::transmute::<Option<ClassObject<'gc>>, Option<ClassObject<'static>>>(
                    bound_super,
                )
            },
            scope_base,
            caller_library: unsafe {
                core::mem::transmute::<Option<CallerLib<'gc>>, Option<CallerLib<'static>>>(
                    caller_library,
                )
            },
        }
    }
}

/// A cached, cx-free per-callee dispatch env for the §8 in-WASM call IC: the callee's
/// [`RunCtx`] plus its `Method` identity (for the AVM2 call-stack push that keeps
/// `Error.getStackTrace()` faithful — mirrors `try_enter`'s `push_call`). Built once per
/// callee on a call-IC miss and leaked (class-lifetime-stable, exactly the invariant the
/// cached `ClassBoundMethod` pointer in the IC already relies on). [`jit_enter`] installs it,
/// [`jit_leave`] tears it down.
pub(crate) struct JitEnv {
    ctx: RunCtx,
    method_ptr: *const (),
}

impl JitEnv {
    pub(crate) fn new(ctx: RunCtx, method_ptr: *const ()) -> Self {
        Self { ctx, method_ptr }
    }
    /// The callee's per-run context (`&self.ctx` — stable while the leaked env lives).
    pub(crate) fn ctx_ptr(&self) -> *const RunCtx {
        &self.ctx
    }
    /// The callee `Method`'s stable identity (`Method::as_ptr`), for the call-stack push.
    pub(crate) fn method_ptr(&self) -> *const () {
        self.method_ptr
    }
}

// Per-run JIT state: the installed [`RunCtx`] and a stashed thrown error.
//
// On **native** these are `thread_local!` — the test harness runs methods on many threads in
// parallel, so each needs its own. On **wasm32** the AVM2 player runs on a SINGLE thread (the
// audio/render workers never execute AVM2), so a plain `static` is used instead: a direct
// load, avoiding the wasm-threads TLS-slot computation on the hottest path (`reify` runs in
// EVERY helper; `run_ctx`/`pending` are touched several times per call).
//
// SAFETY (wasm): sound as long as no two threads run AVM2 concurrently within one module
// instance. All access is synchronous within the one thread; nested runs save/restore
// `RUN_CTX` LIFO, and the interior `&mut` never aliases (helpers run to completion inline).
#[cfg(any(not(target_arch = "wasm32"), target_feature = "atomics"))]
thread_local! {
    static RUN_CTX: Cell<*const RunCtx> = const { Cell::new(std::ptr::null()) };
    /// The ambient `*mut UpdateContext` — set ONCE at the top-level entry, shared by every
    /// (nested) run this tick. Separate from `RUN_CTX` so the latter is per-callee/cacheable.
    static AMBIENT_CX: Cell<*mut ()> = const { Cell::new(std::ptr::null_mut()) };
    static PENDING: RefCell<Option<Error<'static>>> = const { RefCell::new(None) };
    /// This run's LIVE base into the shared `avm2.scope_stack` (`scope_stack_len()` at entry).
    /// Kept SEPARATE from the (per-callee, cacheable) `RunCtx` because it is a PER-CALL value: the
    /// §8 in-WASM fast path installs a cached env whose `RunCtx.scope_base` is a stale 0, so it
    /// sets THIS live instead. `scope_base()`/`reify()` read it; `with_run_ctx` (Rust path) seeds
    /// it from `RunCtx.scope_base`; `jit_enter`/`jit_leave` (in-WASM) swap it LIFO.
    static LIVE_SCOPE_BASE: Cell<usize> = const { Cell::new(0) };
}

#[cfg(all(target_arch = "wasm32", not(target_feature = "atomics")))]
mod single_thread {
    use super::{Error, RunCtx};
    use std::cell::UnsafeCell;
    /// A `Sync` cell relying on the single-AVM2-thread invariant (see the module note above).
    pub struct One<T>(pub UnsafeCell<T>);
    // SAFETY: only ever accessed from the single AVM2 thread.
    unsafe impl<T> Sync for One<T> {}
    pub static RUN_CTX: One<*const RunCtx> = One(UnsafeCell::new(std::ptr::null()));
    pub static AMBIENT_CX: One<*mut ()> = One(UnsafeCell::new(std::ptr::null_mut()));
    pub static PENDING: One<Option<Error<'static>>> = One(UnsafeCell::new(None));
    pub static LIVE_SCOPE_BASE: One<usize> = One(UnsafeCell::new(0));
}

#[inline]
fn ambient_cx_get() -> *mut () {
    #[cfg(any(not(target_arch = "wasm32"), target_feature = "atomics"))]
    {
        AMBIENT_CX.with(|c| c.get())
    }
    // SAFETY: single-threaded read (see the module note).
    #[cfg(all(target_arch = "wasm32", not(target_feature = "atomics")))]
    unsafe {
        *single_thread::AMBIENT_CX.0.get()
    }
}

/// Sets `AMBIENT_CX` to `new`, returning the previous value.
#[inline]
fn ambient_cx_swap(new: *mut ()) -> *mut () {
    #[cfg(any(not(target_arch = "wasm32"), target_feature = "atomics"))]
    {
        AMBIENT_CX.with(|c| c.replace(new))
    }
    // SAFETY: single-threaded swap (see the module note).
    #[cfg(all(target_arch = "wasm32", not(target_feature = "atomics")))]
    unsafe {
        let p = single_thread::AMBIENT_CX.0.get();
        let old = *p;
        *p = new;
        old
    }
}

/// Installs `cx` as the ambient `UpdateContext` for the duration of `f`, restoring the previous
/// after (LIFO — a nested run passes the SAME `cx`, so this is idempotent for nesting). `cx` is
/// erased to `*mut ()`; the frame is GC-quiescent, so the pointer stays valid for the run.
pub(crate) fn with_ambient_cx<'gc, R>(cx: &mut UpdateContext<'gc>, f: impl FnOnce() -> R) -> R {
    let prev = ambient_cx_swap(cx as *mut UpdateContext<'gc> as *mut ());
    let r = f();
    ambient_cx_swap(prev);
    r
}

#[inline]
fn run_ctx_get() -> *const RunCtx {
    #[cfg(any(not(target_arch = "wasm32"), target_feature = "atomics"))]
    {
        RUN_CTX.with(|c| c.get())
    }
    // SAFETY: single-threaded read (see the module note).
    #[cfg(all(target_arch = "wasm32", not(target_feature = "atomics")))]
    unsafe {
        *single_thread::RUN_CTX.0.get()
    }
}

/// Sets `RUN_CTX` to `new`, returning the previous value.
#[inline]
fn run_ctx_swap(new: *const RunCtx) -> *const RunCtx {
    #[cfg(any(not(target_arch = "wasm32"), target_feature = "atomics"))]
    {
        RUN_CTX.with(|c| c.replace(new))
    }
    // SAFETY: single-threaded swap (see the module note).
    #[cfg(all(target_arch = "wasm32", not(target_feature = "atomics")))]
    unsafe {
        let p = single_thread::RUN_CTX.0.get();
        let old = *p;
        *p = new;
        old
    }
}

#[inline]
fn live_sb_get() -> usize {
    #[cfg(any(not(target_arch = "wasm32"), target_feature = "atomics"))]
    {
        LIVE_SCOPE_BASE.with(|c| c.get())
    }
    // SAFETY: single-threaded read (see the module note).
    #[cfg(all(target_arch = "wasm32", not(target_feature = "atomics")))]
    unsafe {
        *single_thread::LIVE_SCOPE_BASE.0.get()
    }
}

/// Sets `LIVE_SCOPE_BASE` to `new`, returning the previous value.
#[inline]
fn live_sb_swap(new: usize) -> usize {
    #[cfg(any(not(target_arch = "wasm32"), target_feature = "atomics"))]
    {
        LIVE_SCOPE_BASE.with(|c| c.replace(new))
    }
    // SAFETY: single-threaded swap (see the module note).
    #[cfg(all(target_arch = "wasm32", not(target_feature = "atomics")))]
    unsafe {
        let p = single_thread::LIVE_SCOPE_BASE.0.get();
        let old = *p;
        *p = new;
        old
    }
}

#[inline]
fn pending_set(e: Option<Error<'static>>) {
    #[cfg(any(not(target_arch = "wasm32"), target_feature = "atomics"))]
    {
        PENDING.with(|p| *p.borrow_mut() = e);
    }
    // SAFETY: single-threaded write (see the module note).
    #[cfg(all(target_arch = "wasm32", not(target_feature = "atomics")))]
    unsafe {
        *single_thread::PENDING.0.get() = e;
    }
}

#[inline]
fn pending_is_some() -> bool {
    #[cfg(any(not(target_arch = "wasm32"), target_feature = "atomics"))]
    {
        PENDING.with(|p| p.borrow().is_some())
    }
    // SAFETY: single-threaded read (see the module note).
    #[cfg(all(target_arch = "wasm32", not(target_feature = "atomics")))]
    unsafe {
        (*single_thread::PENDING.0.get()).is_some()
    }
}

#[inline]
fn pending_take() -> Option<Error<'static>> {
    #[cfg(any(not(target_arch = "wasm32"), target_feature = "atomics"))]
    {
        PENDING.with(|p| p.borrow_mut().take())
    }
    // SAFETY: single-threaded take (see the module note).
    #[cfg(all(target_arch = "wasm32", not(target_feature = "atomics")))]
    unsafe {
        (*single_thread::PENDING.0.get()).take()
    }
}

/// Installs `ctx` for the duration of `f`, restoring the previous one after — so a
/// nested run (a helper re-entering AS3) saves/restores LIFO. One TLS swap in, one out.
pub(crate) fn with_run_ctx<R>(ctx: &RunCtx, f: impl FnOnce() -> R) -> R {
    let prev = run_ctx_swap(ctx as *const RunCtx);
    // The Rust path's `RunCtx` carries the live base — seed `LIVE_SCOPE_BASE` from it so
    // `scope_base()`/`reify()` (which now read `LIVE_SCOPE_BASE`) match the in-WASM path.
    let prev_sb = live_sb_swap(ctx.scope_base);
    let r = f();
    live_sb_swap(prev_sb);
    run_ctx_swap(prev);
    r
}

/// Installs `env` (a `*const RunCtx`) as the current per-callee context, returning the previous
/// one — the caller-bracket primitive for the §8 in-WASM dispatch: a compiled caller does
/// `prev = push_ctx(env); …; call_indirect callee.run(…); pop_ctx(prev)` so the callee's helpers
/// reify from `env` without a Rust `try_enter` bounce. (`with_run_ctx` is the same swap, scoped
/// by the Rust stack; these expose it to WASM.) `env` must be a live `RunCtx` for the call.
pub(crate) fn push_ctx(env: *const RunCtx) -> *const RunCtx {
    run_ctx_swap(env)
}

/// Restores the context swapped out by [`push_ctx`] (LIFO — `prev` is that call's return).
pub(crate) fn pop_ctx(prev: *const RunCtx) {
    run_ctx_swap(prev);
}

// §8 in-WASM dispatch: a LIFO stack of the caller `RUN_CTX` values `jit_enter` swapped out, so
// `jit_leave` (which takes no argument) can restore them. Nesting is LIFO with the wasm call
// stack — a callee may itself make in-WASM calls — so a plain stack is exact. Web-only (the
// in-WASM caller emit is `cfg(wasm32)`); native's `jit_enter`/`jit_leave` never touch it.
#[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
thread_local! {
    static JIT_PREV: RefCell<Vec<*const RunCtx>> = const { RefCell::new(Vec::new()) };
}
#[cfg(all(target_arch = "wasm32", not(target_feature = "atomics")))]
mod jit_prev_store {
    use super::RunCtx;
    use std::cell::UnsafeCell;
    /// A `Sync` cell relying on the single-AVM2-thread invariant (see the RUN_CTX note).
    pub struct One(pub UnsafeCell<Vec<*const RunCtx>>);
    // SAFETY: only ever accessed from the single AVM2 thread.
    unsafe impl Sync for One {}
    pub static JIT_PREV: One = One(UnsafeCell::new(Vec::new()));
}

/// Pushes a swapped-out caller ctx (see [`jit_enter`]).
#[cfg(target_arch = "wasm32")]
pub(crate) fn jit_prev_push(p: *const RunCtx) {
    #[cfg(target_feature = "atomics")]
    JIT_PREV.with(|s| s.borrow_mut().push(p));
    // SAFETY: single AVM2 thread; not held across a call.
    #[cfg(not(target_feature = "atomics"))]
    unsafe {
        (*jit_prev_store::JIT_PREV.0.get()).push(p);
    }
}

/// Pops the most recently swapped-out caller ctx (see [`jit_leave`]).
#[cfg(target_arch = "wasm32")]
pub(crate) fn jit_prev_pop() -> *const RunCtx {
    #[cfg(target_feature = "atomics")]
    {
        JIT_PREV.with(|s| s.borrow_mut().pop().unwrap_or(std::ptr::null()))
    }
    // SAFETY: single AVM2 thread; not held across a call.
    #[cfg(not(target_feature = "atomics"))]
    unsafe {
        (*jit_prev_store::JIT_PREV.0.get()).pop().unwrap_or(std::ptr::null())
    }
}

// §8 in-WASM dispatch: a LIFO stack of the caller `LIVE_SCOPE_BASE` values `jit_enter` swapped
// out, so `jit_leave` can restore them (parallel to `JIT_PREV`). Web-only.
#[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
thread_local! {
    static JIT_SB: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
}
#[cfg(all(target_arch = "wasm32", not(target_feature = "atomics")))]
mod jit_sb_store {
    use std::cell::UnsafeCell;
    /// A `Sync` cell relying on the single-AVM2-thread invariant (see the RUN_CTX note).
    pub struct One(pub UnsafeCell<Vec<usize>>);
    // SAFETY: only ever accessed from the single AVM2 thread.
    unsafe impl Sync for One {}
    pub static JIT_SB: One = One(UnsafeCell::new(Vec::new()));
}

/// Opens the in-WASM scope-base bracket: install `live` as the callee's `LIVE_SCOPE_BASE`
/// (`scope_stack_len()` at entry), saving the caller's for [`jit_pop_scope_base`]. Makes a
/// `scope_base`-reading callee (`getscopeobject`/`newfunction`) sound on the fast path — its
/// cached env's baked `0` is bypassed.
#[cfg(target_arch = "wasm32")]
pub(crate) fn jit_push_scope_base(live: usize) {
    let prev = live_sb_swap(live);
    #[cfg(target_feature = "atomics")]
    JIT_SB.with(|s| s.borrow_mut().push(prev));
    // SAFETY: single AVM2 thread; not held across a call.
    #[cfg(not(target_feature = "atomics"))]
    unsafe {
        (*jit_sb_store::JIT_SB.0.get()).push(prev);
    }
}

/// Closes the [`jit_push_scope_base`] bracket — restores the caller's `LIVE_SCOPE_BASE`.
#[cfg(target_arch = "wasm32")]
pub(crate) fn jit_pop_scope_base() {
    #[cfg(target_feature = "atomics")]
    let prev = JIT_SB.with(|s| s.borrow_mut().pop().unwrap_or(0));
    // SAFETY: single AVM2 thread; not held across a call.
    #[cfg(not(target_feature = "atomics"))]
    let prev = unsafe { (*jit_sb_store::JIT_SB.0.get()).pop().unwrap_or(0) };
    live_sb_swap(prev);
}

/// Pushes the in-WASM-dispatched callee onto the AVM2 call stack so it shows up in
/// `Error.getStackTrace()` — mirroring `try_enter`'s `push_call`. Web-only bracket half.
#[cfg(target_arch = "wasm32")]
pub(crate) fn jit_push_call(method_ptr: *const ()) {
    // SAFETY: called inside the run; `cx` is the installed ambient context, `method_ptr` a live
    // `Method` (its class is alive — the invariant the cached call-IC `ClassBoundMethod` relies on).
    let cx = unsafe { &mut *(cx_ptr() as *mut UpdateContext) };
    let method = unsafe { Method::from_ptr(method_ptr) };
    let gc = cx.gc();
    cx.avm2.push_call(gc, method);
}

/// Pops the call pushed by [`jit_push_call`] (balances the same bracket).
#[cfg(target_arch = "wasm32")]
pub(crate) fn jit_pop_call() {
    // SAFETY: as `jit_push_call`.
    let cx = unsafe { &mut *(cx_ptr() as *mut UpdateContext) };
    let gc = cx.gc();
    cx.avm2.pop_call(gc);
}

/// This run's base into the shared `avm2.scope_stack` (from the installed [`RunCtx`]).
pub(crate) fn scope_base() -> usize {
    live_sb_get()
}

/// Reifies a fresh callee-owned `Activation` from the installed [`RunCtx`].
///
/// # Safety
/// Call only from a helper running inside [`with_run_ctx`]; the returned Activation
/// must not escape the helper call (it holds an aliasing `&mut UpdateContext`).
pub(crate) unsafe fn reify<'gc>() -> Activation<'static, 'gc> {
    let ptr = run_ctx_get();
    debug_assert!(!ptr.is_null(), "avm2-jit3 helper with no RunCtx installed");
    let ctx = unsafe { &*ptr };
    // SAFETY: reverse of `RunCtx::new`'s erasure, within the same run. `cx` comes from the
    // ambient slot (set at the top-level entry), the rest from the per-callee `RunCtx`.
    let cx: &mut UpdateContext<'gc> = unsafe { &mut *(ambient_cx_get() as *mut UpdateContext<'gc>) };
    let scope: ScopeChain<'gc> = unsafe { core::mem::transmute(ctx.scope) };
    let bound_super: Option<ClassObject<'gc>> = unsafe { core::mem::transmute(ctx.bound_super) };
    let caller_library: Option<CallerLib<'gc>> = unsafe { core::mem::transmute(ctx.caller_library) };
    // The callee's own domain comes from its captured scope, not the caller.
    let domain = Some(scope.domain());
    let mut act = Activation::from_builtin(cx, bound_super, scope, domain, caller_library, None);
    // Retarget the scope frame to THIS method's base (not the live stack top), so
    // `newfunction`'s `create_scopechain` captures the method's own local scopes. Read from
    // `LIVE_SCOPE_BASE` (the per-call value both paths set), NOT `ctx.scope_base` — the in-WASM
    // fast path installs a cached env whose `RunCtx.scope_base` is a stale 0.
    act.jit_set_scope_base(live_sb_get());
    act
}

/// The installed run's erased `*mut UpdateContext`, WITHOUT building an `Activation` (unlike
/// [`reify`]). For the call IC's direct dispatch: the caller casts it back to
/// `*mut UpdateContext<'gc>` and reborrows LOCALLY, then hands `cx` to `jit.try_enter` — so a
/// JIT-compiled callee is entered with NO per-call `Activation` (only the declined fallback
/// reifies). Returning a raw pointer (not `&'gc mut`) keeps the reborrow's lifetime independent
/// of the `'gc` brand, avoiding a `'static` over-constraint.
///
/// # Safety
/// Call only from a helper running inside [`with_run_ctx`]; the reborrow must not escape the
/// helper call (it aliases the live `&mut UpdateContext`).
pub(crate) fn cx_ptr() -> *mut () {
    let cx = ambient_cx_get();
    debug_assert!(!cx.is_null(), "avm2-jit3 helper with no ambient cx installed");
    cx
}

/// Stashes a thrown error (erased) for `try_enter` to propagate after the run.
pub(crate) fn stash_error(e: Error<'_>) {
    // SAFETY: erase `'gc` for storage; taken this same run by `take_error`.
    let erased: Error<'static> = unsafe { core::mem::transmute::<Error<'_>, Error<'static>>(e) };
    pending_set(Some(erased));
}

/// Whether an error is currently stashed — the in-wasm bail's flag mirror. A compiled
/// body calls this (`perr`) after a mid-body throwing op and returns early if set, so the
/// error surfaces without running later ops on a bogus result. Non-consuming: `try_enter`
/// still takes it after the run.
pub(crate) fn has_pending() -> bool {
    pending_is_some()
}

/// Takes the stashed error, if any (called by `try_enter` after the run).
pub(crate) fn take_error<'gc>() -> Option<Error<'gc>> {
    pending_take()
        // SAFETY: reconstruct the `'gc` erased in `stash_error`, same run.
        .map(|e| unsafe { core::mem::transmute::<Error<'static>, Error<'gc>>(e) })
}
