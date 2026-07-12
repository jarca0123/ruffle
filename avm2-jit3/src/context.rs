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
    /// `*mut UpdateContext<'gc>` — the ambient GC/update context (NOT an Activation).
    cx: *mut (),
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

impl RunCtx {
    pub(crate) fn new<'gc>(
        cx: &mut UpdateContext<'gc>,
        scope: ScopeChain<'gc>,
        bound_super: Option<ClassObject<'gc>>,
        scope_base: usize,
        caller_library: Option<CallerLib<'gc>>,
    ) -> Self {
        // SAFETY: erase `'gc` for storage; reconstructed only within this run, where
        // the objects are alive (GC-quiescent frame).
        RunCtx {
            cx: cx as *mut UpdateContext<'gc> as *mut (),
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
    static PENDING: RefCell<Option<Error<'static>>> = const { RefCell::new(None) };
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
    pub static PENDING: One<Option<Error<'static>>> = One(UnsafeCell::new(None));
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
    let r = f();
    run_ctx_swap(prev);
    r
}

/// This run's base into the shared `avm2.scope_stack` (from the installed [`RunCtx`]).
pub(crate) fn scope_base() -> usize {
    let ptr = run_ctx_get();
    debug_assert!(!ptr.is_null(), "avm2-jit3 scope helper with no RunCtx installed");
    unsafe { &*ptr }.scope_base
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
    // SAFETY: reverse of `RunCtx::new`'s erasure, within the same run.
    let cx: &mut UpdateContext<'gc> = unsafe { &mut *(ctx.cx as *mut UpdateContext<'gc>) };
    let scope: ScopeChain<'gc> = unsafe { core::mem::transmute(ctx.scope) };
    let bound_super: Option<ClassObject<'gc>> = unsafe { core::mem::transmute(ctx.bound_super) };
    let caller_library: Option<CallerLib<'gc>> = unsafe { core::mem::transmute(ctx.caller_library) };
    // The callee's own domain comes from its captured scope, not the caller.
    let domain = Some(scope.domain());
    let mut act = Activation::from_builtin(cx, bound_super, scope, domain, caller_library, None);
    // Retarget the scope frame to THIS method's base (not the live stack top), so
    // `newfunction`'s `create_scopechain` captures the method's own local scopes.
    act.jit_set_scope_base(ctx.scope_base);
    act
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
