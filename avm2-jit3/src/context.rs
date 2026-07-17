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
    /// Whether the callee reads its scope base (`getscopeobject`/`newfunction`). Only then does the
    /// in-WASM `jit_enter`/`jit_leave` need the LIVE-scope-base bracket (+ scope-stack truncate);
    /// the ~99% of callees that don't skip it entirely — the bracket isn't free.
    scope_base_used: bool,
    /// TEMPORARY (inlining census): the callee's op count, cached here because this env is built
    /// ONCE per callee — so `jit_enter` can bucket every call by callee size for free.
    n_ops: u32,
    /// TEMPORARY (inlining census): whether this callee is safe to splice (see `method_inline_safe`).
    inline_safe: bool,
}

impl JitEnv {
    pub(crate) fn new(
        ctx: RunCtx,
        method_ptr: *const (),
        scope_base_used: bool,
        n_ops: u32,
        inline_safe: bool,
    ) -> Self {
        Self { ctx, method_ptr, scope_base_used, n_ops, inline_safe }
    }
    /// Whether this callee is safe to splice (see the field).
    pub(crate) fn inline_safe(&self) -> bool {
        self.inline_safe
    }
    /// The callee's op count (see the field).
    pub(crate) fn n_ops(&self) -> u32 {
        self.n_ops
    }
    /// The callee's per-run context (`&self.ctx` — stable while the leaked env lives).
    pub(crate) fn ctx_ptr(&self) -> *const RunCtx {
        &self.ctx
    }
    /// The callee `Method`'s stable identity (`Method::as_ptr`), for the call-stack push.
    pub(crate) fn method_ptr(&self) -> *const () {
        self.method_ptr
    }
    /// Whether this callee reads its scope base (see the field).
    pub(crate) fn scope_base_used(&self) -> bool {
        self.scope_base_used
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
    /// Inline-domainMemory descriptor-pointer cache: `(dm_generation, domain_ptr, desc_ptr)`. Read
    /// by [`dm_desc_ptr`](crate::helpers::dm_desc_ptr) to skip a per-entry reify when memory hasn't
    /// been swapped. A `domain_ptr` of `0` means "empty" (a real entry has a non-zero domain AND a
    /// non-zero, shareable `desc_ptr`).
    static DM_DESC_CACHE: Cell<(u64, usize, i64)> = const { Cell::new((0, 0, 0)) };
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
    pub static DM_DESC_CACHE: One<(u64, usize, i64)> = One(UnsafeCell::new((0, 0, 0)));
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

/// A `JIT_SB` sentinel for a callee that does NOT read its scope base: `jit_leave` then leaves
/// `LIVE_SCOPE_BASE` untouched and skips the scope-stack truncate (the common, cheap case). A real
/// saved base is a `scope_stack_len()` — never `usize::MAX` — so this can't collide.
#[cfg(target_arch = "wasm32")]
const SB_NOOP: usize = usize::MAX;

#[cfg(target_arch = "wasm32")]
fn jit_sb_push(v: usize) {
    #[cfg(target_feature = "atomics")]
    JIT_SB.with(|s| s.borrow_mut().push(v));
    // SAFETY: single AVM2 thread; not held across a call.
    #[cfg(not(target_feature = "atomics"))]
    unsafe {
        (*jit_sb_store::JIT_SB.0.get()).push(v);
    }
}

#[cfg(target_arch = "wasm32")]
fn jit_sb_pop() -> usize {
    #[cfg(target_feature = "atomics")]
    {
        JIT_SB.with(|s| s.borrow_mut().pop().unwrap_or(SB_NOOP))
    }
    // SAFETY: single AVM2 thread; not held across a call.
    #[cfg(not(target_feature = "atomics"))]
    unsafe {
        (*jit_sb_store::JIT_SB.0.get()).pop().unwrap_or(SB_NOOP)
    }
}

/// Opens the in-WASM scope-base bracket: install `live` as the callee's `LIVE_SCOPE_BASE`
/// (`scope_stack_len()` at entry), saving the caller's for [`jit_pop_scope_base`]. Makes a
/// `scope_base`-reading callee (`getscopeobject`/`newfunction`) sound on the fast path — its
/// cached env's baked `0` is bypassed.
#[cfg(target_arch = "wasm32")]
pub(crate) fn jit_push_scope_base(live: usize) {
    let prev = live_sb_swap(live);
    jit_sb_push(prev);
}

/// The cheap path for a callee that does NOT read its scope base: leave `LIVE_SCOPE_BASE` alone and
/// record a sentinel so [`jit_pop_scope_base`] skips both the restore and the scope-stack truncate.
#[cfg(target_arch = "wasm32")]
pub(crate) fn jit_push_scope_base_noop() {
    jit_sb_push(SB_NOOP);
}

/// Closes a scope-base bracket. `Some(callee_base)` for a scope-using callee — its LIVE base (for
/// `jit_leave`'s scope-stack truncate); `LIVE_SCOPE_BASE` is restored to the caller's. `None` for
/// the no-op case (nothing to restore or truncate).
#[cfg(target_arch = "wasm32")]
pub(crate) fn jit_pop_scope_base() -> Option<usize> {
    let prev = jit_sb_pop();
    if prev == SB_NOOP {
        None
    } else {
        let callee_base = live_sb_get();
        live_sb_swap(prev);
        Some(callee_base)
    }
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

/// Why an in-WASM fast entry bailed to the Rust path. See [`record_slow`].
#[cfg(target_arch = "wasm32")]
pub(crate) enum Slow {
    /// An argument is an instance of a SUBCLASS of its declared param class. Coercing it is a
    /// no-op — the value passes through unchanged — so this bail is pure loss: it costs a full
    /// `try_enter` to conclude nothing needed doing. `coerces_identically_to` rejects it only
    /// because it compares classes for EQUALITY rather than walking the superclass chain.
    CoerceSubclass,
    /// An argument would need REAL coercion (int→Number, undefined→Object, …) — the raw-copy
    /// fast path cannot do it.
    Coerce,
    /// The frame arena is full (deep nesting).
    Arena,
    /// The site was memoised INELIGIBLE for the in-WASM entry, but re-asking now says it IS
    /// eligible — the callee compiled after this site first ran, and the memo is permanent, so the
    /// site is stuck on the slow path for nothing. A bug, and fixable.
    StaleMemo,
    /// The site is memoised ineligible and still is: `argc != nparams` (a defaulted parameter) or
    /// the callee never compiled. Real, but the `argc` half could be lifted.
    Ineligible,
    /// Ineligible because the callee is not in the compiled cache (declined, or never hot).
    NotCompiled,
    /// Ineligible because this site passes fewer args than the callee declares params — an AS3
    /// DEFAULTED parameter (`function f(a, b = 0)` called as `f(1)`). The in-WASM path raw-copies
    /// `argc` values and cannot materialise the defaults, so it bails. Liftable: the defaults are
    /// static, so the caller could store them into the callee frame at the call site.
    ArgcMismatch,
    /// Ruffle core called an AS3 method directly (event broadcast, frame construction, a timer):
    /// there is no JIT caller to dispatch from, so `try_enter` is the only way in. No JIT change
    /// removes this — it is the floor.
    Boundary,
}

/// TEMPORARY: tally WHY `jit_enter` returned 0, to split the entry-path gap. Starling runs 73.6%
/// fast vs Lua's 99.7%; the call IC is NOT the reason (a POLY census put 97.6% of calls on
/// monomorphic sites), so the remaining 26.4% is either these two bail-outs or a boundary crossing
/// (Ruffle core → AS3: event broadcast, frame construction), which no JIT change can remove.
/// Reported by [`record_entry`].
#[cfg(target_arch = "wasm32")]
thread_local! {
    static SLOW_SUB: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    static SLOW_COERCE: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    static SLOW_ARENA: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    static SLOW_BOUND: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    static SLOW_STALE: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    static SLOW_INELIG: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    static SLOW_NOCOMP: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    static SLOW_ARGC: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Master switch for the entry-path census ([`record_entry`] / [`record_slow`]). OFF: `record_entry`
/// fires on EVERY method entry and measured **3.41%** of the AVM2 worker once the fast path carried
/// ~90% of them — it distorts any profile taken with it on. Flip to `true` to re-run the census.
#[cfg(target_arch = "wasm32")]
pub(crate) const ENTRY_DIAG: bool = false;

/// TEMPORARY census: which coercions does `jit_enter` keep bailing on? `real-coerce` is now the
/// biggest FIXABLE slice of the slow entries (~32%), and the split decides whether it is worth
/// doing on the fast path: `int→Number` is a bit conversion (no alloc, no throw — `jit_enter`
/// could do it exactly as it now writes defaults), `→String` allocates, `→class` throws. Labels
/// are `target<-source`; ordered by [`COERCE_LABELS`].
#[cfg(target_arch = "wasm32")]
pub(crate) const COERCE_LABELS: [&str; 10] = [
    "num<-int",    // 0 — CHEAP: i32 as f64
    "num<-other",  // 1
    "int<-num",    // 2 — cheap-ish: ToInt32
    "uint<-any",   // 3
    "bool<-any",   // 4 — cheap: truthiness, never throws
    "str<-any",    // 5 — ALLOCATES
    "obj<-undef",  // 6 — cheap: undefined → null
    "class<-any",  // 7 — throws #1034 unless null (subclass already handled)
    "builtin<-any",// 8 — the `Some(_) => false` arm (XML/Function/…)
    "other",       // 9
];

#[cfg(target_arch = "wasm32")]
thread_local! {
    static COERCE_KINDS: std::cell::Cell<[u64; 10]> = const { std::cell::Cell::new([0; 10]) };
}

/// TEMPORARY (inlining census): callee-size buckets, by OP COUNT, weighted by CALLS. A JIT→JIT
/// call pays a bracket measured at ~21% of the AVM2 worker (`jit_enter` 12.9 + `jit_leave` 4.1 +
/// `push_call` 4.2); inlining the callee behind the existing vtable guard erases it for that call.
/// This says how much of that is reachable: a call to a 40-op callee is nearly all bracket, a call
/// to a 400-op one is nearly all work. Needs NO deopt — a guard miss just calls the helper — but
/// it does need a second compile tier, so the payoff must be known FIRST.
#[cfg(target_arch = "wasm32")]
pub(crate) const OPS_BUCKETS: [u32; 6] = [32, 64, 128, 256, 512, u32::MAX];

#[cfg(target_arch = "wasm32")]
thread_local! {
    static CALLEE_OPS: std::cell::Cell<[u64; 6]> = const { std::cell::Cell::new([0; 6]) };
    static CALLEE_SAFE: std::cell::Cell<[u64; 6]> = const { std::cell::Cell::new([0; 6]) };
}

/// Tally one in-WASM entry by its callee's op count, and separately whether the callee is
/// INLINE-SAFE (cannot throw / makes no calls / no scope base). Size alone overstates the prize:
/// only the safe ones can be spliced without breaking stack traces or handing a helper the
/// caller's scope. See [`OPS_BUCKETS`].
#[cfg(target_arch = "wasm32")]
pub(crate) fn record_callee_ops(n_ops: u32, inline_safe: bool) {
    if !ENTRY_DIAG {
        return;
    }
    let i = OPS_BUCKETS.iter().position(|&b| n_ops <= b).unwrap_or(5);
    let mut a = CALLEE_OPS.with(|c| c.get());
    a[i] += 1;
    CALLEE_OPS.with(|c| c.set(a));
    if inline_safe {
        let mut s = CALLEE_SAFE.with(|c| c.get());
        s[i] += 1;
        CALLEE_SAFE.with(|c| c.set(s));
    }
}

/// Tally one `jit_enter` coercion bail by kind (an index into [`COERCE_LABELS`]).
#[cfg(target_arch = "wasm32")]
pub(crate) fn record_coerce_kind(kind: usize) {
    if !ENTRY_DIAG {
        return;
    }
    let mut a = COERCE_KINDS.with(|c| c.get());
    a[kind] += 1;
    COERCE_KINDS.with(|c| c.set(a));
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn record_slow(why: Slow) {
    if !ENTRY_DIAG {
        return;
    }
    match why {
        Slow::CoerceSubclass => SLOW_SUB.with(|c| c.set(c.get() + 1)),
        Slow::Coerce => SLOW_COERCE.with(|c| c.set(c.get() + 1)),
        Slow::Arena => SLOW_ARENA.with(|c| c.set(c.get() + 1)),
        Slow::Boundary => SLOW_BOUND.with(|c| c.set(c.get() + 1)),
        Slow::StaleMemo => SLOW_STALE.with(|c| c.set(c.get() + 1)),
        Slow::Ineligible => SLOW_INELIG.with(|c| c.set(c.get() + 1)),
        Slow::NotCompiled => SLOW_NOCOMP.with(|c| c.set(c.get() + 1)),
        Slow::ArgcMismatch => SLOW_ARGC.with(|c| c.set(c.get() + 1)),
    }
}

/// TEMPORARY: tally method ENTRIES by path — the §8 in-WASM `jit_enter` fast path vs the Rust
/// `try_enter` bounce. `try_enter`'s cost is spread evenly over resolve/build/enter (measured
/// ~8/7/3.9/5.6), so it cannot be made much cheaper — the only lever is NOT CALLING IT, which makes
/// this ratio the number that decides whether that lever is worth anything. Plain `Cell`s: a
/// HashMap-based tally once cost ~20% on this path (see the PERF note in memory).
#[cfg(target_arch = "wasm32")]
pub(crate) fn record_entry(inwasm: bool) {
    if !ENTRY_DIAG {
        return;
    }
    use std::cell::Cell;
    thread_local! {
        static INWASM: Cell<u64> = const { Cell::new(0) };
        static RUSTY: Cell<u64> = const { Cell::new(0) };
    }
    if inwasm {
        INWASM.with(|c| c.set(c.get() + 1));
    } else {
        RUSTY.with(|c| c.set(c.get() + 1));
        // Is anything JIT-compiled on the stack? `RUN_CTX` is still the CALLER's here (`try_enter`
        // records before `enter_run` installs the callee's), so null ⇒ Ruffle core called AS3
        // directly and `try_enter` is the only way in. A non-null one is a JIT caller that fell
        // back — counted as guard-miss BY SUBTRACTION below, since `jit_enter`'s own coerce/arena
        // bail-outs land here too and must not be double-counted.
        if run_ctx_get().is_null() {
            record_slow(Slow::Boundary);
        }
    }
    let (i, r) = (INWASM.with(|c| c.get()), RUSTY.with(|c| c.get()));
    if (i + r) % 50000 == 0 {
        let pct = 100.0 * i as f64 / (i + r) as f64;
        // Split the slow half: `coerce`/`arena` are in-WASM guard HITS that `jit_enter` then
        // rejected (fixable in the JIT); the rest is a guard miss or a Ruffle-core → AS3 boundary
        // crossing (event broadcast, frame construction), which no JIT change removes.
        let (sub, co, ar, bo) = (
            SLOW_SUB.with(|c| c.get()),
            SLOW_COERCE.with(|c| c.get()),
            SLOW_ARENA.with(|c| c.get()),
            SLOW_BOUND.with(|c| c.get()),
        );
        let guard = r.saturating_sub(sub + co + ar + bo);
        let (st, il) = (SLOW_STALE.with(|c| c.get()), SLOW_INELIG.with(|c| c.get()));
        let (nc, ag) = (SLOW_NOCOMP.with(|c| c.get()), SLOW_ARGC.with(|c| c.get()));
        let p = |x: u64| 100.0 * x as f64 / r.max(1) as f64;
        crate::runner::diag_log(&format!(
            "JIT3 ENTRY PATH: in-wasm={i} try_enter={r} ({pct:.1}% fast) | of the slow {r}: \
             subclass={sub} ({:.1}%) real-coerce={co} ({:.1}%) arena={ar} ({:.1}%) \
             GUARD-MISS={guard} ({:.1}%) BOUNDARY={bo} ({:.1}%, the floor) \
             | ineligible: STALE(bug)={st} still={il} => NOT-COMPILED={nc} ARGC-MISMATCH={ag} \
             | coerce kinds: {ck} | INLINE CENSUS (calls by callee ops): {oc}",
            p(sub),
            p(co),
            p(ar),
            p(guard),
            p(bo),
            ck = {
                let a = COERCE_KINDS.with(|c| c.get());
                let tot: u64 = a.iter().sum();
                COERCE_LABELS
                    .iter()
                    .zip(a.iter())
                    .filter(|(_, n)| **n > 0)
                    .map(|(l, n)| {
                        format!("{l}={n}({:.0}%)", 100.0 * *n as f64 / tot.max(1) as f64)
                    })
                    .collect::<Vec<_>>()
                    .join(" ")
            },
            oc = {
                let a = CALLEE_OPS.with(|c| c.get());
                let sf = CALLEE_SAFE.with(|c| c.get());
                let tot: u64 = a.iter().sum();
                let (mut cum, mut cums) = (0u64, 0u64);
                let pct = |n: u64| 100.0 * n as f64 / tot.max(1) as f64;
                OPS_BUCKETS
                    .iter()
                    .zip(a.iter())
                    .zip(sf.iter())
                    .map(|((b, n), s)| {
                        cum += n;
                        cums += s;
                        let lbl = if *b == u32::MAX { ">512".into() } else { format!("<={b}") };
                        format!(
                            "{lbl}:{:.1}%(cum {:.1}%)/SAFE {:.1}%(cum {:.1}%)",
                            pct(*n), pct(cum), pct(*s), pct(cums)
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(" ")
            },
        ));
    }
}

/// This run's base into the shared `avm2.scope_stack` (from the installed [`RunCtx`]).
pub(crate) fn scope_base() -> usize {
    live_sb_get()
}

/// The current run's domain pointer, read cheaply from the installed [`RunCtx`]'s captured scope —
/// no full [`reify`]. This is the SAME domain `Activation::domain_memory` would resolve (both go
/// through `scope.domain()`), so it is a sound cache key for the inline-domainMemory descriptor.
/// Returns `0` if no run is installed.
pub(crate) fn current_domain_ptr() -> usize {
    let ptr = run_ctx_get();
    if ptr.is_null() {
        return 0;
    }
    // SAFETY: `RunCtx` installed for this run; its `scope` is a live `ScopeChain` (erased to
    // 'static by `RunCtx::new`). We only read a Gc pointer identity, which does not depend on the
    // lifetime, and the domain outlives the run.
    let ctx = unsafe { &*ptr };
    ctx.scope.domain().as_ptr() as usize
}

/// Read the inline-domainMemory descriptor cache `(dm_generation, domain_ptr, desc_ptr)`.
pub(crate) fn dm_cache_get() -> (u64, usize, i64) {
    #[cfg(any(not(target_arch = "wasm32"), target_feature = "atomics"))]
    {
        DM_DESC_CACHE.with(|c| c.get())
    }
    // SAFETY: single-threaded read (see the module note).
    #[cfg(all(target_arch = "wasm32", not(target_feature = "atomics")))]
    unsafe {
        *single_thread::DM_DESC_CACHE.0.get()
    }
}

/// Store the inline-domainMemory descriptor cache (see [`dm_cache_get`]).
pub(crate) fn dm_cache_set(v: (u64, usize, i64)) {
    #[cfg(any(not(target_arch = "wasm32"), target_feature = "atomics"))]
    {
        DM_DESC_CACHE.with(|c| c.set(v));
    }
    // SAFETY: single-threaded write (see the module note).
    #[cfg(all(target_arch = "wasm32", not(target_feature = "atomics")))]
    unsafe {
        *single_thread::DM_DESC_CACHE.0.get() = v;
    }
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
