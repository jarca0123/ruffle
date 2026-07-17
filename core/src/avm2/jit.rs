//! Pluggable JIT / method-compiler backend for AVM2 bytecode.
//!
//! Before interpreting a verified method, [`Activation::run_actions`] consults
//! the active [`JitBackend`]. A backend may compile the method and run it — a
//! native JIT (e.g. cranelift) on desktop, or a WASM-emitting JIT on the web
//! (generate a WASM module at runtime and let the browser compile it) — or it
//! may decline, in which case the interpreter runs the method as usual.
//!
//! The default [`NullJit`] declines every method, so the interpreter is always
//! used until a real backend is installed via
//! [`Avm2::set_jit_backend`](crate::avm2::Avm2::set_jit_backend).

use crate::avm2::function::FunctionArgs;
use crate::avm2::method::Method;
use crate::avm2::object::ClassObject;
use crate::avm2::scope::ScopeChain;
use crate::avm2::{Activation, Error, Value};
use crate::context::UpdateContext;

/// A strategy for executing a verified AVM2 bytecode method faster than the
/// tree-walking interpreter.
///
/// Stored on [`Avm2`](crate::avm2::Avm2) as `Rc<dyn JitBackend>`. The trait is
/// object-safe: its single method is generic only over the (erased) GC lifetime
/// `'gc`, so concrete backends can be swapped at runtime behind a `dyn`.
///
/// A backend is expected to keep its own cache of compiled methods (behind
/// interior mutability), keyed by method identity, so `try_run` can compile
/// lazily on the first (or Nth "hot") call and reuse the result afterwards.
pub trait JitBackend {
    /// Attempts to run `method` within `activation`.
    ///
    /// Returns `Some(result)` if this backend executed the method (compiling and
    /// caching it on demand). Returns `None` to fall back to the interpreter —
    /// e.g. when the method uses opcodes or control flow the backend doesn't
    /// support yet, or when runtime code generation is unavailable (a strict-CSP
    /// web page that forbids `WebAssembly.Module`).
    ///
    /// An implementation MUST reproduce the interpreter's observable behaviour
    /// for the method: the same operand-stack / local-register effects, the same
    /// return value, and the same thrown errors — otherwise it must return
    /// `None` and let the interpreter handle it.
    fn try_run<'gc>(
        &self,
        activation: &mut Activation<'_, 'gc>,
        method: Method<'gc>,
    ) -> Option<Result<Value<'gc>, Error<'gc>>>;

    /// Attempts to enter `method` **before** the interpreter constructs an
    /// [`Activation`] for it — the avm2-jit3 seam (see `AVM2_JIT_REDESIGN.md`).
    ///
    /// Called at the very top of [`exec`](crate::avm2::function::exec), so a backend
    /// that returns `Some` runs the method with NO `init_from_method` /
    /// `from_builtin` cost: it writes `[this, args]` into its own frame region and
    /// enters compiled code directly. `scope`/`bound_super` are the callee's own
    /// (the material for the per-method `env`); `cx` is the ambient GC/update
    /// context (not an `Activation` — the callee builds its own lazily only if a
    /// slow-path op needs one). Returns `None` to fall through to `exec`'s normal
    /// native/bytecode paths. Unlike [`Self::try_run`] (called from `run_actions`,
    /// after the Activation is already built), this can actually remove the `exec`
    /// floor. Default: decline.
    #[expect(clippy::too_many_arguments)]
    fn try_enter<'gc>(
        &self,
        cx: &mut UpdateContext<'gc>,
        method: Method<'gc>,
        scope: ScopeChain<'gc>,
        receiver: Value<'gc>,
        bound_super: Option<ClassObject<'gc>>,
        args: FunctionArgs<'_, 'gc>,
    ) -> Option<Result<Value<'gc>, Error<'gc>>> {
        let _ = (cx, method, scope, receiver, bound_super, args);
        None
    }

    /// §8 in-WASM dispatch: if `method` is already compiled AND eligible for a direct
    /// caller→callee `call_indirect` at a call site passing `argc` args — a JIT-compiled,
    /// no-scopes, non-arguments, untyped-parameter leaf whose parameter count equals `argc`
    /// (so no coercion / defaulting is needed) — returns the shared-table index of its `run`
    /// funcref, letting the caller's compiled body enter it in-WASM with no Rust dispatch
    /// bounce. Returns `None` otherwise (the caller keeps the Rust call-IC fallback). The
    /// backend fills this into its per-site call IC on a miss. Default: decline.
    fn ic_dispatch_run_idx(&self, method: Method<'_>, argc: usize) -> Option<u32> {
        let _ = (method, argc);
        None
    }

    /// Whether the compiled `method` reads its scope base (`getscopeobject`/`newfunction`) — so an
    /// in-WASM caller's `jit_enter`/`jit_leave` must install the live scope base + truncate the
    /// scope stack for it (the ~99% that don't skip that bracket). Default: `false`.
    fn method_scope_base_used(&self, method: Method<'_>) -> bool {
        let _ = method;
        false
    }

    /// TEMPORARY (inlining census): is `method` safe to SPLICE into its caller? True only if it
    /// cannot throw, makes no calls and does not read its scope base — see `CompiledMethod`. A
    /// caller that inlines such a body needs no ctx install and no call-stack push, because
    /// nothing inside can reify or capture a stack trace. Default: `false`.
    fn method_inline_safe(&self, method: Method<'_>) -> bool {
        let _ = method;
        false
    }

    /// TEMPORARY (inlining census): the compiled `method`'s op count, or `0` if not compiled.
    /// Cached per callee by the in-WASM dispatch env so the hot path can bucket calls by callee
    /// size without a lookup. Default: `0`.
    fn method_n_ops(&self, method: Method<'_>) -> u32 {
        let _ = method;
        0
    }
}

/// The default backend: never compiles anything, so every method is interpreted.
pub struct NullJit;

impl JitBackend for NullJit {
    #[inline]
    fn try_run<'gc>(
        &self,
        _activation: &mut Activation<'_, 'gc>,
        _method: Method<'gc>,
    ) -> Option<Result<Value<'gc>, Error<'gc>>> {
        None
    }
}
