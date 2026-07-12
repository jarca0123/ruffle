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
