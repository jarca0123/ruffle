//! avm2-jit3: a ground-up AVM2 → WebAssembly JIT.
//!
//! See `AVM2_JIT_REDESIGN.md` and the approved plan (`vivid-hopping-flask`). The
//! target architecture mirrors avmplus: every method — AS3 or native builtin — is a
//! funcref of ONE WASM type reached by a single `call_indirect`, with no interpreter
//! [`Activation`](ruffle_core::avm2::Activation) on the hot path. A per-method `env`
//! (arg0) carries the callee's own domain/class/scope/super/tables, so nothing has to
//! reach back into the caller's Activation; slow-path core ops reify a FRESH
//! callee-owned Activation from `(env, frame, cx)`.
//!
//! ## Status: Phase 1 — one leaf method through the type-0 ABI, zero Activation
//!
//! [`Jit3::try_enter`] (called at the very top of
//! [`exec`](ruffle_core::avm2::function), before any Activation is built) compiles and
//! runs **straight-line, primitive-only, matched-arg leaf methods** through the type-0
//! `run(env, argc, args) -> i64` ABI: it writes `[this, args]` into the module's frame
//! memory and calls compiled WASM directly — no `init_from_method`, no `run_actions`,
//! no `Activation`. Everything else declines (returns `None`) and the interpreter runs
//! it unchanged. Later phases add `env` tables, slot/property ops, the reification ABI,
//! native thunks, and the error-sentinel unwinding.

mod cfg;
mod context;
mod emit;
mod helpers;
mod translate;
mod typed;
mod value;

// The runner is the one platform-specific piece: native = wasmtime/cranelift,
// web = the browser's WASM engine via js-sys. Both expose `run_leaf`.
#[cfg(not(target_arch = "wasm32"))]
#[path = "runner.rs"]
mod runner;
#[cfg(target_arch = "wasm32")]
#[path = "runner_web.rs"]
mod runner;

use std::cell::{Cell, RefCell};
use std::mem::MaybeUninit;
use std::rc::Rc;

use fnv::FnvHashMap;

use ruffle_core::avm2::error::make_error_1063;
use ruffle_core::avm2::{
    Activation, ClassObject, Error, FunctionArgs, JitBackend, Method, ScopeChain, Value,
};
use ruffle_core::context::UpdateContext;

/// A compiled method plus the method-constant metadata the per-call entry needs — cached so
/// the hot path avoids re-deriving it (a `body()`/`param_config` walk) every call.
#[derive(Clone)]
struct CompiledMethod {
    /// The type-0 WASM module bytes.
    bytes: Rc<[u8]>,
    /// The runner's per-method entry handle (the instantiated module / entry funcref), lazily
    /// filled on first run. Held HERE so the per-call path takes ONE cache lookup
    /// (`self.compiled`) instead of two — the runner no longer re-looks-up the method by key.
    handle: runner::Handle,
    /// Local count (frame width) — avoids a `method.body()` lookup per call.
    num_locals: u32,
    /// Whether the method reads the local scope stack (`getscopeobject`) — only then must the
    /// per-call scope frame be truncated after the run.
    needs_scopes: bool,
    /// Whether ANY parameter is typed — only then can an arg need coercion, so an untyped
    /// signature skips the per-call `coerces_identically_to` scan entirely.
    has_typed_params: bool,
    /// Declared parameter count (`sig.len()`). The §8 in-WASM dispatch fast path is eligible
    /// only when a call site's `argc` equals this (so args exactly fill params — no defaulting).
    nparams: u32,
    /// Whether the method makes any nested call/construct (`call*`/`construct*` op). BISECTION:
    /// used to restrict §8 in-WASM dispatch to LEAF callees while hunting a Starling-only #1034
    /// (Lua's flat computational callees work; the bug is in OO/nested-call dispatch).
    makes_calls: bool,
    /// Whether the method READS its scope base — `getscopeobject` (local scope frame) or
    /// `newfunction` (`create_scopechain` captures `scope_frame[base..]`). The §8 in-WASM dispatch
    /// bakes a fixed `scope_base = 0` in the cached env (a per-call value it cannot set), so a
    /// nested callee that reads the base would see the wrong frame — such methods are in-WASM
    /// ineligible (they still run via the Rust `try_enter` fallback, which sets `scope_base` live).
    scope_base_used: bool,
    /// TEMPORARY (inlining census): total `JitOp`s across all blocks — the callee's SIZE. A
    /// JIT→JIT call pays a ~21%-of-worker bracket (`jit_enter` + `jit_leave` + `push_call`);
    /// splicing a small callee into its caller behind the existing vtable guard would erase that
    /// for the call. This measures whether it is worth building: what share of CALLS (not sites)
    /// target a callee small enough to inline.
    n_ops: u32,
    /// TEMPORARY (inlining census): whether the body contains NO `BailIfError`/`BailIfPerr` — i.e.
    /// it cannot throw. `translate` emits one after every op that can, so their absence is a proof.
    ///
    /// This is the gate INLINING actually needs, and it is stricter than it first looks. Splicing a
    /// callee's body into its caller erases the callee's `push_call`, so it vanishes from
    /// `Error.getStackTrace()` — which the test suite checks. It also erases `jit_enter`'s ctx
    /// install, so any helper that reifies would read the CALLER's scope/bound_super (the exact
    /// corruption class jit3 exists to avoid). A body that cannot throw and makes no calls has
    /// neither problem: no trace can be captured inside it, and nothing reifies.
    cannot_throw: bool,
}

/// The avm2-jit3 backend. Install via
/// [`Avm2::set_jit_backend`](ruffle_core::avm2::Avm2::set_jit_backend).
#[derive(Default)]
pub struct Jit3 {
    /// Per-method cache keyed by `Method::as_ptr()`. `Ok(compiled)` = a compiled type-0
    /// module + metadata; `Err(reason)` = permanently declined (an unsupported method — don't
    /// re-translate it every call; the `reason` feeds the frequency-weighted decline
    /// profiler). Absent = not yet seen.
    compiled: RefCell<FnvHashMap<usize, Result<CompiledMethod, &'static str>>>,
}

impl Jit3 {
    /// Creates a new avm2-jit3 backend.
    pub fn new() -> Self {
        Self::default()
    }

    /// Phase A of [`try_enter`](JitBackend::try_enter): the hashmap lookup + `CompiledMethod` clone
    /// on the hot path; first-sighting `try_compile` on the cold path. Returns the compiled method,
    /// or `None` when the method is declined (native / unverified-retry / a translate decline) —
    /// reason recorded.
    ///
    /// Hot path takes a SHARED borrow (a compiled cache hit never mutates); only a first-sighting
    /// compile/decline takes the exclusive borrow. The borrow is released before the run (a nested
    /// run aliases this `RefCell`). NB: the cloned lookup is bound to a local FIRST — a `match` on
    /// `self.compiled.borrow()....cloned()` keeps the shared borrow alive across the arms
    /// (temporary lifetime), so the `None` arm's `borrow_mut()` would double-borrow.
    fn resolve_compiled(&self, method: Method<'_>) -> Option<CompiledMethod> {
        let key = method.as_ptr() as usize;
        let cached = self.compiled.borrow().get(&key).cloned();
        match cached {
            Some(Ok(c)) => Some(c),
            Some(Err(reason)) => {
                // Permanently declined — count WHY, weighted by call frequency.
                record_decline_reason(reason);
                None
            }
            None => {
                // Not seen yet. `try_enter` runs at the TOP of `exec`, before `init_from_method`
                // verifies the method — so the FIRST sighting is usually unverified. That is a
                // RETRYABLE decline: don't cache it, so a later call (after the interpreter
                // verified it) can still compile. (Self-`verify()` here corrupts shared state —
                // verify must run in the method's normal entry path — so 2nd-call tier-up is the
                // only sound policy.)
                if method.try_verified_info().is_none() {
                    // A NATIVE method (no bytecode body) can never be JIT'd — cache the decline so
                    // its (often very hot: Math/getters/builtins) calls stop re-checking. A
                    // not-yet-verified BYTECODE method is retryable → don't cache.
                    if method.body().is_none() {
                        self.compiled.borrow_mut().insert(key, Err("native"));
                        record_decline_reason("native");
                    } else {
                        record_decline_reason("unverified(retry)");
                    }
                    return None;
                }
                let compiled = try_compile(method);
                self.compiled.borrow_mut().insert(key, compiled.clone());
                match compiled {
                    Ok(c) => Some(c),
                    Err(reason) => {
                        record_decline_reason(reason);
                        None
                    }
                }
            }
        }
    }
}

/// Phase B of [`try_enter`](JitBackend::try_enter): resolve the param signature and write
/// `[this, params]` into `frame`. Returns the frame length (`1 + nparams`); `None` to DECLINE
/// (more args than params — recorded); `Some(Err)` on a coercion / #1063 throw.
///
/// The fast path (args EXACTLY fill the params, each already the param type — always so for an
/// untyped signature) raw-writes, bit-identical to `init_from_method`. Otherwise the slow path
/// COERCES each provided arg to its param type and fills missing params with their default (mirrors
/// `resolve_parameters`), reifying a temp Activation (a coercion may run AS3 `valueOf` and throw).
/// Only `[this, params]` are written — NOT full-width `undefined` padding; the compiled prologue
/// `undefined`-inits any non-promoted local that could be read before written.
fn build_frame<'gc>(
    cx: &mut UpdateContext<'gc>,
    compiled: &CompiledMethod,
    method: Method<'gc>,
    scope: ScopeChain<'gc>,
    receiver: Value<'gc>,
    bound_super: Option<ClassObject<'gc>>,
    args: FunctionArgs<'_, 'gc>,
    frame: &mut [MaybeUninit<u64>; emit::MAX_LOCALS],
) -> Option<Result<usize, Error<'gc>>> {
    let sig = method.resolved_param_config();
    if args.len() > sig.len() {
        record_decline_reason("args_extra");
        return None;
    }
    frame[0].write(value::to_bits(receiver));
    let all_matched = args.len() == sig.len()
        && (!compiled.has_typed_params
            || args
                .iter()
                .zip(sig.iter())
                .all(|(arg, p)| p.param_type.is_none_or(|c| arg.coerces_identically_to(c))));
    if all_matched {
        for (i, arg) in args.iter().enumerate() {
            frame[1 + i].write(value::to_bits(arg));
        }
    } else {
        let caller_library = method.owner_library();
        let mut act =
            Activation::from_builtin(cx, bound_super, scope, Some(scope.domain()), caller_library, None);
        let mut provided = args.iter();
        for (i, p) in sig.iter().enumerate() {
            let arg = match provided.next() {
                Some(a) => a,
                // Missing param: its default, else `undefined` for an UNCHECKED method (top-level
                // functions / `arguments`-style callees), else #1063 — as `init_from_method` does.
                None => match p.default_value {
                    Some(d) => d,
                    None if method.is_unchecked() => Value::Undefined,
                    None => return Some(Err(make_error_1063(&mut act, method, args.len()))),
                },
            };
            let v = match p.param_type {
                Some(c) if !arg.coerces_identically_to(c) => match arg.coerce_to_type(&mut act, c) {
                    Ok(v) => v,
                    Err(e) => return Some(Err(e)),
                },
                _ => arg,
            };
            frame[1 + i].write(value::to_bits(v));
        }
    }
    Some(Ok(1 + sig.len()))
}

/// Phase C of [`try_enter`](JitBackend::try_enter): install the callee's reification context around
/// the run and execute the compiled body. Returns the raw result bits, or `None` if the run itself
/// declined (`run_leaf` frame-arena overflow / instantiate failure).
///
/// A slow-path helper (`cr` return-coercion; later getproperty/calls) reifies a FRESH callee-owned
/// Activation from `(cx, scope, bound_super)` — never the caller's; `cx` (ambient) is aliased via a
/// raw pointer only for the synchronous run. `scope_base` is the method's base on the shared
/// `avm2.scope_stack`. `push_call` puts the method on the AVM2 call stack (`Error.getStackTrace()`);
/// the jit3 seam runs BEFORE `exec`'s own `push_call`. `caller_library = owner_library()` mirrors
/// `init_from_method` so native callees resolve against this method's SWF. Only a scope-reading
/// method needs the post-run scope truncate.
fn enter_run<'gc>(
    cx: &mut UpdateContext<'gc>,
    compiled: &CompiledMethod,
    method: Method<'gc>,
    scope: ScopeChain<'gc>,
    bound_super: Option<ClassObject<'gc>>,
    frame: &[u64],
    argc: u32,
    num_locals: usize,
) -> Option<u64> {
    let caller_library = method.owner_library();
    let scope_base = cx.avm2.scope_stack_len();
    let gc = cx.gc();
    cx.avm2.push_call(gc, method);
    // `cx` (ambient) is installed separately from the per-callee `RunCtx` (scope/super/…): a
    // nested run reuses the same `cx`, and this decoupling makes `RunCtx` cacheable (§8).
    let run_ctx = context::RunCtx::new(scope, bound_super, scope_base, caller_library);
    let bits = context::with_ambient_cx(cx, || {
        context::with_run_ctx(&run_ctx, || {
            runner::run_leaf(&compiled.handle, &compiled.bytes, frame, argc, num_locals)
        })
    });
    cx.avm2.pop_call(gc);
    if compiled.needs_scopes {
        cx.avm2.truncate_scope_stack(scope_base);
    }
    bits
}

impl JitBackend for Jit3 {
    fn try_run<'gc>(
        &self,
        _activation: &mut Activation<'_, 'gc>,
        _method: Method<'gc>,
    ) -> Option<Result<Value<'gc>, Error<'gc>>> {
        // avm2-jit3's entry point is `try_enter` (pre-Activation), never `try_run`.
        None
    }

    fn try_enter<'gc>(
        &self,
        cx: &mut UpdateContext<'gc>,
        method: Method<'gc>,
        scope: ScopeChain<'gc>,
        receiver: Value<'gc>,
        bound_super: Option<ClassObject<'gc>>,
        args: FunctionArgs<'_, 'gc>,
    ) -> Option<Result<Value<'gc>, Error<'gc>>> {
        // `try_enter` is split into three phases (originally `#[inline(never)]` so a web CPU
        // profile could BISECT the entry cost; that attribution is done, so they're inlinable now):
        //   A `resolve_compiled` — cache lookup / clone / first-sighting compile.
        //   B `build_frame`      — param-config + writing `[this, params]` (fast / coerce paths).
        //   C `enter_run`        — context install + `push_call` + `run_leaf` + unwind.
        let compiled = self.resolve_compiled(method)?;
        #[cfg(target_arch = "wasm32")]
        context::record_entry(false);
        let num_locals = compiled.num_locals as usize; // cached — no `body()` lookup per call
        let argc = args.len() as u32;
        let mut frame_storage: [MaybeUninit<u64>; emit::MAX_LOCALS] =
            [MaybeUninit::uninit(); emit::MAX_LOCALS];
        let frame_len =
            match build_frame(cx, &compiled, method, scope, receiver, bound_super, args, &mut frame_storage) {
                None => return None,                    // decline (more args than params)
                Some(Err(e)) => return Some(Err(e)),    // coercion / #1063 throw
                Some(Ok(n)) => n,
            };
        // SAFETY: slots `[0, frame_len)` are all initialized by `build_frame` (receiver +
        // `nparams` params); verification guarantees `frame_len ≤ num_locals ≤ MAX_LOCALS`.
        let frame: &[u64] =
            unsafe { &*(&frame_storage[..frame_len] as *const [MaybeUninit<u64>] as *const [u64]) };
        let bits = enter_run(cx, &compiled, method, scope, bound_super, frame, argc, num_locals)?;
        // A slow-path coercion may have thrown (`#1034`): propagate it. (The op that can throw
        // — `cr` — is always immediately followed by `Return`, so the run has ended.)
        if let Some(err) = context::take_error::<'gc>() {
            return Some(Err(err));
        }
        // Diagnostic: prove a real method ran through the type-0 ABI with no Activation.
        #[cfg(not(target_arch = "wasm32"))]
        if std::env::var_os("RUFFLE_JIT3_TRACE").is_some() {
            eprintln!("JIT3 RAN method@{:#x} argc={argc} -> {bits:#018x}", method.as_ptr() as usize);
        }
        // SAFETY: `bits` is a `Value` produced by the JIT within this GC-quiescent frame
        // (see `value::from_bits`); any object pointer it encodes is still live.
        let result = unsafe { value::from_bits::<'gc>(bits) };
        Some(Ok(result))
    }

    fn ic_dispatch_run_idx(&self, method: Method<'_>, argc: usize) -> Option<u32> {
        // Eligible only when the callee is compiled AND a direct in-WASM entry is sound:
        //   * does not read `scope_base` (getscopeobject / newfunction) — the per-call base the
        //     fast path bakes as 0 and cannot set live (see `CompiledMethod::scope_base_used`),
        //   * this site's `argc` == the callee's parameter count — args exactly fill params, so no
        //     defaulting / #1063. (Non-arguments/non-variadic are already guaranteed: `try_compile`
        //     declines variadic, and the call-IC cell is filled only for a non-arguments method.)
        // TYPED params are NO LONGER excluded here — `jit_enter` validates at call time that the
        // args already match (else it returns 0 → the caller falls back to the coercing helper), so
        // typed-param callees whose args happen to match (the common case) also take the fast path.
        // Returns the callee `run`'s index in the shared table (web) so the caller `call_indirect`s
        // it. Native has no shared table → `handle_run_idx` is `None`, so native never takes it.
        // Non-leaf callees ARE eligible (the #1034 that once forced a leaf-only bisection was a
        // stale-cell bug in `call_method_ic_bits`, since fixed).
        let key = method.as_ptr() as usize;
        let cache = self.compiled.borrow();
        let Some(Ok(cm)) = cache.get(&key) else {
            #[cfg(target_arch = "wasm32")]
            context::record_slow(context::Slow::NotCompiled);
            return None;
        };
        // `scope_base_used` (getscopeobject/newfunction) is NO LONGER excluded: `jit_enter` now
        // installs this call's LIVE scope base (and `jit_leave` truncates the scope stack), so the
        // cached env's baked `scope_base = 0` is bypassed. This unlocks FlasCC/Alchemy callees
        // (e.g. Lua's `luaV_execute`), which were ~90% of that workload's fast-path misses.
        // TYPED params are NOT excluded — `jit_enter` validates at call time that the args already
        // match (else it returns 0 → the caller falls back to the coercing helper). See the note
        // above `has_typed_params` on `CompiledMethod`.
        // Fewer args than params is fine when every MISSING one has a default that needs no
        // coercion: they are static, so `jit_enter` writes them straight into the callee frame
        // (the caller only ever fills `[this, args[0..argc])`, leaving those slots to us). This is
        // the whole of the remaining fast-path gap — 100% of ineligible sites were this, ~9.6% of
        // all entries, and AS3 libraries default parameters everywhere. A default needing coercion
        // stays out: `jit_enter` must not allocate or throw. `argc > nparams` is `build_frame`'s
        // `args_extra` decline and stays ineligible.
        if argc > cm.nparams as usize {
            #[cfg(target_arch = "wasm32")]
            context::record_slow(context::Slow::ArgcMismatch);
            return None;
        }
        if argc < cm.nparams as usize {
            let sig = method.resolved_param_config();
            let defaults_ok = sig[argc..].iter().all(|p| {
                p.default_value
                    .is_some_and(|d| p.param_type.is_none_or(|c| d.coerces_identically_to(c)))
            });
            if !defaults_ok {
                #[cfg(target_arch = "wasm32")]
                context::record_slow(context::Slow::ArgcMismatch);
                return None;
            }
        }
        runner::handle_run_idx(&cm.handle)
    }

    fn method_scope_base_used(&self, method: Method<'_>) -> bool {
        let key = method.as_ptr() as usize;
        matches!(self.compiled.borrow().get(&key), Some(Ok(cm)) if cm.scope_base_used)
    }

    fn method_inline_safe(&self, method: Method<'_>) -> bool {
        let key = method.as_ptr() as usize;
        matches!(self.compiled.borrow().get(&key),
            Some(Ok(cm)) if cm.cannot_throw && !cm.makes_calls && !cm.scope_base_used)
    }

    fn method_n_ops(&self, method: Method<'_>) -> u32 {
        let key = method.as_ptr() as usize;
        match self.compiled.borrow().get(&key) {
            Some(Ok(cm)) => cm.n_ops,
            _ => 0,
        }
    }
}

/// Translates + compiles `method` to a type-0 module, or `Err(reason)` to permanently
/// decline (the `reason` is a coarse, `'static` label for the decline profiler). Runs only
/// for **verified** methods (`try_enter` fires before `init_from_method` verifies, so a
/// first, unverified sighting declines and the interpreter verifies + runs it; a later call
/// then compiles).
/// Minimum op count a method must have for the [`jitop_histogram`] dump; `0` disables it entirely
/// (zero cost — the whole diagnostic short-circuits). Works on BOTH targets: web has no
/// `std::env`, so the const is the gate there; on native `RUFFLE_JIT3_HISTO=<n>` overrides it.
/// OFF: the dump costs real time on web — `console_log` measured **5.86%** of the AVM2 worker
/// (228ms) on a Lua-bench trace, plus the `record_*` bookkeeping inside the translate fixpoint.
/// It also has to be off for any profile to be trustworthy. Set to e.g. `250` to bring it back.
const HISTO_MIN_OPS: usize = 0;

/// The effective histogram threshold (see [`HISTO_MIN_OPS`]).
pub(crate) fn histo_min_ops() -> usize {
    #[cfg(not(target_arch = "wasm32"))]
    if let Ok(v) = std::env::var("RUFFLE_JIT3_HISTO").unwrap_or_default().parse::<usize>() {
        return v;
    }
    HISTO_MIN_OPS
}

/// TEMPORARY diagnostic (gated by [`HISTO_MIN_OPS`] / `RUFFLE_JIT3_HISTO=<min_ops>`): dump the STATIC JitOp mix of
/// each compiled method with at least `min_ops` ops. Answers "what fraction of the generated code
/// is guard/box vs useful work" — the question the sampling profiler can no longer answer now that
/// everything is inlined into one `wasm-function[0]` symbol. `translate` is architecture-neutral,
/// so a desktop run yields the same op stream the web build compiles. NOTE: this is a STATIC mix
/// over all blocks, NOT an execution-weighted one — read it as the method's shape, not its profile.
fn jitop_histogram(method: &Method<'_>, blocks: &[emit::Block], promoted: &[emit::Promotion]) {
    use emit::JitOp as J;
    let min_ops = histo_min_ops();
    if min_ops == 0 {
        return;
    }
    let ops: Vec<&J> = blocks.iter().flat_map(|b| &b.ops).collect();
    // Always drain (they accumulated during THIS method's translate), even if we don't report.
    let guarded_why = translate::take_guard_log();
    let promo_why = translate::take_promo_log();
    if ops.len() < min_ops {
        return;
    }
    let mut c: std::collections::BTreeMap<&str, usize> = Default::default();
    for op in &ops {
        let bucket = match op {
            // Raw domainMemory access: each carries an inline bounds check (+ tag check unless the
            // address is a proven Int). The target of a range/induction analysis.
            J::MopLoad(_, true) | J::MopStore(_, true) => "mop_addr_proven",
            J::MopLoad(_, false) | J::MopStore(_, false) => "mop_addr_guarded",
            // Arithmetic the typed frame PROVED — native f64/i32, no tag guard, no helper.
            J::BinOpNum { .. } | J::BinOpInt(_) | J::BinOpCmp { .. } | J::NullishEq { .. } => {
                "arith_proven"
            }
            // Arithmetic it did NOT prove — boxed operands, tag guards, may call the helper.
            J::BinOp(_) | J::UnOp(_) => "arith_guarded",
            J::GetLocal(_) | J::SetLocal(_) => "local",
            J::PushBits(_) => "const",
            J::Swap | J::Pop | J::Dup => "stack_shuffle",
            J::GetScope(_) | J::ScopePush | J::ScopePop | J::OuterScope(_) => "scope",
            J::Coerce(_) | J::CoerceReturn(_) => "coerce",
            J::BailIfError | J::BailIfPerr => "bail_check",
            J::GetSlot(_) | J::GetSlotChecked(_) | J::SetSlot(..) => "slot",
            J::GetProperty(_) | J::SetProperty(_) | J::GetPropertyFast(_)
            | J::SetPropertyFast(_) => "property",
            J::CallProperty(..) | J::CallMethod(..) | J::CallNative(..) | J::Call(_)
            | J::TailCallProperty(..) | J::TailCallMethod(..) | J::CallSuper(..) => "call",
            J::ReturnValue | J::ReturnVoid | J::Throw => "return",
            _ => "other",
        };
        *c.entry(bucket).or_default() += 1;
    }
    let total = ops.len();
    let pct = |k: &str| 100.0 * *c.get(k).unwrap_or(&0) as f64 / total as f64;
    let n_f64 = promoted.iter().filter(|p| p.kind == emit::RegKind::F64).count();
    let n_i32 = promoted.len() - n_f64;
    let mix: Vec<String> =
        c.iter().map(|(k, v)| format!("{k}={v} ({:.1}%)", 100.0 * *v as f64 / total as f64)).collect();
    runner::diag_log(&format!(
        "JIT3 HISTO {name:?}: ops={total} blocks={nb} regs=(f64:{n_f64},i32:{n_i32}) \
         | mop={mop:.1}% arith_proven={ap:.1}% arith_guarded={ag:.1}% \
         | {mix}",
        name = method.method_name(),
        nb = blocks.len(),
        mop = pct("mop_addr_proven") + pct("mop_addr_guarded"),
        ap = pct("arith_proven"),
        ag = pct("arith_guarded"),
        mix = mix.join(" "),
    ));
    // How much of the `br_table` dispatch a relooper could remove: `inline` edges vanish, `br_*`
    // become one direct branch, `dispatch` (irreducible only) would still need the state machine.
    let g = cfg::analyze(blocks);
    runner::diag_log(&format!(
        "JIT3 CFG {name:?}: blocks={nb} edges={e} | inline={i} ({ip:.1}%) br_fwd={bf} \
         br_loop={bl} DISPATCH={d} | loops={lp} merges={mg} cross_stack_blocks={cs}",
        name = method.method_name(),
        nb = g.blocks,
        e = g.edges,
        i = g.inline,
        ip = 100.0 * g.inline as f64 / g.edges.max(1) as f64,
        bf = g.br_fwd,
        bl = g.br_loop,
        d = g.dispatch,
        lp = g.loops,
        mg = g.merges,
        cs = g.cross_blocks,
    ));
    // WHY each arithmetic op stayed guarded: the `(binop, repr_a, repr_b)` triple `bin()` saw.
    // This is the actionable half — it names exactly which proof is missing.
    let why: Vec<String> =
        guarded_why.iter().take(12).map(|(k, n)| format!("{k}={n}")).collect();
    runner::diag_log(&format!(
        "JIT3 GUARDED-WHY {:?}: {}",
        method.method_name(),
        why.join(" ")
    ));
    // What PRECEDES each BailIfError — i.e. which throwing op each sentinel check is paying for.
    {
        let mut prev: std::collections::BTreeMap<&str, usize> = Default::default();
        for b in blocks {
            for w in b.ops.windows(2) {
                if matches!(w[1], J::BailIfError | J::BailIfPerr) {
                    let k = match w[0] {
                        J::MopLoad(..) | J::MopStore(..) => "mop",
                        J::BinOp(_) | J::UnOp(_) => "arith",
                        J::GetSlot(_) | J::GetSlotChecked(_) | J::SetSlot(..) => "slot",
                        J::GetProperty(_) | J::SetProperty(_) | J::GetPropertyFast(_)
                        | J::SetPropertyFast(_) => "property",
                        J::CallProperty(..) | J::CallMethod(..) | J::CallNative(..) => "call",
                        J::ScopePush => "scopepush",
                        J::Coerce(_) => "coerce",
                        _ => "other",
                    };
                    *prev.entry(k).or_default() += 1;
                }
            }
        }
        let v: Vec<String> = prev.iter().map(|(k, n)| format!("{k}={n}")).collect();
        runner::diag_log(&format!("JIT3 BAIL-SRC {:?}: {}", method.method_name(), v.join(" ")));
    }
    // WHY each local failed register promotion — the other half of the boxing cost (`local` is the
    // single biggest op bucket, and an unpromoted local is a frame `i64.load` per read).
    let promo: Vec<String> = promo_why.iter().take(10).map(|(k, n)| format!("{k}={n}")).collect();
    runner::diag_log(&format!("JIT3 PROMO-WHY {:?}: {}", method.method_name(), promo.join(" ")));
}

fn try_compile(method: Method<'_>) -> Result<CompiledMethod, &'static str> {
    let verified = method.try_verified_info().ok_or("unverified")?; // native/unverified
    let num_locals = method.body().ok_or("no_body")?.num_locals as usize; // bytecode only
    if method.is_variadic() {
        return Err("variadic"); // varargs frame shape is not the matched-call shape
    }
    if num_locals > emit::MAX_LOCALS {
        return Err("num_locals>256");
    }
    if !verified.exceptions.is_empty() {
        // The method has a try/catch handler table. The JIT translates only the linear op
        // stream and propagates a throw straight out (via the error sentinel); it does NOT
        // model exception ranges, so a throw inside a `try` would escape its `catch` instead
        // of being handled. Decline — the interpreter runs it with correct unwinding.
        return Err("exceptions");
    }
    // Seed each local slot's declared class for the operand-type tracker: `this` (slot 0)
    // and untyped slots stay `None`; param `i` lives in slot `1 + i`.
    let sig = method.resolved_param_config();
    let mut local_types = vec![None; num_locals];
    for (i, param) in sig.iter().enumerate() {
        if let Some(slot) = local_types.get_mut(1 + i) {
            *slot = param.param_type;
        }
    }
    // A declared-`Number` param is a guaranteed canonical inline `Number` UNLESS the method is
    // `unchecked` (a missing param is then `undefined`, not a `Number`) — see `typed::param_repr`.
    let canonical_params = !method.is_unchecked();
    let (blocks, promoted, undefined_init) = translate::translate(
        &verified.parsed_code,
        &verified.null_safe_getslots,
        &verified.number_slots,
        &verified.int_slots,
        &local_types,
        canonical_params,
        sig.len(),
    )
    .ok_or_else(translate::last_decline_reason)?;
    jitop_histogram(&method, &blocks, &promoted);
    let bytes =
        emit::compile(&blocks, num_locals, &promoted, &undefined_init, &method.method_name())
            .ok_or("emit_failed")?;
    // Cache the metadata the per-call entry needs (avoids re-deriving it every call).
    let needs_scopes = verified
        .parsed_code
        .iter()
        .any(|op| matches!(op, ruffle_core::avm2::Op::GetScopeObject { .. }));
    use ruffle_core::avm2::Op;
    let makes_calls = verified.parsed_code.iter().any(|op| {
        matches!(
            op,
            Op::Call { .. }
                | Op::CallMethod { .. }
                | Op::CallNative { .. }
                | Op::CallProperty { .. }
                | Op::CallPropVoid { .. }
                | Op::CallSuper { .. }
                | Op::Construct { .. }
                | Op::ConstructProp { .. }
                | Op::ConstructSlot { .. }
                | Op::ConstructSuper { .. }
        )
    });
    // Reads of the scope base (`getscopeobject` / `newfunction`→`create_scopechain`) that the
    // §8 in-WASM env's baked `scope_base = 0` would answer wrong — see `scope_base_used`.
    let scope_base_used = verified.parsed_code.iter().any(|op| {
        matches!(
            op,
            ruffle_core::avm2::Op::GetScopeObject { .. } | ruffle_core::avm2::Op::NewFunction { .. }
        )
    });
    let has_typed_params = sig.iter().any(|p| p.param_type.is_some());
    Ok(CompiledMethod {
        bytes: Rc::from(bytes.into_boxed_slice()),
        handle: runner::new_handle(),
        num_locals: num_locals as u32,
        needs_scopes,
        has_typed_params,
        nparams: sig.len() as u32,
        scope_base_used,
        makes_calls,
        n_ops: blocks.iter().map(|b| b.ops.len()).sum::<usize>() as u32,
        cannot_throw: !blocks
            .iter()
            .flat_map(|b| &b.ops)
            .any(|op| matches!(op, emit::JitOp::BailIfError | emit::JitOp::BailIfPerr)),
    })
}

thread_local! {
    /// Frequency-weighted decline histogram: how many CALLS fell to the interpreter, by
    /// reason. A hot method that declines every call dominates its reason's count → this
    /// tells us which blocker to remove next (unlike the first-sighting op log). Dumped
    /// every `DECLINE_DUMP_EVERY` declines. Keyed by the reason string's ADDRESS (a `usize`),
    /// NOT its contents — every reason is a stable `&'static str` (a literal or an interned
    /// op name), so a plain integer hash replaces a per-call string hash. That matters because
    /// this runs on EVERY call to a native method (Math/getters/builtins — extremely hot).
    static DECLINE_COUNTS: RefCell<FnvHashMap<usize, (&'static str, u64)>> =
        RefCell::new(FnvHashMap::default());
    static DECLINE_TOTAL: Cell<u64> = const { Cell::new(0) };
}

/// One dump per this many total declines (keeps the log readable under a hot loop).
const DECLINE_DUMP_EVERY: u64 = 2_000_000;

/// Records one interpreter-fallback, keyed by `reason`; periodically dumps the histogram
/// (native: stderr under `RUFFLE_JIT3_TRACE`; web: console) so the biggest blocker of the
/// hot path is visible. Frequency-weighted — every declined CALL counts, not first sighting.
fn record_decline_reason(reason: &'static str) {
    // Off by default: this runs on the HOT decline path (a hot method that declines the JIT
    // executes via the interpreter EVERY call and lands here), and the per-call map entry showed
    // up as ~1.7% in a Starling profile. Flip to `true` to bring the decline histogram back.
    const DECLINE_PROFILE: bool = false;
    if DECLINE_PROFILE {
        let ptr = reason.as_ptr() as usize;
        DECLINE_COUNTS.with(|c| {
            let mut m = c.borrow_mut();
            let e = m.entry(ptr).or_insert((reason, 0));
            e.1 += 1;
        });
        let total = DECLINE_TOTAL.with(|t| {
            let n = t.get() + 1;
            t.set(n);
            n
        });
        if total % DECLINE_DUMP_EVERY == 0 {
            let mut items: Vec<(&'static str, u64)> =
                DECLINE_COUNTS.with(|c| c.borrow().values().copied().collect());
            items.sort_by(|a, b| b.1.cmp(&a.1));
            let mut line = format!("JIT3 DECLINE PROFILE ({total} calls to interp):");
            for (reason, n) in items {
                line.push_str(&format!(" {reason}={n}({}%)", n * 100 / total));
            }
            runner::log_decline(&line);
        }
    }
}

