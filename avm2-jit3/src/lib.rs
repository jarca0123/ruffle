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

mod context;
mod emit;
mod helpers;
mod translate;
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
        let key = method.as_ptr() as usize;

        // Get-or-compile the type-0 module for this method. Hot path takes a SHARED borrow
        // (a compiled cache hit never mutates); only a first-sighting compile/decline takes
        // the exclusive borrow. The borrow is released before we run (a nested run aliases
        // this `RefCell`).
        // NB: bind the cloned lookup to a local FIRST — a `match` on
        // `self.compiled.borrow()....cloned()` would keep the shared borrow alive across the
        // arms (temporary lifetime), so the `None` arm's `borrow_mut()` would double-borrow.
        let cached = self.compiled.borrow().get(&key).cloned();
        let compiled = match cached {
            Some(Ok(c)) => c,
            Some(Err(reason)) => {
                // Permanently declined — count WHY, weighted by call frequency.
                record_decline_reason(reason);
                return None;
            }
            None => {
                // Not seen yet. `try_enter` runs at the TOP of `exec`, before
                // `init_from_method` verifies the method — so the FIRST sighting is
                // usually unverified. That is a RETRYABLE decline: don't cache it, so a
                // later call (after the interpreter verified it) can still compile. (An
                // attempt to self-`verify()` here corrupts shared state — verify must run
                // in the method's normal entry path, not speculatively out-of-band — so
                // a 2nd-call tier-up is the only sound policy.)
                if method.try_verified_info().is_none() {
                    // A NATIVE method (no bytecode body) can never be JIT'd — cache the
                    // decline so its (often very hot: Math, getters, builtins) calls stop
                    // re-checking every time. A not-yet-verified BYTECODE method is
                    // retryable (the interpreter verifies it on this call) → don't cache.
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
                    Ok(c) => c,
                    Err(reason) => {
                        record_decline_reason(reason);
                        return None;
                    }
                }
            }
        };

        // Per-call gate. MORE args than params (a non-variadic method called with extras) is
        // rare — decline. Fewer-or-equal is handled: matched args take the fast raw-write path
        // (bit-identical to `init_from_method`); otherwise a slow path COERCES each provided
        // arg to its param type and fills missing params with their default value (mirrors
        // `resolve_parameters`), reifying a temp Activation (a coercion may throw).
        let sig = method.resolved_param_config();
        if args.len() > sig.len() {
            record_decline_reason("args_extra");
            return None;
        }
        let caller_library = method.owner_library();

        // Build the callee's frame: [this, params…] then `undefined` for the remaining
        // locals, exactly as `init_from_method` initializes a fresh frame. On the STACK —
        // a compiled method has `num_locals ≤ MAX_LOCALS` (`compile` gates it), so this
        // avoids a heap allocation on every JIT call (a measured per-call cost). Only the
        // `num_locals` slots actually used are initialized (via `MaybeUninit`) — a hot method
        // with a handful of locals no longer pays a full `MAX_LOCALS`-wide (2 KiB) init.
        let num_locals = compiled.num_locals as usize; // cached — no `body()` lookup per call
        let mut frame: [MaybeUninit<u64>; emit::MAX_LOCALS] =
            [MaybeUninit::uninit(); emit::MAX_LOCALS];
        frame[0].write(value::to_bits(receiver));

        // Fast path: args EXACTLY fill the params, each already the param type → raw write.
        // An untyped signature can never need coercion, so skip the per-arg scan entirely.
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
            // Slow path (NOT a decline — a successful coerced/defaulted JIT run). Coerce each
            // provided arg to its param type; missing params take their default (a missing
            // param with no default is #1063). A coercion may run AS3 `valueOf` and throw.
            let mut act = Activation::from_builtin(
                cx,
                bound_super,
                scope,
                Some(scope.domain()),
                caller_library,
                None,
            );
            let mut provided = args.iter();
            for (i, p) in sig.iter().enumerate() {
                let arg = match provided.next() {
                    Some(a) => a,
                    // A missing param: its default, else `undefined` for an UNCHECKED method
                    // (top-level functions / `arguments`-style callees), else #1063 — exactly
                    // as `init_from_method` resolves it.
                    None => match p.default_value {
                        Some(d) => d,
                        None if method.is_unchecked() => Value::Undefined,
                        None => return Some(Err(make_error_1063(&mut act, method, args.len()))),
                    },
                };
                let v = match p.param_type {
                    Some(c) if !arg.coerces_identically_to(c) => match arg.coerce_to_type(&mut act, c)
                    {
                        Ok(v) => v,
                        Err(e) => return Some(Err(e)),
                    },
                    _ => arg,
                };
                frame[1 + i].write(value::to_bits(v));
            }
        }
        // Remaining locals past [this, params…] up to `num_locals` are `undefined`.
        for slot in frame.iter_mut().take(num_locals).skip(1 + sig.len()) {
            slot.write(emit::UNDEFINED_BITS);
        }
        // SAFETY: slots `[0, num_locals)` are all initialized above (receiver + `sig.len()`
        // params + undefined padding; verification guarantees `1 + sig.len() ≤ num_locals`).
        let frame: &[u64] =
            unsafe { &*(&frame[..num_locals] as *const [MaybeUninit<u64>] as *const [u64]) };

        // Install the callee's reification context around the run: a slow-path helper
        // (`cr` return-coercion; later getproperty/calls) reifies a FRESH callee-owned
        // Activation from `(cx, scope, bound_super)` — never the caller's. `cx` is aliased
        // via a raw pointer only for the duration of the synchronous run (nothing else
        // touches it meanwhile).
        // The method's scope base on the shared `avm2.scope_stack` (index 0 of its local
        // scope frame for `getscopeobject`); scopes it pushes are truncated after the run.
        let scope_base = cx.avm2.scope_stack_len();
        // `caller_library` (computed above) mirrors `init_from_method`'s
        // `caller_library = method.owner_library()`, so native callees (e.g.
        // `Font.enumerateFonts`, which reads `caller_library().embedded_fonts()`)
        // resolve against this method's SWF instead of `None`.
        // Push the method onto the AVM2 call stack so it shows up in stack traces
        // (`Error.getStackTrace()`). The jit3 seam in `exec` runs BEFORE `exec`'s own
        // `push_call`, so without this a JIT'd method would be invisible to traces.
        let gc = cx.gc();
        cx.avm2.push_call(gc, method);
        let run_ctx = context::RunCtx::new(cx, scope, bound_super, scope_base, caller_library);
        let bits = context::with_run_ctx(&run_ctx, || {
            runner::run_leaf(&compiled.handle, &compiled.bytes, frame, args.len() as u32)
        });
        cx.avm2.pop_call(gc);
        // Only a scope-reading method pushes to the local scope stack, so only it needs the
        // post-run truncate (a `Gc` borrow + `Vec` truncate saved on every other call).
        if compiled.needs_scopes {
            cx.avm2.truncate_scope_stack(scope_base);
        }
        let bits = bits?;
        // A slow-path coercion may have thrown (`#1034`): propagate it. (The op that can
        // throw — `cr` — is always immediately followed by `Return`, so the run has ended.)
        if let Some(err) = context::take_error::<'gc>() {
            return Some(Err(err));
        }
        // Diagnostic: prove a real method ran through the type-0 ABI with no Activation.
        #[cfg(not(target_arch = "wasm32"))]
        if std::env::var_os("RUFFLE_JIT3_TRACE").is_some() {
            eprintln!("JIT3 RAN method@{key:#x} argc={} -> {bits:#018x}", args.len());
        }
        // SAFETY: `bits` is a `Value` produced by the JIT within this GC-quiescent frame
        // (see `value::from_bits`); any object pointer it encodes is still live.
        let result = unsafe { value::from_bits::<'gc>(bits) };
        Some(Ok(result))
    }
}

/// Translates + compiles `method` to a type-0 module, or `Err(reason)` to permanently
/// decline (the `reason` is a coarse, `'static` label for the decline profiler). Runs only
/// for **verified** methods (`try_enter` fires before `init_from_method` verifies, so a
/// first, unverified sighting declines and the interpreter verifies + runs it; a later call
/// then compiles).
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
    let blocks = translate::translate(&verified.parsed_code, &verified.null_safe_getslots, &local_types)
        .ok_or_else(translate::last_decline_reason)?;
    let bytes = emit::compile(&blocks, num_locals).ok_or("emit_failed")?;
    // Cache the metadata the per-call entry needs (avoids re-deriving it every call).
    let needs_scopes = verified
        .parsed_code
        .iter()
        .any(|op| matches!(op, ruffle_core::avm2::Op::GetScopeObject { .. }));
    let has_typed_params = sig.iter().any(|p| p.param_type.is_some());
    Ok(CompiledMethod {
        bytes: Rc::from(bytes.into_boxed_slice()),
        handle: runner::new_handle(),
        num_locals: num_locals as u32,
        needs_scopes,
        has_typed_params,
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

