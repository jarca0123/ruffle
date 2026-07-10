//! Greenfield AVM2 → WebAssembly JIT (`ruffle_avm2_jit2`).
//!
//! This is the clean rebuild of the AVM2 JIT that follows AVM2_JIT_DESIGN.md: the
//! IR is typed, unboxed hot values live in WASM locals (register-allocated by the
//! host engine), and the memory frame exists only for the boundaries a later step
//! needs. The older `ruffle_avm2_jit` crate is kept beside it purely as the
//! reference the brief's LESSONs came from — nothing here depends on it.
//!
//! ## Step 1 (this checkpoint)
//! Compile a small **single-basic-block numeric** subset — `int`/`Number`
//! arithmetic, locals, coercions, return — to a WASM module with typed locals,
//! run it through the host engine, and check it against the interpreter with a
//! differential (verify) mode. Anything outside the subset declines to the
//! interpreter (never a miscompile). Branches, comparisons, calls, and property
//! access are subsequent steps.
//!
//! The pipeline is [`frontend::analyze`] → [`emit::emit`] → [`runner`]. See
//! [`Jit2::try_run`] for how a method is offered, compiled once, cached, and run.

mod emit;
mod frontend;
mod ir;
mod runtime;
mod value_bits;

// The runner turns emitted module bytes into an executable: natively via wasmtime
// (cranelift), on the web via the browser's own `WebAssembly` engine. Both expose
// the same `Runner`/`Program`/`run`/`run_dm` surface.
#[cfg(not(target_arch = "wasm32"))]
#[path = "runner.rs"]
mod runner;
#[cfg(target_arch = "wasm32")]
#[path = "runner_web.rs"]
mod runner;

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use fnv::FnvHashMap;

use ruffle_core::avm2::{Activation, Error, JitBackend, Method, Op, TObject, Value};

use crate::ir::{ReturnCoerce, Ty};

/// A JIT `run` argument, target-agnostic. The native runner converts each to a
/// `wasmtime::Val`; the web runner converts to a JS value (`i64` as `BigInt`).
#[derive(Copy, Clone)]
pub(crate) enum Arg {
    I32(i32),
    F64(f64),
    I64(i64),
}

/// Which runtime helpers a compiled module imports (the union across its amalgam
/// units), so the runner provides exactly those, in canonical order.
#[derive(Copy, Clone, Default)]
pub(crate) struct Helpers {
    pub get_slot: bool,
    pub set_slot: bool,
    pub to_int32: bool,
    pub calls: bool,
}

/// Call count at which a method is first offered to the compiler.
///
/// Step 1 uses `0` — compile on first sight — so the differential harness (which
/// runs a method once) exercises the JIT. A production tiering policy interprets
/// a method several times first; that only changes this constant.
const HOT_THRESHOLD: u32 = 0;

/// A compiled method: the validated module plus the per-call boundary contract
/// (which locals to read as arguments, and how to unbox each). Built once and
/// cached, keyed by method identity.
struct Compiled {
    program: runner::Program,
    /// AS3 local indices to read from the frame and pass as WASM args, in the
    /// module's parameter order.
    param_locals: Vec<u32>,
    /// The type of each `param_locals` entry, for unboxing.
    param_tys: Vec<Ty>,
    /// Whether the method accesses domainMemory (needs the `dm_len` arg + the
    /// memory copy-in/out around the call).
    uses_dm: bool,
    /// Which runtime helpers the module imports (provided at instantiation);
    /// `set_slot` additionally needs the GC mutation via `with_mc`.
    helpers: Helpers,
    /// If the module baked in direct calls, the receiver vtable pointer they were
    /// resolved against. `try_run` declines when the actual `this` has a different
    /// class (a polymorphic re-entry), so a direct call never reaches the wrong
    /// callee. `None` when the module has no direct calls (no guard needed).
    expected_vtable: Option<*const ()>,
}

type CacheEntry = Option<Rc<Compiled>>;

/// The AVM2 → WASM JIT backend. Installed on a `Player` via
/// `Player::set_avm2_jit_backend`.
pub struct Jit2 {
    runner: runner::Runner,
    /// Compiled-method cache keyed by `Method::as_ptr()`; `Some(None)` = a method
    /// examined and declined (don't recompile), `Some(Some(_))` = compiled.
    cache: RefCell<FnvHashMap<*const (), CacheEntry>>,
    /// Per-method call counts for the tiering trigger.
    counts: RefCell<FnvHashMap<*const (), u32>>,
    /// Re-entrancy guard: set while the verify mode re-runs a method through the
    /// interpreter, so the nested `run_actions` declines instead of recursing.
    in_verify: Cell<bool>,
    /// Whether to run every JIT'd method through the interpreter as well and
    /// compare (differential testing). Off in production.
    verify: bool,

    // --- diagnostics (read by tests) ---
    hits: Cell<u64>,
    compiled: Cell<u64>,
    mismatches: Cell<u64>,
    /// Methods for which a direct self-recursive `call` was emitted (step 4's
    /// direct JIT→JIT call, in place of the trampoline).
    direct_calls: Cell<u64>,
}

impl Default for Jit2 {
    fn default() -> Self {
        Self::new()
    }
}

impl Jit2 {
    pub fn new() -> Self {
        Jit2 {
            runner: runner::Runner::new(),
            cache: RefCell::new(FnvHashMap::default()),
            counts: RefCell::new(FnvHashMap::default()),
            in_verify: Cell::new(false),
            verify: false,
            hits: Cell::new(0),
            compiled: Cell::new(0),
            mismatches: Cell::new(0),
            direct_calls: Cell::new(0),
        }
    }

    /// Enable differential (verify) mode: after running a method through the JIT,
    /// also run it through the interpreter and compare the results bit-for-bit,
    /// counting divergences in [`Jit2::mismatches`].
    pub fn with_verify(mut self, verify: bool) -> Self {
        self.verify = verify;
        self
    }

    /// A production backend (verify off) as `Rc<dyn JitBackend>` for
    /// `Player::set_avm2_jit_backend`. The JIT compiles the subset it supports and
    /// declines everything else (including under a strict CSP) to the interpreter,
    /// so installing it is safe.
    pub fn shared() -> Rc<dyn JitBackend> {
        Rc::new(Jit2::new())
    }

    /// Like [`Jit2::shared`] but with differential verify on — a diagnostic: it
    /// re-runs every JIT'd method through the interpreter and counts divergences.
    /// Slow, and (like the old crate's `shared_verified`) unsound for
    /// event/timer-driven content, since call-bearing methods re-run and native
    /// side effects multiply. Use only for compute-heavy validation.
    pub fn shared_verified() -> Rc<dyn JitBackend> {
        Rc::new(Jit2::new().with_verify(true))
    }

    /// Number of method invocations the JIT executed.
    pub fn hits(&self) -> u64 {
        self.hits.get()
    }

    /// Number of distinct methods the JIT compiled.
    pub fn compiled(&self) -> u64 {
        self.compiled.get()
    }

    /// Number of verify-mode invocations where the JIT and interpreter results
    /// differed. Must stay `0`.
    pub fn mismatches(&self) -> u64 {
        self.mismatches.get()
    }

    /// Number of methods compiled with a direct self-recursive `call` (the
    /// direct JIT→JIT call path, not the trampoline).
    pub fn direct_calls(&self) -> u64 {
        self.direct_calls.get()
    }

    /// Bump and return this method's call count.
    fn bump(&self, key: *const ()) -> u32 {
        let mut counts = self.counts.borrow_mut();
        let c = counts.entry(key).or_insert(0);
        *c += 1;
        *c
    }
}

/// Maximum methods co-compiled into one amalgam module (the top method plus its
/// directly-callable call-tree). Bounds compile-time work + code duplication;
/// calls beyond it fall back to the trampoline.
const MAX_AMALGAM_UNITS: usize = 8;

/// A resolved call-tree amalgam: the top method (index 0) plus the methods it (and
/// they) directly call on `this`, all destined for one WASM module.
struct Amalgam<'gc> {
    /// The methods, ordinal order; `[0]` is the top (exported) method. Kept alive
    /// so `compile` can borrow each one's verified ops.
    methods: Vec<Method<'gc>>,
    /// Each method's analysis (owned), aligned with `methods`.
    analyses: Vec<frontend::Analysis>,
    /// Each unit's direct-call targets (disp_id → callee ordinal + signature).
    direct_targets: Vec<Vec<emit::DirectTarget>>,
    /// Whether each unit's `this` (local 0) is never reassigned.
    this_immutable: Vec<bool>,
    /// The receiver vtable pointer the direct calls were resolved against — the
    /// class guard `try_run` checks before running a module with direct calls.
    vtable_ptr: Option<*const ()>,
    /// The module-shared helper imports (union across the units).
    helpers: Helpers,
}

/// Whether a resolved callee can be a *direct-call target*. It must be JIT-analyzable
/// (already ensured by the caller) and not use domainMemory (the trailing `dm_len`
/// param isn't threaded across a direct call yet). Its parameters may be any subset
/// of `{this} ∪ {declared params}` — the emitter maps each to the receiver or a
/// call arg by local index, so an unused `this` (stripped scope prologue) is fine.
fn is_direct_callable(ana: &frontend::Analysis) -> bool {
    !ana.uses_dm
}

impl Jit2 {
    /// Analyze + emit + compile a method (and its directly-callable call-tree), or
    /// `None` to decline it. Called at most once per method; the result (including
    /// a decline) is cached.
    fn compile<'gc>(
        &self,
        activation: &mut Activation<'_, 'gc>,
        method: Method<'gc>,
    ) -> CacheEntry {
        let am = self.build_amalgam(activation, method)?;

        // Borrow each unit's ops (the `methods` vec is now final and immutable).
        let ops_list: Vec<&[Op<'gc>]> = am
            .methods
            .iter()
            .map(|m| m.try_verified_info().expect("unit was verified").parsed_code.as_slice())
            .collect();
        let units: Vec<emit::Unit> = (0..am.methods.len())
            .map(|i| emit::Unit {
                ana: &am.analyses[i],
                ops: ops_list[i],
                direct_targets: am.direct_targets[i].clone(),
                this_immutable: am.this_immutable[i],
            })
            .collect();

        let (bytes, emitted_direct) = emit::emit(&units);
        let program = self.runner.compile(&bytes)?;

        self.compiled.set(self.compiled.get() + 1);
        if emitted_direct {
            self.direct_calls.set(self.direct_calls.get() + 1);
        }

        let top = &am.analyses[0];
        let param_tys = top
            .param_locals
            .iter()
            .map(|&i| top.local_ty[i as usize])
            .collect();
        Some(Rc::new(Compiled {
            program,
            param_locals: top.param_locals.clone(),
            param_tys,
            uses_dm: top.uses_dm,
            helpers: am.helpers,
            // A module with baked-in direct calls resolved disp_ids against a
            // specific receiver class; guard on it so a polymorphic re-entry with a
            // different class declines instead of dispatching to the wrong callee.
            expected_vtable: emitted_direct.then_some(am.vtable_ptr).flatten(),
        }))
    }

    /// Analyze one method into the JIT subset, or `None` to decline. Computes its
    /// entry local types (`this` = Boxed, declared params typed) and return
    /// coercion from the resolved signature. Does not retain the ops borrow.
    fn analyze_method<'gc>(
        &self,
        activation: &mut Activation<'_, 'gc>,
        method: Method<'gc>,
    ) -> Option<frontend::Analysis> {
        let num_locals = method.body()?.num_locals;

        // Resolve parameter/return types up front (needs `&mut activation`), so we
        // don't hold the verified-info borrow across it.
        method.resolve_info(activation).ok()?;
        let params = method.resolved_param_config();
        let cd = activation.avm2().class_defs();

        let mut entry_local_tys = vec![None; num_locals as usize];
        // Local 0 is the receiver (`this`) — an object value. Type it Boxed and
        // pass its bits, so a verifier-resolved `GetSlot` on it works and it can be
        // returned; the scope prologue (`getlocal0; pushscope`) just drops it.
        if num_locals >= 1 {
            entry_local_tys[0] = Some(Ty::Boxed);
        }
        for (i, param) in params.iter().enumerate() {
            let slot = i + 1; // local 0 is `this`
            if slot < num_locals as usize {
                entry_local_tys[slot] = match param.param_type {
                    Some(c) if c == cd.int => Some(Ty::Int),
                    Some(c) if c == cd.number => Some(Ty::Num),
                    Some(c) if c == cd.boolean => Some(Ty::Bool),
                    _ => None,
                };
            }
        }

        let ret_coerce = match method.resolved_return_type() {
            None => Some(ReturnCoerce::Any),
            Some(c) if c == cd.int => Some(ReturnCoerce::Int),
            Some(c) if c == cd.number => Some(ReturnCoerce::Number),
            Some(c) if c == cd.boolean => Some(ReturnCoerce::Boolean),
            // `void` / object / user classes: a `ReturnValue` declines, a
            // `ReturnVoid` is still fine.
            Some(_) => None,
        };

        let verified = method.try_verified_info()?;
        let ops = verified.parsed_code.as_slice();
        frontend::analyze(ops, num_locals, &entry_local_tys, ret_coerce)
    }

    /// Build the call-tree amalgam rooted at `top`: discover the methods it (and
    /// they) directly call on `this` — resolved against the live receiver vtable —
    /// that are themselves JIT-compilable direct-call targets, then compute each
    /// unit's direct targets, `this`-immutability, and the module-shared helpers.
    /// The single top method with no callees is the common (degenerate) case.
    fn build_amalgam<'gc>(
        &self,
        activation: &mut Activation<'_, 'gc>,
        top: Method<'gc>,
    ) -> Option<Amalgam<'gc>> {
        let top_ana = self.analyze_method(activation, top)?;

        // The shared receiver: every direct call in the tree dispatches on `this`,
        // so one vtable resolves them all. Only an object receiver has one.
        let this_val = activation.local_register(0);
        let vtable = this_val.as_object().map(|o| o.vtable());
        let vtable_ptr = vtable.map(|v| v.as_ptr());

        let mut methods = vec![top];
        let mut analyses = vec![top_ana];

        // BFS: pull in each directly-callable this-receiver callee, up to the limit.
        if let Some(vt) = vtable {
            let mut i = 0;
            while i < methods.len() {
                // Collect this unit's resolvable callees (short-lived ops borrow).
                let callees: Vec<Method<'gc>> = {
                    let ops = methods[i].try_verified_info()?.parsed_code.as_slice();
                    ops.iter()
                        .filter_map(|op| match op {
                            Op::CallMethod { index, .. } => vt.get_method(*index),
                            _ => None,
                        })
                        .collect()
                };
                for callee in callees {
                    if methods.len() >= MAX_AMALGAM_UNITS {
                        break;
                    }
                    if methods.iter().any(|m| *m == callee) {
                        continue;
                    }
                    if let Some(cana) = self.analyze_method(activation, callee) {
                        if is_direct_callable(&cana) {
                            methods.push(callee);
                            analyses.push(cana);
                        }
                    }
                }
                i += 1;
            }
        }

        // Per-unit: `this`-immutability + the disp_id → co-compiled-callee targets.
        let mut this_immutable = Vec::with_capacity(methods.len());
        let mut direct_targets = Vec::with_capacity(methods.len());
        for i in 0..methods.len() {
            let ops = methods[i].try_verified_info()?.parsed_code.as_slice();
            // A reassigned `this` breaks the "receiver == resolved vtable" invariant.
            let imm = !ops
                .iter()
                .any(|op| matches!(op, Op::SetLocal { index: 0 } | Op::StoreLocal { index: 0 }));
            let mut targets: Vec<emit::DirectTarget> = Vec::new();
            if imm {
                if let Some(vt) = vtable {
                    for op in ops {
                        let Op::CallMethod { index, .. } = op else {
                            continue;
                        };
                        if targets.iter().any(|t| t.disp_id == *index) {
                            continue;
                        }
                        let Some(callee) = vt.get_method(*index) else {
                            continue;
                        };
                        if let Some(ord) = methods.iter().position(|m| *m == callee) {
                            let a = &analyses[ord];
                            let params = a
                                .param_locals
                                .iter()
                                .map(|&l| (l, a.local_ty[l as usize]))
                                .collect();
                            targets.push(emit::DirectTarget {
                                disp_id: *index,
                                ordinal: ord,
                                params,
                            });
                        }
                    }
                }
            }
            this_immutable.push(imm);
            direct_targets.push(targets);
        }

        // Module-shared helper imports = the union over all units.
        let any = |f: fn(&frontend::Analysis) -> bool| analyses.iter().any(|a| f(a));
        let helpers = Helpers {
            get_slot: any(|a| a.uses_get_slot),
            set_slot: any(|a| a.uses_set_slot),
            to_int32: any(|a| a.uses_to_int32),
            calls: any(|a| a.uses_call),
        };

        Some(Amalgam {
            methods,
            analyses,
            direct_targets,
            this_immutable,
            vtable_ptr,
            helpers,
        })
    }

    /// Read the method's argument locals from the frame and unbox each to a WASM
    /// value. `None` if a value isn't the expected primitive type (declines
    /// safely — the interpreter re-reads the frame).
    fn build_args<'gc>(
        activation: &mut Activation<'_, 'gc>,
        compiled: &Compiled,
    ) -> Option<Vec<Arg>> {
        let mut args = Vec::with_capacity(compiled.param_locals.len());
        for (k, &local) in compiled.param_locals.iter().enumerate() {
            let v = activation.local_register(local);
            let val = match compiled.param_tys[k] {
                // These params are declared `int`/`Number`/`Boolean`, so the frame
                // already holds the coerced primitive; the coercion is exact and
                // side-effect-free.
                Ty::Int => Arg::I32(v.coerce_to_i32(activation).ok()?),
                Ty::Num => Arg::F64(v.coerce_to_number(activation).ok()?),
                Ty::Bool => Arg::I32(v.coerce_to_boolean() as i32),
                // `this` and other object-typed locals pass through as their raw
                // NaN-box bits (a live, rooted pointer during the synchronous call).
                Ty::Boxed => Arg::I64(value_bits::to_bits(v) as i64),
                Ty::Opaque | Ty::NumRaw => return None, // never a parameter type
            };
            args.push(val);
        }
        Some(args)
    }
}

impl JitBackend for Jit2 {
    fn try_run<'gc>(
        &self,
        activation: &mut Activation<'_, 'gc>,
        method: Method<'gc>,
    ) -> Option<Result<Value<'gc>, Error<'gc>>> {
        // Re-entrant call from verify mode: decline so the interpreter runs.
        if self.in_verify.get() {
            return None;
        }

        let key = method.as_ptr();
        if self.bump(key) <= HOT_THRESHOLD {
            return None;
        }

        {
            // Clone the entry out of the cache so we don't hold the borrow across
            // the run (compiling / the verify re-run can re-enter the map).
            let entry = {
                if let Some(e) = self.cache.borrow().get(&key) {
                    e.clone()
                } else {
                    let compiled = self.compile(activation, method);
                    self.cache.borrow_mut().insert(key, compiled.clone());
                    compiled
                }
            };
            let compiled = entry?; // None = declined → interpreter

            // Class guard: a module with baked-in direct calls resolved them
            // against one receiver class. If `this` now has a different class
            // (polymorphic re-entry), decline so the interpreter dispatches
            // correctly instead of us calling the wrong callee.
            if let Some(expected) = compiled.expected_vtable {
                let this = activation.local_register(0);
                match this.as_object() {
                    Some(o) if o.vtable().as_ptr() == expected => {}
                    _ => return None,
                }
            }

            let mut args = Self::build_args(activation, &compiled)?;

            // domainMemory: snapshot the bytes and pass the length (run against a
            // WASM-memory copy).
            let mut dm = if compiled.uses_dm {
                let snap = activation.domain_memory().storage().dm_snapshot();
                args.push(Arg::I32(snap.len() as i32));
                Some(snap)
            } else {
                None
            };

            // Slot writes need the GC mutation available to the helper; `gc()`
            // returns a detached `&'gc Mutation`, so it doesn't tie up `activation`.
            let mc = activation.gc();
            let mut run_once = || -> Option<u64> {
                match &mut dm {
                    Some(dm) => self.runner.run_dm(&compiled.program, &args, dm, compiled.helpers),
                    None => self.runner.run(&compiled.program, &args, compiled.helpers),
                }
            };
            // Establish the runtime contexts the helpers need around the run: the
            // live `Activation` for calls, the GC mutation for slot writes.
            let bits = if compiled.helpers.calls {
                runtime::with_activation(activation, || {
                    if compiled.helpers.set_slot {
                        runtime::with_mc(mc, run_once)
                    } else {
                        run_once()
                    }
                })
            } else if compiled.helpers.set_slot {
                runtime::with_mc(mc, run_once)
            } else {
                run_once()
            }?;

            // Persist domainMemory writes — but only in non-verify mode; verify
            // keeps the interpreter's run as the live one (no double-apply).
            if compiled.uses_dm && !self.verify {
                if let Some(dm) = &dm {
                    let storage = activation.domain_memory();
                    let mut s = storage.storage_mut();
                    for (i, &b) in dm.iter().enumerate() {
                        s.dm_set(i, b);
                    }
                }
            }
            self.hits.set(self.hits.get() + 1);

            // A call may have thrown: turn the stashed error back into `Err`. In
            // verify mode, confirm the interpreter also throws (else a mismatch).
            if let Some(err) = runtime::take_pending_error::<'gc>() {
                if self.verify {
                    self.in_verify.set(true);
                    let interp = activation.run_actions(method);
                    self.in_verify.set(false);
                    if interp.is_ok() {
                        self.mismatches.set(self.mismatches.get() + 1);
                    }
                    return Some(interp);
                }
                return Some(Err(err));
            }

            // SAFETY: a returned value is an Integer/Number/Bool/Undefined or a
            // rooted object bit-pattern — no dangling pointer — and we hold the lock.
            let jit_val = unsafe { value_bits::from_bits::<'gc>(bits) };

            if self.verify {
                self.in_verify.set(true);
                let interp = activation.run_actions(method);
                self.in_verify.set(false);
                let matches = matches!(&interp, Ok(v) if value_bits::to_bits(*v) == bits);
                if !matches {
                    self.mismatches.set(self.mismatches.get() + 1);
                }
                // Keep the interpreter's result/side effects as the live outcome.
                return Some(interp);
            }

            Some(Ok(jit_val))
        }
    }
}

/// Codegen tests that drive the `analyze → emit → runner` pipeline directly on
/// synthetic `Op` lists (no arena / `Player`), checking the emitted WASM produces
/// the exact `Value` bits the interpreter would. The end-to-end agreement with
/// the *real* interpreter is proven separately in `tests/diff.rs`.
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use ruffle_core::avm2::Op;
    use value_bits::{BOOL_BOX_HI, INT_BOX_HI, UNDEFINED_BITS};

    /// Run an op list through the whole pipeline; `None` = declined.
    fn run(
        ops: &[Op<'static>],
        num_locals: u32,
        entry: &[Option<Ty>],
        ret: Option<ReturnCoerce>,
        args: &[Arg],
    ) -> Option<u64> {
        let ana = frontend::analyze(ops, num_locals, entry, ret)?;
        let bytes = emit::emit_single(ops, &ana);
        let runner = runner::Runner::new();
        let program = runner.compile(&bytes)?;
        runner.run(
            &program,
            args,
            Helpers {
                get_slot: ana.uses_get_slot,
                set_slot: ana.uses_set_slot,
                to_int32: ana.uses_to_int32,
                calls: ana.uses_call,
            },
        )
    }

    fn int_bits(i: i32) -> u64 {
        INT_BOX_HI | (i as u32 as u64)
    }
    fn num_bits(n: f64) -> u64 {
        if n.is_nan() {
            value_bits::CANON_NAN
        } else {
            n.to_bits()
        }
    }

    #[test]
    fn addi_two_constants() {
        let ops = vec![
            Op::PushInt { value: 3 },
            Op::PushInt { value: 4 },
            Op::AddI,
            Op::ReturnValue { return_type: None },
        ];
        assert_eq!(
            run(&ops, 1, &[None], Some(ReturnCoerce::Any), &[]),
            Some(int_bits(7))
        );
    }

    #[test]
    fn number_add() {
        let ops = vec![
            Op::PushDouble { value: 1.5 },
            Op::PushDouble { value: 2.0 },
            Op::Add {
                inputs_integral: false,
            },
            Op::ReturnValue { return_type: None },
        ];
        assert_eq!(
            run(&ops, 1, &[None], Some(ReturnCoerce::Any), &[]),
            Some(num_bits(3.5))
        );
    }

    #[test]
    fn multiply_ints_in_range_is_integer() {
        let ops = vec![
            Op::PushInt { value: 6 },
            Op::PushInt { value: 7 },
            Op::Multiply,
            Op::ReturnValue { return_type: None },
        ];
        assert_eq!(
            run(&ops, 1, &[None], Some(ReturnCoerce::Any), &[]),
            Some(int_bits(42))
        );
    }

    #[test]
    fn multiply_ints_overflow_becomes_number() {
        // 50000 * 50000 = 2.5e9 > i32::MAX ⇒ the interpreter yields Number.
        let ops = vec![
            Op::PushInt { value: 50_000 },
            Op::PushInt { value: 50_000 },
            Op::Multiply,
            Op::ReturnValue { return_type: None },
        ];
        assert_eq!(
            run(&ops, 1, &[None], Some(ReturnCoerce::Any), &[]),
            Some(num_bits(2_500_000_000.0))
        );
    }

    #[test]
    fn subtract_ints_overflow_becomes_number() {
        // i32::MIN - 1 underflows ⇒ Number.
        let ops = vec![
            Op::PushInt { value: i32::MIN },
            Op::PushInt { value: 1 },
            Op::Subtract {
                inputs_integral: true,
            },
            Op::ReturnValue { return_type: None },
        ];
        assert_eq!(
            run(&ops, 1, &[None], Some(ReturnCoerce::Any), &[]),
            Some(num_bits(i32::MIN as f64 - 1.0))
        );
    }

    #[test]
    fn int_param_increment() {
        let ops = vec![
            Op::GetLocal { index: 1 },
            Op::PushInt { value: 1 },
            Op::AddI,
            Op::ReturnValue { return_type: None },
        ];
        assert_eq!(
            run(
                &ops,
                2,
                &[None, Some(Ty::Int)],
                Some(ReturnCoerce::Any),
                &[Arg::I32(41)],
            ),
            Some(int_bits(42))
        );
    }

    #[test]
    fn number_param_add() {
        let ops = vec![
            Op::GetLocal { index: 1 },
            Op::PushDouble { value: 0.5 },
            Op::Add {
                inputs_integral: false,
            },
            Op::ReturnValue { return_type: None },
        ];
        assert_eq!(
            run(
                &ops,
                2,
                &[None, Some(Ty::Num)],
                Some(ReturnCoerce::Any),
                &[Arg::F64(1.0f64)],
            ),
            Some(num_bits(1.5))
        );
    }

    #[test]
    fn divide_is_number() {
        let ops = vec![
            Op::PushInt { value: 7 },
            Op::PushInt { value: 2 },
            Op::Divide,
            Op::ReturnValue { return_type: None },
        ];
        assert_eq!(
            run(&ops, 1, &[None], Some(ReturnCoerce::Any), &[]),
            Some(num_bits(3.5))
        );
    }

    #[test]
    fn negate_is_number() {
        let ops = vec![
            Op::PushInt { value: 5 },
            Op::Negate,
            Op::ReturnValue { return_type: None },
        ];
        assert_eq!(
            run(&ops, 1, &[None], Some(ReturnCoerce::Any), &[]),
            Some(num_bits(-5.0))
        );
    }

    #[test]
    fn locals_roundtrip_and_dup() {
        // local2 = param1 * param1 (via dup), return local2.
        let ops = vec![
            Op::GetLocal { index: 1 },
            Op::Dup,
            Op::MultiplyI,
            Op::SetLocal { index: 2 },
            Op::GetLocal { index: 2 },
            Op::ReturnValue { return_type: None },
        ];
        assert_eq!(
            run(
                &ops,
                3,
                &[None, Some(Ty::Int), None],
                Some(ReturnCoerce::Any),
                &[Arg::I32(9)],
            ),
            Some(int_bits(81))
        );
    }

    #[test]
    fn return_void_is_undefined() {
        let ops = vec![Op::ReturnVoid { return_type: None }];
        assert_eq!(run(&ops, 1, &[None], None, &[]), Some(UNDEFINED_BITS));
    }

    #[test]
    fn return_bool() {
        let ops = vec![Op::PushTrue, Op::ReturnValue { return_type: None }];
        assert_eq!(
            run(&ops, 1, &[None], Some(ReturnCoerce::Boolean), &[]),
            Some(BOOL_BOX_HI | 1)
        );
    }

    #[test]
    fn return_coerce_number_from_int() {
        // Declared return `Number`, operand is int ⇒ box as Number(5.0).
        let ops = vec![
            Op::PushInt { value: 5 },
            Op::ReturnValue { return_type: None },
        ];
        assert_eq!(
            run(&ops, 1, &[None], Some(ReturnCoerce::Number), &[]),
            Some(num_bits(5.0))
        );
    }

    /// domainMemory store-then-load: `si32(val=42, addr=0); li32(addr=0)` → 42,
    /// and the byte buffer holds 42 little-endian at offset 0.
    #[test]
    fn dm_store_then_load() {
        // si32 pops addr then val: push val, push addr. li32 pops addr.
        let ops = vec![
            Op::PushInt { value: 42 },
            Op::PushInt { value: 0 },
            Op::Si32,
            Op::PushInt { value: 0 },
            Op::Li32,
            Op::ReturnValue { return_type: None },
        ];
        let ana = frontend::analyze(&ops, 1, &[None], Some(ReturnCoerce::Any)).unwrap();
        assert!(ana.uses_dm);
        let bytes = emit::emit_single(&ops, &ana);
        let runner = runner::Runner::new();
        let program = runner.compile(&bytes).unwrap();
        let mut dm = vec![0u8; 16];
        let bits = runner
            .run_dm(&program, &[Arg::I32(dm.len() as i32)], &mut dm, Helpers::default())
            .unwrap();
        assert_eq!(bits, int_bits(42));
        assert_eq!(&dm[0..4], &42i32.to_le_bytes());
    }

    /// domainMemory OOB store is swallowed (skipped) and OOB load reads 0, per the
    /// inline fast path — no trap.
    #[test]
    fn dm_oob_is_swallowed() {
        // store at addr 1000 into an 8-byte buffer (OOB ⇒ skipped); load there ⇒ 0.
        let ops = vec![
            Op::PushInt { value: 42 },
            Op::PushInt { value: 1000 },
            Op::Si32,
            Op::PushInt { value: 1000 },
            Op::Li32,
            Op::ReturnValue { return_type: None },
        ];
        let ana = frontend::analyze(&ops, 1, &[None], Some(ReturnCoerce::Any)).unwrap();
        let bytes = emit::emit_single(&ops, &ana);
        let runner = runner::Runner::new();
        let program = runner.compile(&bytes).unwrap();
        let mut dm = vec![0u8; 8];
        let bits = runner
            .run_dm(&program, &[Arg::I32(dm.len() as i32)], &mut dm, Helpers::default())
            .unwrap();
        assert_eq!(bits, int_bits(0));
        assert_eq!(dm, vec![0u8; 8]); // untouched
    }

    /// Run a domainMemory method directly (synthetic) and return (result bits, dm).
    fn run_dm_ops(ops: &[Op<'static>], dm_len: usize) -> (u64, Vec<u8>) {
        let ana = frontend::analyze(ops, 1, &[None], Some(ReturnCoerce::Any)).unwrap();
        assert!(ana.uses_dm);
        let bytes = emit::emit_single(ops, &ana);
        let runner = runner::Runner::new();
        let program = runner.compile(&bytes).unwrap();
        let mut dm = vec![0u8; dm_len];
        let bits = runner
            .run_dm(&program, &[Arg::I32(dm.len() as i32)], &mut dm, Helpers::default())
            .unwrap();
        (bits, dm)
    }

    /// `sf64(3.14 @0); lf64(@0) + 0.0` → 3.14, and the bytes land in the buffer.
    #[test]
    fn float_dm_roundtrip() {
        let ops = vec![
            Op::PushDouble { value: 3.14 },
            Op::PushInt { value: 0 },
            Op::Sf64,
            Op::PushInt { value: 0 },
            Op::Lf64,
            Op::PushDouble { value: 0.0 },
            Op::Add {
                inputs_integral: false,
            },
            Op::ReturnValue { return_type: None },
        ];
        let (bits, dm) = run_dm_ops(&ops, 16);
        assert_eq!(bits, num_bits(3.14));
        assert_eq!(&dm[0..8], &3.14f64.to_le_bytes());
    }

    /// A colliding-NaN f64 (written via integer stores) copied `lf64 → sf64` must
    /// survive bit-exact — no NaN canonicalization (the §4/§10 LESSON).
    #[test]
    fn float_dm_lossless_copy() {
        let nan = 0xFFF8_0000_0000_0001u64; // negative quiet NaN — collides with box space
        let ops = vec![
            Op::PushInt { value: nan as u32 as i32 }, // low word
            Op::PushInt { value: 0 },
            Op::Si32,
            Op::PushInt { value: (nan >> 32) as u32 as i32 }, // high word
            Op::PushInt { value: 4 },
            Op::Si32,
            Op::PushInt { value: 0 },
            Op::Lf64, // load the colliding NaN (raw)
            Op::PushInt { value: 8 },
            Op::Sf64, // store it back elsewhere
            Op::ReturnVoid { return_type: None },
        ];
        let (_bits, dm) = run_dm_ops(&ops, 16);
        assert_eq!(&dm[0..8], &nan.to_le_bytes());
        assert_eq!(&dm[8..16], &dm[0..8]); // copy preserved the exact bits
    }

    // ---- control flow ----

    /// `max(a, b)` via a conditional (two blocks + a merge). Exercises
    /// GreaterThan → IfFalse and the dispatch loop.
    #[test]
    fn conditional_max() {
        // 0:getlocal1 1:getlocal2 2:greaterthan 3:iffalse->6
        // 4:getlocal1 5:returnvalue   6:getlocal2 7:returnvalue
        let ops = vec![
            Op::GetLocal { index: 1 },
            Op::GetLocal { index: 2 },
            Op::GreaterThan,
            Op::IfFalse { offset: 6 },
            Op::GetLocal { index: 1 },
            Op::ReturnValue { return_type: None },
            Op::GetLocal { index: 2 },
            Op::ReturnValue { return_type: None },
        ];
        let entry = [None, Some(Ty::Int), Some(Ty::Int)];
        let ret = Some(ReturnCoerce::Any);
        assert_eq!(
            run(&ops, 3, &entry, ret, &[Arg::I32(7), Arg::I32(3)]),
            Some(int_bits(7))
        );
        assert_eq!(
            run(&ops, 3, &entry, ret, &[Arg::I32(2), Arg::I32(9)]),
            Some(int_bits(9))
        );
    }

    /// `a > b ? a : b` — a ternary whose result lives on the operand stack across
    /// the merge block (a NON-EMPTY boundary), so it must cross the CFG edge in a
    /// boundary WASM local (step 2), not boxed memory.
    #[test]
    fn ternary_max_operand_crosses_block() {
        // 0:getlocal1 1:getlocal2 2:greaterthan 3:iffalse->6
        // 4:getlocal1 5:jump->7   6:getlocal2   7:returnvalue (merge, value on stack)
        let ops = vec![
            Op::GetLocal { index: 1 },
            Op::GetLocal { index: 2 },
            Op::GreaterThan,
            Op::IfFalse { offset: 6 },
            Op::GetLocal { index: 1 },
            Op::Jump { offset: 7 },
            Op::GetLocal { index: 2 },
            Op::ReturnValue { return_type: None },
        ];
        let entry = [None, Some(Ty::Int), Some(Ty::Int)];
        let ret = Some(ReturnCoerce::Any);
        assert_eq!(
            run(&ops, 3, &entry, ret, &[Arg::I32(7), Arg::I32(3)]),
            Some(int_bits(7))
        );
        assert_eq!(
            run(&ops, 3, &entry, ret, &[Arg::I32(2), Arg::I32(9)]),
            Some(int_bits(9))
        );
    }

    /// `sum = 0; for (i = 0; i < n; i++) sum += i; return sum` — a real loop with
    /// a back-edge, int counter/accumulator pinned in the entry block.
    #[test]
    fn sum_loop() {
        // 0:pushint0 1:setlocal2  2:pushint0 3:setlocal3           (sum=0, i=0)
        // 4:getlocal3 5:getlocal1 6:lessthan 7:iffalse->14         (header)
        // 8:getlocal2 9:getlocal3 10:addi 11:setlocal2 12:inclocali3 13:jump->4 (body)
        // 14:getlocal2 15:returnvalue                              (exit)
        let ops = vec![
            Op::PushInt { value: 0 },
            Op::SetLocal { index: 2 },
            Op::PushInt { value: 0 },
            Op::SetLocal { index: 3 },
            Op::GetLocal { index: 3 },
            Op::GetLocal { index: 1 },
            Op::LessThan,
            Op::IfFalse { offset: 14 },
            Op::GetLocal { index: 2 },
            Op::GetLocal { index: 3 },
            Op::AddI,
            Op::SetLocal { index: 2 },
            Op::IncLocalI { index: 3 },
            Op::Jump { offset: 4 },
            Op::GetLocal { index: 2 },
            Op::ReturnValue { return_type: None },
        ];
        let entry = [None, Some(Ty::Int), None, None];
        // sum of 0..5 = 10
        assert_eq!(
            run(&ops, 4, &entry, Some(ReturnCoerce::Any), &[Arg::I32(5)]),
            Some(int_bits(10))
        );
        // sum of 0..0 = 0 (loop body never runs)
        assert_eq!(
            run(&ops, 4, &entry, Some(ReturnCoerce::Any), &[Arg::I32(0)]),
            Some(int_bits(0))
        );
    }

    /// The standard `getlocal0; pushscope` prologue is tolerated (no-op) so real
    /// methods that start with it can be JIT'd.
    #[test]
    fn scope_prologue() {
        let ops = vec![
            Op::GetLocal { index: 0 }, // this (untyped ⇒ Opaque)
            Op::PushScope,
            Op::PushInt { value: 5 },
            Op::PushInt { value: 3 },
            Op::AddI,
            Op::ReturnValue { return_type: None },
        ];
        assert_eq!(
            run(&ops, 1, &[None], Some(ReturnCoerce::Any), &[]),
            Some(int_bits(8))
        );
    }

    #[test]
    fn declines_scope_value_in_arithmetic() {
        // Feeding the Opaque `this` into arithmetic must decline, not miscompile.
        let ops = vec![
            Op::GetLocal { index: 0 },
            Op::PushInt { value: 1 },
            Op::AddI,
            Op::ReturnValue { return_type: None },
        ];
        assert_eq!(run(&ops, 1, &[None], Some(ReturnCoerce::Any), &[]), None);
    }

    // ---- ToInt32 (Number → int coercion via the helper) ----

    #[test]
    fn coerce_i_from_number() {
        let coerce = |n: f64| {
            let ops = vec![
                Op::PushDouble { value: n },
                Op::CoerceI,
                Op::ReturnValue { return_type: None },
            ];
            run(&ops, 1, &[None], Some(ReturnCoerce::Any), &[])
        };
        assert_eq!(coerce(3.9), Some(int_bits(3))); // truncates toward zero
        assert_eq!(coerce(-1.5), Some(int_bits(-1)));
        assert_eq!(coerce(4_294_967_297.0), Some(int_bits(1))); // wraps mod 2^32
        assert_eq!(coerce(f64::NAN), Some(int_bits(0)));
        assert_eq!(coerce(f64::INFINITY), Some(int_bits(0)));
    }

    #[test]
    fn int_return_of_number() {
        // Declared return `int`, operand is a Number ⇒ ToInt32.
        let ops = vec![
            Op::PushDouble { value: 2.7 },
            Op::ReturnValue { return_type: None },
        ];
        assert_eq!(
            run(&ops, 1, &[None], Some(ReturnCoerce::Int), &[]),
            Some(int_bits(2))
        );
    }

    #[test]
    fn iftrue_on_number() {
        // if (x) return 1 else return 0, branching on a Number operand.
        let ops = vec![
            Op::GetLocal { index: 1 },
            Op::IfTrue { offset: 4 },
            Op::PushInt { value: 0 },
            Op::ReturnValue { return_type: None },
            Op::PushInt { value: 1 },
            Op::ReturnValue { return_type: None },
        ];
        let go = |x: f64| {
            run(
                &ops,
                2,
                &[None, Some(Ty::Num)],
                Some(ReturnCoerce::Any),
                &[Arg::F64(x)],
            )
        };
        assert_eq!(go(5.0), Some(int_bits(1)));
        assert_eq!(go(0.0), Some(int_bits(0)));
        assert_eq!(go(f64::NAN), Some(int_bits(0))); // NaN is falsy
    }

    // ---- decline cases (out of the step-1 subset) ----

    #[test]
    fn declines_unsupported_op() {
        // StrictEquals is out of subset (strict_eq is a bit-compare) ⇒ decline.
        let ops = vec![
            Op::PushInt { value: 1 },
            Op::PushInt { value: 1 },
            Op::StrictEquals,
            Op::ReturnValue { return_type: None },
        ];
        assert_eq!(run(&ops, 1, &[None], Some(ReturnCoerce::Any), &[]), None);
    }

    #[test]
    fn declines_read_of_untyped_local() {
        // Reading `this` (local 0, untyped) has no numeric type ⇒ decline.
        let ops = vec![
            Op::GetLocal { index: 0 },
            Op::ReturnValue { return_type: None },
        ];
        assert_eq!(run(&ops, 1, &[None], Some(ReturnCoerce::Any), &[]), None);
    }
}
