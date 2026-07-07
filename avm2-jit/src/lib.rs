//! WASM-emitting JIT backend for AVM2 (`ruffle_avm2_jit`).
//!
//! Implements [`ruffle_core::avm2::JitBackend`] by compiling hot AVM2 methods to
//! a WebAssembly module at runtime and running it:
//!
//! - **Web**: emit WASM bytes, hand them to the browser's engine
//!   (`WebAssembly.Module` / `Instance` via a JS host import), and call the
//!   compiled function through a shared table. The generated module imports
//!   Ruffle's linear memory + a table of Rust "runtime" helpers.
//! - **Native** (tests, and a future desktop path): the same emitted module can
//!   be validated / executed through a WASM runtime, so JIT↔interpreter
//!   equivalence is testable without a browser. (The desktop production path
//!   would instead use a native code generator such as cranelift.)
//!
//! ## Execution model (see [`emit`])
//! The whole AVM2 method state — registers `[0..num_locals]` and the operand
//! stack above them — is one contiguous run of 8-byte NaN-boxed `Value` slots
//! (`ruffle_core`'s stack). The JIT receives the frame's base pointer
//! (`state_ptr`) and addresses slot `i` at `state_ptr + i*8`. The operand stack
//! is simulated at *compile* time, so straight-line code uses fixed offsets and
//! needs no runtime stack pointer. Anything GC-aware (property access, calls,
//! allocation, throwing coercions) is emitted as a `call` to an imported Rust
//! helper, keeping GC correctness in Rust.
//!
//! Status: prototype. [`WasmJit::try_run`] compiles the supported numeric +
//! control-flow subset (via [`translate`] → [`lower::compile`]) and runs it
//! natively through [`runner`]; everything else declines to the interpreter. The
//! web execution path (browser bridge over shared memory) is not wired up yet,
//! so on `wasm32` it also declines.

use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet};

use fnv::{FnvHashMap, FnvHashSet};
use std::rc::Rc;

use ruffle_core::avm2::{Activation, Class, Error, JitBackend, Method, Op, TObject, Value};

pub mod analysis;
pub mod inline;
pub mod emit;
pub(crate) mod helpers;
pub mod lower;
pub mod runner;
pub mod translate;

/// A compiled method: the module bytes, its import manifest, and every per-method
/// side table the helpers index at run time — ALL built once at compile time and
/// cached, so `try_run`'s hot path only installs pointers (no per-invocation
/// `parsed_code` scans, no `Vec` allocations; a profile showed the per-run table
/// rebuilds were the JIT's single largest overhead). The tables hold type-erased
/// `Gc` addresses / `Value` bits whose referents are rooted by the method's
/// verified bytecode (multinames, strings, coerce classes) or by the domain
/// (script globals), so they stay valid for the cache's lifetime — the same
/// argument that already carried `mn_table`.
#[derive(Clone)]
struct Compiled {
    bytes: Rc<[u8]>,
    manifest: lower::Manifest,
    /// The **combined** multiname pointer list (the caller's multinames + every
    /// inlined callee's, in the order the final ops index them — inlining remaps
    /// callee `k`s into this list). Empty when the method reads no properties.
    mn_table: Rc<[*const ()]>,
    /// Pre-resolved script-global `Value` bits, one per `GetScriptGlobals` op.
    /// Resolving runs the script initializer (idempotent); a rare init `#error`
    /// is cached as `undefined` (a fatal startup condition either way).
    script_globals: Rc<[u64]>,
    /// Pre-resolved string-constant `Value` bits, one per `PushString` op.
    push_strings: Rc<[u64]>,
    /// Type-erased `Class` addresses, one per `Coerce`/`NewClass`/`NewActivation`
    /// op (they share the table and the `next_coerce` counter, in op order).
    coerce_classes: Rc<[*const ()]>,
    /// Type-erased `NativeMethodImpl` fn pointers, one per `CallNative` op.
    natives: Rc<[*const ()]>,
    /// Erased `Namespace`s, one per `PushNamespace` op.
    namespaces: Rc<[*const ()]>,
    /// The lowered inputs, kept for GENERATION rebuilds (see
    /// [`lower::compile_generation`]): batches of compiled methods are re-emitted
    /// into one shared "amalgam" module against their union import layout.
    gen_src: GenSource,
}

/// The inputs a generation rebuild re-emits a method from.
#[derive(Clone)]
struct GenSource {
    ops: Rc<[lower::JitOp]>,
    switches: Rc<[lower::SwitchTable]>,
    exceptions: Rc<[lower::ExcRange]>,
}

impl GenSource {
    /// The post-install replacement: once a method's generation is live its
    /// sources are never re-emitted — free them (see `build_generation`).
    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
    fn empty() -> Self {
        GenSource {
            ops: Rc::from([] as [lower::JitOp; 0]),
            switches: Rc::from([] as [lower::SwitchTable; 0]),
            exceptions: Rc::from([] as [lower::ExcRange; 0]),
        }
    }
}

type CacheEntry = Option<Compiled>;

/// How many freshly compiled methods accumulate before they are folded into one
/// GENERATION module (web only). Each generation costs one `WebAssembly.Module`
/// + one instance + one entry-table slot for the whole batch — instead of one of
/// each **per method**, which exhausted the browser's page-granular
/// executable-code arena ("failed to allocate executable memory for module") and
/// the reserved entry-slot pool on method-heavy content (OpenTTD).
const GEN_BATCH: usize = 128;

/// Diagnostic: dump every call-bearing method the JIT has run when the FlasCC
/// allocator throws its `#1506` heap-corruption error. Set `true` to hunt the
/// CallMethod corruptor (needs `JIT_CALLMETHOD = true`); off for shipping.
const DUMP_ON_1506: bool = false;

thread_local! {
    /// Ordered log of every **call-bearing** method this thread's JIT has executed
    /// (first run only), as `"[i] name ptr=.. ops=[..]"`. Dumped by [`dump_on_1506`]
    /// when `make_error_1506` first fires — the corrupting method is in here.
    static EXEC_TRACE: RefCell<(std::collections::HashSet<usize>, Vec<String>)> =
        RefCell::new((std::collections::HashSet::new(), Vec::new()));
    /// Guard so the (large) trace is dumped only once, at the first `#1506`.
    static DUMPED_1506: Cell<bool> = const { Cell::new(false) };
    /// Recycled register-snapshot buffers for [`WasmJit::try_run`] — one `Vec`
    /// allocation per JIT invocation was measurable in profiles. Re-entrant runs
    /// pop distinct buffers; [`RegsGuard`] returns them on every exit path.
    static REGS_POOL: RefCell<Vec<Vec<u64>>> = const { RefCell::new(Vec::new()) };
}

/// Returns its register buffer to [`REGS_POOL`] on drop (any `try_run` exit path).
struct RegsGuard(Vec<u64>);

impl Drop for RegsGuard {
    fn drop(&mut self) {
        REGS_POOL.with(|p| p.borrow_mut().push(std::mem::take(&mut self.0)));
    }
}

/// The [`ruffle_core::avm2::error::ERROR_1506_HOOK`] the JIT registers: on the first
/// `#1506`, log every call-bearing method it has run (name + ops) in execution order.
fn dump_on_1506() {
    if DUMPED_1506.with(|d| d.replace(true)) {
        return; // already dumped
    }
    EXEC_TRACE.with(|t| {
        let (_, lines) = &*t.borrow();
        tracing::error!("=== JIT #1506 DUMP: {} call-bearing methods run ===", lines.len());
        for line in lines {
            tracing::error!("{line}");
        }
        tracing::error!("=== end JIT #1506 DUMP ===");
    });
}

/// A [`JitBackend`] that compiles AVM2 methods by emitting WebAssembly at runtime.
///
/// Install with `avm2.set_jit_backend(WasmJit::shared())`.
pub struct WasmJit {
    /// Compiled module bytes keyed by [`Method::as_ptr`]. `Some(None)` records a
    /// method we've already found unsupported so we don't retranslate it.
    cache: RefCell<FnvHashMap<usize, CacheEntry>>,
    /// When set, every JIT run is also executed through the real interpreter and
    /// the two results are asserted equal (differential self-check). Opt-in —
    /// only sound for the side-effect-free methods the JIT accepts.
    verify: bool,
    /// Re-entrancy guard: while the verifier is running the interpreter, nested
    /// [`Self::try_run`] calls decline so the interpreter actually interprets.
    in_verify: Cell<bool>,
    /// Methods compiled since the last GENERATION build (cache keys) — folded
    /// into one amalgam module every [`GEN_BATCH`] (see [`Self::build_generation`]).
    #[cfg(target_arch = "wasm32")]
    gen_pending: RefCell<Vec<usize>>,
    /// Number of methods actually executed by the JIT.
    hits: Cell<u32>,
    /// Number of JIT/interpreter divergences seen under `verify` (should be 0).
    mismatches: Cell<u32>,
    /// Bits of the most recent JIT result (for tests to assert the value, not
    /// just JIT/interpreter agreement).
    last_result: Cell<u64>,
    /// Every distinct `Value`-bits a JIT'd method has returned (for tests to check
    /// a specific value was produced without assuming it was the *last* method to
    /// run — many methods JIT per frame).
    results: RefCell<FnvHashSet<u64>>,
    /// Methods already logged on first JIT execution (diagnostic mode only) — so
    /// the *last* `JIT-EXEC` line before a freeze pinpoints a hanging method.
    executed: RefCell<FnvHashSet<usize>>,
    /// Per-method count of state-comparison verifications done (diagnostic mode).
    /// Bounds the cost of the domain-memory snapshot (verify a method a few times,
    /// then trust it).
    verify_seen: RefCell<FnvHashMap<usize, u32>>,
    /// Histogram of *why* methods declined to JIT, keyed by the first unsupported
    /// core `Op`'s variant name (`"<compile>"` = all ops supported but the lowering
    /// bailed). Recorded once per declined method; logged periodically via
    /// `tracing`. Turns "which ops should we add next" into data. See
    /// [`Self::record_decline`] / [`Self::decline_histogram`].
    declines: RefCell<FnvHashMap<String, u32>>,
}

impl Default for WasmJit {
    fn default() -> Self {
        Self {
            cache: RefCell::new(FnvHashMap::default()),
            verify: false,
            in_verify: Cell::new(false),
            #[cfg(target_arch = "wasm32")]
            gen_pending: RefCell::new(Vec::new()),
            hits: Cell::new(0),
            mismatches: Cell::new(0),
            last_result: Cell::new(0),
            results: RefCell::new(FnvHashSet::default()),
            executed: RefCell::new(FnvHashSet::default()),
            verify_seen: RefCell::new(FnvHashMap::default()),
            declines: RefCell::new(FnvHashMap::default()),
        }
    }
}

impl WasmJit {
    pub fn new() -> Self {
        if DUMP_ON_1506 {
            // Register the diagnostic dump (idempotent — first setter wins).
            let _ = ruffle_core::avm2::error::ERROR_1506_HOOK.set(dump_on_1506);
        }
        Self::default()
    }

    /// Enables the differential self-check (compare every JIT run against the
    /// interpreter). For testing/validation; do not enable in production.
    pub fn with_verify(mut self, verify: bool) -> Self {
        self.verify = verify;
        self
    }


    /// Boxed as `Rc<dyn JitBackend>` for
    /// [`Avm2::set_jit_backend`](ruffle_core::avm2::Avm2::set_jit_backend).
    pub fn shared() -> Rc<dyn JitBackend> {
        Rc::new(Self::new())
    }

    /// Like [`Self::shared`] but with the differential self-check enabled — every
    /// JIT run is also interpreted and compared, logging any divergence (with the
    /// method's ops) via `tracing::error!`. Much slower (double execution); swap it
    /// in temporarily to hunt a JIT/interpreter mismatch (e.g. suspected
    /// corruption) in a real build. Uses **state-comparison** verify: it snapshots
    /// domain memory + the receiver's written slots, restores them, re-runs the
    /// interpreter, and compares side effects (not just the return value — most
    /// JIT'd FlasCC methods return `void`). Returns the interpreter's result, so the
    /// game stays correct on verified methods even where the JIT diverges. Bounded
    /// to a few checks per method.
    pub fn shared_verified() -> Rc<dyn JitBackend> {
        Rc::new(Self::new().with_verify(true))
    }

    /// Number of methods the JIT has executed.
    pub fn hits(&self) -> u32 {
        self.hits.get()
    }

    /// Number of JIT/interpreter divergences seen under [`Self::with_verify`].
    pub fn mismatches(&self) -> u32 {
        self.mismatches.get()
    }

    /// Bits of the most recent JIT result `Value` (for tests).
    pub fn last_result(&self) -> u64 {
        self.last_result.get()
    }

    /// Whether some JIT'd method returned exactly `bits` (for tests — robust to
    /// other methods JIT'ing in the same frame, unlike [`Self::last_result`]).
    pub fn produced(&self, bits: u64) -> bool {
        self.results.borrow().contains(&bits)
    }

    /// Records that a method declined to JIT because of `reason` (the first
    /// unsupported op's name), and logs the running histogram when a *new* blocking
    /// op first appears (so apps with few declines still surface their walls) or
    /// every 200 declines thereafter.
    fn record_decline(&self, reason: String) {
        let (total, is_new_reason) = {
            let mut declines = self.declines.borrow_mut();
            let is_new = !declines.contains_key(&reason);
            *declines.entry(reason).or_insert(0) += 1;
            (declines.values().sum::<u32>(), is_new)
        };
        if is_new_reason || total.is_multiple_of(200) {
            let hist = self.decline_histogram();
            let top: Vec<String> = hist
                .iter()
                .take(20)
                .map(|(name, count)| format!("{name}={count}"))
                .collect();
            tracing::info!("AVM2 JIT declines ({total} methods): {}", top.join(", "));
        }
    }

    /// The decline histogram, `(op-name, count)` sorted most-frequent first —
    /// which unsupported ops block the most methods (add these to the JIT next).
    pub fn decline_histogram(&self) -> Vec<(String, u32)> {
        let mut v: Vec<(String, u32)> = self
            .declines
            .borrow()
            .iter()
            .map(|(name, count)| (name.clone(), *count))
            .collect();
        v.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        v
    }

    /// Returns the compiled module for `method`, compiling+caching on first use.
    /// `None` means unsupported (cached so we don't retry). Building the entry on
    /// a miss needs `activation` to resolve the signature for the int-soundness
    /// seed; cache hits never touch it.
    fn compiled<'gc>(
        &self,
        activation: &mut Activation<'_, 'gc>,
        method: Method<'gc>,
    ) -> Option<Compiled> {
        let key = method.as_ptr() as usize;
        if let Some(entry) = self.cache.borrow().get(&key) {
            return entry.clone();
        }
        // Low-memory guard (web): compiling allocates (wasm-encoder buffers,
        // caches, tables), and a failed allocation inside `try_run` ABORTS the
        // whole player (`handle_alloc_error` → `unreachable`, seen in the field
        // mid-`CodeSection::function` once content neared the 2 GiB wasm cap).
        // When headroom runs out, stop compiling — the interpreter keeps the
        // content running. wasm memory never shrinks, so cache the decline.
        #[cfg(target_arch = "wasm32")]
        if heap_exhausted() {
            warn_heap_exhausted_once();
            self.cache.borrow_mut().insert(key, None);
            return None;
        }
        let entry: CacheEntry = locals_typed_seed(activation, method)
            .and_then(|(int_seed, double_seed)| compile_method(method, &int_seed, &double_seed))
            .map(|(bytes, manifest, mn_list, gen_src)| {
                // Build the per-method side tables ONCE, here (the only place with
                // both the compile result and an activation for script-globals
                // resolution) — the hot path then just installs the cached slices.
                Compiled {
                    script_globals: if manifest.has_script_globals {
                        script_globals_table(activation, method).into()
                    } else {
                        Rc::from([])
                    },
                    push_strings: if manifest.has_push_strings {
                        push_string_table(method).into()
                    } else {
                        Rc::from([])
                    },
                    // `newclass`/`newactivation` (vcall kinds) ride the same table
                    // as `coerce`, so either flag builds it.
                    coerce_classes: if manifest.has_coerce || manifest.has_vcall {
                        coerce_class_table(method).into()
                    } else {
                        Rc::from([])
                    },
                    natives: if manifest.has_vcall {
                        natives_table(method).into()
                    } else {
                        Rc::from([])
                    },
                    namespaces: if manifest.has_vcall {
                        namespaces_table(method).into()
                    } else {
                        Rc::from([])
                    },
                    bytes,
                    manifest,
                    mn_table: mn_list,
                    gen_src,
                }
            });
        // On the first (and only) miss for a method that declined, record why —
        // the first op the boxed path can't lower (`<compile>` if all ops are
        // supported but the lowering bailed). Skips body-less methods.
        if entry.is_none() && method.body().is_some() {
            let core_ops = &method.get_verified_info().parsed_code;
            let reason = translate::first_unsupported_boxed(core_ops)
                .unwrap_or_else(|| "<compile>".to_string());
            self.record_decline(reason);
        }
        self.cache.borrow_mut().insert(key, entry.clone());
        // Web: fold freshly compiled methods into a GENERATION module per batch
        // (one module/instance/entry-slot for all of them — see [`GEN_BATCH`]).
        #[cfg(target_arch = "wasm32")]
        if entry.is_some() {
            self.gen_pending.borrow_mut().push(key);
            if self.gen_pending.borrow().len() >= GEN_BATCH {
                self.build_generation();
            }
        }
        entry
    }

    /// Builds and installs one GENERATION module from the pending batch (see
    /// [`GEN_BATCH`]). A failure (compile/instantiate/CSP) just leaves the batch's
    /// methods on their per-method path — correct, only less economical.
    #[cfg(target_arch = "wasm32")]
    fn build_generation(&self) {
        // A generation build re-emits the whole batch (wasm-encoder buffers) —
        // don't risk an OOM abort near the heap cap; the per-method modules
        // keep working (see `heap_exhausted`).
        if heap_exhausted() {
            warn_heap_exhausted_once();
            return;
        }
        let keys = std::mem::take(&mut *self.gen_pending.borrow_mut());
        let (bytes, union, member_keys) = {
            let cache = self.cache.borrow();
            let compiled: Vec<&Compiled> = keys
                .iter()
                .filter_map(|k| cache.get(k).and_then(|e| e.as_ref()))
                .collect();
            let members: Vec<lower::GenMember<'_>> = compiled
                .iter()
                .map(|c| lower::GenMember {
                    ops: &c.gen_src.ops,
                    switches: &c.gen_src.switches,
                    exceptions: &c.gen_src.exceptions,
                })
                .collect();
            let Some((bytes, union)) = lower::compile_generation(&members) else {
                return;
            };
            // The runner keys methods by their (cached, never-moving) module bytes.
            let member_keys: Vec<usize> =
                compiled.iter().map(|c| c.bytes.as_ptr() as usize).collect();
            (bytes, union, member_keys)
        };
        if runner::install_generation(&bytes, &union, &member_keys) {
            // The members' generation sources (per-method `JitOp`/switch/exception
            // copies) exist only to re-emit this batch — dead weight once the
            // generation is live. Drop them; the standing JIT memory then stays
            // bounded. (The per-method module `bytes` must stay allocated: the
            // runner keys generation members by their address.)
            let mut cache = self.cache.borrow_mut();
            for k in &keys {
                if let Some(Some(c)) = cache.get_mut(k) {
                    c.gen_src = GenSource::empty();
                }
            }
        }
    }
}

/// Whether the wasm linear memory is close enough to its build cap
/// (`build_wasm.ts --max-memory`, currently 4 GiB — keep in sync) that further
/// JIT compilation must stop. Leaves 256 MiB of headroom for the content's own
/// allocations — an allocation failure anywhere in wasm is a player-killing
/// abort, so the JIT (a pure accelerator) backs off first. Memory never
/// shrinks, so once true, always true.
#[cfg(target_arch = "wasm32")]
fn heap_exhausted() -> bool {
    // In 64 KiB pages (byte counts overflow 32-bit `usize` at the 4 GiB cap).
    const MAX_PAGES: usize = 65536; // 4 GiB
    const HEADROOM_PAGES: usize = 256 * 1024 * 1024 / 65536;
    core::arch::wasm32::memory_size(0) >= MAX_PAGES - HEADROOM_PAGES
}

/// Logs the heap-exhaustion backoff once (it explains a sudden fps drop:
/// methods stop entering the JIT and run interpreted from that point on).
#[cfg(target_arch = "wasm32")]
fn warn_heap_exhausted_once() {
    thread_local! {
        static WARNED: Cell<bool> = const { Cell::new(false) };
    }
    WARNED.with(|w| {
        if !w.replace(true) {
            tracing::warn!(
                "AVM2 JIT: wasm heap near its build cap — JIT compilation \
                 disabled (new methods run interpreted) to avoid an OOM abort"
            );
        }
    });
}

/// Per-local entry types for the fast-path soundness analyses: returns
/// `(is_int, is_double)` per local. `this` and object params are neither;
/// `int`/`uint` params are int; `Number` params are double; fresh locals neither.
/// `None` if the method has no body.
fn locals_typed_seed<'gc>(
    activation: &mut Activation<'_, 'gc>,
    method: Method<'gc>,
) -> Option<(Vec<bool>, Vec<bool>)> {
    let num_locals = method.body()?.num_locals as usize;
    method.resolve_info(activation).ok()?;
    let params = method.resolved_param_config();
    let class_defs = activation.avm2().class_defs();
    let (int_class, number_class) = (class_defs.int, class_defs.number);

    let mut int_seed = vec![false; num_locals];
    let mut double_seed = vec![false; num_locals];
    for (i, param) in params.iter().enumerate() {
        let slot = i + 1; // local 0 is `this`
        if slot < num_locals {
            match param.param_type {
                // `uint` is deliberately NOT seeded int: the raw-i32 model compares
                // signed (`I32LtS`) and re-boxes as `Integer`, but a `uint > i32::MAX`
                // is a positive `Number` in AS3 — signed compare / int-box would
                // diverge. `uint` methods fall to the boxed path (CoerceU/abstract_lt).
                Some(c) if c == int_class => int_seed[slot] = true,
                Some(c) if c == number_class => double_seed[slot] = true,
                _ => {}
            }
        }
    }
    Some((int_seed, double_seed))
}

/// Translates, type-checks, and compiles a method. `None` if any op is
/// unsupported or the raw-`i32` model would be unsound for `seed`.
fn compile_method<'gc>(
    method: Method<'gc>,
    int_seed: &[bool],
    double_seed: &[bool],
) -> Option<(Rc<[u8]>, lower::Manifest, Rc<[*const ()]>, GenSource)> {
    method.body()?;
    let core_ops = &method.get_verified_info().parsed_code;

    // `mn_list`: the combined multiname pointer list (see [`Compiled`]). The fast paths
    // read no properties → empty; the boxed path passes the caller's + inlined callees'.
    let finish = |ops: &[lower::JitOp],
                  switches: &[lower::SwitchTable],
                  exceptions: &[lower::ExcRange],
                  mn_list: Vec<*const ()>|
     -> Option<(Rc<[u8]>, lower::Manifest, Rc<[*const ()]>, GenSource)> {
        let bytes =
            Rc::from(lower::compile_full(ops, switches, exceptions)?.into_boxed_slice());
        let gen_src = GenSource {
            ops: ops.into(),
            switches: switches.into(),
            exceptions: exceptions.into(),
        };
        Some((bytes, lower::manifest(ops), mn_list.into(), gen_src))
    };

    // Int fast path (raw i32) — only when provably int-sound.
    if let Some(ops) = translate::translate(core_ops) {
        if analysis::int_sound(&ops, int_seed) {
            return finish(&ops, &[], &[], Vec::new());
        }
    }

    // Double fast path (unboxed f64, inline arithmetic) — only when Number-sound.
    if let Some(ops) = translate::translate_double(core_ops) {
        if analysis::double_sound(&ops, double_seed) {
            return finish(&ops, &[], &[], Vec::new());
        }
    }

    // The method's exception handlers (op-index ranges). A throw — explicit or from
    // a throwing call/dm — inside a range dispatches through the handler table.
    let exceptions = exc_ranges(method);

    // Boxed (GC-aware) path — raw `Value`s + imported helper `call`s.
    let (mut ops, switches) = translate::translate_boxed(core_ops)?;
    // The caller's multiname pointers; the inline pass appends each inlined callee's
    // (and remaps that callee's `k`s), keeping this list in sync with the final ops.
    let mut mn_list = multiname_table(method);
    // Inline pass: splice statically-resolvable, small callees so their calls become
    // wasm-internal, avoiding the per-invocation `try_run` overhead. Correct either way
    // — a failed resolve/inline leaves the call helper. Skipped when the method has a
    // `lookupswitch`: inlining shifts op indices, which would desync the switch
    // side-table's (absolute) targets — not worth remapping for a rare combination.
    // Also skipped when the method has exception handlers: their op-index ranges
    // must stay aligned with `core_ops` (inlining shifts indices).
    if JIT_INLINE && switches.is_empty() && exceptions.is_empty() {
        inline_pass(&mut ops, &mut mn_list, method);
    }
    // Decline helper-dominated methods: when a boxed method is mostly JS-boundary
    // crossings (getproperty/getslot/callmethod/generic helpers) with little inline
    // compute, the interpreter's fast native dispatch beats the JIT's per-call reg
    // copy + per-op boundary crossings. Falling back to the interpreter is always
    // correct — this only trades a JIT loss for the faster path.
    if DECLINE_HELPER_DOMINATED && lower::helper_dominated(&ops) {
        return None;
    }
    finish(&ops, &switches, &exceptions, mn_list)
}

/// The method's exception handlers as [`lower::ExcRange`]s (op-index ranges).
fn exc_ranges<'gc>(method: Method<'gc>) -> Vec<lower::ExcRange> {
    method
        .get_verified_info()
        .exceptions
        .iter()
        .map(|e| lower::ExcRange {
            from: e.from_offset,
            to: e.to_offset,
            target: e.target_offset,
        })
        .collect()
}

/// Whether the inline pass runs (phase 1: super constructors). Splices small,
/// statically-resolvable callees so their calls avoid the per-invocation `try_run`
/// overhead (the dominant cost once JIT coverage is broad).
const JIT_INLINE: bool = true;

/// The maximum local slot the frame supports (`STRIDE / 8`); an inline must keep the
/// caller's + callee's locals within it (see `runner::web` `STRIDE`).
const MAX_FRAME_SLOTS: u32 = 512;

/// Splices statically-resolvable, small callees into `ops` so their calls become
/// wasm-internal. Phase 1: `constructsuper` (super ctor — unambiguous, non-virtual).
/// Phase 2: `this.callmethod(disp_id)` when the bound class is **final** (so the
/// vtable slot can't be overridden → the callee is exact) and the receiver is provably
/// `this`. Each splice removes one call and inlinable callees contain no calls, so the
/// scan terminates. All inlined callees share one local base (`caller_locals`) — their
/// scratch locals are never live simultaneously (calls are sequential, not nested).
fn inline_pass<'gc>(
    ops: &mut Vec<lower::JitOp>,
    mn_list: &mut Vec<*const ()>,
    method: Method<'gc>,
) {
    let Some(bound) = method.bound_class() else { return };
    let Some(caller_locals) = method.body().map(|b| b.num_locals) else { return };
    let base = caller_locals;
    // Repeatedly splice the first inlinable call until none remain (bounded).
    loop {
        let mut progressed = false;
        for idx in 0..ops.len() {
            let Some((callee_method, callee, callee_locals, argc, result)) =
                resolve_inlinable(ops, idx, method, bound)
            else {
                continue;
            };
            if base + callee_locals > MAX_FRAME_SLOTS {
                continue;
            }
            // The callee's multinames are appended to the combined list; its `GetProperty`
            // `k`s are remapped by `mn_base` (the list length *before* appending) so they
            // index the appended entries.
            let mn_base = mn_list.len() as u32;
            if let Some(spliced) = inline::splice(ops, idx, &callee, base, mn_base, argc, result) {
                *ops = spliced;
                mn_list.extend(multiname_table(callee_method));
                progressed = true;
                break;
            }
        }
        if !progressed {
            break;
        }
    }
}

/// If the op at `idx` is an inlinable call, returns the callee method, its boxed ops,
/// local count, the call's arg count, and how its result is consumed. `None` otherwise.
#[allow(clippy::type_complexity)]
fn resolve_inlinable<'gc>(
    ops: &[lower::JitOp],
    idx: usize,
    method: Method<'gc>,
    bound: ruffle_core::avm2::Class<'gc>,
) -> Option<(Method<'gc>, Vec<lower::JitOp>, u32, u32, inline::ResultMode)> {
    use lower::JitOp;
    let (callee_method, argc, result) = match ops[idx] {
        // Super ctor: unambiguous (non-virtual), no receiver analysis needed.
        JitOp::ConstructSuper(argc) => {
            let ctor = bound.super_class()?.instance_init()?;
            (ctor, argc, inline::ResultMode::Discard)
        }
        // `this.method()`: sound only if the class is final (no override) and the
        // receiver is provably `this`.
        JitOp::CallMethod(disp_id, argc, push) => {
            if !bound.is_final() || !inline::receiver_is_this(ops, idx, argc) {
                return None;
            }
            let callee = bound.vtable().get_method(disp_id as usize)?;
            let result = if push {
                inline::ResultMode::Push
            } else {
                inline::ResultMode::Discard
            };
            (callee, argc, result)
        }
        _ => return None,
    };
    // Don't inline a method into itself (direct recursion).
    if callee_method.as_ptr() == method.as_ptr() {
        return None;
    }
    let callee_locals = callee_method.body()?.num_locals;
    // Only already-verified bytecode callees (never force verification here).
    let info = callee_method.try_verified_info()?;
    let (callee, callee_switches) = translate::translate_boxed(&info.parsed_code)?;
    // A callee with a `lookupswitch` carries a side-table of absolute op targets
    // that splicing would have to remap — don't inline it (its call stays a helper).
    if !callee_switches.is_empty() {
        return None;
    }
    inline::callee_inlinable(&callee)?;
    Some((callee_method, callee, callee_locals, argc, result))
}

/// Whether to decline compiling helper-dominated boxed methods (see
/// [`lower::helper_dominated`]). **Off** — the heuristic was calibrated when each
/// helper op was an expensive JS-boundary crossing (~26% of the compute worker in a
/// profile). Now that the web runner binds helper imports to Ruffle's own wasm
/// functions (a direct **wasm→wasm** call — no JS trampoline, no i64↔BigInt), a
/// helper call is cheap, so declining crossing-heavy methods only pushes them onto
/// the (slower) interpreter — measured as a big JIT-coverage drop (51.8%→11.6% on
/// gameplay) and much slower loading. The right axis is call-count (amortize the
/// one-time compile + per-call JIT-entry crossing), not helper density — future work.
/// Kept wired (+ tested) so it can be re-enabled/repurposed.
const DECLINE_HELPER_DOMINATED: bool = false;

/// Reinterprets a `Value` as its 8-byte NaN-boxed bit pattern.
/// The `Value` bits a `returnvoid` yields for a declared `return_type`, mirroring
/// [`Activation::return_void`]: `void`/none → `undefined`, a numeric type → `0`, a
/// boolean → `false`, any other class → `null`. The JIT previously always returned
/// `undefined`, silently miscompiling e.g. `function():int { return }` (→ `0`).
pub(crate) fn return_void_bits(return_type: Option<Class<'_>>) -> u64 {
    let v: Value<'_> = match return_type {
        Some(c) if c.is_builtin_void() => Value::Undefined,
        Some(c) if c.is_builtin_numeric() => Value::from(0),
        Some(c) if c.is_builtin_boolean() => Value::from(false),
        Some(_) => Value::Null,
        None => Value::Undefined,
    };
    value_to_bits(v)
}

fn value_to_bits(v: Value<'_>) -> u64 {
    debug_assert_eq!(std::mem::size_of::<Value<'_>>(), 8);
    // SAFETY: `Value` is a NaN-boxed `u64`; transmuting to its bits is total.
    unsafe { std::mem::transmute(v) }
}

/// Reconstructs a `Value` from bits produced by the JIT.
///
/// SAFETY / soundness: `bits` must be a `Value` the JIT actually obtained — a
/// local slot's bits, or a value a helper produced (`get_property`, `increment`,
/// …) — never a fabricated pointer. Reconstructing a *real* `Gc` pointer this way
/// is sound: it's the same address the collector already knows, we don't *store*
/// it into another `Gc` (a by-value return needs no write barrier), and gc-arena
/// can't collect mid-run (it only collects between `mutate`s — see
/// `Player::run_frame`), so the object can't have been freed. Fabricating a
/// pointer from arbitrary integer bits would still be unsound; the translators
/// only feed genuine `Value` bits here.
unsafe fn value_from_bits<'gc>(bits: u64) -> Value<'gc> {
    // SAFETY: delegated to the caller (see doc comment).
    unsafe { std::mem::transmute(bits) }
}

/// The method's multiname table for the getproperty helper: one live
/// `Gc<Multiname>` address (type-erased) per `GetPropertyStatic`/`Slow` op, in op
/// order — matching the `k` indices [`translate::translate_boxed`] assigns. Built
/// fresh each run from the still-live verified ops, so the pointers are valid for
/// the run's duration (see [`helpers::with_multinames`]). Empty (no getproperty).
fn multiname_table(method: Method<'_>) -> Vec<*const ()> {
    method
        .get_verified_info()
        .parsed_code
        .iter()
        .filter_map(|op| match op {
            // Keep in sync with the ops `translate::boxed_op` assigns a `k` to (it
            // bumps `next_mn` for exactly these, in op order): property reads and
            // writes, multiname calls, and the super/delete vcall kinds. A compiled
            // method's non-`Fast` multinames are all non-lazy (lazy ones — including
            // every `GetPropertySlow`/`SetPropertySlow` — make `boxed_op` decline),
            // so the sets align.
            Op::GetPropertyStatic { multiname }
            | Op::GetPropertyFast { multiname }
            | Op::CallProperty { multiname, .. }
            | Op::CallPropVoid { multiname, .. }
            | Op::ConstructProp { multiname, .. }
            | Op::CallSuper { multiname, .. }
            | Op::GetSuper { multiname }
            | Op::SetSuper { multiname }
            | Op::DeleteProperty { multiname }
            | Op::SetPropertyStatic { multiname }
            | Op::SetPropertyFast { multiname } => {
                Some(std::ptr::from_ref(&**multiname) as *const ())
            }
            _ => None,
        })
        .collect()
}

/// The method's **pre-resolved** script-globals table: the global-object `Value`
/// bits for each `GetScriptGlobals` op, in op order (matching `boxed_op`'s
/// `next_script`). Resolved here (we have the `context`) so the `get_script_globals`
/// helper is a plain table read. `globals()` is idempotent (runs the script
/// initializer once, then returns the cached global); a rare init `#error` is
/// swallowed to `undefined` (a fatal startup condition either way). Empty unless the
/// method reads script globals.
fn script_globals_table<'gc>(activation: &mut Activation<'_, 'gc>, method: Method<'gc>) -> Vec<u64> {
    // Collect the scripts first (releases the `parsed_code` borrow before we borrow
    // `activation.context` to resolve them).
    let scripts: Vec<_> = method
        .get_verified_info()
        .parsed_code
        .iter()
        .filter_map(|op| match op {
            Op::GetScriptGlobals { script } => Some(*script),
            _ => None,
        })
        .collect();
    scripts
        .iter()
        .map(|script| match script.globals(activation.context) {
            Ok(obj) => value_to_bits(Value::Object(obj)),
            Err(_) => value_to_bits(Value::Undefined),
        })
        .collect()
}

/// The method's **pre-resolved** string table: the `Value` bits for each
/// `PushString` op, in op order (matching `boxed_op`'s `next_string`). The string
/// atoms are already interned in the method's constant pool (no `context` needed);
/// a string `Value` holds a `Gc` that stays valid for the run (no GC mid-`try_run`),
/// so the `get_push_string` helper is a plain table read. Empty unless the method
/// pushes strings.
fn push_string_table<'gc>(method: Method<'gc>) -> Vec<u64> {
    method
        .get_verified_info()
        .parsed_code
        .iter()
        .filter_map(|op| match op {
            Op::PushString { string } => Some(value_to_bits(Value::from(*string))),
            _ => None,
        })
        .collect()
}

/// The method's coerce-class table: a type-erased `Class` address for each
/// `Op::Coerce`, in op order (matching `boxed_op`'s `next_coerce`). The classes are
/// resolved in the verified bytecode and stay live for the run (no GC mid-`try_run`),
/// so the `coerce` helper reverses the erasure and calls `coerce_to_type` directly.
/// Empty unless the method has a `coerce` op.
fn coerce_class_table<'gc>(method: Method<'gc>) -> Vec<*const ()> {
    method
        .get_verified_info()
        .parsed_code
        .iter()
        .filter_map(|op| match op {
            // SAFETY: `Class` is `#[repr(transparent)]` over a `Gc` (a single pointer);
            // the class is alive for the run, so erasing its pointer is sound (the
            // `coerce` helper reverses it within the same run). `NewClass` and
            // `NewActivation` share the table (and translate's `next_coerce` counter).
            Op::Coerce { class }
            | Op::NewClass { class }
            | Op::NewActivation { activation_class: class } => {
                Some(unsafe { std::mem::transmute::<Class<'gc>, *const ()>(*class) })
            }
            _ => None,
        })
        .collect()
}

/// The method's native-fn table: a type-erased `NativeMethodImpl` fn pointer per
/// `Op::CallNative`, in op order (matching `boxed_op`'s `next_native`). Fn pointers
/// are process-stable, so the erasure round-trips trivially (`vcall`'s
/// `CALL_NATIVE`). Empty unless the method has a `callnative`.
fn natives_table(method: Method<'_>) -> Vec<*const ()> {
    method
        .get_verified_info()
        .parsed_code
        .iter()
        .filter_map(|op| match op {
            Op::CallNative { method: native, .. } => Some(*native as *const ()),
            _ => None,
        })
        .collect()
}

/// The method's namespace table: an erased `Namespace` per `Op::PushNamespace`, in
/// op order (matching `boxed_op`'s `next_ns`). `Namespace` is a niche-optimized
/// `Option<Gc>` (pointer-sized) alive for the run, like the coerce classes.
fn namespaces_table(method: Method<'_>) -> Vec<*const ()> {
    method
        .get_verified_info()
        .parsed_code
        .iter()
        .filter_map(|op| match op {
            // SAFETY: pointer-sized handle, alive for the run; `vcall`'s
            // `PUSH_NAMESPACE` reverses the erasure within the same run.
            Op::PushNamespace { namespace } => Some(unsafe {
                std::mem::transmute::<ruffle_core::avm2::Namespace<'_>, *const ()>(*namespace)
            }),
            _ => None,
        })
        .collect()
}

/// The slot ids the call-aware verifier tracks on `this` (local0): the FlasCC ABI
/// globals ESP (1) / eax (2) plus every slot the method itself writes. FlasCC callees
/// mutate only ESP/eax (all other C state lives in domain memory, tracked via the
/// write-log), so this *small* set covers the bracket — snapshotting all ~19k slots
/// of the package global was needlessly slow (and forced `VERIFY_LIMIT` too low).
fn tracked_slot_ids(method: Method<'_>) -> BTreeSet<usize> {
    let mut ids = BTreeSet::from([1usize, 2]);
    for op in &method.get_verified_info().parsed_code {
        if let Op::SetSlot { index } | Op::SetSlotNoCoerce { index } | Op::SetSlotCoerceI { index } =
            op
        {
            ids.insert(*index as usize);
        }
    }
    ids
}

/// Snapshot the tracked slots (id → bits) on `this`, skipping any out of range.
fn slots_snapshot<'gc>(
    activation: &mut Activation<'_, 'gc>,
    ids: &BTreeSet<usize>,
) -> BTreeMap<usize, u64> {
    match activation.local_register(0).as_object() {
        Some(obj) => {
            let n = obj.vtable().slot_count();
            ids.iter()
                .filter(|&&i| i < n)
                .map(|&i| (i, value_to_bits(obj.get_slot(i))))
                .collect()
        }
        None => BTreeMap::new(),
    }
}

/// Restore the tracked slots that drifted from `baseline` (call-aware reset).
fn slots_restore<'gc>(activation: &mut Activation<'_, 'gc>, baseline: &BTreeMap<usize, u64>) {
    if let Some(obj) = activation.local_register(0).as_object() {
        let mc = activation.gc();
        for (&i, &b) in baseline {
            if value_to_bits(obj.get_slot(i)) != b {
                // SAFETY: `b` came from a real `Value` we read from this slot.
                obj.set_slot_no_coerce(i, unsafe { value_from_bits(b) }, mc);
            }
        }
    }
}

impl JitBackend for WasmJit {
    fn try_run<'gc>(
        &self,
        activation: &mut Activation<'_, 'gc>,
        method: Method<'gc>,
    ) -> Option<Result<Value<'gc>, Error<'gc>>> {
        // Decline while the verifier is driving the interpreter (avoid recursion).
        if self.in_verify.get() {
            return None;
        }

        let num_locals = method.body()?.num_locals as usize;
        // Every per-method side table (multinames, script globals, strings, coerce
        // classes) is built once at compile time and cached in `Compiled` — the hot
        // path below only installs the cached slices.
        let compiled = self.compiled(activation, method)?;
        let (bytes, manifest) = (&compiled.bytes, &compiled.manifest);

        // The register-snapshot source. PRODUCTION: **zero-copy** — the frame's
        // local registers are contiguous 8-byte `Value` words in Ruffle's own
        // memory and the JIT only READS them (the wasm prologue copies them into
        // the frame memory; on native the runner writes them into the wasmi
        // memory), so pass the live storage directly — no per-call Vec fill.
        // VERIFY: materialize a snapshot through the pool — the interpreter
        // re-runs mutate the live frame, and JIT run 2 must see pristine values.
        let regs_guard;
        let regs: &[u64] = if self.verify {
            let mut buf = REGS_POOL.with(|p| p.borrow_mut().pop()).unwrap_or_default();
            buf.clear();
            for i in 0..num_locals as u32 {
                buf.push(value_to_bits(activation.local_register(i)));
            }
            regs_guard = RegsGuard(buf);
            &regs_guard.0
        } else {
            // SAFETY: points at ≥ `num_locals` contiguous `Value` words alive
            // for the whole synchronous run; nothing writes through this slice.
            unsafe { std::slice::from_raw_parts(activation.local_registers_ptr(), num_locals) }
        };

        let key = method.as_ptr() as usize;

        // Diagnostic: log each method's name + ops the first time it JIT-executes,
        // so if a JIT'd method hangs (infinite loop), the *last* `JIT-EXEC` line
        // before the freeze names it. Only in verify/diagnostic mode.
        if self.verify && self.executed.borrow_mut().insert(key) {
            tracing::info!(
                "JIT-EXEC {} {:p} ops={:?}",
                method.method_name(),
                method.as_ptr(),
                method.get_verified_info().parsed_code
            );
        }
        // Diagnostic: on this method's first run, if it's **call-bearing** (a `#1506`
        // heap-corruption suspect — those methods only JIT with `callmethod`), record
        // its name + ops for [`dump_on_1506`] to dump when the allocator throws #1506.
        if DUMP_ON_1506 {
            let ops = &method.get_verified_info().parsed_code;
            if ops.iter().any(|o| matches!(o, Op::CallMethod { .. })) {
                EXEC_TRACE.with(|t| {
                    let mut t = t.borrow_mut();
                    if t.0.insert(key) {
                        let i = t.1.len();
                        let line =
                            format!("[{i}] {} ptr={key:#x} ops={ops:?}", method.method_name());
                        t.1.push(line);
                    }
                });
            }
        }

        // **Call-aware verify, gated to call-bearing methods only.** Both engines'
        // domain-memory writes funnel through the core write-log
        // (`ByteArrayStorage::dm_set`/`dm_write`), so a call-bearing method's *callee*
        // writes are captured and rolled back precisely (per run, our addresses only —
        // never a whole-buffer restore that would clobber other workers). All slots +
        // the heap break are snapshot/restored too; each engine runs twice (bracket)
        // so a byte is a fault only when both JIT runs agree, both interp runs agree,
        // and the clusters differ. **Leaf methods are skipped** — their JIT is already
        // proven correct, and verifying them (the vast majority) is what made the
        // verify too slow to reach the `ScanPath`/`#1506` corruption. Call-only keeps
        // the game correct (leaf=JIT, call-bearing=interp-masked) *and* fast enough to
        // reach and catch the corrupting allocator method. Requires helper dm
        // (`INLINE_DM = false`) so `si*` routes through the logging `dm_store`.
        // High: the corrupting invocation of a hot allocator method is far past the
        // 40th call, so a low limit never verifies it. Cheap now (tracked slots only).
        const VERIFY_LIMIT: u32 = 20_000;
        // Verify is off in production, so the call-op scan + slot-id scan below run
        // ONLY under verify — `self.verify` short-circuits the `&&` before the
        // `parsed_code` scan, keeping the hot path free of per-call bytecode scans.
        let will_verify = self.verify
            && *self.verify_seen.borrow().get(&key).unwrap_or(&0) < VERIFY_LIMIT
            && method.get_verified_info().parsed_code.iter().any(|o| {
                matches!(
                    o,
                    Op::CallMethod { .. } | Op::CallProperty { .. } | Op::CallPropVoid { .. } | Op::ConstructSuper { .. } | Op::Call { .. }
                )
            });
        // Snapshot the tracked slots (ESP/eax + the method's own targets) + the heap
        // break (`sbrk` length) *before* the JIT run, and arm the dm write-log. The
        // length isn't a `dm_set`/`dm_write` so it isn't in the write-log. All gated on
        // `will_verify` so the `tracked_slot_ids` scan never runs in production.
        let slot_ids = will_verify.then(|| tracked_slot_ids(method));
        let pre_slots = slot_ids
            .as_ref()
            .map(|ids| slots_snapshot(activation, ids));
        let pre_len = will_verify.then(|| helpers::dm_len(activation));
        if will_verify {
            helpers::dm_log_start();
        }

        // Clear any pending error left over from a prior run that returned `None`
        // (e.g. a web trap after a call already threw) so it can't leak into this one.
        let _ = helpers::take_pending_error();


        // The method's declared return type, for `coerce_return` (a `ReturnValueCoerced`
        // op). Only methods that actually have that op need it — skip the per-call
        // signature resolution otherwise. Resolve first (idempotent, cached) so
        // `resolved_return_type` can't panic on a boxed method that compiled without seeding.
        let return_type = if manifest.has_coerced_return {
            method
                .resolve_info(activation)
                .ok()
                .and_then(|_| method.resolved_return_type())
        } else {
            None
        };

        // Install the whole run context (activation, side tables, return type,
        // method) with ONE thread-local swap, then run. (The regs snapshot above
        // is taken first, so it doesn't observe helper-caused mutation.)
        let ctx = helpers::RunCtx::new(
            activation,
            return_type,
            &compiled.mn_table,
            &compiled.script_globals,
            &compiled.push_strings,
            &compiled.coerce_classes,
            &compiled.natives,
            &compiled.namespaces,
            method,
        );
        let result_bits = helpers::with_run_ctx(&ctx, || runner::run(bytes, regs, manifest))?;

        // A `callmethod` that threw is captured out-of-band (the ABI is infallible
        // and the emitted code bails after the throwing call). Propagate it — the
        // interpreter would unwind at the same point with the same prior side
        // effects, so the JIT's captured error is the correct result.
        if let Some(err) = helpers::take_pending_error() {
            self.hits.set(self.hits.get() + 1);
            return Some(Err(err));
        }

        // SAFETY: the bits are a genuine `Value` the JIT/helpers produced. See above.
        let jit_value = unsafe { value_from_bits(result_bits) };
        self.hits.set(self.hits.get() + 1);
        self.last_result.set(result_bits);
        // Test-only result recording — MUST stay out of production: every distinct
        // result accumulates forever, and continuously-varying f64 results (times,
        // rotations, positions) grow the set unboundedly. In a real app this
        // exhausted the 2 GiB wasm heap — the HashSet's doubling rehash allocates
        // a new table beside the old one, and that spike aborted the player in
        // `handle_alloc_error` (observed with Starling: seconds after a benchmark
        // started mutating rotation/scale every frame).
        if self.verify {
            self.results.borrow_mut().insert(result_bits);
        }

        let (Some(pre_slots), Some(pre_len)) = (pre_slots, pre_len) else {
            let _ = helpers::dm_log_take();
            return Some(Ok(jit_value));
        };
        // Reaching here means `will_verify` was true, so `slot_ids` is `Some`.
        let slot_ids = slot_ids.expect("slot_ids present when verifying");
        *self.verify_seen.borrow_mut().entry(key).or_insert(0) += 1;

        // Each run's pristine (pre-write) bytes, accumulated across runs; all runs
        // start from the same state (each rolls its own writes back), so a byte another
        // worker mutates at an address none of our runs touch never enters the compare.
        let mut pre: BTreeMap<usize, u8> = BTreeMap::new();
        // Read a finished run's `addr -> final byte` map from the shared write-log,
        // record pristine bytes into `pre`, and (unless `keep`) roll the writes back so
        // the next run sees the same inputs. `keep` leaves the last interp run live.
        let capture = |act: &mut Activation<'_, 'gc>,
                       pre: &mut BTreeMap<usize, u8>,
                       keep: bool|
         -> (BTreeMap<usize, u8>, usize) {
            // Post-run heap break (`sbrk` length): compared across runs, since the
            // length is *not* in the dm write-log yet a JIT/interp grow divergence
            // directly triggers `#1506` (a chunk size implying an out-of-range addr).
            let len = helpers::dm_len(act);
            let writes = helpers::dm_log_take();
            let mut fin = BTreeMap::new();
            for (addr, old) in &writes {
                let cur = helpers::dm_read_range(act, *addr, old.len());
                for i in 0..old.len() {
                    fin.insert(addr + i, cur.get(i).copied().unwrap_or(0));
                    pre.entry(addr + i).or_insert(old[i]);
                }
            }
            if !keep {
                for (addr, old) in writes.iter().rev() {
                    helpers::dm_write_range(act, *addr, old);
                }
            }
            (fin, len)
        };

        // Reset to the pristine pre-run state: restore every drifted slot (caller's and
        // callees') and roll the heap break back to `pre_len` (undo the run's `sbrk`
        // growth). `capture` has already rolled the dm byte writes back.
        let reset = |act: &mut Activation<'_, 'gc>| {
            slots_restore(act, &pre_slots);
            helpers::dm_restore_len(act, pre_len);
            act.clear_scope();
        };

        // JIT run 1 already executed with the log armed. Capture + roll back.
        let jit_slots = slots_snapshot(activation, &slot_ids);
        let (jit1, jit1_len) = capture(activation, &mut pre, false);
        reset(activation);

        // Interp run 1 — deliberately INTERLEAVED between the two JIT runs (order:
        // jit1, interp1, jit2, interp2). A monotonic external input read by the
        // method (e.g. `getTimer()`, ms resolution) then advances *inside* each
        // cluster, breaking jit1==jit2 agreement and getting filtered — whereas the
        // clustered order (jit,jit,interp,interp) let the clock tick exactly between
        // the clusters and framed Starling's `nextFrame` `_frameTimestamp` slot as a
        // fake divergence. A real deterministic JIT fault agrees within each cluster
        // regardless of interleaving, so detection is unaffected.
        helpers::dm_log_start();
        self.in_verify.set(true);
        let interp = activation.run_actions(method);
        self.in_verify.set(false);
        let interp_slots = slots_snapshot(activation, &slot_ids);
        let (i1, i1_len) = capture(activation, &mut pre, false);
        reset(activation);

        // JIT run 2 (bracket): a divergence only counts if both JIT runs agree, so a
        // cross-thread write to one of the JIT's *inputs* between runs can't frame it.
        // The context stack must MIRROR run 1's exactly — a missing table here (this
        // used to omit script-globals/push-strings/method) made run 2 read a stale
        // outer table, feeding e.g. `undefined` where run 1 saw a global object, and
        // a downstream `callmethod` on that bogus receiver panicked ("Method should
        // exist") — crashing the verify build before any divergence could be logged.
        helpers::dm_log_start();
        let ctx2 = helpers::RunCtx::new(
            activation,
            return_type,
            &compiled.mn_table,
            &compiled.script_globals,
            &compiled.push_strings,
            &compiled.coerce_classes,
            &compiled.natives,
            &compiled.namespaces,
            method,
        );
        let _ = helpers::with_run_ctx(&ctx2, || runner::run(bytes, regs, manifest));
        let _ = helpers::take_pending_error();
        let jit2_slots = slots_snapshot(activation, &slot_ids);
        let (jit2, jit2_len) = capture(activation, &mut pre, false);
        reset(activation);

        // Interp run 2 — runs last and is **kept**, so its side effects are the live
        // state and stay coherent even where the JIT diverges (it's the returned value).
        helpers::dm_log_start();
        self.in_verify.set(true);
        let interp2 = activation.run_actions(method);
        self.in_verify.set(false);
        let i2_slots = slots_snapshot(activation, &slot_ids);
        let (i2, i2_len) = capture(activation, &mut pre, true);

        let result_ok = matches!(&interp, Ok(v) if value_to_bits(*v) == result_bits);
        // First address where both JIT runs agree, both interp runs agree, and the two
        // clusters differ — a deterministic JIT fault the shared heap can't explain.
        // A run that didn't write an address holds the pristine byte there.
        let at = |m: &BTreeMap<usize, u8>, a: usize| {
            m.get(&a).copied().unwrap_or_else(|| pre.get(&a).copied().unwrap_or(0))
        };
        let mut addrs: BTreeSet<usize> = BTreeSet::new();
        for m in [&jit1, &jit2, &i1, &i2] {
            addrs.extend(m.keys().copied());
        }
        let dm_diff = addrs.into_iter().find_map(|a| {
            let (j1, j2, x1, x2) = (at(&jit1, a), at(&jit2, a), at(&i1, a), at(&i2, a));
            (j1 == j2 && x1 == x2 && j1 != x1).then_some((a, j1, x1))
        });
        // Heap-break (`sbrk` length) divergence — the JIT grows the heap to a
        // different length than the interpreter, on identical inputs (bracketed).
        let len_diff = (jit1_len == jit2_len && i1_len == i2_len && jit1_len != i1_len)
            .then_some((jit1_len, i1_len));
        // Slot divergence, bracketed like dm: a slot both JIT runs agree on, both
        // interp runs agree on, and the clusters differ. A run that didn't touch a
        // slot holds the baseline there. A slot holding an **object** (NaN-box tag
        // `0xFFFD`) in *both* clusters is a re-alloc artifact (each run allocates
        // fresh addresses), so it's filtered out.
        let is_obj = |b: u64| (b & 0xFFFF_0000_0000_0000) == 0xFFFD_0000_0000_0000;
        let slot_at = |m: &BTreeMap<usize, u64>, id: usize| {
            m.get(&id).copied().unwrap_or_else(|| pre_slots.get(&id).copied().unwrap_or(0))
        };
        let mut slot_ids: BTreeSet<usize> = BTreeSet::new();
        for m in [&jit_slots, &jit2_slots, &interp_slots, &i2_slots] {
            slot_ids.extend(m.keys().copied());
        }
        let slot_diff = slot_ids.into_iter().find_map(|id| {
            let (j1, j2, x1, x2) =
                (slot_at(&jit_slots, id), slot_at(&jit2_slots, id), slot_at(&interp_slots, id), slot_at(&i2_slots, id));
            (j1 == j2 && x1 == x2 && j1 != x1 && !(is_obj(j1) && is_obj(x1))).then_some((id, j1, x1))
        });
        // Return-value divergence — but two differing **object** returns are a re-alloc
        // artifact (the verify's slot rollback makes the interp re-run allocate a fresh
        // object with a different address), exactly like the slot filter. Trigger only
        // when the results aren't both objects (a real value/type/`undefined` mismatch).
        let interp_bits = if let Ok(v) = &interp { Some(value_to_bits(*v)) } else { None };
        let result_diff = match interp_bits {
            Some(ib) => ib != result_bits && !(is_obj(ib) && is_obj(result_bits)),
            None => true, // interp threw but the JIT returned a value (or vice versa)
        };
        if result_diff || slot_diff.is_some() || dm_diff.is_some() || len_diff.is_some() {
            self.mismatches.set(self.mismatches.get() + 1);
            let has_call = method.get_verified_info().parsed_code.iter().any(|o| {
                matches!(
                    o,
                    Op::CallMethod { .. } | Op::CallProperty { .. } | Op::CallPropVoid { .. } | Op::ConstructSuper { .. } | Op::Call { .. }
                )
            });
            let dm_desc = match &dm_diff {
                Some((addr, j, i)) => format!("dm_DIFF@{addr:#x} jit={j:#04x} interp={i:#04x}"),
                None => "dm_ok".to_string(),
            };
            let len_desc = match &len_diff {
                Some((j, i)) => format!(" len_DIFF jit={j:#x} interp={i:#x}"),
                None => String::new(),
            };
            let slot_desc = match &slot_diff {
                Some((id, j, i)) => format!(" slot_DIFF[{id}] jit={j:#x} interp={i:#x}"),
                None => String::new(),
            };
            let kind = match (slot_diff.is_some(), len_diff.is_some(), dm_diff.is_some(), has_call) {
                _ if result_diff => "RESULT",
                (true, _, _, _) => "SLOT",
                (_, true, _, _) => "LEN",
                (_, _, true, true) => "CALL-DM",
                (_, _, true, false) => "LEAF-DM",
                _ => "?",
            };
            let result_desc = if result_diff {
                format!(" result jit={result_bits:#x} interp={interp_bits:#x?}")
            } else {
                String::new()
            };
            tracing::error!(
                "JIT DIVERGENCE ({kind}) result_ok={result_ok}{result_desc} {dm_desc}{len_desc}{slot_desc} \
                 jit_slots={jit_slots:x?} interp_slots={interp_slots:x?} ops={:?}",
                &method.get_verified_info().parsed_code,
            );
        }

        Some(interp2)
    }
}
