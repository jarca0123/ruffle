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

#![feature(thread_local)]

use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet};

use fnv::{FnvHashMap, FnvHashSet};
use std::rc::Rc;

use ruffle_core::avm2::{Activation, Class, Error, JitBackend, Method, Op, TObject, Value};

pub mod analysis;
pub(crate) mod hoist;
pub(crate) mod direct;
pub mod inline;
pub mod emit;
pub(crate) mod helpers;
pub mod lower;
pub mod runner;
pub mod translate;
pub(crate) mod typed;

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
    /// Ops for the **direct-exec tier** — tiny straight-line methods run as a
    /// Rust match-loop over their `JitOp`s, skipping the wasm engine entirely
    /// (see [`direct`]). `None` = ineligible → the ordinary wasm path.
    direct_ops: Option<Rc<[lower::JitOp]>>,
    /// The method's `num_locals`, cached so `try_run` skips the per-call
    /// `method.body()` deref chain (a `Gc` pointer walk on EVERY invocation).
    num_locals: u32,
    /// This method's property-IC cache cells — one zeroed `u64` per
    /// [`lower::JitOp::GetPropertyIc`] site (`{ class_word: u32, slot_id: u32 }`).
    /// Held here so the buffer stays alive for the method's lifetime; behind an
    /// `Rc` so a `Compiled` clone shares the SAME address (the emitted code holds
    /// a raw offset into it). Empty for methods without IC sites.
    ic_cells: Rc<[std::cell::Cell<u64>]>,
    /// Memory-1 byte offset of `ic_cells` (the `ic_base` `run` param). 0 when the
    /// method has no IC site (`!has_ic`) — then no emitted code reads it.
    ic_base: u32,
    /// Whether this compiled method is a safe JIT→JIT **direct-call target**
    /// (`Manifest::directable`) — a caller's `emit_call_direct` may `call_indirect`
    /// its `run` without building an `Activation`. Consulted by the call-cache refill.
    directable: bool,
    /// The method's declared parameter count. A direct call sets up the callee frame
    /// itself (this + the call's args); it is only sound when the call provides
    /// EXACTLY the params (`argc == param_count` → no defaults to fill) and the method
    /// has no locals beyond them (`num_locals == param_count + 1` → no stale
    /// uninitialized temporaries). The refill checks both before caching a target.
    param_count: u32,
    /// Whether this method is a valid **direct-call target** w.r.t. its parameters: not
    /// variadic, at most [`lower::MAX_DIRECT_ARGC`] params, and every param a type the
    /// inline HIT guard can tag-check ([`helpers::param_check_words`] resolves — untyped
    /// `*`, `int`/`uint`/`Number`/`Boolean`/`String`/`Object`). A direct call writes the
    /// caller's args into the callee frame **uncoerced**; the guard verifies each already
    /// matches its param type (`Value::coerces_identically_to`) so that is bit-identical
    /// to the interpreter's coerced entry (`init_from_method` → `resolve_parameters`),
    /// else it misses to the coercing helper. A concrete-class or `void` param (needs an
    /// exact-class compare) keeps the callee on the helper. Checked before caching a target.
    all_params_directable: bool,
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
    /// The AVM2 method name, emitted into the amalgam module's wasm name section
    /// so profilers label each JIT-compiled method (otherwise every method shows
    /// as `wasm-function[N]` and its samples mis-attribute to its caller).
    name: Rc<str>,
    /// Per-local "is numeric on entry" (param types) — seeds check-elision when
    /// this method is re-emitted into a generation amalgam.
    numeric_seed: Rc<[bool]>,
    /// Resolved `Gc<Multiname>` pointers baked at getproperty sites, re-supplied when
    /// this method is re-emitted into an amalgam (see [`lower::compile_full`]).
    mn_list: Rc<[*const ()]>,
    /// Resolved `Class` pointers baked at `Coerce` sites, re-supplied on amalgam re-emit.
    coerce_classes: Rc<[*const ()]>,
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
            name: Rc::from(""),
            numeric_seed: Rc::from([] as [bool; 0]),
            mn_list: Rc::from([] as [*const (); 0]),
            coerce_classes: Rc::from([] as [*const (); 0]),
        }
    }
}

/// `Rc`: `try_run` clones the entry OUT of the cache map on every call (the
/// borrow can't be held across the run — compiling a callee re-enters the
/// cache). Cloning the former by-value `Compiled` (10 fields, 8 of them `Rc`)
/// per call showed up in `try_run`'s ~18% self-time on Starling gameplay
/// profiles; an `Rc` clone is one refcount bump.
type CacheEntry = Option<Rc<Compiled>>;

/// How many freshly compiled methods accumulate before they are folded into one
/// GENERATION module (web only). Each generation costs one `WebAssembly.Module`
/// + one instance + one entry-table slot for the whole batch — instead of one of
/// each **per method**, which exhausted the browser's page-granular
/// executable-code arena ("failed to allocate executable memory for module") and
/// the reserved entry-slot pool on method-heavy content (OpenTTD).
// SMALL amalgams: a method becomes a JIT→JIT direct-call target ONLY once it is an
// installed generation member (`WasmJit::direct_target` caches only a member's permanent
// dispatcher slot, never a recyclable standalone slot — see the stale-slot note there), so
// a small batch installs sooner → the direct-call cache warms up faster (measured: a large
// batch left Starling callees standalone and missing to the coercing helper for far too
// long → a busy/frame regression). The earlier bump to 256 chased a `WebAssembly.Instance`
// hotspot that turned out to be an unrelated import-binding bug (`pca` for `has_call_direct`
// went unbound → per-call re-instantiation), now fixed — so batch size is back to favoring
// warmup. The re-instantiation cost is gone, so more (smaller) generations is cheap.
const GEN_BATCH: usize = 32;

/// Decline a method whose emitted module exceeds this many bytes. A JIT module is one
/// wasm function plus small sections, and the browsers cap a **single function** at
/// ≈7.65 MB (`kV8MaxWasmFunctionSize` = 7_654_321; SpiderMonkey similar). Past that,
/// `WebAssembly.Module` rejects the module — and, fatally, rejects any GENERATION amalgam
/// that function is folded into (skipping the whole batch). Declining oversized methods
/// to the interpreter keeps amalgams valid and stops the per-call `Module()` retry. The
/// margin below the hard limit covers the (small) non-function module sections + the
/// amalgam's shared dispatcher/import overhead.
const MAX_EMITTED_MODULE_BYTES: usize = 7_000_000;

/// Inline-cache table size (power of two — the mask is `IC_SIZE - 1`). Direct-
/// mapped method-ptr → `Compiled`, fronting the `cache` HashMap on the hot path.
const IC_SIZE: usize = 4096;

/// Diagnostic: dump every call-bearing method the JIT has run when the FlasCC
/// allocator throws its `#1506` heap-corruption error. Set `true` to hunt the
/// CallMethod corruptor (needs `JIT_CALLMETHOD = true`); off for shipping.
const DUMP_ON_1506: bool = false;

/// Diagnostic: log each method's name + op count the first time the JIT decides
/// its fate (compiled vs declined). Grep the run for a specific method (e.g.
/// `F_lua_newstate`) to learn whether the JIT translated it or left it to the
/// interpreter. Off for shipping.
const LOG_JIT_METHODS: bool = false;

/// Diagnostic/bisection: decline to JIT any method whose name contains one of
/// these substrings, leaving it to the interpreter while the JIT stays on for
/// everything else. Used to localise a JIT miscompile: if declining method X
/// makes a fault vanish, X's translation is at fault; if it persists, X merely
/// inherited already-corrupt state from an upstream JIT'd method. Empty = ship.
const JIT_DENY_SUBSTRINGS: &[&str] = &[];

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
    /// Monomorphic-ish **inline cache** fronting `cache`: a direct-mapped table
    /// (method-ptr → its `Compiled`) so the hot per-call path skips the `HashMap`
    /// hash+probe (`WasmJit` dispatch was the single biggest JIT-entry cost on
    /// call-dense content). Only *successfully compiled* entries are cached (never
    /// a decline — those are rare in the hot path and must stay retry-able for the
    /// script-init deferral). Per-thread (each worker's `WasmJit` is thread-local),
    /// so no cross-thread aliasing. Kept in sync trivially: a key's `Compiled` `Rc`
    /// is set once and never changes identity.
    ic: RefCell<Box<[Option<(usize, Rc<Compiled>)>]>>,
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
    /// Histogram of *why* a JIT-compiled method is not a **direct-call target** (so its
    /// callers pay the `exec`/Activation call path instead of a `call_indirect`), keyed
    /// by [`lower::Manifest::directable_decline_reason`]. Which remaining feature to make
    /// directable next — turns the `exec` wall into data. See [`Self::record_nondirectable`].
    nondirectable: RefCell<FnvHashMap<&'static str, u32>>,
}

impl Default for WasmJit {
    fn default() -> Self {
        Self {
            cache: RefCell::new(FnvHashMap::default()),
            ic: RefCell::new(vec![None; IC_SIZE].into_boxed_slice()),
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
            nondirectable: RefCell::new(FnvHashMap::default()),
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

    /// Resolve a callee (by its `Method` `Gc` pointer) to a JIT→JIT direct-call
    /// target `(table_slot, ic_base, member_idx)` — it must be compiled,
    /// [`Compiled::directable`], have **tag-checkable params**
    /// ([`Compiled::all_params_directable`] — the HIT guard verifies each arg matches so
    /// the uncoerced write equals the interpreter), receive EXACTLY its params
    /// (`argc == param_count`,
    /// `num_locals == argc + 1`), and be a **generation member** holding its amalgam's
    /// (never-recycled) dispatcher slot — a standalone `run` slot is recycled when the
    /// method is amalgamated, so caching it risks a stale-slot misdispatch. `None`
    /// otherwise, so the call-cache refill records the sentinel and the site keeps taking
    /// the coercing helper. Called by the `cmi` helper via the `RunCtx`'s erased `self`.
    pub(crate) fn direct_target(&self, callee_ptr: usize, argc: u32) -> Option<(u32, u32, u32)> {
        let cache = self.cache.borrow();
        let compiled = cache.get(&callee_ptr)?.as_deref()?;
        // Sound only when the caller's direct-call frame — exactly `{this, argc args}` —
        // provides every param (no defaults to fill: `param_count == argc`). Extra locals
        // ARE allowed now: the callee prologue zeroes them to `undefined` on the direct
        // path (see `emit_body`), matching `init_from_method`. The frame must still fit one
        // nesting slice (`num_locals * 8 <= FRAME_STRIDE`); otherwise fall to the helper.
        if !compiled.directable
            || !compiled.all_params_directable
            || compiled.param_count != argc
            || compiled.num_locals < argc + 1
            || compiled.num_locals * 8 > lower::FRAME_STRIDE
        {
            return None;
        }
        // `run_index_for` returns only PERMANENT slots (a member's own slot, or its
        // generation dispatcher) — never the recyclable standalone slot — so whatever it
        // yields is safe to cache in a direct-call cell (no stale-slot misdispatch). A
        // callee thus becomes direct-callable once it is amalgamated; until then its callers
        // take the (correct) coercing helper. `member_idx == CALL_NO_MEMBER` here means the
        // member's own 6-param slot (no dispatcher); any other value is the 7-param
        // dispatcher fallback (pool exhausted).
        let (slot, member_idx) = runner::run_index_for(compiled.bytes.as_ptr() as usize)?;
        Some((slot, compiled.ic_base, member_idx))
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

    /// Records that a JIT-compiled method is NOT a direct-call target because of
    /// `reason` (see [`lower::Manifest::directable_decline_reason`]), logging the running
    /// histogram on a new reason or every 200. Names which feature to make directable
    /// next to shrink the `exec`/Activation wall.
    fn record_nondirectable(&self, reason: &'static str) {
        let (total, is_new) = {
            let mut m = self.nondirectable.borrow_mut();
            let is_new = !m.contains_key(reason);
            *m.entry(reason).or_insert(0) += 1;
            (m.values().sum::<u32>(), is_new)
        };
        if is_new || total.is_multiple_of(200) {
            let mut v: Vec<(&'static str, u32)> =
                self.nondirectable.borrow().iter().map(|(k, c)| (*k, *c)).collect();
            v.sort_by(|a, b| b.1.cmp(&a.1));
            let top: Vec<String> = v.iter().take(20).map(|(n, c)| format!("{n}={c}")).collect();
            tracing::info!("AVM2 direct-call declines ({total} methods): {}", top.join(", "));
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
    ) -> Option<Rc<Compiled>> {
        let key = method.as_ptr() as usize;
        if let Some(entry) = self.cache.borrow().get(&key) {
            return entry.clone();
        }
        // Bisection denylist: leave named methods to the interpreter (see
        // `JIT_DENY_SUBSTRINGS`).
        if !JIT_DENY_SUBSTRINGS.is_empty() {
            let name = method.method_name();
            if JIT_DENY_SUBSTRINGS.iter().any(|s| name.contains(s)) {
                if LOG_JIT_METHODS {
                    tracing::info!("JIT DENYLISTED {name}");
                }
                self.cache.borrow_mut().insert(key, None);
                return None;
            }
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
        // Init-ordering guard: don't compile a method that references a script
        // (`getscriptglobals`) which hasn't been initialized yet. Building the side
        // tables resolves script globals eagerly (`script_globals_table` →
        // `Script::globals`), which RUNS the script initializer — at method *entry*
        // rather than at the `getscriptglobals` op, reordering observable class/
        // script inits (the `lazyinit` / `loader_duplicate_class` tests). Let the
        // interpreter run this method once (initializing scripts in op order); a
        // later invocation compiles safely, when the eager resolution is a pure
        // cache hit. Retry-able, so DON'T cache the decline.
        if method
            .get_verified_info()
            .parsed_code
            .iter()
            .any(|op| matches!(op, Op::GetScriptGlobals { script } if !script.is_initialized()))
        {
            return None;
        }
        // Pre-resolve the script-globals / push-string `Value` bits HERE (we have the
        // `activation`); `compile_method` bakes them into `PushBits` constants. Both are
        // empty for methods without those ops. Scripts are initialized (the decline check
        // above guarantees it), so the resolution is a pure read.
        let script_globals = script_globals_table(activation, method);
        let push_strings = push_string_table(method);
        // The declared return-type class pointer (0 = none) — for baking `ReturnValueCoerced`.
        // Resolve like `try_run` does (idempotent/cached); `0` leaves the `coerce_return` path.
        let return_type: u64 = method
            .resolve_info(activation)
            .ok()
            .and_then(|_| method.resolved_return_type())
            // MUST match `coerce_class_table`'s erasure (the `Gc`/`GcBox` pointer the
            // `coerce` helper reverses) — NOT `Class::as_ptr()`, which is the data pointer
            // (offset by the GcBox header) and would make the helper read a bogus class.
            .map(|c| unsafe { std::mem::transmute::<Class<'gc>, *const ()>(c) } as usize as u64)
            .unwrap_or(0);
        let entry: CacheEntry = locals_typed_seed(activation, method)
            .and_then(|(int_seed, double_seed, numeric_seed)| {
                compile_method(
                    method,
                    &int_seed,
                    &double_seed,
                    &numeric_seed,
                    &script_globals,
                    &push_strings,
                    return_type,
                )
            })
            .map(|(bytes, manifest, mn_list, mut gen_src, directable)| {
                // Record the method name for the amalgam's wasm name section
                // (this is the only place with the `Method` in scope).
                gen_src.name = method.method_name().as_ref().into();
                // Diagnostic: a compiled-but-not-directable method still pays the
                // `exec`/Activation path on every call. Record WHY (the first blocker) so
                // the histogram names the next feature to make directable. `reason(true)`
                // (assume getslots null-safe) finds the first NON-getslot blocker; if it's
                // `None` yet the method isn't directable, the block was getslot null-safety.
                if !directable {
                    let reason = manifest
                        .directable_decline_reason(true)
                        .unwrap_or("getslot_not_null_safe");
                    self.record_nondirectable(reason);
                }
                // Property-IC cache: one zeroed cell per `GetPropertyIc` site. The
                // cells live in memory 1 (a normal Rust allocation IS in Ruffle's
                // single linear memory on wasm32), so the slice's data pointer IS
                // the `ic_base` memory-1 offset the emitted guard/miss address. Only
                // meaningful where the inline IC is emitted (`has_ic`, wasm32); the
                // pointer is truncated to 0 elsewhere (nothing reads it).
                // Shared per-method cache buffer, `[property-IC cells | call-cache
                // cells]` in memory 1. IC cell = `IC_CELL_SIZE` (8 bytes, 1 `u64`); call
                // cell = `CALL_CELL_SIZE` (16-byte header + per-arg check-words).
                // `ic_base` (the run param) points at the start; the call region begins
                // at `ic_sites * IC_CELL_SIZE` (see `emit_body`).
                let ic_sites = gen_src
                    .ops
                    .iter()
                    .filter(|op| matches!(op, lower::JitOp::GetPropertyIc(..)))
                    .count();
                let call_sites = gen_src
                    .ops
                    .iter()
                    .filter(|op| matches!(op, lower::JitOp::CallMethodDirect(..)))
                    .count();
                let total_u64 = ic_sites + call_sites * (lower::CALL_CELL_SIZE as usize / 8);
                let ic_cells: Rc<[std::cell::Cell<u64>]> =
                    (0..total_u64).map(|_| std::cell::Cell::new(0u64)).collect();
                let ic_base = if manifest.has_ic || manifest.has_call_direct {
                    ic_cells.as_ptr() as usize as u32
                } else {
                    0
                };
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
                    direct_ops: direct::eligible(&gen_src.ops).then(|| gen_src.ops.clone()),
                    num_locals: method.body().map(|b| b.num_locals).unwrap_or(0),
                    ic_cells,
                    ic_base,
                    directable,
                    param_count: method.resolved_param_config().len() as u32,
                    // Direct-call param gate: not variadic, ≤ MAX_DIRECT_ARGC params, and
                    // every param tag-checkable by the inline HIT guard. Each checkable
                    // param's uncoerced pass is guarded to be bit-identical to the
                    // interpreter (see `all_params_directable` / `param_check_words`).
                    all_params_directable: !method.is_variadic()
                        && method.resolved_param_config().len() as u32 <= lower::MAX_DIRECT_ARGC
                        && method
                            .resolved_param_config()
                            .iter()
                            .all(|p| helpers::param_check(p.param_type).is_some()),
                    gen_src,
                }
            })
            .map(Rc::new);
        // On the first (and only) miss for a method that declined, record why —
        // the first op the boxed path can't lower (`<compile>` if all ops are
        // supported but the lowering bailed). Skips body-less methods.
        let decline_reason = (entry.is_none() && method.body().is_some()).then(|| {
            let core_ops = &method.get_verified_info().parsed_code;
            translate::first_unsupported_boxed(core_ops).unwrap_or_else(|| "<compile>".to_string())
        });
        if let Some(reason) = &decline_reason {
            self.record_decline(reason.clone());
        }
        if LOG_JIT_METHODS {
            let op_count = method.body().map(|b| b.code.len()).unwrap_or(0);
            tracing::info!(
                "JIT {} {} (locals={}, ops={}){}",
                if entry.is_some() { "compiled" } else { "DECLINED" },
                method.method_name(),
                method.body().map(|b| b.num_locals).unwrap_or(0),
                op_count,
                decline_reason
                    .as_deref()
                    .map(|r| format!(" reason={r}"))
                    .unwrap_or_default(),
            );
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
        let (bytes, union, member_keys, directable) = {
            let cache = self.cache.borrow();
            let compiled: Vec<&Compiled> = keys
                .iter()
                .filter_map(|k| cache.get(k).and_then(|e| e.as_deref()))
                .collect();
            let members: Vec<lower::GenMember<'_>> = compiled
                .iter()
                .map(|c| lower::GenMember {
                    ops: &c.gen_src.ops,
                    switches: &c.gen_src.switches,
                    exceptions: &c.gen_src.exceptions,
                    name: &c.gen_src.name,
                    numeric_seed: &c.gen_src.numeric_seed,
                    mn_list: &c.gen_src.mn_list,
                    coerce_classes: &c.gen_src.coerce_classes,
                })
                .collect();
            let Some((bytes, union)) = lower::compile_generation(&members) else {
                return;
            };
            // The runner keys methods by their (cached, never-moving) module bytes.
            let member_keys: Vec<usize> =
                compiled.iter().map(|c| c.bytes.as_ptr() as usize).collect();
            // Which members are worth their own permanent direct-call slot: the
            // method-intrinsic direct-call gates (the per-call `argc` gates are checked in
            // `direct_target`). Parallel to `member_keys`.
            let directable: Vec<bool> = compiled
                .iter()
                .map(|c| c.directable && c.all_params_directable)
                .collect();
            (bytes, union, member_keys, directable)
        };
        if runner::install_generation(&bytes, &union, &member_keys, &directable) {
            // The members' generation sources (per-method `JitOp`/switch/exception
            // copies) exist only to re-emit this batch — dead weight once the
            // generation is live. Drop them; the standing JIT memory then stays
            // bounded. (The per-method module `bytes` must stay allocated: the
            // runner keys generation members by their address.)
            let mut cache = self.cache.borrow_mut();
            for k in &keys {
                // `Rc::get_mut` fails only while a (nested) run still holds the
                // entry's clone — then that member just keeps its sources
                // (memory-only, bounded).
                if let Some(Some(c)) = cache.get_mut(k) {
                    if let Some(c) = Rc::get_mut(c) {
                        c.gen_src = GenSource::empty();
                    }
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
) -> Option<(Vec<bool>, Vec<bool>, Vec<bool>)> {
    let num_locals = method.body()?.num_locals as usize;
    method.resolve_info(activation).ok()?;
    let params = method.resolved_param_config();
    let class_defs = activation.avm2().class_defs();
    let (int_class, number_class, uint_class) =
        (class_defs.int, class_defs.number, class_defs.uint);

    let mut int_seed = vec![false; num_locals];
    let mut double_seed = vec![false; num_locals];
    // For check-elision: `int`, `uint`, AND `Number` params are all provably numeric
    // (safe to skip the `is_numeric` guard). This is broader than the int/double
    // path seeds — `uint` is numeric even though it can't use the signed-`i32` path.
    let mut numeric_seed = vec![false; num_locals];
    for (i, param) in params.iter().enumerate() {
        let slot = i + 1; // local 0 is `this`
        if slot < num_locals {
            match param.param_type {
                // `uint` is deliberately NOT seeded int: the raw-i32 model compares
                // signed (`I32LtS`) and re-boxes as `Integer`, but a `uint > i32::MAX`
                // is a positive `Number` in AS3 — signed compare / int-box would
                // diverge. `uint` methods fall to the boxed path (CoerceU/abstract_lt).
                Some(c) if c == int_class => {
                    int_seed[slot] = true;
                    numeric_seed[slot] = true;
                }
                Some(c) if c == number_class => {
                    double_seed[slot] = true;
                    numeric_seed[slot] = true;
                }
                Some(c) if c == uint_class => numeric_seed[slot] = true,
                _ => {}
            }
        }
    }
    Some((int_seed, double_seed, numeric_seed))
}

/// Translates, type-checks, and compiles a method. `None` if any op is
/// unsupported or the raw-`i32` model would be unsound for `seed`.
fn compile_method<'gc>(
    method: Method<'gc>,
    int_seed: &[bool],
    double_seed: &[bool],
    numeric_seed: &[bool],
    // Pre-resolved `Value` bits per `GetScriptGlobals` / `PushString` op (built by the
    // caller, which has the activation). The boxed path rewrites those ops to
    // `JitOp::PushBits` so they bake to a constant and stop blocking directability.
    script_globals: &[u64],
    push_strings: &[u64],
    // The method's resolved return-type `Class` pointer (0 = none/`*`). Non-zero →
    // `ReturnValueCoerced` is rewritten to `ReturnValueCoerceBaked` so a typed-return
    // method becomes directable (bakes the class, drops the `coerce_return` helper).
    return_type: u64,
) -> Option<(Rc<[u8]>, lower::Manifest, Rc<[*const ()]>, GenSource, bool)> {
    method.body()?;
    let core_ops = &method.get_verified_info().parsed_code;

    // `mn_list`: the combined multiname pointer list (see [`Compiled`]). The fast paths
    // read no properties → empty; the boxed path passes the caller's + inlined callees'.
    // `directable` = safe as a JIT→JIT direct-call target (see `Manifest::directable`).
    // `all_getslots_null_safe`: every `GetSlot` on the FINAL ops is verifier-proven
    // not-null → the activation-reading throw path of `helpers::get_slot` is dead, so
    // a leaf whose only "helper" is getslot is a safe direct-call target
    // ([`lower::Manifest::directable`]). The fast paths have no getslots (vacuously
    // true); the boxed path computes it below.
    let finish = |ops: &[lower::JitOp],
                  switches: &[lower::SwitchTable],
                  exceptions: &[lower::ExcRange],
                  mn_list: Vec<*const ()>,
                  coerce_classes: Vec<*const ()>,
                  all_getslots_null_safe: bool|
     -> Option<(Rc<[u8]>, lower::Manifest, Rc<[*const ()]>, GenSource, bool)> {
        // Share one `Rc` for the three consumers: `compile_full` (bakes `mn_list[k]` at
        // getproperty sites), the `GenSource` (re-emitted into an amalgam), and the
        // returned `mn_table`. `coerce_classes` likewise feeds the `Coerce`-site bake.
        let mn_list: Rc<[*const ()]> = mn_list.into();
        let coerce_classes: Rc<[*const ()]> = coerce_classes.into();
        let bytes: Rc<[u8]> = Rc::from(
            lower::compile_full(ops, switches, exceptions, numeric_seed, &mn_list, &coerce_classes)?
                .into_boxed_slice(),
        );
        // A method that emits a single wasm function larger than the browser's
        // per-function size limit (V8/SpiderMonkey ≈ 7.65 MB) can't be validated by
        // `WebAssembly.Module` — and worse, it POISONS any generation amalgam it joins
        // (the whole batch's install is rejected: "size … > maximum function size …"),
        // and the per-method path retries `Module()` on every call. A few enormous
        // CrossBridge C/C++ functions (OpenTTD, Lua's `F_luaV_execute`) hit this. The
        // module is one function plus small sections, so `bytes.len()` tracks it; decline
        // (→ interpreter) below the limit with margin, so amalgams stay valid.
        if bytes.len() > MAX_EMITTED_MODULE_BYTES {
            tracing::warn!(
                "AVM2 JIT: method emits {} bytes (> {} per-function limit) — declining to interpreter",
                bytes.len(),
                MAX_EMITTED_MODULE_BYTES,
            );
            return None;
        }
        let gen_src = GenSource {
            ops: ops.into(),
            switches: switches.into(),
            exceptions: exceptions.into(),
            // Filled in by the caller (which has the `Method` for `method_name`).
            name: Rc::from(""),
            numeric_seed: numeric_seed.into(),
            mn_list: mn_list.clone(),
            coerce_classes: coerce_classes.clone(),
        };
        let manifest = lower::manifest(ops);
        let directable = manifest.directable(all_getslots_null_safe);
        Some((bytes, manifest, mn_list, gen_src, directable))
    };

    // Int fast path (raw i32) — only when provably int-sound. Pure register/frame
    // arithmetic, no helpers → a direct-call target.
    if let Some(ops) = translate::translate(core_ops) {
        if analysis::int_sound(&ops, int_seed) {
            return finish(&ops, &[], &[], Vec::new(), Vec::new(), true);
        }
    }

    // Double fast path (unboxed f64, inline arithmetic) — only when Number-sound.
    // Also pure numeric → a direct-call target.
    if let Some(ops) = translate::translate_double(core_ops) {
        if analysis::double_sound(&ops, double_seed) {
            return finish(&ops, &[], &[], Vec::new(), Vec::new(), true);
        }
    }

    // The method's exception handlers (op-index ranges). A throw — explicit or from
    // a throwing call/dm — inside a range dispatches through the handler table.
    let exceptions = exc_ranges(method);

    // Boxed (GC-aware) path — raw `Value`s + imported helper `call`s.
    let (mut ops, switches) =
        translate::translate_boxed(core_ops, &method.get_verified_info().number_slots)?;
    // Whether every `GetSlot` is verifier-proven not-null — computed HERE, while op
    // indices are still 1:1 with `parsed_code` (the `null_safe_getslots` indices refer
    // to those). A direct-call-eligible method is a leaf (no calls → inlining never
    // fires) and `hoist` only ever moves already-null-safe getslots, so this survives
    // both passes. Gates `getslot` leaves into `directable`.
    let all_getslots_null_safe = {
        let null_safe = &method.get_verified_info().null_safe_getslots;
        ops.iter().enumerate().all(|(i, op)| {
            !matches!(op, lower::JitOp::GetSlot(..)) || null_safe.contains(&(i as u32))
        })
    };
    // The caller's multiname pointers; the inline pass appends each inlined callee's
    // (and remaps that callee's `k`s), keeping this list in sync with the final ops.
    let mut mn_list = multiname_table(method);
    // Loop-invariant `getslot` hoisting (Starling's transform loop re-reads six
    // matrix fields per iteration). Runs BEFORE inlining, while op indices are
    // still 1:1 with `parsed_code` (the verifier's null-safe indices refer to
    // those); its scratch locals sit right above `num_locals`, so the inline
    // pass's callee locals start above them. Gated like inlining: splicing
    // shifts indices, which a switch side-table / exception ranges can't follow.
    let mut hoisted_locals = 0;
    if JIT_HOIST && switches.is_empty() && exceptions.is_empty() {
        let num_locals = method.body().map(|b| b.num_locals).unwrap_or(0);
        let null_safe = &method.get_verified_info().null_safe_getslots;
        // Guard: a null-safe index must still point at a GetSlot (dead-code
        // elimination can shift a removed op's remapped index onto a neighbor).
        let null_safe: Vec<u32> = null_safe
            .iter()
            .copied()
            .filter(|&i| matches!(ops.get(i as usize), Some(lower::JitOp::GetSlot(_, _))))
            .collect();
        hoisted_locals = hoist::hoist_pass(&mut ops, &null_safe, num_locals, MAX_FRAME_SLOTS);
    }
    // Inline pass: splice statically-resolvable, small callees so their calls become
    // wasm-internal, avoiding the per-invocation `try_run` overhead. Correct either way
    // — a failed resolve/inline leaves the call helper. Skipped when the method has a
    // `lookupswitch`: inlining shifts op indices, which would desync the switch
    // side-table's (absolute) targets — not worth remapping for a rare combination.
    // Also skipped when the method has exception handlers: their op-index ranges
    // must stay aligned with `core_ops` (inlining shifts indices).
    if JIT_INLINE && switches.is_empty() && exceptions.is_empty() {
        inline_pass(&mut ops, &mut mn_list, method, hoisted_locals);
    }
    // Property-IC `site` ids must be a dense `0..n` over the FINAL op stream so each
    // maps to a distinct cache cell. Inlining splices a callee's sites (also `0..`)
    // into the caller, colliding — a global renumber after all splicing restores
    // uniqueness. Idempotent for un-inlined methods (translate already numbers them
    // sequentially) and a no-op where no IC is emitted (native).
    renumber_cache_sites(&mut ops);
    // Bake `GetScriptGlobals`/`PushString` to a constant `PushBits`: their pre-resolved
    // `Value` bits are known now (the script is initialized — else `compiled` declined —
    // and string atoms are pool-interned; both are non-moving-GC and alive for the
    // method's lifetime). This drops the `h17`/`h18` helper, so a method whose only
    // "unsafe" helper was one of these becomes a direct-call target. Inlining never
    // splices these ops (see `inline`), so their `k`s still index the caller's tables.
    for op in ops.iter_mut() {
        match *op {
            lower::JitOp::GetScriptGlobals(k) => {
                if let Some(&bits) = script_globals.get(k as usize) {
                    *op = lower::JitOp::PushBits(bits);
                }
            }
            lower::JitOp::PushString(k) => {
                if let Some(&bits) = push_strings.get(k as usize) {
                    *op = lower::JitOp::PushBits(bits);
                }
            }
            // Typed return: bake the return-type class so `coerce` (already self-contained)
            // handles it — no `coerce_return`/`RUN_CTX.return_type` → directable.
            lower::JitOp::ReturnValueCoerced if return_type != 0 => {
                *op = lower::JitOp::ReturnValueCoerceBaked(return_type);
            }
            _ => {}
        }
    }
    // Decline helper-dominated methods: when a boxed method is mostly JS-boundary
    // crossings (getproperty/getslot/callmethod/generic helpers) with little inline
    // compute, the interpreter's fast native dispatch beats the JIT's per-call reg
    // copy + per-op boundary crossings. Falling back to the interpreter is always
    // correct — this only trades a JIT loss for the faster path.
    if DECLINE_HELPER_DOMINATED && lower::helper_dominated(&ops) {
        return None;
    }
    // Boxed path: not a direct-call target in v1 (its `getslot`/`dm`/coerce helpers
    // are activation-coupled; the null-safe-getslot widening comes later).
    finish(&ops, &switches, &exceptions, mn_list, coerce_class_table(method), all_getslots_null_safe)
}

/// Re-assign property-IC and direct-call `site` indices to dense `0..n` sequences
/// over the final op stream (see the call site). Each op family is one-to-one with
/// its region of the per-method cache buffer, so the two counters are independent.
fn renumber_cache_sites(ops: &mut [lower::JitOp]) {
    let mut ic = 0u32;
    let mut call = 0u32;
    for op in ops.iter_mut() {
        match op {
            lower::JitOp::GetPropertyIc(_, site) => {
                *site = ic;
                ic += 1;
            }
            lower::JitOp::CallMethodDirect(_, _, _, site) => {
                *site = call;
                call += 1;
            }
            _ => {}
        }
    }
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

/// Whether the loop-invariant `getslot` hoist pass runs (see [`hoist`]).
const JIT_HOIST: bool = true;

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
    // Scratch locals the hoist pass consumed above `num_locals` — the inlined
    // callees' locals start above THOSE.
    extra_locals: u32,
) {
    let Some(bound) = method.bound_class() else { return };
    let Some(caller_locals) = method.body().map(|b| b.num_locals) else { return };
    let base = caller_locals + extra_locals;
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
    let (callee, callee_switches) = translate::translate_boxed(&info.parsed_code, &info.number_slots)?;
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
            | Op::FindPropStrict { multiname }
            | Op::FindProperty { multiname }
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
            // SAFETY: `Class` is a single-`Gc` wrapper (one pointer); the class is
            // alive for the run, so erasing its pointer is sound (the `coerce`
            // helper reverses it within the same run). `NewClass`, `NewActivation`
            // and `NewFunction` share the table (and translate's `next_coerce`
            // counter) — the `NewFunction` entries are erased `Method`s (the same
            // pointer-sized `Gc` wrapper shape), reversed by `method_at`.
            Op::Coerce { class }
            | Op::NewClass { class }
            | Op::NewActivation { activation_class: class } => {
                Some(unsafe { std::mem::transmute::<Class<'gc>, *const ()>(*class) })
            }
            Op::NewFunction { method } => {
                Some(unsafe { std::mem::transmute::<Method<'gc>, *const ()>(*method) })
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

        // Every per-method side table (multinames, script globals, strings, coerce
        // classes) is built once at compile time and cached in `Compiled` — the hot
        // path below only installs the cached slices (incl. `num_locals`, so the
        // per-call `method.body()` deref chain is gone).
        //
        // Inline cache: try the direct-mapped `ic` slot first, so a hot method skips
        // the `cache` HashMap hash+probe entirely. On a miss (or the slot holds a
        // different method), fall through to `compiled()` and refill the slot — but
        // only with a real `Compiled` (`?` already returned for a decline), so a
        // retry-able decline is never cached here.
        let key = method.as_ptr() as usize;
        let slot = (key >> 4) & (IC_SIZE - 1);
        let compiled = {
            let hit = match &self.ic.borrow()[slot] {
                Some((k, c)) if *k == key => Some(c.clone()),
                _ => None,
            };
            match hit {
                Some(c) => c,
                None => {
                    let c = self.compiled(activation, method)?;
                    self.ic.borrow_mut()[slot] = Some((key, c.clone()));
                    c
                }
            }
        };
        let (bytes, manifest) = (&compiled.bytes, &compiled.manifest);
        let num_locals = compiled.num_locals as usize;

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

        // (`key` was computed above for the inline-cache probe.)

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
            self as *const WasmJit as *const (),
        );
        let result_bits = helpers::with_run_ctx(&ctx, || match &compiled.direct_ops {
            // Direct-exec tier: a tiny straight-line method runs as a Rust
            // match-loop over its ops — same helpers, same side tables, no wasm
            // engine (whose per-call setup dominates 3-op accessors like `LI8`).
            Some(ops) => direct::run(ops, regs),
            None => runner::run(bytes, regs, manifest, compiled.ic_base),
        })?;

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
            self as *const WasmJit as *const (),
        );
        let _ = helpers::with_run_ctx(&ctx2, || runner::run(bytes, regs, manifest, compiled.ic_base));
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
