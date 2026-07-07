//! Execution of a compiled JIT module.
//!
//! - **Native** (desktop / tests): the emitted module is compiled to real
//!   machine code by **wasmtime** (cranelift) and its `run(state_ptr) -> i64`
//!   is called. The frame is *copied* into the instance's WASM memory:
//!   registers `[0..num_locals]` are written as 8-byte `Value` slots at
//!   offset 0, `run(0)` is called, and the returned `i64` is the result
//!   `Value`'s bits. No copy-back is needed — the method's frame is discarded
//!   once it returns, so only the return value matters. (wasmi — an
//!   interpreter, ~40x slower on exception-heavy content — remains only in the
//!   lowering unit tests.)
//! - **Web**: execution goes through the browser's own WASM engine, with the
//!   helper imports wired to a JS trampoline back into Ruffle's wasm (see the
//!   `wasm32` `run` below).

/// Runs the compiled module `bytes` with `regs` as the initial frame slots
/// (register `i` = `regs[i]`, an 8-byte `Value` bit pattern), returning the
/// result `Value`'s bits. The [`Manifest`](crate::lower::Manifest) records which
/// helper imports the module declares; they are bound (in `lower::compile`'s
/// order) to [`crate::helpers`], which reach the current activation via
/// [`crate::helpers::with_activation`] set by the caller. `None` if the module
/// fails to compile/instantiate/run.
/// A ready-to-call wasmtime instance for one compiled method: the store owns the
/// instance, its frame memory, and the bound helper imports; `run` is the typed
/// export. Pooled per method and reused across calls — the module has **no
/// globals**, so its only state is the frame memory, which every call overwrites
/// (registers at offset 0) before running.
#[cfg(not(target_arch = "wasm32"))]
struct PooledRun {
    store: wasmtime::Store<()>,
    memory: wasmtime::Memory,
    run: wasmtime::TypedFunc<(i32, i32, i32, i32, i32), i64>,
}

// The wasmtime state lives in `ManuallyDrop`: a thread's TLS destructors must
// NOT tear it down. Dropping a `Module`/`Store` deregisters its unwind info
// (`CodeMemory` → `UnwindRegistration` → libgcc `__deregister_frame`), and when
// that TLS destructor races the process's own `exit()` (libtest calls
// `process::exit` while a player thread is still unwinding its TLS), libgcc
// aborts — seen as flaky post-"test result: ok" SIGABRTs across random SWF
// tests. Leaking the pool at thread exit is harmless: the memory goes away
// with the process, and long-lived player threads keep using theirs.
#[cfg(not(target_arch = "wasm32"))]
thread_local! {
    /// The single shared wasmtime engine — modules and stores must share one.
    static NATIVE_ENGINE: std::mem::ManuallyDrop<wasmtime::Engine> =
        std::mem::ManuallyDrop::new(wasmtime::Engine::default());
    /// Compiled wasmtime modules, keyed by `bytes.as_ptr()` (stable/unique — the
    /// `Rc<[u8]>` lives in `WasmJit`'s never-evicted cache, mirroring the web
    /// `MODULE_CACHE`). `Module::new` (validation + translation) is the costly
    /// step and used to run **per call** — hot methods called thousands of times
    /// (the avmplus `Date` suites, `rng`) re-translated their module every
    /// invocation and blew the script-execution time limit.
    static NATIVE_MODULES: std::mem::ManuallyDrop<
        std::cell::RefCell<fnv::FnvHashMap<usize, wasmtime::Module>>,
    > = std::mem::ManuallyDrop::new(std::cell::RefCell::new(fnv::FnvHashMap::default()));
    /// Ready instances per method (same key). Popped for the duration of a call
    /// and pushed back after, so a re-entrant call (a helper running AS3 that
    /// re-enters the same JIT'd method) finds the pool empty and builds a fresh
    /// instance instead of aliasing a live frame. Building one
    /// (`Store` + `Memory` + ~a dozen `Func::wrap` + `Instance::new`) per CALL —
    /// millions of them in the avmplus mops range tests, which throw/catch
    /// `#1506` ~800k times through JIT'd `LI*` — was the dominant cost; pooled,
    /// a call is a regs memcpy + the call.
    static NATIVE_INSTANCES: std::mem::ManuallyDrop<
        std::cell::RefCell<fnv::FnvHashMap<usize, Vec<PooledRun>>>,
    > = std::mem::ManuallyDrop::new(std::cell::RefCell::new(fnv::FnvHashMap::default()));
}

#[cfg(not(target_arch = "wasm32"))]
pub fn run(bytes: &[u8], regs: &[u64], m: &crate::lower::Manifest) -> Option<u64> {
    let key = bytes.as_ptr() as usize;
    let pooled = NATIVE_INSTANCES.with(|p| p.borrow_mut().get_mut(&key).and_then(Vec::pop));
    let mut pr = match pooled {
        Some(pr) => pr,
        None => build_instance(bytes, m)?,
    };

    // Write the frame registers at offset 0. `Value` bit patterns are written
    // little-endian; on LE hosts that's the in-memory representation, so write
    // the slice directly.
    #[cfg(target_endian = "little")]
    // SAFETY: viewing `[u64]` as bytes is always valid.
    let buf = unsafe { std::slice::from_raw_parts(regs.as_ptr().cast::<u8>(), regs.len() * 8) };
    #[cfg(not(target_endian = "little"))]
    let buf = &{
        let mut v = Vec::with_capacity(regs.len() * 8);
        for r in regs {
            v.extend_from_slice(&r.to_le_bytes());
        }
        v
    }[..];
    pr.memory.write(&mut pr.store, 0, buf).ok()?;

    // `run(state_ptr, dm_base, dm_len, regs_ptr, regs_len)`. Native production
    // modules use the helper path for domainMemory (never `has_dm`) and the frame
    // was written into the instance memory above (no register-copy prologue on
    // native), so everything but `state_ptr` is unused → 0.
    let result = pr.run.call(&mut pr.store, (0, 0, 0, 0, 0)).ok();
    // Return the instance to the pool even on a trap — a trap doesn't poison
    // the store, and the next call rewrites the frame anyway.
    NATIVE_INSTANCES.with(|p| p.borrow_mut().entry(key).or_default().push(pr));
    result.map(|v| v as u64)
}

/// Builds a [`PooledRun`] for `bytes`: compiles the module (cached), creates a
/// store + frame memory, binds the helper imports, and instantiates.
#[cfg(not(target_arch = "wasm32"))]
fn build_instance(bytes: &[u8], m: &crate::lower::Manifest) -> Option<PooledRun> {
    use wasmtime::{Extern, Func, Instance, Memory, MemoryType, Module, Store};

    let engine = NATIVE_ENGINE.with(Clone::clone);
    let module = NATIVE_MODULES.with(|cache| {
        match cache.borrow_mut().entry(bytes.as_ptr() as usize) {
            std::collections::hash_map::Entry::Occupied(e) => Some(e.get().clone()),
            std::collections::hash_map::Entry::Vacant(v) => {
                let module = Module::new(&engine, bytes).ok()?;
                Some(v.insert(module).clone())
            }
        }
    })?;
    let mut store = Store::new(&engine, ());

    // One 64 KiB page holds 8192 slots — far more than any real frame.
    let memory = Memory::new(&mut store, MemoryType::new(1, None)).ok()?;

    // Imports in declaration order — matching `lower::compile`: arity-1 helpers
    // h0..h{N-1}, then (if used) arity-2 `gp`/`gs`, then the used arity-3 set
    // helpers in kind order, then the memory.
    let mut externs: Vec<Extern> = Vec::new();
    for i in 0..m.num_helpers as usize {
        let helper = *crate::helpers::HELPERS.get(i)?;
        externs.push(Func::wrap(&mut store, move |a: i64| -> i64 { helper(a) }).into());
    }
    if m.has_getprop {
        externs.push(
            Func::wrap(&mut store, |recv: i64, k: i64| -> i64 {
                crate::helpers::get_property(recv, k)
            })
            .into(),
        );
    }
    if m.has_getslot {
        externs.push(
            Func::wrap(&mut store, |recv: i64, slot_id: i64| -> i64 {
                crate::helpers::get_slot(recv, slot_id)
            })
            .into(),
        );
    }
    if m.has_getprop_fast {
        externs.push(
            Func::wrap(&mut store, |recv: i64, name: i64, k: i64| -> i64 {
                crate::helpers::get_property_fast(recv, name, k)
            })
            .into(),
        );
    }
    for i in 0..m.num_helpers2 as usize {
        let helper = *crate::helpers::HELPERS2.get(i)?;
        externs.push(Func::wrap(&mut store, move |a: i64, b: i64| -> i64 { helper(a, b) }).into());
    }
    for (k, &helper) in crate::helpers::HELPERS3.iter().enumerate() {
        if m.set3_mask & (1 << k) != 0 {
            externs.push(
                Func::wrap(&mut store, move |r: i64, v: i64, imm: i64| -> i64 { helper(r, v, imm) })
                    .into(),
            );
        }
    }
    // Call imports in layout order: `cm` (call_method), `cp` (call_property), then
    // the shared `pca` (arg spill) and `perr` (error bail).
    if m.has_call {
        externs.push(
            Func::wrap(&mut store, |r: i64, id: i64, argc: i64| -> i64 {
                crate::helpers::call_method(r, id, argc)
            })
            .into(),
        );
    }
    if m.has_callprop {
        externs.push(
            Func::wrap(&mut store, |r: i64, k: i64, argc: i64| -> i64 {
                crate::helpers::call_property(r, k, argc)
            })
            .into(),
        );
    }
    if m.has_construct_super {
        externs.push(
            Func::wrap(&mut store, |r: i64, argc: i64| -> i64 {
                crate::helpers::construct_super(r, argc)
            })
            .into(),
        );
    }
    if m.has_call_value {
        externs.push(
            Func::wrap(&mut store, |f: i64, r: i64, argc: i64| -> i64 {
                crate::helpers::call_value(f, r, argc)
            })
            .into(),
        );
    }
    let any_call =
        m.has_call || m.has_callprop || m.has_construct_super || m.has_call_value || m.has_vcall;
    if any_call {
        externs.push(Func::wrap(&mut store, |v: i64| crate::helpers::push_call_arg(v)).into());
    }
    // `perr` (pending_error) — see `Manifest::needs_perr` (the single source of
    // truth): every op that throws out of band via `PENDING_ERROR` is followed by a
    // `perr` check in the emitted code, bailing promptly.
    if m.needs_perr() {
        externs.push(Func::wrap(&mut store, || -> i32 { crate::helpers::pending_error() }).into());
    }
    // `coerce` (arity-2 `(value, class_idx) -> result`) follows `perr`.
    if m.has_coerce {
        externs.push(
            Func::wrap(&mut store, |v: i64, k: i64| -> i64 { crate::helpers::coerce(v, k) }).into(),
        );
    }
    // `vc` (arity-4, the generic variadic helper) follows `coerce`.
    if m.has_vcall {
        externs.push(
            Func::wrap(&mut store, |a: i64, imm: i64, spill: i64, kind: i64| -> i64 {
                crate::helpers::vcall(a, imm, spill, kind)
            })
            .into(),
        );
    }
    externs.push(memory.into());

    let instance = Instance::new(&mut store, &module, &externs).ok()?;
    // (The inline dm path + memory 1 is exercised by `lower::tests::lowers_dm_inline`.)
    let run = instance
        .get_typed_func::<(i32, i32, i32, i32, i32), i64>(&mut store, "run")
        .ok()?;
    Some(PooledRun { store, memory, run })
}

/// Web execution: hand the emitted bytes to the browser's own WASM engine.
///
/// Compiles `bytes` via `WebAssembly.Module` (**cached per method** — codegen is
/// the costly step), backs the frame with a fresh `WebAssembly.Memory` (the module
/// imports it as `("env","memory")`), writes `regs` at offset 0, wires the helper
/// imports to a **JS trampoline** back into Ruffle's own wasm (see below),
/// instantiates, and calls the exported `run(state_ptr) -> i64`. Returns the
/// result `Value`'s bits, or `None` if anything in the pipeline fails (e.g. a
/// strict CSP without `wasm-unsafe-eval`), keeping the interpreter authoritative.
/// Only the module is cached; instantiation is per-call so recursion is safe.
///
/// ## Helper trampoline
/// The emitted module imports the same helpers the native path binds — arity-1
/// `h{i}` and arity-2 `gp`/`gs`. The browser runs the module in a *separate*
/// `WebAssembly.Instance`, so its imports must be JS callables; we build them from
/// Rust closures (`Closure`) that call [`crate::helpers`] directly. The `i64`
/// arguments/results cross the boundary as `BigInt` (WASM JS-BigInt-integration).
/// This is a synchronous, single-threaded re-entry into Ruffle's wasm while
/// [`crate::WasmJit::try_run`]'s `with_activation`/`with_multinames` are still on
/// the stack, so the helpers' thread-locals are valid — exactly as on native. The
/// closures are kept alive until after the call returns.
///
/// This mirrors the native (copy-based) path, but with **one shared frame memory**
/// reused across all calls, so there's no per-call allocation (GC) and no
/// per-method retention (which — since Firefox reserves a large virtual region per
/// `WebAssembly.Memory` *regardless of `maximum`* — exhausts address space:
/// "failed to reserve a large virtual memory region", a V8 abort/SIGILL on
/// Chromium). Each JIT nesting level uses a disjoint slice at `depth * STRIDE`, so
/// recursion (via `callmethod`, getters, …) can't alias frames, and instances are
/// cached per method (`Rc`, re-entrant-safe — WASM funcs run at any `state_ptr`).
/// A later zero-copy step (importing Ruffle's own memory + the real frame pointer)
/// would drop the reg copy but is web-only/untested.
#[cfg(target_arch = "wasm32")]
mod web {
    use js_sys::{Object, Reflect, Uint8Array, WebAssembly};
    use std::cell::{Cell, RefCell};
    use fnv::FnvHashMap;
    use std::rc::Rc;
    use wasm_bindgen::{JsCast, JsValue};

    /// Bytes per frame in the shared memory; a method's `num_locals` slots must fit
    /// (checked — else it declines). 512 slots ≫ any real `num_locals` (the operand
    /// stack lives on the WASM value stack, not here).
    const STRIDE: usize = 4096;
    /// Shared memory size in 64 KiB pages — fully committed, bounded. `MEM_BYTES /
    /// STRIDE` = max JIT nesting depth (1024) before a run declines.
    const PAGES: u32 = 64;
    const MEM_BYTES: usize = PAGES as usize * 65536;

    /// A cached instance. `run` (the exported function) is kept alive so its
    /// instance — and its slot in Ruffle's indirect function table — stays valid;
    /// `run_index` is that table slot, which [`run`] calls **wasm→wasm** via a
    /// transmuted fn-pointer (`call_indirect`, no JS trampoline, no i64→BigInt). The
    /// helper imports are likewise Ruffle's own wasm funcrefs (nothing to keep alive).
    /// No per-instance memory — all share the single [`memory`].
    struct Cached {
        run: js_sys::Function,
        /// The `run` export's slot in Ruffle's indirect function table, if it could be
        /// registered — then [`run`] calls it wasm→wasm. `None` (table not growable)
        /// falls back to the JS `run.call3` path (correct, just slower).
        run_index: Option<usize>,
    }

    /// A live GENERATION instance (see [`crate::lower::compile_generation`]): ONE
    /// module holding many methods behind a 6-param dispatcher export
    /// `run(method_idx, state_ptr, dm_base, dm_len, regs_ptr, regs_len)`. One
    /// instance + one reserved-table slot serve every member — the per-method
    /// modules/instances/slots they replace are dropped on install.
    struct Generation {
        /// The dispatcher export — kept alive so the instance (and its table slot)
        /// stays valid; also the JS fallback entry.
        run: js_sys::Function,
        /// The dispatcher's reserved `__indirect_function_table` slot, if any.
        run_index: Option<usize>,
    }

    thread_local! {
        /// Compiled modules, keyed by `bytes.as_ptr()` (stable/unique — the
        /// `Rc<[u8]>` lives in `WasmJit`'s never-evicted cache). Codegen is the
        /// costly step; do it once per method. Thread-local is also required — a
        /// `WebAssembly.Module` belongs to its agent and can't cross workers.
        static MODULE_CACHE: RefCell<FnvHashMap<usize, WebAssembly::Module>> =
            RefCell::new(FnvHashMap::default());
        /// One instance per method (`Rc` so a run clones it out and drops the map
        /// borrow before the call, which re-enters `run`).
        static INSTANCES: RefCell<FnvHashMap<usize, Rc<Cached>>> = RefCell::new(FnvHashMap::default());
        /// The single shared frame memory (lazily created).
        static MEMORY: RefCell<Option<WebAssembly::Memory>> = RefCell::new(None);
        /// Current JIT nesting depth → frame base `depth * STRIDE`.
        static DEPTH: Cell<usize> = const { Cell::new(0) };
        /// Whether the wasm→wasm-entry diagnostic has been logged once (per thread).
        static ENTRY_MODE_LOGGED: Cell<bool> = const { Cell::new(false) };
        /// Free reserved `__indirect_function_table` slots (see [`reserved_slot`]),
        /// lazily filled on first use. A method's `build()` pops one for its `run`
        /// funcref; methods never evict, so slots aren't returned.
        static FREE_SLOTS: RefCell<Option<Vec<usize>>> = const { RefCell::new(None) };
        /// Methods that live in a GENERATION module: bytes-key → (generation,
        /// member index). Checked before the per-method path; entries replace the
        /// member's `MODULE_CACHE`/`INSTANCES` entries when the generation installs.
        static GEN_ENTRIES: RefCell<FnvHashMap<usize, (Rc<Generation>, u32)>> =
            RefCell::new(FnvHashMap::default());
    }

    /// Pops a free reserved table slot (lazily initializing the pool), or `None` when
    /// the pool ([`RESERVED_SLOT_COUNT`]) is exhausted → that method uses `call3`.
    fn alloc_slot() -> Option<usize> {
        FREE_SLOTS.with(|slot| {
            slot.borrow_mut()
                .get_or_insert_with(reserved_slot_indices)
                .pop()
        })
    }


    /// Number of pre-reserved `__indirect_function_table` slots for JIT `run`
    /// exports. `grow` fails on this toolchain's fixed-size function table (wasm-bindgen
    /// keeps JS closures in a separate externref table), but `set` on an *existing* slot
    /// works — so we force this many distinct dummy fns into the table at link time and
    /// overwrite their slots with JIT `run` funcrefs, giving a wasm→wasm entry
    /// (`call_indirect`) without growth. Methods past the pool fall back to `call3`.
    const RESERVED_SLOT_COUNT: usize = 2048;

    /// A distinct reserved-slot fn per `N` (the body depends on `N` so identical-code
    /// folding can't merge them into one slot). Never called — only its address is taken
    /// (forcing it into the table); the slot is later `set` to a JIT `run` export.
    extern "C" fn reserved_slot<const N: usize>(_: i32, _: i32, _: i32, _: i32, _: i32) -> i64 {
        N as i64
    }

    /// The link-assigned `__indirect_function_table` indices of the reserved slots. On
    /// wasm32 a fn-pointer's integer value IS its table index. The pool is spelled as
    /// a 32×64 grid (`hi * 64 + lo` in a const-generic argument) — 2048 distinct
    /// monomorphizations without listing them all (8192 of them measurably slowed
    /// rustc). Two macro levels because `macro_rules` can't cross-product two
    /// independent repetitions in one transcription: the inner (`lo`) list travels
    /// as a single `tt`. 2048 suffices WITH generations: an amalgam's dispatcher
    /// takes one slot per `GEN_BATCH` methods, and members' per-method slots are
    /// recycled on install.
    fn reserved_slot_indices() -> Vec<usize> {
        macro_rules! slots_row {
            ($v:ident, $hi:literal, ($($lo:literal),* $(,)?)) => {
                $( $v.push(
                    reserved_slot::<{ $hi * 64 + $lo }>
                        as extern "C" fn(i32, i32, i32, i32, i32) -> i64 as usize,
                ); )*
            };
        }
        macro_rules! slots_grid {
            ($v:ident, $lo:tt, $($hi:literal),* $(,)?) => {
                $( slots_row!($v, $hi, $lo); )*
            };
        }
        let mut v = Vec::with_capacity(RESERVED_SLOT_COUNT);
        slots_grid!(
            v,
            (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63),
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
        );
        debug_assert_eq!(v.len(), RESERVED_SLOT_COUNT);
        v
    }

    /// The one shared frame memory, created (bounded) on first use.
    fn memory() -> Option<WebAssembly::Memory> {
        MEMORY.with(|slot| {
            if let Some(m) = slot.borrow().as_ref() {
                return Some(m.clone());
            }
            let desc = Object::new();
            Reflect::set(&desc, &"initial".into(), &JsValue::from(PAGES)).ok()?;
            Reflect::set(&desc, &"maximum".into(), &JsValue::from(PAGES)).ok()?;
            let m = WebAssembly::Memory::new(&desc).ok()?;
            *slot.borrow_mut() = Some(m.clone());
            Some(m)
        })
    }

    /// Compiles (cached) and instantiates an instance for `m`, importing the shared
    /// `mem`.
    fn build(bytes: &[u8], m: &crate::lower::Manifest, mem: &WebAssembly::Memory) -> Option<Cached> {
        let key = bytes.as_ptr() as usize;
        let module = MODULE_CACHE.with(|cache| {
            // Held only across `Module::new` (which doesn't re-enter), no reentry.
            let mut cache = cache.borrow_mut();
            if let Some(m) = cache.get(&key) {
                return Some(m.clone());
            }
            let byte_view = Uint8Array::from(bytes);
            let module = WebAssembly::Module::new(byte_view.as_ref()).ok()?;
            cache.insert(key, module.clone());
            Some(module)
        })?;

        let (instance, run) = instantiate(&module, m, mem)?;
        let _ = instance;
        // Put `run` in a **pre-reserved** `__indirect_function_table` slot so it can be
        // called **wasm→wasm** via a transmuted fn-pointer — no JS `call` trampoline, and
        // the i64 return stays i64 (no BigInt round-trip). `grow` fails on this toolchain's
        // fixed-size function table, but `set` on an existing (reserved) slot works. `run`
        // is kept in `Cached` to hold the instance alive. Pool exhausted → JS fallback.
        let table = wasm_bindgen::function_table().unchecked_into::<WebAssembly::Table>();
        let run_index = alloc_slot().filter(|&idx| table.set(idx as u32, &run).is_ok());
        // One-time diagnostic: is the wasm→wasm JIT entry active, or did we fall back to
        // the slower JS path (pool exhausted / `set` failed)?
        ENTRY_MODE_LOGGED.with(|logged| {
            if !logged.replace(true) {
                match run_index {
                    Some(_) => tracing::info!("AVM2 JIT entry: wasm→wasm (call_indirect) active"),
                    None => tracing::info!("AVM2 JIT entry: FALLBACK to JS apply"),
                }
            }
        });
        Some(Cached { run, run_index })
    }

    /// Binds `m`'s imports and instantiates `module`, returning the instance and
    /// its `run` export. Shared by the per-method [`build`] and
    /// [`install_generation`].
    fn instantiate(
        module: &WebAssembly::Module,
        m: &crate::lower::Manifest,
        mem: &WebAssembly::Memory,
    ) -> Option<(WebAssembly::Instance, js_sys::Function)> {
        let env = Object::new();
        Reflect::set(&env, &"memory".into(), mem).ok()?;
        // Memory 1 = `dm`: Ruffle's *own* linear memory — ALWAYS imported on web:
        // the prologue copies the register snapshot out of it (`regs_ptr`), and
        // inline `li*`/`si*` (when present) `i32.load`/`store` domainMemory at
        // `dm_base + addr`.
        Reflect::set(&env, &"dm".into(), &wasm_bindgen::memory()).ok()?;

        // Bind every helper import to Ruffle's OWN wasm function — a funcref pulled
        // from the module's indirect function table — instead of a JS `Closure`
        // trampoline. The engine links a wasm funcref import as a **direct wasm→wasm
        // call**: no JS frame, and i64 args stay i64 (no `BigInt` marshaling — that
        // only happens at a JS boundary). On wasm32 a fn pointer's integer value IS
        // its index in `__indirect_function_table`, so `HELPERS[i] as usize` (and
        // `foo as fn(..) as usize`, which also forces `foo` into the table) is the
        // table slot. This is the fix for the ~26% of the compute worker that a
        // profile showed spent in libxul JS↔wasm boundary machinery per helper call.
        let table = wasm_bindgen::function_table().unchecked_into::<WebAssembly::Table>();
        for i in 0..m.num_helpers as usize {
            let helper = *crate::helpers::HELPERS.get(i)?;
            bind_fn(&env, &table, &format!("h{i}"), helper as usize)?;
        }
        if m.has_getprop {
            bind_fn(&env, &table, "gp", crate::helpers::get_property as fn(i64, i64) -> i64 as usize)?;
        }
        if m.has_getslot {
            bind_fn(&env, &table, "gs", crate::helpers::get_slot as fn(i64, i64) -> i64 as usize)?;
        }
        if m.has_getprop_fast {
            bind_fn(&env, &table, "gpf", crate::helpers::get_property_fast as fn(i64, i64, i64) -> i64 as usize)?;
        }
        for i in 0..m.num_helpers2 as usize {
            let helper = *crate::helpers::HELPERS2.get(i)?;
            bind_fn(&env, &table, &format!("t{i}"), helper as usize)?;
        }
        for (k, &helper) in crate::helpers::HELPERS3.iter().enumerate() {
            if m.set3_mask & (1 << k) != 0 {
                bind_fn(&env, &table, &format!("s{k}"), helper as usize)?;
            }
        }
        // Call imports: `cm`/`cp` (ternary), then the shared `pca` (i64->()) and
        // `perr` (()->i32).
        if m.has_call {
            bind_fn(&env, &table, "cm", crate::helpers::call_method as fn(i64, i64, i64) -> i64 as usize)?;
        }
        if m.has_callprop {
            bind_fn(&env, &table, "cp", crate::helpers::call_property as fn(i64, i64, i64) -> i64 as usize)?;
        }
        if m.has_construct_super {
            bind_fn(&env, &table, "csup", crate::helpers::construct_super as fn(i64, i64) -> i64 as usize)?;
        }
        if m.has_call_value {
            bind_fn(&env, &table, "callv", crate::helpers::call_value as fn(i64, i64, i64) -> i64 as usize)?;
        }
        let any_call = m.has_call
            || m.has_callprop
            || m.has_construct_super
            || m.has_call_value
            || m.has_vcall;
        if any_call {
            bind_fn(&env, &table, "pca", crate::helpers::push_call_arg as fn(i64) as usize)?;
        }
        // `perr` — see `Manifest::needs_perr` (the single source of truth for the
        // import's presence, shared with the import section and the native runner).
        if m.needs_perr() {
            bind_fn(&env, &table, "perr", crate::helpers::pending_error as fn() -> i32 as usize)?;
        }
        // `coerce` (arity-2 `(value, class_idx) -> result`) follows `perr`.
        if m.has_coerce {
            bind_fn(&env, &table, "coerce", crate::helpers::coerce as fn(i64, i64) -> i64 as usize)?;
        }
        // `vc` (arity-4, the generic variadic helper) follows `coerce`.
        if m.has_vcall {
            bind_fn(&env, &table, "vc", crate::helpers::vcall as fn(i64, i64, i64, i64) -> i64 as usize)?;
        }

        let imports = Object::new();
        Reflect::set(&imports, &"env".into(), &env).ok()?;
        let instance = WebAssembly::Instance::new(module, &imports).ok()?;
        let run = Reflect::get(&instance.exports(), &"run".into())
            .ok()?
            .dyn_into::<js_sys::Function>()
            .ok()?;
        Some((instance, run))
    }

    /// Installs a GENERATION module (see [`crate::lower::compile_generation`]):
    /// instantiates it once against the union manifest, registers its dispatcher
    /// in ONE reserved table slot, points every member's bytes-key at
    /// `(generation, index)`, and drops the members' per-method modules and
    /// instances — releasing their executable memory and instance-cache entries.
    /// Returns `false` (leaving members on their per-method path) if anything
    /// fails (e.g. a strict CSP).
    pub(super) fn install_generation(
        gen_bytes: &[u8],
        union: &crate::lower::Manifest,
        member_keys: &[usize],
    ) -> bool {
        let Some(mem) = memory() else { return false };
        let byte_view = Uint8Array::from(gen_bytes);
        let module = match WebAssembly::Module::new(byte_view.as_ref()) {
            Ok(m) => m,
            Err(e) => {
                // MUST stay loud: a silently failing install leaves every member on
                // its per-method slot forever — the pool exhausts and later methods
                // degrade to the slow JS entry (seen as huge `js-to-wasm` profile
                // time when a Manifest field was missing from the generation union).
                // Include the engine's validation error — it names the function
                // index + byte offset, which identifies the miscompiled member.
                tracing::warn!(
                    "AVM2 JIT: generation module INVALID ({} members) — install skipped: {e:?}",
                    member_keys.len()
                );
                return false;
            }
        };
        let Some((instance, run)) = instantiate(&module, union, &mem) else {
            tracing::warn!(
                "AVM2 JIT: generation instantiation failed ({} members) — install skipped",
                member_keys.len()
            );
            return false;
        };
        let _ = instance;
        let table = wasm_bindgen::function_table().unchecked_into::<WebAssembly::Table>();
        let run_index = alloc_slot().filter(|&idx| table.set(idx as u32, &run).is_ok());
        let generation = Rc::new(Generation { run, run_index });
        GEN_ENTRIES.with(|g| {
            let mut g = g.borrow_mut();
            for (i, &key) in member_keys.iter().enumerate() {
                g.insert(key, (generation.clone(), i as u32));
            }
        });
        // The members' standalone modules/instances are now dead weight — drop
        // them so the browser can reclaim their executable memory, and RECYCLE
        // their reserved table slots (this keeps total slot usage bounded by
        // `pending cold methods + generations`, so the pool stays small).
        MODULE_CACHE.with(|c| {
            let mut c = c.borrow_mut();
            for key in member_keys {
                c.remove(key);
            }
        });
        INSTANCES.with(|c| {
            let mut c = c.borrow_mut();
            for key in member_keys {
                if let Some(cached) = c.remove(key) {
                    if let Some(slot) = cached.run_index {
                        FREE_SLOTS.with(|s| {
                            if let Some(pool) = s.borrow_mut().as_mut() {
                                pool.push(slot);
                            }
                        });
                    }
                }
            }
        });
        true
    }

    /// Sets `env[name]` to the wasm function at `table_index` in Ruffle's indirect
    /// function table — a wasm funcref the JIT module imports and calls directly
    /// (see [`build`]).
    fn bind_fn(env: &Object, table: &WebAssembly::Table, name: &str, table_index: usize) -> Option<()> {
        let f = table.get(table_index as u32).ok()?;
        Reflect::set(env, &name.into(), f.as_ref()).ok()?;
        Some(())
    }

    pub(super) fn run(bytes: &[u8], regs: &[u64], m: &crate::lower::Manifest) -> Option<u64> {
        let frame_bytes = regs.len() * 8;
        let depth = DEPTH.with(|d| d.get());
        let state_ptr = depth * STRIDE;
        // Decline (→ interpreter) rather than alias frames or run off the memory: a
        // method with more than STRIDE-worth of locals, or nesting past the memory.
        if frame_bytes > STRIDE || state_ptr + frame_bytes > MEM_BYTES {
            return None;
        }
        let mem = memory()?;
        let key = bytes.as_ptr() as usize;

        // The registers are read by the module's prologue (see below), and the
        // dm base/len feed the inline `li*`/`si*` path — shared by both entries.
        let regs_ptr = regs.as_ptr() as u32;
        let regs_len = frame_bytes as u32;
        let (dm_base, dm_len) = if m.has_dm {
            crate::helpers::dm_base_len()
        } else {
            (0, 0)
        };

        // GENERATION path: the method lives in an amalgam module — call its
        // dispatcher with the member index prepended.
        if let Some((generation, member)) = GEN_ENTRIES.with(|g| g.borrow().get(&key).cloned()) {
            DEPTH.with(|d| d.set(depth + 1));
            let result = match generation.run_index {
                Some(idx) => {
                    type RunFn6 = unsafe extern "C" fn(i32, i32, i32, i32, i32, i32) -> i64;
                    let run: RunFn6 = unsafe { core::mem::transmute::<usize, RunFn6>(idx) };
                    Some(unsafe {
                        run(
                            member as i32,
                            state_ptr as i32,
                            dm_base as i32,
                            dm_len as i32,
                            regs_ptr as i32,
                            regs_len as i32,
                        )
                    } as u64)
                }
                None => {
                    // (`js_sys::Array` has constructors only up to `of5`.)
                    let args = js_sys::Array::of5(
                        &JsValue::from(member),
                        &JsValue::from(state_ptr as u32),
                        &JsValue::from(dm_base),
                        &JsValue::from(dm_len),
                        &JsValue::from(regs_ptr),
                    );
                    args.push(&JsValue::from(regs_len));
                    generation
                        .run
                        .apply(&JsValue::NULL, &args)
                        .ok()
                        .and_then(|r| r.dyn_into::<js_sys::BigInt>().ok())
                        .and_then(|b| i64::try_from(b).ok())
                        .map(|v| v as u64)
                }
            };
            DEPTH.with(|d| d.set(depth));
            return result;
        }

        // Get (or build+cache) the method's instance, cloning the `Rc` out so no
        // `INSTANCES` borrow is held across the call (which re-enters `run`).
        let inst = match INSTANCES.with(|c| c.borrow().get(&key).cloned()) {
            Some(i) => i,
            None => {
                let built = Rc::new(build(bytes, m, &mem)?);
                INSTANCES.with(|c| c.borrow_mut().insert(key, built.clone()));
                built
            }
        };

        // The registers are NOT copied here: the module's prologue does a
        // wasm→wasm `memory.copy` from Ruffle's own memory (imported as memory 1)
        // at `regs_ptr` into the frame memory at `state_ptr`. On wasm32 a slice's
        // address IS its linear-memory offset, and `regs` stays borrowed across
        // the call, so the source can't move.
        //
        // Enter one nesting level, run, leave — a re-entrant run reads the
        // incremented depth → a disjoint frame.
        DEPTH.with(|d| d.set(depth + 1));
        let result = match inst.run_index {
            // Preferred: call the JIT `run` **wasm→wasm** through its indirect-table
            // slot. On wasm32 a fn-pointer's integer value IS its
            // `__indirect_function_table` index, so transmuting `run_index` to a
            // fn-pointer and calling it emits a `call_indirect` straight to the JIT
            // module — no JS `call` frame, and the i64 return stays i64 (the BigInt
            // round-trip that a JS call forces is gone). The `(i32,i32,i32,i32,i32)->i64`
            // signature matches the `run` export so `call_indirect`'s type check passes;
            // a JIT trap aborts here (no catchable fallback) — acceptable, the JIT
            // shouldn't trap (bounds explicit, throws go out-of-band via `perr`).
            Some(idx) => {
                type RunFn = unsafe extern "C" fn(i32, i32, i32, i32, i32) -> i64;
                let run: RunFn = unsafe { core::mem::transmute::<usize, RunFn>(idx) };
                Some(unsafe {
                    run(
                        state_ptr as i32,
                        dm_base as i32,
                        dm_len as i32,
                        regs_ptr as i32,
                        regs_len as i32,
                    )
                } as u64)
            }
            // Fallback (table not growable): the JS `Function.apply` path. The i64
            // return crosses as a **signed** BigInt (any NaN-boxed Value arrives
            // negative), so reinterpret the signed i64 back to raw u64 bits — never
            // a u64 range-check.
            None => {
                let args = js_sys::Array::of5(
                    &JsValue::from(state_ptr as u32),
                    &JsValue::from(dm_base),
                    &JsValue::from(dm_len),
                    &JsValue::from(regs_ptr),
                    &JsValue::from(regs_len),
                );
                inst.run
                    .apply(&JsValue::NULL, &args)
                    .ok()
                    .and_then(|r| r.dyn_into::<js_sys::BigInt>().ok())
                    .and_then(|b| i64::try_from(b).ok())
                    .map(|v| v as u64)
            }
        };
        DEPTH.with(|d| d.set(depth));
        result
    }
}

#[cfg(target_arch = "wasm32")]
pub fn run(bytes: &[u8], regs: &[u64], m: &crate::lower::Manifest) -> Option<u64> {
    web::run(bytes, regs, m)
}

/// Installs a GENERATION (amalgam) module — see [`web::install_generation`].
#[cfg(target_arch = "wasm32")]
pub fn install_generation(
    gen_bytes: &[u8],
    union: &crate::lower::Manifest,
    member_keys: &[usize],
) -> bool {
    web::install_generation(gen_bytes, union, member_keys)
}

/// In-browser end-to-end test of the web runner: emit a real `run` module and
/// execute it through the browser engine. Run with
/// `wasm-pack test --headless --firefox -p ruffle_avm2_jit`.
#[cfg(all(test, target_arch = "wasm32"))]
mod wasm_tests {
    use super::*;
    use crate::lower::{compile, JitOp};
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    fn int_bits(n: i32) -> u64 {
        0xFFFB_0000_0000_0000 | (n as u32 as u64)
    }

    #[wasm_bindgen_test]
    fn browser_runs_local_add() {
        // return local1 + local2
        let ops = [
            JitOp::GetLocal(1),
            JitOp::GetLocal(2),
            JitOp::AddI,
            JitOp::ReturnValue,
        ];
        let bytes = compile(&ops).expect("compiles");
        let regs = [int_bits(0), int_bits(10), int_bits(20)];
        assert_eq!(run(&bytes, &regs, &crate::lower::manifest(&ops)), Some(int_bits(30)));
    }

    #[wasm_bindgen_test]
    fn browser_runs_counted_loop() {
        // sum(5) = 10 through the dispatch-loop lowering.
        let ops = [
            JitOp::PushInt(0),
            JitOp::SetLocal(2),
            JitOp::PushInt(0),
            JitOp::SetLocal(3),
            JitOp::GetLocal(3),
            JitOp::GetLocal(1),
            JitOp::IfGe(16),
            JitOp::GetLocal(2),
            JitOp::GetLocal(3),
            JitOp::AddI,
            JitOp::SetLocal(2),
            JitOp::GetLocal(3),
            JitOp::PushInt(1),
            JitOp::AddI,
            JitOp::SetLocal(3),
            JitOp::Jump(4),
            JitOp::GetLocal(2),
            JitOp::ReturnValue,
        ];
        let bytes = compile(&ops).expect("compiles");
        let regs = [int_bits(0), int_bits(5), int_bits(0), int_bits(0)];
        assert_eq!(run(&bytes, &regs, &crate::lower::manifest(&ops)), Some(int_bits(10)));
    }
}
