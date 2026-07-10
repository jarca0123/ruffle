//! Web execution: hand the emitted bytes to the browser's own WASM engine via the
//! `WebAssembly` JS API (js-sys). The counterpart to the native `runner` (wasmtime);
//! both expose the same `Runner`/`Program`/`run`/`run_dm` surface so `lib.rs` is
//! target-agnostic.
//!
//! ## Helper imports
//! The emitted module imports the same helpers the native path binds
//! (`gs`/`ss`/`ti`/`ca`/`cm`/`pe`). The browser runs the module in a *separate*
//! `WebAssembly.Instance`, but rather than JS `Closure` trampolines we bind each
//! import to **Ruffle's own wasm function** — a funcref pulled from Ruffle's
//! `__indirect_function_table`. The engine links a wasm-funcref import as a direct
//! wasm→wasm call: no JS frame, and `i64` args stay `i64` (no `BigInt` marshaling,
//! which only happens at a JS boundary). On wasm32 a fn pointer's integer value IS
//! its table index, so `f as fn(..) -> .. as usize` (which also forces `f` into the
//! table) is the slot. This is a synchronous, single-threaded re-entry into
//! Ruffle's wasm while `try_run`'s `with_activation`/`with_mc` are still on the
//! stack, so the helpers' thread-locals are valid — exactly as on native.
//!
//! Only the outer `run` call crosses the JS boundary (its signature varies per
//! method, so it goes through `Function.apply`, `i64` as `BigInt`); the helpers it
//! calls do not. Instantiation is per-call (the module is compiled once and
//! cached), so recursion via the amalgam's wasm→wasm `call`s stays in one instance
//! and re-entrant trampoline calls get their own — no frame aliasing.

use std::cell::RefCell;
use std::rc::Rc;

use js_sys::{Array, Function, Object, Reflect, Uint8Array, WebAssembly};
use wasm_bindgen::{JsCast, JsValue};

use crate::emit::{ENTRY_NAME, MEMORY_NAME};
use crate::{Arg, Helpers};

const WASM_PAGE: usize = 65536;

/// A compiled-and-validated module (a browser `WebAssembly.Module`), cached per
/// AVM2 method, plus a lazily-built, reused instance for the non-dm path.
pub struct Program {
    module: WebAssembly::Module,
    /// A cached instance + its `run` export, built on first (non-dm) run and
    /// reused. A non-dm module has no mutable module-global state (only WASM
    /// locals + funcref imports), so reusing one instance across calls — including
    /// re-entrant / recursive ones — is sound and avoids a per-call
    /// `WebAssembly.Instance` allocation (a measured GC hotspot). dm methods do NOT
    /// use this (their shared linear memory would alias across a re-entrant run).
    instance: RefCell<Option<Rc<(WebAssembly::Instance, Function)>>>,
}

/// The web runner. Stateless — the browser owns the engine — but kept as a struct
/// so the target-agnostic `lib.rs` can hold a `runner::Runner` on both targets.
pub struct Runner {}

impl Runner {
    pub fn new() -> Self {
        Runner {}
    }

    /// Compile+validate emitted module bytes via `WebAssembly.Module`. `None` if
    /// the bytes fail to validate or the environment forbids runtime compilation
    /// (e.g. a strict CSP without `wasm-unsafe-eval`) — either way the interpreter
    /// stays authoritative.
    pub fn compile(&self, bytes: &[u8]) -> Option<Program> {
        let view = Uint8Array::from(bytes);
        WebAssembly::Module::new(view.as_ref()).ok().map(|module| Program {
            module,
            instance: RefCell::new(None),
        })
    }

    /// Call `run(args) -> i64` on the program's cached instance (built on first
    /// use), returning the result's `Value` bits. `None` on any failure (declines
    /// to the interpreter).
    pub fn run(&self, program: &Program, args: &[Arg], h: Helpers) -> Option<u64> {
        // Reuse the instance across calls; clone the `run` handle out and drop the
        // borrow before calling, so a re-entrant call can re-borrow the cell.
        let cached = {
            let mut slot = program.instance.borrow_mut();
            if slot.is_none() {
                *slot = Some(Rc::new(instantiate(&program.module, h)?));
            }
            slot.as_ref().unwrap().clone()
        };
        call_run(&cached.1, args)
    }

    /// Run a domainMemory method: fill the module's exported linear memory with
    /// `dm`, call `run(args)` (whose last arg is `dm_len`), then read the memory
    /// back into `dm` so the caller can persist stores. Returns the result's bits.
    pub fn run_dm(&self, program: &Program, args: &[Arg], dm: &mut [u8], h: Helpers) -> Option<u64> {
        let (instance, run) = instantiate(&program.module, h)?;
        let memory = Reflect::get(&instance.exports(), &JsValue::from_str(MEMORY_NAME))
            .ok()?
            .dyn_into::<WebAssembly::Memory>()
            .ok()?;

        // Grow to fit the domain memory if needed, then copy it in. Growth detaches
        // the old buffer, so the view is taken *after* any grow.
        let need = dm.len();
        let have = Uint8Array::new(&memory.buffer()).length() as usize;
        if need > have {
            let pages = (need - have).div_ceil(WASM_PAGE) as u32;
            memory.grow(pages);
        }
        let view = Uint8Array::new(&memory.buffer());
        view.subarray(0, need as u32).copy_from(dm);

        let bits = call_run(&run, args)?;

        // The emitted dm code never grows memory, so the buffer is still valid.
        view.subarray(0, need as u32).copy_to(dm);
        drop(instance);
        Some(bits)
    }
}

/// Bind `program`'s helper imports (as Ruffle's own wasm funcrefs) and instantiate,
/// returning the instance and its `run` export.
fn instantiate(module: &WebAssembly::Module, h: Helpers) -> Option<(WebAssembly::Instance, Function)> {
    let env = Object::new();
    let table = wasm_bindgen::function_table().unchecked_into::<WebAssembly::Table>();

    // Canonical order matches the emitted import section: gs, ss, ti, ca, cm, pe.
    if h.get_slot {
        bind(&env, &table, "gs", crate::runtime::get_slot as fn(u64, u32) -> u64 as usize)?;
    }
    if h.set_slot {
        bind(&env, &table, "ss", crate::runtime::set_slot as fn(u64, u64, u32) as usize)?;
    }
    if h.to_int32 {
        bind(&env, &table, "ti", crate::runtime::to_int32 as fn(f64) -> i32 as usize)?;
    }
    if h.calls {
        bind(&env, &table, "ca", crate::runtime::push_call_arg as fn(u64) as usize)?;
        bind(&env, &table, "cm", crate::runtime::call_method as fn(u64, u32, u32) -> u64 as usize)?;
        bind(&env, &table, "pe", crate::runtime::pending_error as fn() -> i32 as usize)?;
    }

    let imports = Object::new();
    Reflect::set(&imports, &JsValue::from_str("env"), &env).ok()?;
    let instance = WebAssembly::Instance::new(module, &imports).ok()?;
    let run = Reflect::get(&instance.exports(), &JsValue::from_str(ENTRY_NAME))
        .ok()?
        .dyn_into::<Function>()
        .ok()?;
    Some((instance, run))
}

/// Set `env[name]` to the wasm function at `table_index` in Ruffle's indirect
/// function table (a funcref the JIT module imports and calls directly).
fn bind(env: &Object, table: &WebAssembly::Table, name: &str, table_index: usize) -> Option<()> {
    let f = table.get(table_index as u32).ok()?;
    Reflect::set(env, &JsValue::from_str(name), f.as_ref()).ok()?;
    Some(())
}

/// Call `run(args)` across the JS boundary and decode the `i64` result. The bits
/// cross as a **signed** `BigInt` (a NaN-boxed `Value` arrives negative), so
/// reinterpret the signed `i64` back to raw `u64` — never a `u64` range-check.
fn call_run(run: &Function, args: &[Arg]) -> Option<u64> {
    let arr = Array::new();
    for a in args {
        let v = match *a {
            Arg::I32(x) => JsValue::from(x),
            Arg::F64(x) => JsValue::from(x),
            Arg::I64(x) => JsValue::from(x), // BigInt (JS-BigInt-integration)
        };
        arr.push(&v);
    }
    let result = run.apply(&JsValue::NULL, &arr).ok()?;
    let big = result.dyn_into::<js_sys::BigInt>().ok()?;
    i64::try_from(big).ok().map(|v| v as u64)
}
