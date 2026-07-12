//! Browser execution of a compiled type-0 module (the WASM engine in the JS agent /
//! worker). Mirrors the native [`runner`](crate::runner) `run_leaf` interface.
//!
//! ## The frame lives in the MAIN module's memory
//!
//! Every compiled module imports the SAME memory as `env.memory` — and that memory is the
//! **main Ruffle module's** linear memory ([`wasm_bindgen::memory`]). The frame is a Rust
//! buffer ([`FRAME_BUF`]) inside it; a call runs at `buf_base + DEPTH * STRIDE`. Because the
//! frame is main memory, the `cp` (callproperty) helper — Rust code in the main module —
//! reads a method's outgoing call args straight from it by pointer, so a call is ONE helper
//! crossing rather than one `pca` per argument. (An earlier design gave each module its own
//! memory, which reserved address space per method and OOM'd; a later one used a standalone
//! shared memory, which fixed OOM but couldn't be read by the helpers — hence this.)
//!
//! ## GC + re-entrancy discipline
//!
//! The cache borrow is released (the `run` funcref cloned out) BEFORE the call, so a
//! re-entrant `run_leaf` can't double-borrow the thread-local. `DEPTH` places each nested
//! run at a fresh stride, so a helper that re-enters AS3 never aliases a live frame.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use js_sys::{Object, Reflect, Uint8Array, WebAssembly};
use wasm_bindgen::prelude::*;
use wasm_bindgen::{JsCast, JsValue};

use crate::emit::{FRAME_PAGES, FRAME_STRIDE};
use crate::helpers;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console, js_name = log)]
    fn console_log(s: &str);
}

/// Logs a first-sighting declined op to the browser console (see
/// `translate::record_decline`) — this is where real content reveals which ops to add next.
pub fn log_decline(name: &str) {
    console_log(&format!("JIT3 DECLINE (new op): {name}"));
}

/// Total frame arena, in bytes (matches the old shared-memory size: 4 MiB = many strides).
const FRAME_BUF_BYTES: usize = (FRAME_PAGES as usize) * 65536;

/// How a compiled method's `run` is entered.
#[derive(Clone)]
enum Callee {
    /// The `run` funcref was placed in the main indirect function table at this index, so it
    /// can be entered by a transmuted fn-pointer — a wasm→wasm `call_indirect`, with the i64
    /// result returned directly (no JS boundary, no BigInt marshaling). The fast path.
    Indirect(u32),
    /// Fallback (the main table couldn't grow): enter via `Function::call3`, whose i64 result
    /// crosses back as a BigInt.
    Js(js_sys::Function),
}

/// A compiled method's per-instance entry state, held by the JIT in `CompiledMethod` (one per
/// method). The JIT hands this to [`run_leaf`] directly, so the runner no longer re-hashes the
/// method by key on every call — the double cache lookup is gone. Opaque to the JIT (private
/// field): it only stores/clones it (a cheap `Rc` refcount bump per call).
#[derive(Clone)]
pub struct Handle(Rc<RefCell<HandleState>>);

enum HandleState {
    /// Not yet instantiated (first call will instantiate).
    Uninit,
    /// Instantiated; entered via this [`Callee`].
    Ready(Callee),
    /// `WebAssembly.Instance()` failed — declined permanently so it never retries (a failure
    /// re-instantiating every call amplified a single mismatch into a frame-dominating blowup).
    Failed,
}

/// Allocates a fresh (uninstantiated) entry handle for a newly compiled method.
pub fn new_handle() -> Handle {
    Handle(Rc::new(RefCell::new(HandleState::Uninit)))
}

// Global runner state (NOT per-method): the frame arena and nesting depth — one PER AVM2 THREAD.
//
// The THREADED build (`target_feature = "atomics"`) supports Flash **Workers**, which run AVM2
// on multiple threads sharing ONE wasm module instance. A plain `static` there would be shared
// across those threads — two workers writing frames at the same `DEPTH` stride corrupt each
// other → heap/RefCell corruption. So the threaded build uses `thread_local!` (per-worker), as
// does native (the test harness runs methods on many threads).
//
// The SINGLE-THREAD (non-atomics) build has exactly one AVM2 thread, so it uses a plain `static`
// — avoiding the wasm TLS-slot computation `run_leaf` (inlined into `try_enter`) would otherwise
// pay several times per JIT call (profiled: ~19% of the dispatch). Access is synchronous and
// never held across a call, so the single-thread `&mut` never aliases.
#[cfg(target_feature = "atomics")]
thread_local! {
    static FRAME_BUF: RefCell<Vec<u64>> = const { RefCell::new(Vec::new()) };
    static DEPTH: Cell<u32> = const { Cell::new(0) };
}

#[cfg(not(target_feature = "atomics"))]
mod single_thread {
    use std::cell::UnsafeCell;
    /// A `Sync` cell relying on the single-AVM2-thread invariant (see the module note above).
    pub struct One<T>(pub UnsafeCell<T>);
    // SAFETY: only ever accessed from the single AVM2 thread.
    unsafe impl<T> Sync for One<T> {}
    /// The frame arena — a stable Rust allocation inside the main module's linear memory.
    pub static FRAME_BUF: One<Vec<u64>> = One(UnsafeCell::new(Vec::new()));
    /// Current frame nesting level.
    pub static DEPTH: One<u32> = One(UnsafeCell::new(0));
}

/// Runs `f` with the frame arena (mutable). Never hold the reference across a call.
#[inline]
fn with_frame_buf<R>(f: impl FnOnce(&mut Vec<u64>) -> R) -> R {
    #[cfg(target_feature = "atomics")]
    {
        FRAME_BUF.with(|fb| f(&mut fb.borrow_mut()))
    }
    // SAFETY: single AVM2 thread; `&mut` is not held across a call.
    #[cfg(not(target_feature = "atomics"))]
    unsafe {
        f(&mut *single_thread::FRAME_BUF.0.get())
    }
}

#[inline]
fn depth_get() -> u32 {
    #[cfg(target_feature = "atomics")]
    {
        DEPTH.with(|d| d.get())
    }
    // SAFETY: single AVM2 thread.
    #[cfg(not(target_feature = "atomics"))]
    unsafe {
        *single_thread::DEPTH.0.get()
    }
}

#[inline]
fn depth_set(v: u32) {
    #[cfg(target_feature = "atomics")]
    {
        DEPTH.with(|d| d.set(v));
    }
    // SAFETY: single AVM2 thread.
    #[cfg(not(target_feature = "atomics"))]
    unsafe {
        *single_thread::DEPTH.0.get() = v;
    }
}

/// Runs a compiled leaf method: instantiate-or-reuse via the method's own `handle`, write
/// `frame` at `buf_base + DEPTH*STRIDE`, call `run(0, argc, args)`, return the result bits.
pub fn run_leaf(handle: &Handle, bytes: &[u8], frame: &[u64], argc: u32) -> Option<u64> {
    // Resolve the entry from the method's OWN handle — no hashmap lookup. Clone the `Callee`
    // out and release the borrow before running (a callee re-enters run_leaf on OTHER handles;
    // and this handle's borrow must not span the call). On first use we instantiate OUTSIDE any
    // borrow, then take a short `borrow_mut` to record the result. `main_memory()` crosses into
    // JS (`wasm_bindgen::memory()` + `dyn_into`), so it is fetched ONLY on that miss path — the
    // hot cache-hit path (every subsequent call) never pays that crossing.
    let ready = match &*handle.0.borrow() {
        HandleState::Ready(c) => Some(c.clone()),
        HandleState::Failed => return None, // known-bad module: never re-attempt
        HandleState::Uninit => None,
    };
    let callee = match ready {
        Some(c) => c,
        None => {
            let memory = main_memory()?;
            match instantiate(bytes, &memory) {
                Some(f) => {
                    // Prefer the wasm→wasm entry: park `run` in the main table and enter it by
                    // fn-pointer. If the table can't grow, fall back to the JS `call3` path.
                    let callee = match add_run_to_table(&f) {
                        Some(idx) => {
                            record_entry_kind(true);
                            Callee::Indirect(idx)
                        }
                        None => {
                            record_entry_kind(false);
                            Callee::Js(f)
                        }
                    };
                    *handle.0.borrow_mut() = HandleState::Ready(callee.clone());
                    callee
                }
                None => {
                    *handle.0.borrow_mut() = HandleState::Failed;
                    return None;
                }
            }
        }
    };

    let depth = depth_get();
    let stride_bytes = depth.checked_mul(FRAME_STRIDE)? as usize;
    if stride_bytes + frame.len() * 8 > FRAME_BUF_BYTES {
        return None; // frame nesting overflow → decline
    }

    // Write `[this, args…]` into the arena at this depth, and compute the absolute byte
    // offset (into main memory) that the compiled module reads/writes as its frame base.
    let args = with_frame_buf(|buf| {
        if buf.is_empty() {
            *buf = vec![0u64; FRAME_BUF_BYTES / 8];
        }
        let base_slot = stride_bytes / 8;
        buf[base_slot..base_slot + frame.len()].copy_from_slice(frame);
        // `buf` lives in main linear memory; its data pointer is the byte offset the module
        // uses. Stable: the arena is allocated once and never resized.
        buf.as_ptr() as usize + stride_bytes
    });

    depth_set(depth + 1);
    let bits = match &callee {
        Callee::Indirect(idx) => {
            // SAFETY: `*idx` is a main-`__indirect_function_table` slot holding the compiled
            // `run` (type `(i32,i32,i32)->i64`). On wasm32 a fn-pointer's integer value IS its
            // table index, so transmuting the index yields a callable that `call_indirect`s
            // that slot IN-WASM — i64 stays i64, no JS boundary, no BigInt.
            let run: extern "C" fn(i32, i32, i32) -> i64 =
                unsafe { core::mem::transmute(*idx as usize) };
            Some(run(0, argc as i32, args as u32 as i32) as u64)
        }
        Callee::Js(func) => func
            .call3(
                &JsValue::NULL,
                &JsValue::from(0),
                &JsValue::from(argc),
                &JsValue::from(args as u32 as i32),
            )
            .ok()
            .and_then(|r| r.dyn_into::<js_sys::BigInt>().ok())
            .and_then(|b| i64::try_from(b).ok())
            .map(|v| v as u64),
    };
    depth_set(depth);
    bits
}

/// Tallies compiled-method ENTRY kinds (wasm→wasm `Indirect` vs JS `call3` fallback) and
/// logs the split each time a new method instantiates — a high `js` count means the main
/// function table can't grow, so entries pay a JS-boundary + BigInt crossing per call.
fn record_entry_kind(indirect: bool) {
    thread_local! {
        static INDIRECT_N: Cell<u32> = const { Cell::new(0) };
        static JS_N: Cell<u32> = const { Cell::new(0) };
    }
    if indirect {
        INDIRECT_N.with(|c| c.set(c.get() + 1));
    } else {
        JS_N.with(|c| c.set(c.get() + 1));
    }
    let (i, j) = (INDIRECT_N.with(|c| c.get()), JS_N.with(|c| c.get()));
    if (i + j) % 200 == 0 {
        console_log(&format!("JIT3 ENTRY KINDS: indirect={i} js={j}"));
    }
}

/// Parks a compiled module's `run` funcref in the main indirect function table and returns
/// its index (so it can be entered by fn-pointer). `None` if the table can't grow.
fn add_run_to_table(run: &js_sys::Function) -> Option<u32> {
    let table = wasm_bindgen::function_table().unchecked_into::<WebAssembly::Table>();
    let idx = table.grow(1).ok()?; // returns the previous length = the new slot
    table.set(idx, run).ok()?;
    Some(idx)
}

/// The main Ruffle module's linear memory (shared with every compiled module).
fn main_memory() -> Option<WebAssembly::Memory> {
    wasm_bindgen::memory().dyn_into::<WebAssembly::Memory>().ok()
}

/// Instantiates `bytes` with the main module's function table + memory and the helper
/// index globals, returning `run`. Helpers are reached by `call_indirect` on the table
/// (wasm→wasm — no JS boundary, so the i64 `Value`s never marshal through BigInt).
fn instantiate(bytes: &[u8], memory: &WebAssembly::Memory) -> Option<js_sys::Function> {
    let module = WebAssembly::Module::new(Uint8Array::from(bytes).as_ref()).ok()?;

    let env = Object::new();
    // The compiled module's `env.table` IS the main module's indirect function table — the
    // one holding the helper functions. A `fn`'s integer value on wasm32 is its index there.
    let table = wasm_bindgen::function_table();
    Reflect::set(&env, &"table".into(), &table).ok()?;
    bind_idx(&env, "gs_i", helpers::get_slot as fn(i64, i64) -> i64 as usize)?;
    bind_idx(&env, "cr_i", helpers::coerce_return as fn(i64, i64) -> i64 as usize)?;
    bind_idx(&env, "gp_i", helpers::get_property as fn(i64, i64) -> i64 as usize)?;
    bind_idx(&env, "cp_i", helpers::call_property as fn(i64, i64, i64, i64) -> i64 as usize)?;
    bind_idx(&env, "perr_i", helpers::pending_error as fn() -> i32 as usize)?;
    bind_idx(&env, "binop_i", helpers::binop as fn(i64, i64, i32) -> i64 as usize)?;
    bind_idx(&env, "unop_i", helpers::unop as fn(i64, i32) -> i64 as usize)?;
    bind_idx(&env, "truthy_i", helpers::truthy as fn(i64) -> i32 as usize)?;
    bind_idx(&env, "setprop_i", helpers::set_property as fn(i64, i64, i64) as usize)?;
    bind_idx(&env, "setslot_i", helpers::set_slot as fn(i64, i32, i64, i32) as usize)?;
    bind_idx(&env, "callmethod_i", helpers::call_method as fn(i64, i32, i64, i32) -> i64 as usize)?;
    bind_idx(&env, "newarray_i", helpers::new_array as fn(i64, i32) -> i64 as usize)?;
    bind_idx(&env, "outerscope_i", helpers::outer_scope as fn(i32) -> i64 as usize)?;
    bind_idx(&env, "scriptglobals_i", helpers::script_globals as fn(i64) -> i64 as usize)?;
    bind_idx(&env, "newactivation_i", helpers::new_activation as fn(i64) -> i64 as usize)?;
    bind_idx(&env, "pushscope_i", helpers::push_scope as fn(i64) as usize)?;
    bind_idx(&env, "popscope_i", helpers::pop_scope as fn() as usize)?;
    bind_idx(&env, "getscope_i", helpers::get_scope as fn(i32) -> i64 as usize)?;
    bind_idx(&env, "construct_i", helpers::construct as fn(i64, i64, i32) -> i64 as usize)?;
    bind_idx(&env, "delprop_i", helpers::delete_property as fn(i64, i64) -> i64 as usize)?;
    bind_idx(&env, "istype_i", helpers::is_type_late as fn(i64, i64) -> i64 as usize)?;
    bind_idx(&env, "astype_i", helpers::as_type_late as fn(i64, i64) -> i64 as usize)?;
    bind_idx(&env, "getsuper_i", helpers::get_super as fn(i64, i64) -> i64 as usize)?;
    bind_idx(&env, "setsuper_i", helpers::set_super as fn(i64, i64, i64) as usize)?;
    bind_idx(&env, "callsuper_i", helpers::call_super as fn(i64, i64, i64, i64) -> i64 as usize)?;
    bind_idx(&env, "constructsuper_i", helpers::construct_super as fn(i64, i64, i32) -> i64 as usize)?;
    bind_idx(&env, "callnative_i", helpers::call_native as fn(i64, i64, i64, i64) -> i64 as usize)?;
    bind_idx(&env, "getpropfast_i", helpers::get_prop_index as fn(i64, i64, i64) -> i64 as usize)?;
    bind_idx(&env, "setpropfast_i", helpers::set_prop_index as fn(i64, i64, i64, i64) -> i64 as usize)?;
    bind_idx(&env, "in_i", helpers::op_in as fn(i64, i64) -> i64 as usize)?;
    bind_idx(&env, "nextvalue_i", helpers::next_value as fn(i64, i64) -> i64 as usize)?;
    bind_idx(&env, "nextname_i", helpers::next_name as fn(i64, i64) -> i64 as usize)?;
    bind_idx(&env, "hasnext_i", helpers::has_next as fn(i64, i64) -> i64 as usize)?;
    bind_idx(&env, "newfunction_i", helpers::new_function as fn(i64) -> i64 as usize)?;
    bind_idx(&env, "applytype_i", helpers::apply_type as fn(i64, i64, i32) -> i64 as usize)?;
    bind_idx(&env, "constructslot_i", helpers::construct_slot as fn(i64, i32, i64, i32) -> i64 as usize)?;
    bind_idx(&env, "constructprop_i", helpers::construct_prop as fn(i64, i64, i64, i64) -> i64 as usize)?;
    bind_idx(&env, "call_i", helpers::call_fn as fn(i64, i64, i64, i64) -> i64 as usize)?;
    bind_idx(&env, "newobject_i", helpers::new_object as fn(i64, i32) -> i64 as usize)?;
    bind_idx(&env, "hasnext2_i", helpers::has_next_2 as fn(i32, i32, i64) -> i64 as usize)?;
    bind_idx(&env, "throw_i", helpers::throw_value as fn(i64) -> i64 as usize)?;
    bind_idx(&env, "gschecked_i", helpers::get_slot_checked as fn(i64, i64) -> i64 as usize)?;
    bind_idx(&env, "mopload_i", helpers::mop_load as fn(i64, i32) -> i64 as usize)?;
    bind_idx(&env, "mopstore_i", helpers::mop_store as fn(i64, i64, i32) -> i64 as usize)?;
    bind_idx(&env, "gp_ic_i", helpers::get_property_ic as fn(i64, i64, i64) -> i64 as usize)?;
    bind_idx(&env, "dmdesc_i", helpers::dm_desc_ptr as fn() -> i64 as usize)?;
    Reflect::set(&env, &"memory".into(), memory).ok()?;
    let imports = Object::new();
    Reflect::set(&imports, &"env".into(), &env).ok()?;

    let instance = WebAssembly::Instance::new(&module, &imports).ok()?;
    Reflect::get(&instance.exports(), &"run".into())
        .ok()?
        .dyn_into::<js_sys::Function>()
        .ok()
}

/// Binds an imported i32 index global `name = table_index` (a plain Number satisfies an
/// immutable i32 global import).
fn bind_idx(env: &Object, name: &str, table_index: usize) -> Option<()> {
    Reflect::set(env, &name.into(), &JsValue::from(table_index as u32)).ok()?;
    Some(())
}
