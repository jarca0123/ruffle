//! Native execution of a compiled type-0 module (wasmtime/cranelift → machine code).
//!
//! One instantiated module per method, keyed by method identity (thread-local cache).
//! The module imports its frame memory + the `gs`/`cr` helpers; each call writes the
//! frame at `DEPTH * STRIDE` and invokes `run(env, argc, args)`. Placing the frame at a
//! per-nesting-level offset means a helper that re-enters AS3 (a `cr` coercion's
//! `valueOf` calling another JIT method) never aliases a live frame. (Native uses a
//! per-method memory — no address-space pressure there; the browser runner shares one.)

use std::cell::Cell;
use std::cell::RefCell;
use std::rc::Rc;

use wasmtime::{
    Caller, Engine, Func, Global, GlobalType, HeapType, Instance, Memory, MemoryType, Module,
    Mutability, Ref, RefType, Store, Table, TableType, TypedFunc, Val, ValType,
};

use crate::emit::{FRAME_PAGES, FRAME_STRIDE, UNDEFINED_BITS};
use crate::helpers;

struct Compiled {
    store: Store<()>,
    memory: Memory,
    run: TypedFunc<(i32, i32, i32), i64>,
}

/// A compiled method's per-instance state, held by the JIT in `CompiledMethod` (one per
/// method) and handed to [`run_leaf`] directly — so the runner no longer re-hashes the method
/// by key on every call (the double cache lookup is gone). Opaque to the JIT (private field):
/// it only stores/clones it. `None` until first run (lazy `build`). `Rc<RefCell<..>>` so a
/// re-entrant callee can hold its own handle while this one is borrowed, and so direct
/// recursion is detected via `try_borrow_mut` (see [`run_leaf`]).
#[derive(Clone)]
pub struct Handle(Rc<RefCell<Option<Compiled>>>);

/// Allocates a fresh (unbuilt) handle for a newly compiled method.
pub fn new_handle() -> Handle {
    Handle(Rc::new(RefCell::new(None)))
}

/// Logs a first-sighting declined op (see `translate::record_decline`). Native: stderr,
/// gated on `RUFFLE_JIT3_TRACE` so the test suite stays quiet.
pub fn log_decline(name: &str) {
    if std::env::var_os("RUFFLE_JIT3_TRACE").is_some() {
        eprintln!("JIT3 DECLINE (new op): {name}");
    }
}

thread_local! {
    // `Engine::default()` compiles every emitted method module with Cranelift, which
    // dominated the native thread (profiled ~63%: `cranelift_codegen` optimization +
    // `regalloc2` register allocation) — we compile one tiny module PER METHOD and rarely
    // re-run it enough to amortize an OPTIMIZING backend. Note `OptLevel::None` is NOT
    // enough: Cranelift runs `regalloc2` at every opt level (you must allocate registers to
    // emit code), so it only removes the mid-end passes (~half). WINCH is the fix — wasmtime's
    // baseline SINGLE-PASS compiler with NO `regalloc2` (trivial stack allocation), dropping
    // both hot subsystems. Slightly slower code, dramatically faster compile — the right
    // trade for a per-method JIT. A module Winch can't compile just fails `Module::new` →
    // `run_leaf` returns `None` → that method declines to the interpreter (correctness kept).
    // Web is unaffected (the browser compiles the emitted modules, not wasmtime).
    static ENGINE: Engine = {
        let mut cfg = wasmtime::Config::new();
        cfg.strategy(wasmtime::Strategy::Winch);
        // Fall back to the default (Cranelift) engine if Winch isn't available (feature off,
        // or an unsupported host arch) instead of panicking the whole player.
        Engine::new(&cfg).unwrap_or_default()
    };
    /// Current frame nesting level; a call runs at `DEPTH * STRIDE` and bumps it.
    static DEPTH: Cell<u32> = const { Cell::new(0) };
}

/// Builds the method's instance on first use (lazily, into its own `handle`), writes `frame`
/// at `DEPTH*STRIDE`, calls `run(0, argc, args)`, and returns the result `Value` bits. `None`
/// on failure. Takes the method's `handle` directly — no per-call key hashing.
pub fn run_leaf(handle: &Handle, bytes: &[u8], frame: &[u64], argc: u32) -> Option<u64> {
    // The common A→B re-entry borrows a DIFFERENT method's handle. Only DIRECT recursion (this
    // method already running → its handle already borrowed) hits the `Err` arm → run a
    // throwaway instance for that level (rare; churn not correctness).
    match handle.0.try_borrow_mut() {
        Ok(mut slot) => {
            if slot.is_none() {
                *slot = Some(build(bytes)?); // first call: compile + instantiate
            }
            // SAFETY of unwrap: just ensured `Some`.
            run_compiled(slot.as_mut().unwrap(), frame, argc)
        }
        Err(_) => run_compiled(&mut build(bytes)?, frame, argc),
    }
}

/// Writes the frame at `DEPTH*STRIDE`, calls `run`, restores DEPTH. No CACHE borrow held.
fn run_compiled(c: &mut Compiled, frame: &[u64], argc: u32) -> Option<u64> {
    let depth = DEPTH.with(|d| d.get());
    let args = depth.checked_mul(FRAME_STRIDE)?;
    if (args as usize) + frame.len() * 8 > (FRAME_PAGES as usize) * 65536 {
        return None; // frame nesting overflow → decline (interpreter runs it)
    }
    // The frame `Value`s are already little-endian `u64`s in memory (as is the wasm frame);
    // reinterpret as bytes instead of allocating + copying a fresh `Vec` every call.
    // SAFETY: `[u64]` → `[u8]` is a valid reinterpret; all Ruffle targets are little-endian
    // (wasm + x86), matching the previous `to_le_bytes` encoding.
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(frame.as_ptr() as *const u8, frame.len() * 8) };
    c.memory.write(&mut c.store, args as usize, bytes).ok()?;
    DEPTH.with(|d| d.set(depth + 1));
    let result = c.run.call(&mut c.store, (0, argc as i32, args as i32)).ok();
    DEPTH.with(|d| d.set(depth));
    result.map(|r| r as u64)
}

/// Instantiates a fresh module instance with all helper imports (once per method normally).
fn build(bytes: &[u8]) -> Option<Compiled> {
    ENGINE.with(|engine| -> Option<Compiled> {
                let module = Module::new(engine, bytes).ok()?;
                let mut store = Store::new(engine, ());
                let gs = Func::wrap(&mut store, |o: i64, s: i64| -> i64 { helpers::get_slot(o, s) });
                let cr = Func::wrap(&mut store, |v: i64, c: i64| -> i64 { helpers::coerce_return(v, c) });
                let gp = Func::wrap(&mut store, |r: i64, m: i64| -> i64 { helpers::get_property(r, m) });
                // `cp` reads the `n` outgoing args from the (re-exported) frame memory at
                // `off` via `Caller`, then delegates — one crossing, not one per arg.
                let cp = Func::wrap(
                    &mut store,
                    |mut caller: Caller<'_, ()>, r: i64, m: i64, off: i64, n: i64| -> i64 {
                        let Some(mem) =
                            caller.get_export("memory").and_then(|e| e.into_memory())
                        else {
                            return UNDEFINED_BITS as i64;
                        };
                        let mut bits = Vec::with_capacity(n as usize);
                        for j in 0..n as usize {
                            let mut b = [0u8; 8];
                            if mem.read(&caller, off as usize + j * 8, &mut b).is_err() {
                                return UNDEFINED_BITS as i64;
                            }
                            bits.push(i64::from_le_bytes(b));
                        }
                        // SAFETY: `bits` are `Value`s the JIT stored this frame; `m` a live mn.
                        unsafe { helpers::call_property_bits(r, m, &bits) }
                    },
                );
                let perr = Func::wrap(&mut store, || -> i32 { helpers::pending_error() });
                let binop = Func::wrap(&mut store, |a: i64, b: i64, op: i32| -> i64 {
                    helpers::binop(a, b, op)
                });
                let unop =
                    Func::wrap(&mut store, |a: i64, op: i32| -> i64 { helpers::unop(a, op) });
                let truthy = Func::wrap(&mut store, |a: i64| -> i32 { helpers::truthy(a) });
                let setprop = Func::wrap(&mut store, |r: i64, m: i64, v: i64| {
                    helpers::set_property(r, m, v)
                });
                let setslot = Func::wrap(&mut store, |r: i64, i: i32, v: i64, mode: i32| {
                    helpers::set_slot(r, i, v, mode)
                });
                // `call_method` reads its args from the (re-exported) frame memory via `Caller`,
                // like `cp`.
                let callmethod = Func::wrap(
                    &mut store,
                    |mut caller: Caller<'_, ()>, r: i64, id: i32, off: i64, n: i32| -> i64 {
                        let Some(mem) = caller.get_export("memory").and_then(|e| e.into_memory())
                        else {
                            return UNDEFINED_BITS as i64;
                        };
                        let mut bits = Vec::with_capacity(n as usize);
                        for j in 0..n as usize {
                            let mut b = [0u8; 8];
                            if mem.read(&caller, off as usize + j * 8, &mut b).is_err() {
                                return UNDEFINED_BITS as i64;
                            }
                            bits.push(i64::from_le_bytes(b));
                        }
                        // SAFETY: `bits` are `Value`s the JIT stored this frame.
                        unsafe { helpers::call_method_bits(r, id, &bits) }
                    },
                );
                let newarray = Func::wrap(
                    &mut store,
                    |mut caller: Caller<'_, ()>, off: i64, n: i32| -> i64 {
                        let Some(mem) = caller.get_export("memory").and_then(|e| e.into_memory())
                        else {
                            return UNDEFINED_BITS as i64;
                        };
                        let mut bits = Vec::with_capacity(n as usize);
                        for j in 0..n as usize {
                            let mut b = [0u8; 8];
                            if mem.read(&caller, off as usize + j * 8, &mut b).is_err() {
                                return UNDEFINED_BITS as i64;
                            }
                            bits.push(i64::from_le_bytes(b));
                        }
                        // SAFETY: `bits` are `Value`s the JIT stored this frame.
                        unsafe { helpers::new_array_bits(&bits) }
                    },
                );
                let outerscope = Func::wrap(&mut store, |i: i32| -> i64 { helpers::outer_scope(i) });
                let scriptglobals =
                    Func::wrap(&mut store, |p: i64| -> i64 { helpers::script_globals(p) });
                let newactivation =
                    Func::wrap(&mut store, |p: i64| -> i64 { helpers::new_activation(p) });
                let pushscope = Func::wrap(&mut store, |b: i64| helpers::push_scope(b));
                let popscope = Func::wrap(&mut store, || helpers::pop_scope());
                let getscope = Func::wrap(&mut store, |i: i32| -> i64 { helpers::get_scope(i) });
                let construct = Func::wrap(
                    &mut store,
                    |mut caller: Caller<'_, ()>, ctor: i64, off: i64, n: i32| -> i64 {
                        let Some(mem) = caller.get_export("memory").and_then(|e| e.into_memory())
                        else {
                            return UNDEFINED_BITS as i64;
                        };
                        let mut bits = Vec::with_capacity(n as usize);
                        for j in 0..n as usize {
                            let mut b = [0u8; 8];
                            if mem.read(&caller, off as usize + j * 8, &mut b).is_err() {
                                return UNDEFINED_BITS as i64;
                            }
                            bits.push(i64::from_le_bytes(b));
                        }
                        // SAFETY: `bits` are `Value`s the JIT stored this frame.
                        unsafe { helpers::construct_bits(ctor, &bits) }
                    },
                );
                let delprop = Func::wrap(&mut store, |r: i64, m: i64| -> i64 {
                    helpers::delete_property(r, m)
                });
                let istype = Func::wrap(&mut store, |v: i64, t: i64| -> i64 {
                    helpers::is_type_late(v, t)
                });
                let astype = Func::wrap(&mut store, |v: i64, c: i64| -> i64 {
                    helpers::as_type_late(v, c)
                });
                let getsuper =
                    Func::wrap(&mut store, |r: i64, m: i64| -> i64 { helpers::get_super(r, m) });
                let setsuper = Func::wrap(&mut store, |r: i64, m: i64, v: i64| {
                    helpers::set_super(r, m, v)
                });
                let callsuper = Func::wrap(
                    &mut store,
                    |mut caller: Caller<'_, ()>, r: i64, m: i64, off: i64, n: i64| -> i64 {
                        let Some(mem) = caller.get_export("memory").and_then(|e| e.into_memory())
                        else {
                            return UNDEFINED_BITS as i64;
                        };
                        let mut bits = Vec::with_capacity(n as usize);
                        for j in 0..n as usize {
                            let mut b = [0u8; 8];
                            if mem.read(&caller, off as usize + j * 8, &mut b).is_err() {
                                return UNDEFINED_BITS as i64;
                            }
                            bits.push(i64::from_le_bytes(b));
                        }
                        // SAFETY: `bits` are `Value`s the JIT stored this frame.
                        unsafe { helpers::call_super_bits(r, m, &bits) }
                    },
                );
                let constructsuper = Func::wrap(
                    &mut store,
                    |mut caller: Caller<'_, ()>, r: i64, off: i64, n: i32| -> i64 {
                        let Some(mem) = caller.get_export("memory").and_then(|e| e.into_memory())
                        else {
                            return UNDEFINED_BITS as i64;
                        };
                        let mut bits = Vec::with_capacity(n as usize);
                        for j in 0..n as usize {
                            let mut b = [0u8; 8];
                            if mem.read(&caller, off as usize + j * 8, &mut b).is_err() {
                                return UNDEFINED_BITS as i64;
                            }
                            bits.push(i64::from_le_bytes(b));
                        }
                        // SAFETY: `bits` are `Value`s the JIT stored this frame.
                        unsafe { helpers::construct_super_bits(r, &bits) }
                    },
                );
                let callnative = Func::wrap(
                    &mut store,
                    |mut caller: Caller<'_, ()>, r: i64, m: i64, off: i64, n: i64| -> i64 {
                        let Some(mem) = caller.get_export("memory").and_then(|e| e.into_memory())
                        else {
                            return UNDEFINED_BITS as i64;
                        };
                        let mut bits = Vec::with_capacity(n as usize);
                        for j in 0..n as usize {
                            let mut b = [0u8; 8];
                            if mem.read(&caller, off as usize + j * 8, &mut b).is_err() {
                                return UNDEFINED_BITS as i64;
                            }
                            bits.push(i64::from_le_bytes(b));
                        }
                        // SAFETY: `bits` are `Value`s the JIT stored this frame; `m` a live fn-ptr.
                        unsafe { helpers::call_native_bits(r, m, &bits) }
                    },
                );
                // Helpers are reached via `call_indirect` through an imported table (mirrors
                // web, where the table is the main module's — keeping helper calls in-wasm).
                // Slots: [gs,cr,gp,cp,perr,binop,unop,truthy,setprop,setslot]; globals name each.
                let getpropfast = Func::wrap(&mut store, |o: i64, n: i64, m: i64| -> i64 {
                    helpers::get_prop_index(o, n, m)
                });
                let setpropfast = Func::wrap(&mut store, |o: i64, n: i64, v: i64, m: i64| -> i64 {
                    helpers::set_prop_index(o, n, v, m)
                });
                let op_in = Func::wrap(&mut store, |n: i64, v: i64| -> i64 { helpers::op_in(n, v) });
                let nextvalue =
                    Func::wrap(&mut store, |v: i64, i: i64| -> i64 { helpers::next_value(v, i) });
                let nextname =
                    Func::wrap(&mut store, |v: i64, i: i64| -> i64 { helpers::next_name(v, i) });
                let hasnext =
                    Func::wrap(&mut store, |v: i64, i: i64| -> i64 { helpers::has_next(v, i) });
                let newfunction =
                    Func::wrap(&mut store, |p: i64| -> i64 { helpers::new_function(p) });
                let applytype = Func::wrap(
                    &mut store,
                    |mut caller: Caller<'_, ()>, base: i64, off: i64, n: i32| -> i64 {
                        let Some(mem) = caller.get_export("memory").and_then(|e| e.into_memory())
                        else {
                            return UNDEFINED_BITS as i64;
                        };
                        let mut bits = Vec::with_capacity(n as usize);
                        for j in 0..n as usize {
                            let mut b = [0u8; 8];
                            if mem.read(&caller, off as usize + j * 8, &mut b).is_err() {
                                return UNDEFINED_BITS as i64;
                            }
                            bits.push(i64::from_le_bytes(b));
                        }
                        // SAFETY: `bits` are `Value`s the JIT stored this frame.
                        unsafe { helpers::apply_type_bits(base, &bits) }
                    },
                );
                let constructslot = Func::wrap(
                    &mut store,
                    |mut caller: Caller<'_, ()>, src: i64, index: i32, off: i64, n: i32| -> i64 {
                        let Some(mem) = caller.get_export("memory").and_then(|e| e.into_memory())
                        else {
                            return UNDEFINED_BITS as i64;
                        };
                        let mut bits = Vec::with_capacity(n as usize);
                        for j in 0..n as usize {
                            let mut b = [0u8; 8];
                            if mem.read(&caller, off as usize + j * 8, &mut b).is_err() {
                                return UNDEFINED_BITS as i64;
                            }
                            bits.push(i64::from_le_bytes(b));
                        }
                        // SAFETY: `bits` are `Value`s the JIT stored this frame.
                        unsafe { helpers::construct_slot_bits(src, index, &bits) }
                    },
                );
                let constructprop = Func::wrap(
                    &mut store,
                    |mut caller: Caller<'_, ()>, src: i64, m: i64, off: i64, n: i64| -> i64 {
                        let Some(mem) = caller.get_export("memory").and_then(|e| e.into_memory())
                        else {
                            return UNDEFINED_BITS as i64;
                        };
                        let mut bits = Vec::with_capacity(n as usize);
                        for j in 0..n as usize {
                            let mut b = [0u8; 8];
                            if mem.read(&caller, off as usize + j * 8, &mut b).is_err() {
                                return UNDEFINED_BITS as i64;
                            }
                            bits.push(i64::from_le_bytes(b));
                        }
                        // SAFETY: `bits` are `Value`s the JIT stored this frame.
                        unsafe { helpers::construct_prop_bits(src, m, &bits) }
                    },
                );
                let call = Func::wrap(
                    &mut store,
                    |mut caller: Caller<'_, ()>, f: i64, r: i64, off: i64, n: i64| -> i64 {
                        let Some(mem) = caller.get_export("memory").and_then(|e| e.into_memory())
                        else {
                            return UNDEFINED_BITS as i64;
                        };
                        let mut bits = Vec::with_capacity(n as usize);
                        for j in 0..n as usize {
                            let mut b = [0u8; 8];
                            if mem.read(&caller, off as usize + j * 8, &mut b).is_err() {
                                return UNDEFINED_BITS as i64;
                            }
                            bits.push(i64::from_le_bytes(b));
                        }
                        // SAFETY: `bits` are `Value`s the JIT stored this frame.
                        unsafe { helpers::call_fn_bits(f, r, &bits) }
                    },
                );
                let newobject = Func::wrap(
                    &mut store,
                    |mut caller: Caller<'_, ()>, off: i64, num_pairs: i32| -> i64 {
                        let Some(mem) = caller.get_export("memory").and_then(|e| e.into_memory())
                        else {
                            return UNDEFINED_BITS as i64;
                        };
                        let count = num_pairs as usize * 2;
                        let mut bits = Vec::with_capacity(count);
                        for j in 0..count {
                            let mut b = [0u8; 8];
                            if mem.read(&caller, off as usize + j * 8, &mut b).is_err() {
                                return UNDEFINED_BITS as i64;
                            }
                            bits.push(i64::from_le_bytes(b));
                        }
                        // SAFETY: `bits` are `Value`s the JIT stored this frame.
                        unsafe { helpers::new_object_bits(&bits) }
                    },
                );
                let hasnext2 = Func::wrap(
                    &mut store,
                    |mut caller: Caller<'_, ()>, obj_reg: i32, idx_reg: i32, frame_off: i64| -> i64 {
                        let Some(mem) = caller.get_export("memory").and_then(|e| e.into_memory())
                        else {
                            return UNDEFINED_BITS as i64;
                        };
                        let obj_off = frame_off as usize + obj_reg as usize * 8;
                        let idx_off = frame_off as usize + idx_reg as usize * 8;
                        let (mut ob, mut ib) = ([0u8; 8], [0u8; 8]);
                        if mem.read(&caller, obj_off, &mut ob).is_err()
                            || mem.read(&caller, idx_off, &mut ib).is_err()
                        {
                            return UNDEFINED_BITS as i64;
                        }
                        // SAFETY: `ob`/`ib` are `Value`s the JIT stored in this frame's locals.
                        let (result, new_idx, new_obj) = unsafe {
                            helpers::has_next_2_bits(
                                i64::from_le_bytes(ob),
                                i64::from_le_bytes(ib),
                            )
                        };
                        let _ = mem.write(&mut caller, idx_off, &new_idx.to_le_bytes());
                        let _ = mem.write(&mut caller, obj_off, &new_obj.to_le_bytes());
                        result
                    },
                );
                let throw = Func::wrap(&mut store, |v: i64| -> i64 { helpers::throw_value(v) });
                let gschecked =
                    Func::wrap(&mut store, |o: i64, s: i64| -> i64 { helpers::get_slot_checked(o, s) });
                let mopload =
                    Func::wrap(&mut store, |a: i64, c: i32| -> i64 { helpers::mop_load(a, c) });
                let mopstore = Func::wrap(&mut store, |v: i64, a: i64, c: i32| -> i64 {
                    helpers::mop_store(v, a, c)
                });
                // The property inline cache is web-only (it reads the GC object layout in the
                // shared linear memory); the native build never emits the inline guard, but the
                // table slot must exist so the helper index globals line up (NUM_HELPERS).
                let gp_ic = Func::wrap(&mut store, |r: i64, m: i64, c: i64| -> i64 {
                    helpers::get_property_ic(r, m, c)
                });
                // Inline domainMemory (`li*`/`si*`) is web-only; native never emits it, but the
                // table slot must exist so the helper index globals line up (NUM_HELPERS).
                let dm_desc = Func::wrap(&mut store, || -> i64 { helpers::dm_desc_ptr() });
                let call_prop_ic = Func::wrap(&mut store, |r: i64, m: i64, o: i64, n: i64, ic: i64| -> i64 {
                    helpers::call_prop_ic(r, m, o, n, ic)
                });
                let call_method_ic = Func::wrap(&mut store, |r: i64, d: i64, o: i64, n: i64, ic: i64| -> i64 {
                    helpers::call_method_ic(r, d, o, n, ic)
                });
                let table = Table::new(
                    &mut store,
                    TableType::new(RefType::new(true, HeapType::Func), 48, Some(48)),
                    Ref::Func(None),
                )
                .ok()?;
                let helpers_tbl: [Func; 48] = [
                    gs, cr, gp, cp, perr, binop, unop, truthy, setprop, setslot, callmethod,
                    newarray, outerscope, scriptglobals, newactivation, pushscope, popscope,
                    getscope, construct, delprop, istype, astype, getsuper, setsuper, callsuper,
                    constructsuper, callnative, getpropfast, setpropfast, op_in, nextvalue,
                    nextname, hasnext, newfunction, applytype, constructslot, constructprop, call,
                    newobject, hasnext2, throw, gschecked, mopload, mopstore, gp_ic, dm_desc,
                    call_prop_ic, call_method_ic,
                ];
                for (i, f) in helpers_tbl.into_iter().enumerate() {
                    table.set(&mut store, i as u64, Ref::Func(Some(f))).ok()?;
                }
                let idx = |store: &mut Store<()>, i: i32| {
                    Global::new(store, GlobalType::new(ValType::I32, Mutability::Const), Val::I32(i))
                };
                let mut imports: Vec<wasmtime::Extern> = vec![table.into()];
                for i in 0..48 {
                    imports.push(idx(&mut store, i).ok()?.into());
                }
                let memory =
                    Memory::new(&mut store, MemoryType::new(FRAME_PAGES, Some(FRAME_PAGES))).ok()?;
                imports.push(memory.into());
                let instance = Instance::new(&mut store, &module, &imports).ok()?;
                let run = instance
                    .get_typed_func::<(i32, i32, i32), i64>(&mut store, "run")
                    .ok()?;
                Some(Compiled { store, memory, run })
            })
}
