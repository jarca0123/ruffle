//! Native execution of an emitted module via wasmtime (cranelift → machine code).
//!
//! This is the differential-harness / desktop execution path; on the web the same
//! emitted bytes are handed to the browser's `WebAssembly` engine instead. The
//! step-1 module imports nothing (arguments arrive as typed WASM params, the
//! result leaves as one `i64`), so instantiation needs no host imports.

use wasmtime::{Engine, Extern, Func, Instance, Module, Store, Val};

use crate::emit::{ENTRY_NAME, MEMORY_NAME};
use crate::{Arg, Helpers};

const WASM_PAGE: usize = 65536;

/// Convert a target-agnostic [`Arg`] to a wasmtime `Val` (wasmtime holds an `f64`
/// as its raw `u64` bits).
fn to_val(a: &Arg) -> Val {
    match *a {
        Arg::I32(v) => Val::I32(v),
        Arg::F64(v) => Val::F64(v.to_bits()),
        Arg::I64(v) => Val::I64(v),
    }
}

/// A compiled-and-validated module, cached per AVM2 method.
pub struct Program {
    module: Module,
}

/// Owns the wasmtime `Engine` (its compiled-code cache and allocator) for the
/// lifetime of the JIT backend.
pub struct Runner {
    engine: Engine,
}

impl Runner {
    pub fn new() -> Self {
        Runner {
            engine: Engine::default(),
        }
    }

    /// Compile and validate emitted module bytes. `None` if the bytes fail to
    /// validate — which would be an emitter bug, so callers treat it as a decline
    /// rather than trusting unvalidated code.
    pub fn compile(&self, bytes: &[u8]) -> Option<Program> {
        Module::new(&self.engine, bytes)
            .ok()
            .map(|module| Program { module })
    }

    /// Instantiate, providing the helper imports the module declares, in the
    /// canonical order `[get_slot, set_slot]`.
    fn instantiate(
        &self,
        store: &mut Store<()>,
        program: &Program,
        h: Helpers,
    ) -> Option<Instance> {
        // Canonical import order: [get_slot, set_slot, to_int32].
        let mut imports: Vec<Extern> = Vec::new();
        if h.get_slot {
            // env.gs = get_slot(obj_bits, slot_id) -> value_bits (pure).
            let gs = Func::wrap(&mut *store, |bits: i64, id: i32| -> i64 {
                crate::runtime::get_slot(bits as u64, id as u32) as i64
            });
            imports.push(gs.into());
        }
        if h.set_slot {
            // env.ss = set_slot(obj_bits, value_bits, slot_id) -> (); reads the GC
            // mutation from the `with_mc` thread-local the caller established.
            let ss = Func::wrap(&mut *store, |obj: i64, val: i64, id: i32| {
                crate::runtime::set_slot(obj as u64, val as u64, id as u32)
            });
            imports.push(ss.into());
        }
        if h.to_int32 {
            // env.ti = to_int32(n: f64) -> i32 (ECMAScript ToInt32, pure).
            let ti = Func::wrap(&mut *store, |n: f64| -> i32 { crate::runtime::to_int32(n) });
            imports.push(ti.into());
        }
        if h.calls {
            // ca/cm/pe: the call helpers (need the live Activation via with_activation).
            let ca = Func::wrap(&mut *store, |bits: i64| crate::runtime::push_call_arg(bits as u64));
            imports.push(ca.into());
            let cm = Func::wrap(&mut *store, |recv: i64, disp: i32, argc: i32| -> i64 {
                crate::runtime::call_method(recv as u64, disp as u32, argc as u32) as i64
            });
            imports.push(cm.into());
            let pe = Func::wrap(&mut *store, || -> i32 { crate::runtime::pending_error() });
            imports.push(pe.into());
        }
        Instance::new(store, &program.module, &imports).ok()
    }

    /// Instantiate and call `run(args) -> i64`, returning the result's `Value`
    /// bits. `None` on any instantiation/trap error (declines to the interpreter).
    pub fn run(&self, program: &Program, args: &[Arg], h: Helpers) -> Option<u64> {
        let mut store = Store::new(&self.engine, ());
        let instance = self.instantiate(&mut store, program, h)?;
        let func = instance.get_func(&mut store, ENTRY_NAME)?;
        let vals: Vec<Val> = args.iter().map(to_val).collect();
        let mut results = [Val::I64(0)];
        func.call(&mut store, &vals, &mut results).ok()?;
        match results[0] {
            Val::I64(b) => Some(b as u64),
            _ => None,
        }
    }

    /// Run a domainMemory method: fill the module's linear memory with `dm`, call
    /// `run(args)` (whose last arg is `dm_len`), then read the memory back into
    /// `dm` so the caller can persist stores. Returns the result's `Value` bits.
    pub fn run_dm(&self, program: &Program, args: &[Arg], dm: &mut [u8], h: Helpers) -> Option<u64> {
        let mut store = Store::new(&self.engine, ());
        let instance = self.instantiate(&mut store, program, h)?;
        let memory = instance.get_memory(&mut store, MEMORY_NAME)?;

        // Grow to fit the domain memory if needed, then copy it in.
        let need = dm.len();
        let have = memory.data_size(&store);
        if need > have {
            let pages = (need - have).div_ceil(WASM_PAGE) as u64;
            memory.grow(&mut store, pages).ok()?;
        }
        memory.data_mut(&mut store)[..need].copy_from_slice(dm);

        let func = instance.get_func(&mut store, ENTRY_NAME)?;
        let vals: Vec<Val> = args.iter().map(to_val).collect();
        let mut results = [Val::I64(0)];
        func.call(&mut store, &vals, &mut results).ok()?;

        dm.copy_from_slice(&memory.data(&store)[..need]);
        match results[0] {
            Val::I64(b) => Some(b as u64),
            _ => None,
        }
    }
}
