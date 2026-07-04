//! Execution of a compiled JIT module.
//!
//! - **Native** (desktop / tests): run the emitted `run(state_ptr) -> i64`
//!   through the `wasmi` interpreter. The frame is *copied* into a fresh WASM
//!   memory: registers `[0..num_locals]` are written as 8-byte `Value` slots at
//!   offset 0, `run(0)` is called, and the returned `i64` is the result
//!   `Value`'s bits. No copy-back is needed — the method's frame is discarded
//!   once it returns, so only the return value matters. (A production desktop
//!   JIT would instead lower to native code via e.g. cranelift; wasmi keeps the
//!   prototype honest and gives us JIT↔interpreter equivalence tests.)
//! - **Web**: execution goes through the browser's own WASM engine over
//!   Ruffle's *shared* linear memory (zero-copy), driven from JS. Not wired up
//!   here yet, so the web path declines.

/// Runs the compiled module `bytes` with `regs` as the initial frame slots
/// (register `i` = `regs[i]`, an 8-byte `Value` bit pattern), returning the
/// result `Value`'s bits. `None` if the module fails to compile/instantiate/run.
#[cfg(not(target_arch = "wasm32"))]
pub fn run(bytes: &[u8], regs: &[u64]) -> Option<u64> {
    use wasmi::{Engine, Instance, Memory, MemoryType, Module, Store};

    let engine = Engine::default();
    let module = Module::new(&engine, bytes).ok()?;
    let mut store = Store::new(&engine, ());

    // One 64 KiB page holds 8192 slots — far more than any real frame.
    let memory = Memory::new(&mut store, MemoryType::new(1, None).ok()?).ok()?;
    let mut buf = Vec::with_capacity(regs.len() * 8);
    for r in regs {
        buf.extend_from_slice(&r.to_le_bytes());
    }
    memory.write(&mut store, 0, &buf).ok()?;

    // The module has exactly one import — the shared memory as ("env","memory").
    let instance = Instance::new(&mut store, &module, &[memory.into()]).ok()?;
    let run = instance.get_typed_func::<i32, i64>(&store, "run").ok()?;
    Some(run.call(&mut store, 0).ok()? as u64)
}

/// Web execution is not implemented yet — the JS/browser bridge (shared memory +
/// `WebAssembly.Instance`) lands separately. Declining here keeps the
/// interpreter authoritative on the web.
#[cfg(target_arch = "wasm32")]
pub fn run(_bytes: &[u8], _regs: &[u64]) -> Option<u64> {
    None
}
