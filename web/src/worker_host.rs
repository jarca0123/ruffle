//! Web implementation of [`WorkerHost`](ruffle_core::worker_host::WorkerHost).
//!
//! Spawns each Flash worker as a Web Worker running the same wasm module over a
//! shared `WebAssembly.Memory` (via `wasm_thread`). Requires a threaded wasm
//! build (`+atomics`, shared memory) and a cross-origin-isolated page — hence it
//! is gated behind the `threads` cargo feature. Without that feature the player
//! falls back to `NullWorkerHost` and `Worker.isSupported` is `false`.

#[cfg(feature = "threads")]
use ruffle_core::worker_host::WorkerHost;

/// Runs worker runtimes on Web Workers sharing this module's memory.
#[cfg(feature = "threads")]
pub struct WasmWorkerHost;

#[cfg(feature = "threads")]
impl WorkerHost for WasmWorkerHost {
    fn spawn(&self, entry: Box<dyn FnOnce() + Send + 'static>) {
        // `wasm_thread` packages the wasm-bindgen worker/TLS/stack bootstrap; the
        // closure runs on a fresh Web Worker thread, where blocking (Atomics.wait,
        // std Mutex/Condvar) is permitted.
        tracing::info!("WasmWorkerHost::spawn: creating nested Web Worker via wasm_thread");
        let result = wasm_thread::Builder::new().spawn(move || {
            // Runs *inside* the freshly-spawned worker — if this line never logs,
            // the worker was created but failed to bootstrap the wasm module.
            tracing::info!("WasmWorkerHost: nested worker thread started");
            entry();
        });
        match result {
            Ok(_) => tracing::info!("WasmWorkerHost::spawn: wasm_thread::spawn returned Ok"),
            Err(e) => tracing::error!("WasmWorkerHost::spawn: wasm_thread::spawn failed: {e:?}"),
        }
    }
}
