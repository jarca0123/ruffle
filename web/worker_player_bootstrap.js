// Worker bootstrap for the player-in-worker path (browser OpenTTD).
//
// This module runs on the dedicated worker that hosts the *primordial* Ruffle
// player. It shares the main thread's `WebAssembly.Memory` (so all the `Arc`
// handles — the `WorkerBridge`, the `WorkerInit` box — resolve to the same
// bytes) and receives the on-page canvas transferred as an `OffscreenCanvas`.
//
// The main thread starts it via `ruffle_start_worker_player(canvas, swf, url,
// workerUrl)` (Rust, in `worker_shell.rs`), which does:
//   const worker = new Worker(workerUrl, { type: "module" });
//   worker.postMessage({ module, memory, canvas, initPtr }, [canvas]);
//
// where `module` = `wasm_bindgen::module()` (the compiled WebAssembly.Module),
// `memory` = `wasm_bindgen::memory()` (the shared memory), `canvas` = the
// transferred OffscreenCanvas, `initPtr` = a pointer into the shared memory.
//
// IMPORTANT — build wiring:
//   * Serve this file (and the wasm glue it imports) same-origin, under the same
//     COOP/COEP cross-origin-isolation headers as the main page (required for
//     SharedArrayBuffer / threads).
//   * build_wasm.ts copies this file into `dist/ruffle_worker_player.js`, next
//     to the wasm-bindgen glue (`ruffle_web.js`) it imports below.
//   * The init API here targets wasm-bindgen `--target web` (what build_wasm.ts
//     uses): `initSync({ module, memory })`. If you change the target, adjust.

// Use the async default `init` (not `initSync`): it is exactly how `wasm_thread`
// initialises its own workers, and it sets up the state `wasm_thread` needs to
// spawn *nested* workers (notably retaining the compiled module that
// `wasm_bindgen::module()` returns and hands to spawned workers). `initSync` is
// enough to run this player, but nested `Worker.start()` then silently fails to
// bootstrap.
import init, { ruffle_worker_player_entry } from "./ruffle_web.js";

self.onmessage = async (event) => {
    const { module, memory, canvas, initPtr } = event.data;

    // Initialise over the SHARED compiled module + memory so this worker's linear
    // memory *is* the main thread's. `--target web` (0.2.x): the init options key
    // is `module_or_path` (initSync uses `module`).
    await init({ module_or_path: module, memory });

    // Hand off to Rust. This never returns to the JS event loop while a frame is
    // ticking — the whole point of being on a worker (blocking / Atomics.wait is
    // allowed here).
    ruffle_worker_player_entry(canvas, initPtr);
};
