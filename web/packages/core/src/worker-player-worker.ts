// Worker entry for the player-in-worker path (browser OpenTTD).
//
// Created by `start-worker-player.ts` via `new Worker(new URL(...,
// import.meta.url), { type: "module" })`, which the bundler (Vite) recognises —
// so this file *and* the wasm-bindgen glue it imports are packaged into a worker
// chunk that resolves in dev and in the hashed production build alike.
//
// It receives the main thread's compiled module + shared memory + the
// transferred OffscreenCanvas + a WorkerInit pointer, initialises this thread's
// wasm-bindgen instance over that same module/memory (so its linear memory *is*
// the main thread's), and runs the primordial player, which never returns.

import { initSync, ruffle_worker_player_entry } from "../dist/ruffle_web";

self.onmessage = (event: MessageEvent) => {
    const { module, memory, canvas, initPtr } = event.data;
    // Synchronous init over the shared module + memory (`--target web` option
    // keys are `module`/`memory`).
    initSync({ module, memory });
    ruffle_worker_player_entry(canvas, initPtr);
};
