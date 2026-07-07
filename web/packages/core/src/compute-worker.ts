// Worker entry for a `flash.system.Worker` (OpenTTD's compute thread).
//
// `wasm_thread`'s own worker bootstrap does not run in a bundled build, so
// instead the primordial queues each `Worker.start()` on the shared bridge, the
// main thread launches THIS bundler-packaged worker (Vite recognises
// `new Worker(new URL(...), {type:module})`), and here we `initSync` over the
// shared module + memory and run the queued `run_worker` closure — which loops
// as OpenTTD's C `thread_entry`, blocking on its Flash Condition (allowed on a
// worker).

import { initSync, ruffle_run_worker_entry } from "../dist/ruffle_web";

self.onmessage = (event: MessageEvent) => {
    const { module, memory, entryPtr } = event.data;
    initSync({ module, memory });
    ruffle_run_worker_entry(entryPtr);
};
