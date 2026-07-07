/**
 * Player-in-worker entry point (experimental; browser OpenTTD).
 *
 * Runs the *primordial* Ruffle player on a dedicated worker so it may block
 * (`Atomics.wait`), which the UI thread forbids — required for SWFs whose
 * `flash.system.Worker` handshakes park the primordial (e.g. FlasCC / OpenTTD).
 * The worker renders straight to the page canvas via a transferred
 * `OffscreenCanvas`; this module kicks it off.
 *
 * Requires the threaded (`BUILD_WASM_THREADS=true`) build served on a
 * cross-origin-isolated page (COOP/COEP), with the GL renderer feature enabled.
 *
 * Standalone helper, separate from the normal `RufflePlayer` element:
 *
 *   import { startWorkerPlayer } from "ruffle-core/dist/start-worker-player";
 *   const handle = await startWorkerPlayer(canvasElement, "openttd.swf");
 *   // handle.set_viewport(cssW, cssH); handle.destroy();
 */

import init, {
    ruffle_prepare_worker_player,
    ruffle_wasm_module,
    ruffle_wasm_memory,
    ruffle_terminate_all_workers,
} from "../dist/ruffle_web";
import type { WorkerPlayerHandle } from "../dist/ruffle_web";

let initPromise: Promise<unknown> | undefined;

/**
 * Teardown for the worker player currently running, if any.
 *
 * Every worker player owns a set of Web Workers — one primordial plus one per
 * `flash.system.Worker.start()` (OpenTTD's compute thread). Nothing in the
 * browser reclaims those automatically, and a FlasCC compute worker spends its
 * whole life parked on the wasm futex inside `flash.concurrent.Condition.wait`,
 * so if it outlives its SWF it lingers forever — stealing a Web Worker slot and
 * perturbing the *next* player's `Atomics.wait` scheduling. So starting another
 * player (e.g. drag-dropping a different SWF) runs this first.
 */
let activeTeardown: (() => void) | undefined;

/**
 * Loads/initialises the threaded wasm module (once) and starts a worker-hosted
 * player rendering into `canvas`.
 *
 * @param canvas The on-page canvas; rendering control is transferred to the
 *   worker, so it must not already have a 2D/WebGL context.
 * @param swfUrl URL of the movie to load (fetched here, on the main thread).
 * @returns A handle exposing `set_viewport(cssWidth, cssHeight)` and `destroy()`.
 */
export async function startWorkerPlayer(
    canvas: HTMLCanvasElement,
    swfUrl: string,
): Promise<WorkerPlayerHandle> {
    if (!initPromise) {
        initPromise = init();
    }
    await initPromise;

    // Reclaim the previous worker player's workers before starting a new one —
    // this is what stops the last SWF's compute workers from zombie-blocking on
    // `Condition.wait`. Safe to call unconditionally (it is a no-op the first
    // time and idempotent thereafter).
    activeTeardown?.();

    // Absolute URL: Ruffle's sandbox / SharedObject / VFS reject a bare relative
    // path ("relative URL without a base").
    const absoluteUrl = new URL(swfUrl, location.href).href;
    const swf = new Uint8Array(await (await fetch(absoluteUrl)).arrayBuffer());

    // Transfer rendering control of the page canvas to the worker.
    const offscreen = canvas.transferControlToOffscreen();

    // Rust builds the shared bridge + WorkerInit and wires DOM input; we get back
    // the WorkerInit pointer to hand across.
    const handle = ruffle_prepare_worker_player(canvas, swf, absoluteUrl);

    // Create the worker in JS so the bundler packages it (and its wasm glue)
    // consistently — works in dev and in the hashed production build.
    // `.js`: consumers import the compiled `dist/`, where this sibling is
    // `worker-player-worker.js` (tsc-emitted). Vite bundles it (and its wasm
    // glue) as a worker chunk.
    const primordial = new Worker(
        new URL("./worker-player-worker.js", import.meta.url),
        { type: "module" },
    );
    primordial.postMessage(
        {
            module: ruffle_wasm_module(),
            memory: ruffle_wasm_memory(),
            canvas: offscreen,
            initPtr: handle.init_ptr(),
        },
        [offscreen],
    );

    // The `flash.system.Worker` compute workers spawned below. Tracked so
    // teardown can reclaim them; a discarded reference would leave them
    // unreachable and unkillable.
    const computeWorkers: Worker[] = [];
    let pumping = true;

    // Fulfil the primordial's `flash.system.Worker` spawns (OpenTTD's compute
    // thread): poll the shared queue and launch each as another bundler-packaged
    // worker over the shared memory. (`wasm_thread`'s own bootstrap doesn't run
    // in a bundled build, so we reuse the mechanism that does.)
    const pumpSpawns = () => {
        if (!pumping) {
            return; // teardown stopped the pump
        }
        for (const entryPtr of handle.take_spawn_requests()) {
            const compute = new Worker(
                new URL("./compute-worker.js", import.meta.url),
                { type: "module" },
            );
            compute.postMessage({
                module: ruffle_wasm_module(),
                memory: ruffle_wasm_memory(),
                entryPtr,
            });
            computeWorkers.push(compute);
        }
        requestAnimationFrame(pumpSpawns);
    };
    requestAnimationFrame(pumpSpawns);

    // Full teardown: stop spawning, then shut every worker down *gracefully*.
    //
    // `ruffle_terminate_all_workers()` sets each spawned worker's terminate flag
    // and wakes anything parked in `Condition.wait` / `Mutex.lock`; the woken
    // wait raises an uncatchable error that unwinds the worker's AS3/FlasCC stack
    // out to its run loop, which exits — releasing the shared allocator/mutex
    // locks on the way (avmplus's interrupt model). `handle.destroy()` likewise
    // asks the primordial to stop. We then reclaim the (now idle) Web Workers a
    // moment later; by then each has unwound past any allocation it was in, so the
    // JS `terminate()` — only a backstop to free the worker slots — can't leave
    // the shared linear memory's allocator lock held for the next player.
    let destroyed = false;
    const rustDestroy = handle.destroy.bind(handle);
    const teardown = () => {
        if (destroyed) {
            return;
        }
        destroyed = true;
        pumping = false;
        ruffle_terminate_all_workers(); // graceful: signal + wake every worker
        rustDestroy(); // primordial exits its loop at the next tick boundary
        setTimeout(() => {
            for (const w of computeWorkers) {
                w.terminate();
            }
            computeWorkers.length = 0;
            primordial.terminate();
        }, 250);
        if (activeTeardown === teardown) {
            activeTeardown = undefined;
        }
    };
    activeTeardown = teardown;

    // Route the handle's own destroy() through full teardown, so an external
    // `handle.destroy()` reclaims every worker too — not just the primordial.
    handle.destroy = teardown;

    return handle;
}
