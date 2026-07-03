//! wasm-threads infra smoke test for Ruffle "Path A" workers.
//!
//! Proves the end-to-end path we need for real Flash workers on the web:
//!   * threaded wasm build (`+atomics`, shared memory),
//!   * a second thread running Rust on a Web Worker (`wasm_thread`),
//!   * cross-thread shared memory + `Atomics.wait`/`notify`.
//!
//! Success criterion (visible in the browser devtools console):
//!   [main]   ... spawned worker
//!   [worker] started; blocking on shared cell (Atomics.wait)
//!   [main]   stored 42 and notified worker
//!   [worker] woke up; shared cell = 42   <-- cross-thread wakeup worked
#![feature(stdarch_wasm_atomic_wait)]

use std::sync::Arc;
use std::sync::atomic::{AtomicI32, Ordering};
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;

fn log(s: &str) {
    web_sys::console::log_1(&JsValue::from_str(s));
}

/// Entry point, called from the page's main thread.
#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
    log("[main] start(); spawning worker thread");

    // A single 32-bit cell in *shared* linear memory. Because wasm threads share
    // one `SharedArrayBuffer`, this `AtomicI32` has a fixed address visible from
    // every thread — this is exactly how a shareable `ByteArray` / domainMemory
    // will be aliased across Flash workers.
    let cell = Arc::new(AtomicI32::new(0));

    // --- worker thread: block until the cell becomes non-zero ---
    let cell_worker = cell.clone();
    wasm_thread::Builder::new()
        .spawn(move || {
            log("[worker] started; blocking on shared cell (Atomics.wait)");
            let ptr = cell_worker.as_ref() as *const AtomicI32 as *mut i32;
            // Loop guards against spurious wakeups: wait while value == 0.
            while cell_worker.load(Ordering::SeqCst) == 0 {
                // SAFETY: `ptr` addresses a live AtomicI32 in shared memory.
                unsafe {
                    core::arch::wasm32::memory_atomic_wait32(ptr, 0, -1);
                }
            }
            let got = cell_worker.load(Ordering::SeqCst);
            log(&format!("[worker] woke up; shared cell = {got}"));
        })
        .expect("spawn worker");

    // --- main thread: it may NOT Atomics.wait, but it can store + notify.
    // Do it after a short timeout so the worker is definitely parked in the
    // wait first, proving a real cross-thread wakeup (not just a value race).
    let cell_main = cell.clone();
    let poke = Closure::<dyn FnMut()>::new(move || {
        cell_main.store(42, Ordering::SeqCst);
        let ptr = cell_main.as_ref() as *const AtomicI32 as *mut i32;
        // SAFETY: same live cell, wake one waiter.
        let woken = unsafe { core::arch::wasm32::memory_atomic_notify(ptr, 1) };
        log(&format!("[main] stored 42 and notified worker (woke {woken})"));
    });
    web_sys::window()
        .expect("window")
        .set_timeout_with_callback_and_timeout_and_arguments_0(poke.as_ref().unchecked_ref(), 300)
        .expect("set_timeout");
    // The closure must outlive the timeout.
    poke.forget();
}
