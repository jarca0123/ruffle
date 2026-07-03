//! Spawning background worker execution contexts for `flash.system.Worker`.
//!
//! Real Flash workers are OS threads. Ruffle runs each spawned worker as its own
//! headless runtime (its own GC arena + `Avm2`) on a background thread:
//!   * native builds use [`std::thread`],
//!   * the web uses `wasm_thread` (Web Workers over a shared `WebAssembly.Memory`).
//!
//! Because the threaded wasm build implements `std::sync` primitives (`Mutex`,
//! `Condvar`, `Arc<Atomic…>`) over the wasm futex, worker-side synchronization
//! code is identical on both platforms — only the spawn entry (and the message
//! transport) is platform-specific, which is exactly what this trait abstracts.

/// Host capable of spawning background worker runtimes.
///
/// Provided by the frontend (like the render/audio backends) and reached through
/// the player so `Worker.start()` can launch a real execution thread.
pub trait WorkerHost: 'static {
    /// Whether background workers are actually supported. Backs
    /// `Worker.isSupported` / `WorkerDomain.isSupported`.
    fn is_supported(&self) -> bool {
        true
    }

    /// Run `entry` on a fresh worker thread.
    ///
    /// `entry` must build its *own* headless runtime on that thread: a Ruffle GC
    /// arena is `!Send`, so nothing GC-resident may be captured. Only `Send`
    /// inputs may cross — the SWF bytes and shared `Arc` handles (shareable
    /// `ByteArray` backing, `Mutex`/`Condition` state, message queues).
    fn spawn(&self, entry: Box<dyn FnOnce() + Send + 'static>);
}

/// Spawns workers on native OS threads via [`std::thread`].
#[cfg(not(target_family = "wasm"))]
#[derive(Default)]
pub struct ThreadWorkerHost;

#[cfg(not(target_family = "wasm"))]
impl WorkerHost for ThreadWorkerHost {
    fn spawn(&self, entry: Box<dyn FnOnce() + Send + 'static>) {
        std::thread::spawn(entry);
    }
}

/// Rejects worker creation. Used by frontends (and headless/test hosts) that do
/// not support background workers; `Worker.isSupported` stays `false`.
#[derive(Default)]
pub struct NullWorkerHost;

impl WorkerHost for NullWorkerHost {
    fn is_supported(&self) -> bool {
        false
    }

    fn spawn(&self, _entry: Box<dyn FnOnce() + Send + 'static>) {
        tracing::warn!("WorkerHost::spawn called on NullWorkerHost; ignoring");
    }
}
