//! Shared control block between the main-thread shell and the primordial player
//! running on a worker (the player-in-worker path; see the module comment in
//! `worker_host`).
//!
//! On the web the primordial player cannot run on the main thread: Ruffle's
//! Flash sync primitives park on the wasm futex (`Atomics.wait`), which traps on
//! the UI thread. So the player is moved to a dedicated worker where blocking is
//! allowed, and the main thread keeps only DOM input capture + the canvas.
//!
//! JS objects (the `OffscreenCanvas`) are transferred to the worker once via
//! `postMessage`; everything else — input events, viewport changes, shutdown —
//! flows through *this* struct, which lives in the shared `WebAssembly.Memory`.
//! The main thread creates it, hands the worker a raw pointer
//! ([`into_shared_ptr`](WorkerBridge::into_shared_ptr)) across `postMessage`, and
//! the worker reclaims a second `Arc` to the same allocation
//! ([`from_shared_ptr`](WorkerBridge::from_shared_ptr)). Every field is plain
//! data, so the block is `Send`/`Sync` and needs no GC awareness.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use ruffle_core::worker_host::WorkerHost;
use ruffle_core::PlayerEvent;
use ruffle_render::backend::ViewportDimensions;

/// Upper bound on buffered input events. If the worker stalls we drop the
/// *oldest* events rather than grow without bound; input latency is preferable
/// to unbounded memory. Generous enough that normal play never trims.
const MAX_QUEUED_EVENTS: usize = 4096;

/// How many `try_lock` attempts a **main-thread** producer makes before giving up (it
/// cannot block on `Atomics.wait`). The lock is held only for a fast drain, so a
/// couple of thousand spins comfortably outlast any real contention window.
const MAIN_THREAD_SPIN: u32 = 4096;

/// Main-thread ⇄ worker control block. See the module docs.
pub struct WorkerBridge {
    /// Input events captured on the main thread, drained by the worker each tick.
    input: Mutex<VecDeque<PlayerEvent>>,
    /// Latest viewport, applied by the worker when [`Self::viewport_generation`]
    /// changes.
    viewport: Mutex<ViewportDimensions>,
    /// Bumped on every viewport change so the worker can cheaply detect one
    /// without locking.
    viewport_gen: AtomicU64,
    /// Set by the main thread to ask the worker's run loop to exit.
    shutdown: AtomicBool,
    /// Pending nested-worker spawn requests: raw pointers to leaked
    /// `Box<Box<dyn FnOnce() + Send>>` closures. The primordial (on a worker)
    /// cannot spawn nested `wasm_thread` workers — they never bootstrap — so it
    /// queues each request here for the main thread to fulfil (where
    /// `wasm_thread::spawn` works). All threads share one linear memory, so the
    /// pointer is valid on the main thread and the spawned worker sees the same
    /// shared buffers/mutexes/conditions.
    spawn_requests: Mutex<VecDeque<usize>>,
}

impl WorkerBridge {
    /// Creates a bridge seeded with the initial viewport. Returns an `Arc` the
    /// main thread keeps; hand the worker a second one via [`Self::into_shared_ptr`].
    pub fn new(viewport: ViewportDimensions) -> Arc<Self> {
        Arc::new(Self {
            input: Mutex::new(VecDeque::new()),
            viewport: Mutex::new(viewport),
            viewport_gen: AtomicU64::new(0),
            shutdown: AtomicBool::new(false),
            spawn_requests: Mutex::new(VecDeque::new()),
        })
    }

    /// Queues a nested-worker spawn request (worker thread). `ptr` is a leaked
    /// `Box<Box<dyn FnOnce() + Send>>` for the main thread to reclaim exactly once.
    pub fn push_spawn_request(&self, ptr: usize) {
        self.spawn_requests
            .lock()
            .expect("bridge spawn_requests poisoned")
            .push_back(ptr);
    }

    /// Takes all pending spawn-request pointers (main thread).
    pub fn drain_spawn_requests(&self, out: &mut Vec<usize>) {
        out.extend(
            self.spawn_requests
                .lock()
                .expect("bridge spawn_requests poisoned")
                .drain(..),
        );
    }

    /// Queues an input event for the worker (**main thread**). Non-blocking: the main
    /// thread must never park on `Atomics.wait` (it traps on the UI thread), so this
    /// spins briefly on `try_lock` — the worker holds this lock only for a fast drain,
    /// so it almost always succeeds first try — and **drops the event** if the lock
    /// stays contended past the spin budget (a lost input beats a trapped/hung UI
    /// thread; contention this sustained is not expected in normal play).
    pub fn push_event(&self, event: PlayerEvent) {
        for _ in 0..MAIN_THREAD_SPIN {
            if let Ok(mut queue) = self.input.try_lock() {
                if queue.len() >= MAX_QUEUED_EVENTS {
                    queue.pop_front();
                }
                queue.push_back(event);
                return;
            }
            std::hint::spin_loop();
        }
    }

    /// Moves all queued events into `out` (worker thread), preserving order.
    pub fn drain_events(&self, out: &mut Vec<PlayerEvent>) {
        let mut queue = self.input.lock().expect("bridge input poisoned");
        out.extend(queue.drain(..));
    }

    /// Records a new viewport (**main thread**) and bumps the generation. Non-blocking
    /// for the same reason as [`Self::push_event`] — spins on `try_lock`; only bumps the
    /// generation once the write lands, so the worker never observes a bumped generation
    /// without the matching viewport. A dropped resize (sustained contention) is
    /// corrected by the next one.
    pub fn set_viewport(&self, viewport: ViewportDimensions) {
        for _ in 0..MAIN_THREAD_SPIN {
            if let Ok(mut slot) = self.viewport.try_lock() {
                *slot = viewport;
                drop(slot);
                self.viewport_gen.fetch_add(1, Ordering::Release);
                return;
            }
            std::hint::spin_loop();
        }
    }

    /// The current viewport generation (worker thread); compare against a locally
    /// held value to detect a resize without locking.
    pub fn viewport_generation(&self) -> u64 {
        self.viewport_gen.load(Ordering::Acquire)
    }

    /// The latest viewport (worker thread).
    pub fn viewport(&self) -> ViewportDimensions {
        *self.viewport.lock().expect("bridge viewport poisoned")
    }

    /// Asks the worker's run loop to stop (main thread).
    pub fn request_shutdown(&self) {
        self.shutdown.store(true, Ordering::Release);
    }

    /// Whether shutdown has been requested (worker thread).
    pub fn is_shutdown(&self) -> bool {
        self.shutdown.load(Ordering::Acquire)
    }

    /// Consumes an `Arc` into a raw pointer to pass across `postMessage` as a
    /// `u32` (a wasm32 address into the shared memory). Pair with exactly one
    /// [`Self::from_shared_ptr`] on the worker to avoid leaking the refcount.
    pub fn into_shared_ptr(self: Arc<Self>) -> u32 {
        Arc::into_raw(self) as usize as u32
    }

    /// Reclaims the `Arc` handed over by [`Self::into_shared_ptr`] (worker thread).
    ///
    /// # Safety
    /// `ptr` must be a value returned by `into_shared_ptr` on the same shared
    /// memory, reclaimed exactly once.
    pub unsafe fn from_shared_ptr(ptr: u32) -> Arc<Self> {
        // SAFETY: delegated to the caller (see above).
        unsafe { Arc::from_raw(ptr as usize as *const Self) }
    }
}

/// The [`WorkerHost`] the primordial-in-worker installs. `flash.system.Worker`
/// spawns can't run nested (`wasm_thread` workers spawned from a worker never
/// bootstrap), so this queues each spawn on the [`WorkerBridge`] for the main
/// thread to fulfil instead.
pub struct BridgeWorkerHost {
    bridge: Arc<WorkerBridge>,
}

impl BridgeWorkerHost {
    pub fn new(bridge: Arc<WorkerBridge>) -> Self {
        Self { bridge }
    }
}

impl WorkerHost for BridgeWorkerHost {
    fn spawn(&self, entry: Box<dyn FnOnce() + Send + 'static>) {
        // `Box<dyn FnOnce>` is a fat pointer, so box it once more to get a thin
        // pointer that fits a `usize`; the main thread reclaims and runs it.
        let ptr = Box::into_raw(Box::new(entry)) as usize;
        self.bridge.push_spawn_request(ptr);
        tracing::info!("worker: queued Worker.start spawn for the main thread");
    }
}
