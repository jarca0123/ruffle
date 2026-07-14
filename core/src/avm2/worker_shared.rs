//! Send-able shared state for `flash.system.Worker` under the real-threads
//! ("Path A") model.
//!
//! Each Flash worker runs on its own OS thread with its own GC arena. gc_arena
//! objects are `!Send`, so anything shared *by reference* across workers —
//! `shareable` ByteArray backing / domainMemory, `Mutex`, `Condition`, and
//! `setSharedProperty` values — must live *outside* every arena, behind `Arc`.
//! That is what this module provides. The AVM2 `ByteArray`/`Mutex`/`Condition`
//! objects in each arena hold a handle into one of these shared primitives.

use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, Weak};
use std::time::Duration;

/// The backing store of a `shareable` ByteArray (and of FlasCC's shared "RAM"
/// used as domainMemory), shared by reference across worker threads.
///
/// The allocation carries **no reservation** and may move on growth (geometric
/// reallocation, like a `Vec`). Correctness of moves: every accessor
/// serializes on the `grow` lock, and the one access that cannot lock per
/// access — the JIT's inline `base + addr` fast path, which hits domainMemory
/// millions of times per frame — instead registers its frame on the buffer
/// ([`enter_dm_frame`]), pinning it in place for the run. Overlapping DATA
/// access is synchronized at the AS3 `Mutex`/`Condition` layer, as on real
/// Flash.
#[derive(Clone)]
pub struct SharedByteBuffer(Arc<SharedBuf>);

/// The largest a ByteArray (hence domainMemory / FlasCC's shared "RAM") can ever
/// grow, taken verbatim from avmplus:
///
/// ```c
/// // core/ByteArrayGlue.cpp
/// #define MAX_BYTEARRAY_STORE_LENGTH (MMgc::GCHeap::kMaxObjectSize - MMgc::GCHeap::kBlockSize*2)
/// // MMgc/GCHeap.h: kMaxObjectSize = 0xFFFFFFFF, kBlockSize = 4096
/// ```
///
/// i.e. `0xFFFF_FFFF - 2*4096 = 0xFFFF_DFFF` (~4 GiB − 8 KiB), and avmplus
/// `static_assert`s it stays `< 2^32`. We reserve exactly this so the base
/// pointer stays stable for the buffer's whole lifetime and the hot path never
/// reallocates (which would move memory out from under lock-free readers). The
/// reservation is virtual — the OS commits physical pages lazily as the heap
/// grows — so it costs address space, not memory.
#[cfg(not(target_family = "wasm"))]
const SHARED_RESERVE: usize = 0xFFFF_FFFF - 4096 * 2; // avmplus MAX_BYTEARRAY_STORE_LENGTH

struct SharedBuf {
    /// Base of the current `cap`-byte allocation. May move on growth — always
    /// under `grow` and only while no JIT frame holds the base (`jit_frames`);
    /// every other accessor takes `grow`, so no one can observe a stale base.
    /// Atomics for formal `Sync`; relaxed loads suffice (mutation is always
    /// under the lock).
    ptr: AtomicUsize,
    cap: AtomicUsize,
    /// The buffer's stable **descriptor**: `[base, cap, len]` words at a FIXED
    /// heap address for the buffer's whole lifetime. The JIT's inline domainMemory
    /// path loads base+len from here on EVERY access (from one hot cache line)
    /// instead of caching them per frame — so a growth reallocation just rewrites
    /// the cell and even a frame that never exits (FlasCC's dispatch loop) observes
    /// the move on its next access. No pinning, no clamping. `len` (word 2, offset
    /// 8 on wasm32) is the exact logical length the inline `li*`/`si*` bounds-check
    /// against (`addr + width <= len`, else bail to the throwing helper) — mirrored
    /// from `len` on every store below. On wasm32 `usize` words are the `u32`s the
    /// emitted `i32.load`s expect (offsets 0/4/8); native never reads it inline.
    desc: Box<[AtomicUsize; 3]>,
    /// Logical length (grows via `sbrk`); always `<= cap`.
    len: AtomicUsize,
    /// Serializes every accessor against growth moves (and a grow's zero-fill
    /// against concurrent CAS). Uncontended in practice: the hot dm traffic is
    /// the JIT inline path, which holds a registered frame instead.
    grow: Mutex<()>,
}

// SAFETY: `ptr` models raw shared RAM. Concurrent access to disjoint bytes is
// race-free; overlapping access is synchronized at the AS3 (Mutex/Condition)
// layer, exactly as on real Flash. The base can only move under the `grow`
// lock, which every accessor (or, for the JIT fast path, a registered frame)
// synchronizes with — so no thread can observe a stale base.
unsafe impl Send for SharedBuf {}
unsafe impl Sync for SharedBuf {}

impl Drop for SharedBuf {
    fn drop(&mut self) {
        // Reclaim the allocation (`u8` has no destructor, so len 0 is fine).
        let ptr = *self.ptr.get_mut() as *mut u8;
        let cap = *self.cap.get_mut();
        unsafe { drop(Vec::from_raw_parts(ptr, 0, cap)) };
    }
}

impl std::fmt::Debug for SharedByteBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SharedByteBuffer({:#x}, len={})",
            self.ptr_id(),
            self.len()
        )
    }
}

impl SharedByteBuffer {
    /// The initial allocation for a new buffer. **Native**: the full
    /// `SHARED_RESERVE` of lazily-committed address space (an mmap reservation
    /// is free, and the buffer then never moves — the wasm move machinery
    /// simply never triggers). **Wasm**: exactly what the seed needs — NO
    /// reservation; growth reallocates (`ensure_cap`).
    fn reserve(initial_len: usize) -> (*mut u8, usize) {
        #[cfg(not(target_family = "wasm"))]
        let want = SHARED_RESERVE.max(initial_len);
        #[cfg(target_family = "wasm")]
        let want = initial_len.max(4096);
        let mut v: Vec<u8> = Vec::with_capacity(want);
        let ptr = v.as_mut_ptr();
        let cap = v.capacity();
        std::mem::forget(v);
        (ptr, cap)
    }

    pub fn with_len(len: usize) -> Self {
        let (ptr, cap) = Self::reserve(len);
        // `sbrk` memory reads as zero.
        unsafe { std::ptr::write_bytes(ptr, 0, len) };
        Self(Arc::new(SharedBuf {
            ptr: AtomicUsize::new(ptr as usize),
            cap: AtomicUsize::new(cap),
            desc: Box::new([
                AtomicUsize::new(ptr as usize),
                AtomicUsize::new(cap),
                AtomicUsize::new(len),
            ]),
            len: AtomicUsize::new(len),
            grow: Mutex::new(()),
        }))
    }

    pub fn from_vec(bytes: Vec<u8>) -> Self {
        let (ptr, cap) = Self::reserve(bytes.len());
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len()) };
        Self(Arc::new(SharedBuf {
            ptr: AtomicUsize::new(ptr as usize),
            cap: AtomicUsize::new(cap),
            desc: Box::new([
                AtomicUsize::new(ptr as usize),
                AtomicUsize::new(cap),
                AtomicUsize::new(bytes.len()),
            ]),
            len: AtomicUsize::new(bytes.len()),
            grow: Mutex::new(()),
        }))
    }

    /// Moves the allocation to a fresh one, preserving contents. Tries `target_cap` (the amortized
    /// growth size) first; if the allocator can't satisfy it — near the shared-memory ceiling,
    /// where the geometric HEADROOM is what overflows — falls back to `min_cap`, the smallest size
    /// that actually fits the request. So a buffer that would fit at its real size doesn't OOM
    /// merely because the growth slack didn't (the FlasCC/domainMemory `sbrk` failure mode). Only a
    /// genuine can't-fit-even-minimally allocation aborts. Caller must hold `grow` and have verified
    /// the buffer may move (no live JIT frame on its base).
    fn realloc(b: &SharedBuf, target_cap: usize, min_cap: usize) {
        let old_ptr = b.ptr.load(Ordering::Relaxed) as *mut u8;
        let old_cap = b.cap.load(Ordering::Relaxed);
        // Fallible: `try_reserve_exact` returns `Err` instead of aborting, so we can retry smaller.
        let mut v: Vec<u8> = Vec::new();
        if v.try_reserve_exact(target_cap).is_err() && v.try_reserve_exact(min_cap).is_err() {
            std::alloc::handle_alloc_error(
                std::alloc::Layout::array::<u8>(min_cap)
                    .unwrap_or_else(|_| std::alloc::Layout::new::<u8>()),
            );
        }
        let new_ptr = v.as_mut_ptr();
        let new_cap = v.capacity();
        std::mem::forget(v);
        let old_len = b.len.load(Ordering::Acquire);
        // SAFETY: both regions are live allocations; `old_len <= old_cap <= new_cap`.
        unsafe { std::ptr::copy_nonoverlapping(old_ptr, new_ptr, old_len) };
        b.ptr.store(new_ptr as usize, Ordering::Relaxed);
        b.cap.store(new_cap, Ordering::Relaxed);
        // Publish the move to the JIT's inline path: base first, then cap —
        // a torn observer sees (new base, old smaller cap), which is in bounds.
        b.desc[0].store(new_ptr as usize, Ordering::Release);
        b.desc[1].store(new_cap, Ordering::Release);
        // SAFETY: reconstructs the leaked source Vec to free it.
        unsafe { drop(Vec::from_raw_parts(old_ptr, 0, old_cap)) };
    }

    /// Identity of the underlying buffer, so two handles to the *same* shared
    /// buffer compare equal (used for by-reference `setSharedProperty`).
    pub fn ptr_id(&self) -> usize {
        Arc::as_ptr(&self.0) as usize
    }

    pub fn len(&self) -> usize {
        self.0.len.load(Ordering::Acquire)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The base address of the buffer's allocation. For a PINNED (cross-worker)
    /// buffer it never moves; for a worker-local one it is stable while a JIT
    /// frame holds it (see [`enter_dm_frame`]) — so, on wasm32, where this is an
    /// offset into the module's own linear memory, the JIT can
    /// `i32.load`/`store` domainMemory inline (`base + addr`) for the duration
    /// of a run instead of calling a helper.
    pub fn base_ptr(&self) -> usize {
        self.0.ptr.load(Ordering::Relaxed)
    }

    /// The allocation capacity — the whole `[0, cap)` range is valid, committed
    /// memory (`len()` grows within it). The JIT's inline `li*`/`si*`
    /// bounds-check against this bound, not `len()`: a method that grows the
    /// heap (via a call) must not then treat the newly-valid region as out of
    /// bounds using a `len` captured before the grow. Stable for the duration
    /// of a JIT run (same argument as [`Self::base_ptr`]).
    pub fn cap(&self) -> usize {
        self.0.cap.load(Ordering::Relaxed)
    }

    /// Address of the buffer's stable `[base, cap]` descriptor cell (see
    /// `SharedBuf::desc`) — what the JIT's inline domainMemory path is handed:
    /// it loads base and cap from here on every access, so growth moves are
    /// safe even under a live (or never-exiting) JIT frame.
    pub fn desc_ptr(&self) -> usize {
        self.0.desc.as_ptr() as usize
    }

    /// Make room for `new_len`, growing the allocation when needed — by
    /// GEOMETRIC REALLOCATION, like a `Vec`: there is NO up-front reservation,
    /// ever. Moving is safe: every accessor serializes on `grow` (held here by
    /// the caller), and the JIT's inline fast path re-reads base+cap from the
    /// stable `desc` cell on every access, so it observes the move immediately
    /// (same-thread: the grow happens inside a helper call, which never splits
    /// one emitted dm op; cross-worker: overlapping grow/access is content-
    /// synchronized at the AS3 layer, like every other data race here).
    /// Returns the (possibly moved) base and the length.
    fn ensure_cap(&self, new_len: usize) -> (*mut u8, usize) {
        let b = &*self.0;
        let cap = b.cap.load(Ordering::Relaxed);
        if new_len <= cap {
            return (b.ptr.load(Ordering::Relaxed) as *mut u8, new_len);
        }
        // Amortized growth at 1.5× (not 2×): still geometric — O(log) reallocs — but bounds the
        // steady-state over-allocation to ~1.5× the live size and shrinks the transient old+new
        // peak during the copy (old + 1.5·old vs old + 2·old), both of which matter on wasm where
        // there is no address-space reservation and every byte is committed. `realloc` falls back
        // to exactly `new_len` if even 1.5× can't be allocated.
        let target = new_len.max(cap.saturating_add(cap / 2));
        Self::realloc(b, target, new_len);
        (b.ptr.load(Ordering::Relaxed) as *mut u8, new_len)
    }

    /// Grow (zero-filled) or shrink the logical length.
    pub fn set_len(&self, new_len: usize) {
        let b = &*self.0;
        let _g = b.grow.lock().unwrap();
        let old = b.len.load(Ordering::Acquire);
        let (ptr, new_len) = self.ensure_cap(new_len);
        if new_len > old {
            // SAFETY: `new_len <= cap`, so `[old, new_len)` is within the
            // allocation. Zeroed before the length store below publishes it.
            unsafe { std::ptr::write_bytes(ptr.add(old), 0, new_len - old) };
        }
        b.len.store(new_len, Ordering::Release);
        b.desc[2].store(new_len, Ordering::Release); // keep the JIT's inline len in sync
    }

    /// `ByteArray.atomicCompareAndSwapLength`: if the length equals `expected`,
    /// set it to `new`. Returns the previous length regardless.
    pub fn cas_len(&self, expected: usize, new: usize) -> usize {
        let b = &*self.0;
        let _g = b.grow.lock().unwrap();
        let old = b.len.load(Ordering::Acquire);
        if old == expected {
            let (ptr, new) = self.ensure_cap(new);
            if new > old {
                // SAFETY: `new <= cap`; zero the freshly-grown range before it
                // becomes visible via the length store.
                unsafe { std::ptr::write_bytes(ptr.add(old), 0, new - old) };
            }
            b.len.store(new, Ordering::Release);
            b.desc[2].store(new, Ordering::Release); // keep the JIT's inline len in sync
            tracing::trace!(
                "ram {:#x}: t{:#x} sbrk {old} -> {new}",
                self.ptr_id(),
                thread_id_u64()
            );
        }
        old
    }

    /// Copy `out.len()` bytes starting at `index`. Returns `false` (no copy) if
    /// the range is out of bounds.
    pub fn read(&self, index: usize, out: &mut [u8]) -> bool {
        let b = &*self.0;
        // Serialized with growth moves (JIT inline accesses don't come here —
        // they hold the base via a registered frame instead).
        let _g = b.grow.lock().unwrap();
        let Some(end) = index.checked_add(out.len()) else {
            return false;
        };
        if end > b.len.load(Ordering::Acquire) {
            return false;
        }
        // SAFETY: `end <= len <= cap`, and `ptr` is valid for `[0, cap)`.
        unsafe {
            std::ptr::copy_nonoverlapping(
                (b.ptr.load(Ordering::Relaxed) as *const u8).add(index),
                out.as_mut_ptr(),
                out.len(),
            )
        };
        true
    }

    /// Overwrite bytes starting at `index`. Returns `false` (no write) if the
    /// range is out of bounds.
    pub fn write(&self, index: usize, data: &[u8]) -> bool {
        let b = &*self.0;
        // Serialized with growth moves (see `read`).
        let _g = b.grow.lock().unwrap();
        let Some(end) = index.checked_add(data.len()) else {
            return false;
        };
        if end > b.len.load(Ordering::Acquire) {
            return false;
        }
        // SAFETY: `end <= len <= cap`, and `ptr` is valid for `[0, cap)`.
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                (b.ptr.load(Ordering::Relaxed) as *mut u8).add(index),
                data.len(),
            )
        };
        true
    }

    /// Atomic compare-and-swap of the raw bytes at `index`: reads the current
    /// `expected.len()` bytes; if they equal `expected`, replaces them with
    /// `new` (which must be the same length). Returns the bytes that were there
    /// before, or `None` if the range is out of bounds. Endianness is handled by
    /// the caller (the AVM2 `ByteArray` layer) before encoding to bytes.
    pub fn cas_bytes(&self, index: usize, expected: &[u8], new: &[u8]) -> Option<Vec<u8>> {
        debug_assert_eq!(expected.len(), new.len());
        let b = &*self.0;
        // Atomic against other CAS and against length changes.
        let _g = b.grow.lock().unwrap();
        let end = index.checked_add(expected.len())?;
        if end > b.len.load(Ordering::Acquire) {
            return None;
        }
        let mut old = vec![0u8; expected.len()];
        // SAFETY: `end <= len <= cap`.
        unsafe {
            std::ptr::copy_nonoverlapping(
                (b.ptr.load(Ordering::Relaxed) as *const u8).add(index),
                old.as_mut_ptr(),
                old.len(),
            )
        };
        if old == expected {
            // SAFETY: same bounds; `new.len() == expected.len()`.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    new.as_ptr(),
                    (b.ptr.load(Ordering::Relaxed) as *mut u8).add(index),
                    new.len(),
                )
            };
        }
        Some(old)
    }

    /// Snapshot the whole buffer (e.g. to seed a worker's domainMemory view).
    pub fn snapshot(&self) -> Vec<u8> {
        let b = &*self.0;
        // Serialized with growth moves (see `read`).
        let _g = b.grow.lock().unwrap();
        let len = b.len.load(Ordering::Acquire);
        let mut v = vec![0u8; len];
        // SAFETY: `len <= cap`.
        unsafe {
            std::ptr::copy_nonoverlapping(
                b.ptr.load(Ordering::Relaxed) as *const u8,
                v.as_mut_ptr(),
                len,
            )
        };
        v
    }
}

/// A `flash.concurrent.Mutex`, shared by reference across workers.
///
/// Flash's `Mutex.lock()`/`unlock()` are *not* scope-bound (unlike a Rust
/// `MutexGuard`), so this is a hand-rolled lock over a `Condvar`: `lock` blocks
/// until free, `unlock` releases and wakes a waiter. It is recursive, matching
/// Flash (a thread may re-lock a mutex it already owns).
#[derive(Clone)]
pub struct SharedMutex(Arc<SharedMutexInner>);

impl std::fmt::Debug for SharedMutex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SharedMutex({:#x})", self.ptr_id())
    }
}

struct SharedMutexInner {
    state: Mutex<MutexState>,
    /// `Arc` so it can be registered in [`WAKEABLES`] (termination wakes lockers
    /// parked here too, not just `Condition` waiters).
    cvar: Arc<Condvar>,
}

#[derive(Default)]
struct MutexState {
    /// Owning thread's id (as `u64`), or `None` when free.
    owner: Option<u64>,
    /// Recursive lock depth held by the owner.
    depth: u32,
}

fn thread_id_u64() -> u64 {
    // `ThreadId` has no stable numeric accessor; hash it for a stable-per-thread
    // key. Collisions are astronomically unlikely and only affect fairness.
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    std::thread::current().id().hash(&mut h);
    h.finish()
}

// ---------------------------------------------------------------------------
// Worker interruption / termination (avmplus `Isolate` model)
//
// A spawned `flash.system.Worker` blocks on `Condition.wait` / `Mutex.lock`,
// which park on the wasm futex indefinitely. To terminate such a worker — on
// `Worker.terminate()`, or when the page tears the player down on a SWF change —
// we mirror what avmplus does (`core/Isolate.cpp` `signalInterruptibleState`):
//
//   1. set the worker's terminate flag,
//   2. `notify_all` every live wait primitive so the parked worker wakes,
//   3. the woken `wait`/`lock` re-checks the flag and, if set, reports
//      interruption so the native layer raises an *uncatchable* `RustError` that
//      unwinds the whole AS3 (+ FlasCC C) stack back out to the `run_worker`
//      loop, which then exits.
//
// This is why termination is graceful (locks release as the stack unwinds)
// rather than a hard `Worker.terminate()` in JS, which could kill a thread
// mid-`malloc` and poison the shared linear-memory allocator.
// ---------------------------------------------------------------------------

thread_local! {
    /// Terminate flag of the worker running on *this* thread, bound by
    /// [`bind_worker_terminate`] in `run_worker`. `None` on the primordial/main
    /// thread (which is never interrupted this way), so its waits behave normally.
    static WORKER_TERMINATE: RefCell<Option<Arc<AtomicBool>>> = const { RefCell::new(None) };
}

/// Every live wait primitive's condvar, so terminating a worker can wake it even
/// when parked in an indefinite wait. `Weak` so dropped primitives self-prune.
static WAKEABLES: Mutex<Vec<Weak<Condvar>>> = Mutex::new(Vec::new());

/// Every spawned worker's terminate flag, so [`terminate_all_workers`] can stop
/// them all at once (page teardown / SWF change). `Weak` so exited workers prune.
static WORKER_FLAGS: Mutex<Vec<Weak<AtomicBool>>> = Mutex::new(Vec::new());

fn register_wakeable(cv: &Arc<Condvar>) {
    let mut g = WAKEABLES.lock().expect("wakeables poisoned");
    // FlasCC creates many transient mutexes/conditions; prune dead `Weak`s
    // (cheap `strong_count`, no upgrade) before the vec would grow so the
    // registry stays bounded to the live set rather than leaking one entry per
    // primitive ever created.
    if g.len() == g.capacity() {
        g.retain(|w| w.strong_count() > 0);
    }
    g.push(Arc::downgrade(cv));
}

/// Bind the current worker thread to `flag` so a blocked `wait`/`lock` on this
/// thread reports interruption once `flag` is set. Call once at worker startup.
pub fn bind_worker_terminate(flag: Arc<AtomicBool>) {
    WORKER_TERMINATE.with(|f| *f.borrow_mut() = Some(flag));
}

/// Register a spawned worker's terminate flag for [`terminate_all_workers`].
pub fn register_worker_flag(flag: &Arc<AtomicBool>) {
    WORKER_FLAGS
        .lock()
        .expect("worker flags poisoned")
        .push(Arc::downgrade(flag));
}

/// Whether the worker on this thread has been asked to terminate. Always `false`
/// off a spawned worker (no flag bound).
pub fn worker_terminate_requested() -> bool {
    WORKER_TERMINATE.with(|f| {
        f.borrow()
            .as_ref()
            .is_some_and(|a| a.load(Ordering::Relaxed))
    })
}

/// Wake every live wait primitive so any parked worker re-checks its terminate
/// flag. Call after setting a flag. Prunes dropped primitives.
pub fn wake_blocked_workers() {
    let mut g = WAKEABLES.lock().expect("wakeables poisoned");
    g.retain(|w| match w.upgrade() {
        Some(cv) => {
            cv.notify_all();
            true
        }
        None => false,
    });
}

/// Terminate every spawned worker: set all their flags, then wake anything
/// parked so it unwinds and exits. Idempotent; prunes exited workers. Used on
/// page teardown / SWF change.
pub fn terminate_all_workers() {
    {
        let mut g = WORKER_FLAGS.lock().expect("worker flags poisoned");
        g.retain(|w| match w.upgrade() {
            Some(flag) => {
                flag.store(true, Ordering::Relaxed);
                true
            }
            None => false,
        });
    }
    wake_blocked_workers();
}

impl SharedMutex {
    pub fn new() -> Self {
        let cvar = Arc::new(Condvar::new());
        register_wakeable(&cvar);
        Self(Arc::new(SharedMutexInner {
            state: Mutex::new(MutexState::default()),
            cvar,
        }))
    }

    pub fn ptr_id(&self) -> usize {
        Arc::as_ptr(&self.0) as usize
    }

    /// Block until the mutex is acquired by the current thread. Returns `true`
    /// once acquired, or `false` if the worker was interrupted while parked
    /// (terminate requested) — in which case the mutex is **not** held and the
    /// caller must unwind (see [`worker_terminate_requested`]).
    #[must_use]
    pub fn lock(&self) -> bool {
        let me = thread_id_u64();
        let mut state = self.0.state.lock().unwrap();
        let mut parked = false;
        loop {
            match state.owner {
                None => {
                    state.owner = Some(me);
                    state.depth = 1;
                    if parked {
                        tracing::trace!("mutex {:#x}: acquired after park (t{me:#x})", self.ptr_id());
                    }
                    return true;
                }
                Some(o) if o == me => {
                    state.depth += 1;
                    return true;
                }
                Some(other) => {
                    // Interrupted before/while parked: give up without acquiring.
                    if worker_terminate_requested() {
                        return false;
                    }
                    tracing::trace!(
                        "mutex {:#x}: t{me:#x} parks (held by t{other:#x})",
                        self.ptr_id()
                    );
                    parked = true;
                    state = self.0.cvar.wait(state).unwrap();
                }
            }
        }
    }

    /// Try to acquire without blocking. Returns `true` on success.
    pub fn try_lock(&self) -> bool {
        let me = thread_id_u64();
        let mut state = self.0.state.lock().unwrap();
        match state.owner {
            None => {
                state.owner = Some(me);
                state.depth = 1;
                true
            }
            Some(o) if o == me => {
                state.depth += 1;
                true
            }
            Some(_) => false,
        }
    }

    /// Release one level of ownership; wakes a waiter when fully released.
    /// No-op if the current thread does not own the mutex.
    pub fn unlock(&self) {
        let me = thread_id_u64();
        let mut state = self.0.state.lock().unwrap();
        if state.owner != Some(me) {
            return;
        }
        state.depth -= 1;
        if state.depth == 0 {
            state.owner = None;
            // Wake *all* blocked threads, not just one: both `lock()` waiters and
            // `Condition::wait` re-acquirers park on this cvar, and with a single
            // shared predicate (`owner`) a `notify_one` can wake a thread that
            // then loses the race and re-parks, dropping the wakeup — a lost-wakeup
            // deadlock. avmplus's `MutexObject::State::unlock` likewise `notifyAll`s.
            self.0.cvar.notify_all();
        }
    }
}

impl Default for SharedMutex {
    fn default() -> Self {
        Self::new()
    }
}

/// A `flash.concurrent.Condition`, shared by reference across workers. Bound to
/// a [`SharedMutex`]; `wait` atomically releases the mutex and blocks.
///
/// Correctness note: the condition parks on its own `Condvar` but *over the
/// mutex's internal state guard*, which it holds continuously from the moment it
/// releases Flash ownership until `wait` atomically parks. `notify` takes that
/// same guard. This is what makes a `notify` unlose-able — a naive design that
/// parks over an *unrelated* lock has a window where a `notify` between the
/// unlock and the park is dropped, which turns FlasCC's `yield()` (a timed
/// `wait`) into a busy 100%-CPU spin.
impl std::fmt::Debug for SharedCondition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SharedCondition({:#x})", self.ptr_id())
    }
}

#[derive(Clone)]
pub struct SharedCondition {
    mutex: SharedMutex,
    cond_cvar: Arc<Condvar>,
}

impl SharedCondition {
    pub fn new(mutex: SharedMutex) -> Self {
        let cond_cvar = Arc::new(Condvar::new());
        register_wakeable(&cond_cvar);
        Self { mutex, cond_cvar }
    }

    pub fn ptr_id(&self) -> usize {
        Arc::as_ptr(&self.cond_cvar) as usize
    }

    pub fn mutex(&self) -> &SharedMutex {
        &self.mutex
    }

    /// Wake one waiter, holding the mutex's state guard so a waiter about to park
    /// cannot miss it.
    pub fn notify(&self) {
        let _guard = self.mutex.0.state.lock().unwrap();
        tracing::trace!("cond {:#x}: notify", self.ptr_id());
        self.cond_cvar.notify_one();
    }

    /// Wake all waiters.
    pub fn notify_all(&self) {
        let _guard = self.mutex.0.state.lock().unwrap();
        tracing::trace!("cond {:#x}: notifyAll", self.ptr_id());
        self.cond_cvar.notify_all();
    }

    /// Atomically release the associated mutex and block until notified (or
    /// `timeout_ms` elapses), then re-acquire it. Returns `true` if woken by a
    /// notify, `false` on timeout.
    ///
    /// If the worker is interrupted (terminate requested) at any point, this
    /// returns *without* re-acquiring the mutex — it stays released (so other
    /// workers can proceed) and the caller must check
    /// [`worker_terminate_requested`] and unwind. Termination wakes the park via
    /// [`wake_blocked_workers`].
    pub fn wait(&self, timeout_ms: Option<u64>) -> bool {
        let inner = &*self.mutex.0;
        let me = thread_id_u64();
        let mut state = inner.state.lock().unwrap();

        // Fully release Flash ownership (remember the depth to restore on wake)
        // and wake anyone blocked in `lock()`.
        let saved_depth = state.depth.max(1);
        state.owner = None;
        state.depth = 0;
        inner.cvar.notify_all();

        // Interrupted before parking: stay released and bail; the caller unwinds.
        if worker_terminate_requested() {
            return false;
        }

        tracing::trace!("cond {:#x}: t{me:#x} waits", self.ptr_id());

        // Park on the condition, atomically releasing the state guard. The guard
        // was held continuously since the ownership release, so a concurrent
        // `notify` cannot be lost.
        let woken = match timeout_ms {
            Some(ms) => {
                let (g, res) = self
                    .cond_cvar
                    .wait_timeout(state, Duration::from_millis(ms))
                    .unwrap();
                state = g;
                !res.timed_out()
            }
            None => {
                state = self.cond_cvar.wait(state).unwrap();
                true
            }
        };

        // Interrupted while parked (the wake may have been the termination
        // signal): don't re-acquire — the owner may be the terminating side — and
        // stay released so it can drain.
        if worker_terminate_requested() {
            return woken;
        }

        // Re-acquire Flash ownership.
        while state.owner.is_some() {
            if worker_terminate_requested() {
                return woken;
            }
            state = inner.cvar.wait(state).unwrap();
        }
        state.owner = Some(me);
        state.depth = saved_depth;

        tracing::trace!("cond {:#x}: t{me:#x} resumes (woken={woken})", self.ptr_id());
        woken
    }
}

/// A `flash.system.MessageChannel`, shared by reference across workers. Carries
/// a FIFO queue of messages; `send` enqueues, the receiving worker drains it via
/// `receive`. Messages are stored in the same `Send` [`SharedValue`] form used
/// for shared properties (AMF-style copy, or by-reference for shared primitives).
#[derive(Clone, Default)]
pub struct SharedMessageChannel(Arc<Mutex<VecDeque<SharedValue>>>);

impl std::fmt::Debug for SharedMessageChannel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SharedMessageChannel({:#x})", self.ptr_id())
    }
}

impl SharedMessageChannel {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn ptr_id(&self) -> usize {
        Arc::as_ptr(&self.0) as usize
    }

    /// Enqueue a message.
    pub fn send(&self, value: SharedValue) {
        self.0.lock().expect("message channel poisoned").push_back(value);
    }

    /// Dequeue the next message, if any.
    pub fn receive(&self) -> Option<SharedValue> {
        self.0.lock().expect("message channel poisoned").pop_front()
    }

    /// Whether at least one message is queued.
    pub fn message_available(&self) -> bool {
        !self.0.lock().expect("message channel poisoned").is_empty()
    }
}

/// A value stored via `Worker.setSharedProperty`, in a form that can cross the
/// worker boundary. Scalars and non-shareable `ByteArray`s are copied; shared
/// primitives (`shareable` ByteArray, `Mutex`, `Condition`) are passed by
/// reference (their `Arc` handle is cloned).
#[derive(Clone)]
pub enum SharedValue {
    Undefined,
    Null,
    Bool(bool),
    Int(i32),
    Number(f64),
    Str(String),
    /// A non-shareable `ByteArray`, copied by value (its raw bytes).
    ByteArrayCopy(Vec<u8>),
    /// A `shareable` ByteArray, shared by reference.
    ByteBuffer(SharedByteBuffer),
    /// A `Mutex`, shared by reference.
    Mutex(SharedMutex),
    /// A `Condition`, shared by reference.
    Condition(SharedCondition),
    /// A `MessageChannel`, shared by reference.
    MessageChannel(SharedMessageChannel),
}

/// The per-worker shared-property store, addressed by the primordial thread and
/// the worker thread alike (both hold a clone of this `Arc`).
pub type SharedProperties = Arc<Mutex<HashMap<String, SharedValue>>>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_buffer_cas_and_len() {
        let buf = SharedByteBuffer::with_len(16);
        assert_eq!(buf.len(), 16);

        // Write, read back.
        assert!(buf.write(4, &[1, 2, 3, 4]));
        let mut out = [0u8; 4];
        assert!(buf.read(4, &mut out));
        assert_eq!(out, [1, 2, 3, 4]);

        // CAS success (matches) then failure (does not).
        assert_eq!(buf.cas_bytes(4, &[1, 2, 3, 4], &[9, 9, 9, 9]).unwrap(), vec![1, 2, 3, 4]);
        assert_eq!(buf.cas_bytes(4, &[0, 0, 0, 0], &[7, 7, 7, 7]).unwrap(), vec![9, 9, 9, 9]);
        let mut out = [0u8; 4];
        buf.read(4, &mut out);
        assert_eq!(out, [9, 9, 9, 9]); // unchanged by the failed CAS

        // Out of range.
        assert!(buf.cas_bytes(14, &[0; 4], &[1; 4]).is_none());

        // Length CAS.
        assert_eq!(buf.cas_len(16, 32), 16);
        assert_eq!(buf.len(), 32);
        assert_eq!(buf.cas_len(16, 8), 32); // mismatch -> no change
        assert_eq!(buf.len(), 32);
    }

    #[test]
    fn mutex_recursive_and_shared() {
        let m = SharedMutex::new();
        let m2 = m.clone();
        assert_eq!(m.ptr_id(), m2.ptr_id());

        assert!(m.lock());
        assert!(m.lock()); // recursive
        assert!(!std::thread::spawn({
            let m = m.clone();
            move || m.try_lock()
        })
        .join()
        .unwrap()); // another thread cannot acquire
        m.unlock();
        m.unlock();

        assert!(std::thread::spawn(move || m2.try_lock()).join().unwrap());
    }

    #[test]
    fn condition_notify_wakes_waiter() {
        let mutex = SharedMutex::new();
        let cond = SharedCondition::new(mutex.clone());
        let cond2 = cond.clone();

        let waiter = std::thread::spawn(move || {
            assert!(cond2.mutex().lock());
            let woken = cond2.wait(Some(5000));
            cond2.mutex().unlock();
            woken
        });

        // Give the waiter time to park, then notify.
        std::thread::sleep(std::time::Duration::from_millis(100));
        cond.notify();

        assert!(waiter.join().unwrap(), "waiter should be woken, not timed out");
    }

    #[test]
    fn wait_interrupted_by_terminate() {
        let mutex = SharedMutex::new();
        let cond = SharedCondition::new(mutex.clone());
        let cond2 = cond.clone();
        let flag = Arc::new(AtomicBool::new(false));
        let flag2 = flag.clone();

        // A worker parked in an *indefinite* wait must still return once its
        // terminate flag is set and the park is woken.
        let waiter = std::thread::spawn(move || {
            bind_worker_terminate(flag2);
            assert!(cond2.mutex().lock());
            let _ = cond2.wait(None);
            // The native layer throws here; we just confirm the worker sees it.
            worker_terminate_requested()
        });

        std::thread::sleep(std::time::Duration::from_millis(100));
        flag.store(true, Ordering::Relaxed);
        wake_blocked_workers();

        assert!(
            waiter.join().unwrap(),
            "interrupted waiter must return and observe termination"
        );
    }

    #[test]
    fn terminate_all_sets_registered_flag() {
        let flag = Arc::new(AtomicBool::new(false));
        register_worker_flag(&flag);
        assert!(!flag.load(Ordering::Relaxed));
        terminate_all_workers();
        assert!(
            flag.load(Ordering::Relaxed),
            "terminate_all_workers must set every registered flag"
        );
    }
}
