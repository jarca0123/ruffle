//! Send-able shared state for `flash.system.Worker` under the real-threads
//! ("Path A") model.
//!
//! Each Flash worker runs on its own OS thread with its own GC arena. gc_arena
//! objects are `!Send`, so anything shared *by reference* across workers —
//! `shareable` ByteArray backing / domainMemory, `Mutex`, `Condition`, and
//! `setSharedProperty` values — must live *outside* every arena, behind `Arc`.
//! That is what this module provides. The AVM2 `ByteArray`/`Mutex`/`Condition`
//! objects in each arena hold a handle into one of these shared primitives.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

/// The backing store of a `shareable` ByteArray (and of FlasCC's shared "RAM"
/// used as domainMemory), shared by reference across worker threads.
///
/// A fixed-capacity, stable-pointer allocation: the base pointer never moves, so
/// `read`/`write`/`len` are **lock-free** — essential because every `si*`/`li*`
/// domainMemory opcode hits this millions of times per frame. This models raw
/// shared process RAM (like a `SharedArrayBuffer` / Flash `domainMemory`); any
/// overlapping access is synchronized at the AS3 `Mutex`/`Condition` layer, so
/// only length changes (`sbrk`) and byte-CAS take the (rarely contended) `grow`
/// lock.
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
const SHARED_RESERVE: usize = 0xFFFF_FFFF - 4096 * 2; // avmplus MAX_BYTEARRAY_STORE_LENGTH

struct SharedBuf {
    /// Base of a `cap`-byte reservation. Never moves for the buffer's lifetime,
    /// so a load/store is always valid for `[0, cap)`.
    ptr: *mut u8,
    cap: usize,
    /// Logical length (grows via `sbrk`); always `<= cap`.
    len: AtomicUsize,
    /// Serializes length changes and byte-CAS so a grow's zero-fill stays
    /// consistent. Plain reads/writes take *no* lock.
    grow: Mutex<()>,
}

// SAFETY: `ptr` is a fixed allocation modelling raw shared RAM. Concurrent access
// to disjoint bytes is race-free; overlapping access is synchronized at the AS3
// (Mutex/Condition) layer, exactly as on real Flash. The base never moves, so a
// handle to it stays valid across threads for the buffer's whole lifetime.
unsafe impl Send for SharedBuf {}
unsafe impl Sync for SharedBuf {}

impl Drop for SharedBuf {
    fn drop(&mut self) {
        // Reclaim the reservation (`u8` has no destructor, so len 0 is fine).
        unsafe { drop(Vec::from_raw_parts(self.ptr, 0, self.cap)) };
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
    /// Reserve `SHARED_RESERVE` (or more, if the seed is larger) bytes of stable,
    /// lazily-committed address space and return the base pointer + real capacity.
    fn reserve(initial_len: usize) -> (*mut u8, usize) {
        let want = SHARED_RESERVE.max(initial_len);
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
            ptr,
            cap,
            len: AtomicUsize::new(len),
            grow: Mutex::new(()),
        }))
    }

    pub fn from_vec(bytes: Vec<u8>) -> Self {
        let (ptr, cap) = Self::reserve(bytes.len());
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len()) };
        Self(Arc::new(SharedBuf {
            ptr,
            cap,
            len: AtomicUsize::new(bytes.len()),
            grow: Mutex::new(()),
        }))
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

    /// Clamp a requested length to the reservation. Since `SHARED_RESERVE` is
    /// avmplus's own `MAX_BYTEARRAY_STORE_LENGTH`, no valid ByteArray can exceed
    /// it, so this is unreachable in correct operation — but we cannot move
    /// memory out from under live lock-free readers, so if it ever happens we
    /// clamp and warn rather than reallocate.
    fn clamp(b: &SharedBuf, new_len: usize) -> usize {
        if new_len > b.cap {
            tracing::error!(
                "shared RAM grow to {new_len} exceeds reservation {} — clamping",
                b.cap
            );
            b.cap
        } else {
            new_len
        }
    }

    /// Grow (zero-filled) or shrink the logical length.
    pub fn set_len(&self, new_len: usize) {
        let b = &*self.0;
        let _g = b.grow.lock().unwrap();
        let old = b.len.load(Ordering::Acquire);
        let new_len = Self::clamp(b, new_len);
        if new_len > old {
            // SAFETY: `new_len <= cap`, so `[old, new_len)` is within the
            // reservation. Zeroed before the length store below publishes it.
            unsafe { std::ptr::write_bytes(b.ptr.add(old), 0, new_len - old) };
        }
        b.len.store(new_len, Ordering::Release);
    }

    /// `ByteArray.atomicCompareAndSwapLength`: if the length equals `expected`,
    /// set it to `new`. Returns the previous length regardless.
    pub fn cas_len(&self, expected: usize, new: usize) -> usize {
        let b = &*self.0;
        let _g = b.grow.lock().unwrap();
        let old = b.len.load(Ordering::Acquire);
        if old == expected {
            let new = Self::clamp(b, new);
            if new > old {
                // SAFETY: `new <= cap`; zero the freshly-grown range before it
                // becomes visible via the length store.
                unsafe { std::ptr::write_bytes(b.ptr.add(old), 0, new - old) };
            }
            b.len.store(new, Ordering::Release);
            tracing::trace!(
                "ram {:#x}: t{:#x} sbrk {old} -> {new}",
                self.ptr_id(),
                thread_id_u64()
            );
        }
        old
    }

    /// Copy `out.len()` bytes starting at `index`. Returns `false` (no copy) if
    /// the range is out of bounds. Lock-free.
    pub fn read(&self, index: usize, out: &mut [u8]) -> bool {
        let b = &*self.0;
        let Some(end) = index.checked_add(out.len()) else {
            return false;
        };
        if end > b.len.load(Ordering::Acquire) {
            return false;
        }
        // SAFETY: `end <= len <= cap`, and `ptr` is valid for `[0, cap)`.
        unsafe { std::ptr::copy_nonoverlapping(b.ptr.add(index), out.as_mut_ptr(), out.len()) };
        true
    }

    /// Overwrite bytes starting at `index`. Returns `false` (no write) if the
    /// range is out of bounds. Lock-free.
    pub fn write(&self, index: usize, data: &[u8]) -> bool {
        let b = &*self.0;
        let Some(end) = index.checked_add(data.len()) else {
            return false;
        };
        if end > b.len.load(Ordering::Acquire) {
            return false;
        }
        // SAFETY: `end <= len <= cap`, and `ptr` is valid for `[0, cap)`.
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), b.ptr.add(index), data.len()) };
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
        unsafe { std::ptr::copy_nonoverlapping(b.ptr.add(index), old.as_mut_ptr(), old.len()) };
        if old == expected {
            // SAFETY: same bounds; `new.len() == expected.len()`.
            unsafe { std::ptr::copy_nonoverlapping(new.as_ptr(), b.ptr.add(index), new.len()) };
        }
        Some(old)
    }

    /// Snapshot the whole buffer (e.g. to seed a worker's domainMemory view).
    pub fn snapshot(&self) -> Vec<u8> {
        let b = &*self.0;
        let len = b.len.load(Ordering::Acquire);
        let mut v = vec![0u8; len];
        // SAFETY: `len <= cap`.
        unsafe { std::ptr::copy_nonoverlapping(b.ptr, v.as_mut_ptr(), len) };
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
    cvar: Condvar,
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

impl SharedMutex {
    pub fn new() -> Self {
        Self(Arc::new(SharedMutexInner {
            state: Mutex::new(MutexState::default()),
            cvar: Condvar::new(),
        }))
    }

    pub fn ptr_id(&self) -> usize {
        Arc::as_ptr(&self.0) as usize
    }

    /// Block until the mutex is acquired by the current thread.
    pub fn lock(&self) {
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
                    return;
                }
                Some(o) if o == me => {
                    state.depth += 1;
                    return;
                }
                Some(other) => {
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
        Self {
            mutex,
            cond_cvar: Arc::new(Condvar::new()),
        }
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

        // Re-acquire Flash ownership.
        while state.owner.is_some() {
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

        m.lock();
        m.lock(); // recursive
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
            cond2.mutex().lock();
            let woken = cond2.wait(Some(5000));
            cond2.mutex().unlock();
            woken
        });

        // Give the waiter time to park, then notify.
        std::thread::sleep(std::time::Duration::from_millis(100));
        cond.notify();

        assert!(waiter.join().unwrap(), "waiter should be woken, not timed out");
    }
}
