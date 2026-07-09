//! A TLSF (two-level segregated fit) global allocator for the wasm build.
//!
//! Rust's default wasm allocator (`dlmalloc`) searches free lists on every
//! allocation. Under Ruffle's player-in-worker workload — MB-scale
//! `BitmapData` pixel buffers interleaved with a torrent of tiny
//! tessellation/`Gc` allocations, all on the one worker thread — the heap
//! fragments badly and those searches dominate: traces show ~77% of the worker
//! thread's time inside `dlmalloc::malloc` self-time, with no `memory.grow`
//! storm (the shared memory reserves its 4 GiB maximum up front, so growth is
//! cheap) and no lock contention (only the worker allocates). The cost is pure
//! free-list traversal.
//!
//! TLSF allocates, frees and (when adjacent) reallocates in O(1) regardless of
//! fragmentation — it never walks a free list — which removes that cost.
//!
//! We drive `rlsf`'s core `Tlsf` engine ourselves rather than use its bundled
//! `GlobalTlsf`, so the memory source and locking are explicit and correct for
//! the shared, multi-threaded linear memory:
//!
//! * Memory is claimed from the wasm linear memory via `memory.grow`. We only
//!   ever manage pages we obtained that way — never the region below the
//!   initial memory size, which holds the linker's static data and shadow
//!   stack — so we can't corrupt them. Consecutive grows are contiguous (we are
//!   the only grower, and only under the lock), so they extend one pool and
//!   TLSF can coalesce a freed tail across the grow boundary.
//! * Access is serialised by a spinlock. Contention is effectively nil (one
//!   thread does virtually all allocation), so a spinlock avoids `std`
//!   machinery and any futex wait in the allocation path.

use core::alloc::{GlobalAlloc, Layout};
use core::cell::UnsafeCell;
use core::ptr::{self, NonNull};
use core::sync::atomic::{AtomicBool, Ordering};
use rlsf::Tlsf;

/// `Tlsf<'pool, FLBitmap, SLBitmap, FLLEN, SLLEN>`.
///
/// `GRANULARITY = size_of::<usize>() * 4 = 16` bytes on wasm32, and the largest
/// size class is `GRANULARITY << FLLEN = 16 << 28 = 4 GiB`, so a single free
/// block or allocation can span the whole addressable linear memory. The
/// second level splits each class into `SLLEN = 16` sub-classes. Bitmap widths
/// must cover the level lengths: `u32` (≥ 28 bits) and `u16` (= 16 bits).
type Heap = Tlsf<'static, u32, u16, 28, 16>;

/// wasm linear-memory page size.
const PAGE: usize = 65536;

/// Grow the linear memory by at least this many pages at a time, to amortise
/// `memory.grow` calls and keep the managed regions coarse. 256 pages = 16 MiB.
const MIN_GROW_PAGES: usize = 256;

struct SpinLock {
    locked: AtomicBool,
}

impl SpinLock {
    const fn new() -> Self {
        Self {
            locked: AtomicBool::new(false),
        }
    }

    #[inline]
    fn lock(&self) {
        while self
            .locked
            .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            core::hint::spin_loop();
        }
    }

    #[inline]
    fn unlock(&self) {
        self.locked.store(false, Ordering::Release);
    }
}

struct State {
    heap: Heap,
    /// Byte offset one past the end of the last region handed to the pool —
    /// i.e. the end of the linear memory we manage. `0` until the first grow.
    end: usize,
}

pub struct TlsfAlloc {
    lock: SpinLock,
    state: UnsafeCell<State>,
}

// Access to `state` is serialised by `lock`, so it is sound to share.
unsafe impl Sync for TlsfAlloc {}

impl TlsfAlloc {
    pub const fn new() -> Self {
        Self {
            lock: SpinLock::new(),
            state: UnsafeCell::new(State {
                heap: Tlsf::new(),
                end: 0,
            }),
        }
    }

    /// Claim at least `min_bytes` more of linear memory and hand it to the pool.
    /// Returns `false` if the memory can't be grown (out of address space).
    ///
    /// # Safety
    /// The caller must hold `self.lock` and pass the unique `&mut State`.
    unsafe fn grow(state: &mut State, min_bytes: usize) -> bool {
        // Round the request up to whole pages (plus a page of slack for TLSF's
        // per-block overhead and alignment waste), and grow by at least the
        // coarse minimum.
        let need_pages = min_bytes.div_ceil(PAGE) + 1;
        let pages = need_pages.max(MIN_GROW_PAGES);

        let prev = core::arch::wasm32::memory_grow::<0>(pages);
        if prev == usize::MAX {
            return false;
        }

        let start = prev * PAGE;
        let len = pages * PAGE;
        let Some(block) = NonNull::new(ptr::slice_from_raw_parts_mut(start as *mut u8, len)) else {
            return false;
        };

        if state.end == start {
            // Contiguous with the previous region: extend the same pool so a
            // freed tail can coalesce across the grow boundary.
            unsafe { state.heap.append_free_block_ptr(block) };
        } else {
            // First grow (or a non-contiguous one, which shouldn't happen since
            // we are the only grower): start a new pool.
            unsafe { state.heap.insert_free_block_ptr(block) };
        }
        state.end = start + len;
        true
    }
}

unsafe impl GlobalAlloc for TlsfAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        self.lock.lock();
        let state = unsafe { &mut *self.state.get() };

        let ptr = match state.heap.allocate(layout) {
            Some(p) => p.as_ptr(),
            None => {
                // Pool exhausted: grow (claiming at least this request) and retry.
                if unsafe { Self::grow(state, layout.size() + layout.align()) } {
                    state
                        .heap
                        .allocate(layout)
                        .map_or(ptr::null_mut(), |p| p.as_ptr())
                } else {
                    ptr::null_mut()
                }
            }
        };

        self.lock.unlock();
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let Some(nn) = NonNull::new(ptr) else { return };
        self.lock.lock();
        let state = unsafe { &mut *self.state.get() };
        unsafe { state.heap.deallocate(nn, layout.align()) };
        self.lock.unlock();
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let Some(nn) = NonNull::new(ptr) else {
            return unsafe { self.alloc(Layout::from_size_align_unchecked(new_size, layout.align())) };
        };
        let Ok(new_layout) = Layout::from_size_align(new_size, layout.align()) else {
            return ptr::null_mut();
        };

        self.lock.lock();
        let state = unsafe { &mut *self.state.get() };

        // `reallocate` grows/shrinks in place when possible (O(1)); otherwise it
        // moves. On out-of-memory it returns `None` and leaves the original
        // block intact, so we can grow and retry.
        if let Some(p) = unsafe { state.heap.reallocate(nn, new_layout) } {
            self.lock.unlock();
            return p.as_ptr();
        }
        if unsafe { Self::grow(state, new_size + layout.align()) } {
            if let Some(p) = unsafe { state.heap.reallocate(nn, new_layout) } {
                self.lock.unlock();
                return p.as_ptr();
            }
        }
        self.lock.unlock();
        ptr::null_mut()
    }
}
