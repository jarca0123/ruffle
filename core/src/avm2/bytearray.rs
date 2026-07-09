use crate::avm2::Activation;
use crate::avm2::Error;
use crate::avm2::error::{Error2006Type, make_error_2006, make_error_2030};
use crate::avm2::worker_shared::SharedByteBuffer;
use crate::context::UpdateContext;
use flate2::Compression;
use flate2::read::*;
use gc_arena::Collect;
use ruffle_macros::Avm2Enum;
use std::cell::Cell;
use std::cmp;
use std::io::prelude::*;
use std::io::{self, SeekFrom};

#[derive(Clone, Collect, Debug, Copy, PartialEq, Eq)]
#[collect(no_drop)]
pub enum Endian {
    Big,
    Little,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Avm2Enum)]
pub enum CompressionAlgorithm {
    #[avm2_variant("zlib")]
    Zlib,
    #[avm2_variant("deflate")]
    Deflate,
    #[avm2_variant("lzma")]
    Lzma,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ByteArrayError {
    EndOfFile,
    IndexOutOfBounds,
}

impl ByteArrayError {
    #[inline(never)]
    pub fn to_avm<'gc>(self, activation: &mut Activation<'_, 'gc>) -> Error<'gc> {
        match self {
            ByteArrayError::EndOfFile => make_error_2030(activation),
            ByteArrayError::IndexOutOfBounds => {
                make_error_2006(activation, Error2006Type::RangeError)
            }
        }
    }
}

#[derive(Clone, Collect, Debug, Copy, PartialEq, Eq)]
#[collect(no_drop)]
pub enum ObjectEncoding {
    Amf0 = 0,
    Amf3 = 3,
}

thread_local! {
    /// Verify-mode domain-memory write log: `(addr, pre-write bytes)`, most-recent
    /// last. Armed by [`dm_log_start`]. Both the interpreter's `si*` and the JIT's
    /// `dm_store` helper funnel through [`ByteArrayStorage::dm_set`]/
    /// [`ByteArrayStorage::dm_write`], so this single per-thread log captures the
    /// writes of *both* engines — letting the JIT's call-aware verifier roll back a
    /// run's own (and its callees') domain-memory writes precisely, without a
    /// whole-buffer restore that would clobber other workers' concurrent writes.
    static DM_WRITE_LOG: std::cell::RefCell<Option<Vec<(usize, Vec<u8>)>>> =
        const { std::cell::RefCell::new(None) };
}

/// Arms the domain-memory write log (JIT verify). See [`DM_WRITE_LOG`].
pub fn dm_log_start() {
    DM_WRITE_LOG.with(|l| *l.borrow_mut() = Some(Vec::new()));
}

/// Disarms and returns the `(addr, pre-write bytes)` writes since [`dm_log_start`].
pub fn dm_log_take() -> Vec<(usize, Vec<u8>)> {
    DM_WRITE_LOG.with(|l| l.borrow_mut().take().unwrap_or_default())
}

#[derive(Clone, Debug)]
pub struct ByteArrayStorage {
    /// Underlying ByteArray
    bytes: Vec<u8>,

    /// The current position to read/write from
    position: Cell<usize>,

    /// This represents what endian to use while reading/writing data.
    endian: Endian,

    /// The encoding used when serializing/deserializing using readObject/writeObject
    object_encoding: ObjectEncoding,

    /// When this ByteArray is `shareable`, its bytes live SOLELY in an
    /// arena-external buffer shared by reference across worker threads (see
    /// [`SharedByteBuffer`]) — the single source of truth. `bytes` above is then
    /// NOT a full mirror (that doubled memory for every shareable ByteArray, e.g.
    /// the multi-MB FlasCC domainMemory RAM): it's a transient **read scratch**,
    /// materialized on demand only by the slice-returning read API
    /// ([`Self::materialize`]) so those borrows have stable backing (the shared
    /// buffer may move on wasm growth). Value reads/writes (`li*`/`si*`, `get`/
    /// `set`, atomics) go straight to the shared buffer, never through `bytes`.
    shared: Option<SharedByteBuffer>,
}

impl ByteArrayStorage {
    /// Create a new ByteArrayStorage
    pub fn new(context: &mut UpdateContext<'_>) -> ByteArrayStorage {
        ByteArrayStorage {
            bytes: Vec::new(),
            position: Cell::new(0),
            endian: Endian::Big,
            object_encoding: context.avm2.default_bytearray_encoding,
            shared: None,
        }
    }

    /// Create a new ByteArrayStorage using an already existing vector
    pub fn from_vec(context: &mut UpdateContext<'_>, bytes: Vec<u8>) -> ByteArrayStorage {
        ByteArrayStorage {
            bytes,
            position: Cell::new(0),
            endian: Endian::Big,
            object_encoding: context.avm2.default_bytearray_encoding,
            shared: None,
        }
    }

    /// Mark this ByteArray as `shareable`, MOVING its contents into a new
    /// arena-external shared buffer. Idempotent. The local `bytes` is dropped
    /// (freed): once shareable, the shared buffer is the sole store and `bytes`
    /// is only a transient read scratch (see the field doc / [`Self::materialize`]).
    /// This is the memory win — no more full duplicate per shareable ByteArray.
    pub fn make_shareable(&mut self) {
        if self.shared.is_none() {
            self.shared = Some(SharedByteBuffer::from_vec(std::mem::take(&mut self.bytes)));
            // Drop the (now-copied-into-shared) buffer; `bytes` becomes the scratch.
            self.bytes = Vec::new();
        }
    }

    /// Materializes the whole shared buffer into the `bytes` read scratch and
    /// returns it — the stable backing for a slice-returning read on a shareable
    /// ByteArray (the shared buffer itself may move on growth). For a non-shareable
    /// ByteArray `bytes` already IS the data, so this is a no-op. Only the
    /// slice-returning read API needs it; value ops go straight to the shared buffer.
    fn materialize(&mut self) -> &[u8] {
        if let Some(s) = &self.shared {
            let slen = s.len();
            if self.bytes.len() != slen {
                self.bytes.resize(slen, 0);
            }
            if slen > 0 {
                s.read(0, &mut self.bytes[..slen]);
            }
        }
        &self.bytes
    }

    /// Whether this ByteArray is backed by a shared buffer.
    pub fn is_shareable(&self) -> bool {
        self.shared.is_some()
    }

    /// The shared backing buffer, if this ByteArray is `shareable`. Cloning the
    /// handle is how a shared ByteArray crosses the worker boundary by reference.
    pub fn shared_buffer(&self) -> Option<SharedByteBuffer> {
        self.shared.clone()
    }

    /// Adopt an existing shared buffer (worker side of by-reference sharing):
    /// route all traffic through it. No mirror is seeded — the read scratch is
    /// materialized on demand (see the field doc).
    pub fn attach_shared(&mut self, buffer: SharedByteBuffer) {
        self.bytes = Vec::new();
        self.shared = Some(buffer);
    }

    // --- domain-memory fast path (`si*`/`li*`) ---
    // These route straight to the shared buffer when present so cross-thread
    // domainMemory access (FlasCC's shared "RAM") is coherent.

    /// Length as seen by domain-memory ops (authoritative shared length).
    pub fn dm_len(&self) -> usize {
        match &self.shared {
            Some(s) => s.len(),
            None => self.bytes.len(),
        }
    }

    /// Set the shared logical length (JIT verify: roll a run's heap growth back so
    /// each bracketed re-run starts from the same `sbrk` break). Shared only —
    /// the allocator's grow decision reads [`Self::dm_len`], i.e. the shared length.
    pub fn dm_set_len(&mut self, len: usize) {
        if let Some(s) = &self.shared {
            s.set_len(len);
        }
    }

    /// For the JIT's **inline** `li*`/`si*` fast path: ensures domainMemory is
    /// `shareable` and returns the address of its stable `[base, cap]`
    /// **descriptor cell** (see [`SharedByteBuffer::desc_ptr`]). The emitted
    /// code loads base+cap from the cell on every access, so the buffer itself
    /// may move on growth (no reservation) — even under a live JIT frame. The
    /// second element is unused (kept for the run ABI). `None` if not shareable.
    pub fn dm_base_len(&mut self) -> Option<(usize, usize)> {
        self.make_shareable();
        self.shared.as_ref().map(|s| (s.desc_ptr(), 0))
    }

    /// Snapshot the whole domainMemory (the *shared* buffer when shareable — where
    /// the JIT's inline `si*` writes land — else the local bytes). For the JIT's
    /// full-domain-memory differential verifier.
    pub fn dm_snapshot(&self) -> Vec<u8> {
        match &self.shared {
            Some(s) => s.snapshot(),
            None => self.bytes.clone(),
        }
    }

    /// Restore domainMemory from a [`Self::dm_snapshot`] (verifier: reset to the
    /// pre-JIT state before re-running the interpreter).
    pub fn dm_restore(&mut self, data: &[u8]) {
        match &self.shared {
            Some(s) => {
                s.write(0, data);
            }
            None => {
                let n = data.len().min(self.bytes.len());
                self.bytes[..n].copy_from_slice(&data[..n]);
            }
        }
    }

    /// Read a single domain-memory byte.
    pub fn dm_get(&self, index: usize) -> Option<u8> {
        match &self.shared {
            Some(s) => {
                let mut b = [0u8; 1];
                s.read(index, &mut b).then_some(b[0])
            }
            None => self.bytes.get(index).copied(),
        }
    }

    /// Read `N` little-endian domain-memory bytes into a fixed array.
    pub fn dm_read<const N: usize>(&self, index: usize) -> Option<[u8; N]> {
        match &self.shared {
            Some(s) => {
                let mut b = [0u8; N];
                s.read(index, &mut b).then_some(b)
            }
            None => index
                .checked_add(N)
                .and_then(|end| self.bytes.get(index..end))
                .map(|s| s.try_into().unwrap()),
        }
    }

    /// Write domain-memory bytes at `index` (non-growing); also updates the
    /// local mirror.
    pub fn dm_write(&mut self, index: usize, data: &[u8]) -> Result<(), ByteArrayError> {
        self.dm_log_write(index, data.len());
        if let Some(s) = &self.shared {
            if !s.write(index, data) {
                return Err(ByteArrayError::IndexOutOfBounds);
            }
        }
        self.write_at_nongrowing(data, index)
    }

    /// Write a single domain-memory byte at `index` (non-growing).
    pub fn dm_set(&mut self, index: usize, value: u8) {
        self.dm_log_write(index, 1);
        if let Some(s) = &self.shared {
            s.write(index, &[value]);
        }
        if index < self.bytes.len() {
            self.bytes[index] = value;
        }
    }

    /// If the verify write-log is armed, record the `n` pre-write bytes at `index`
    /// so the caller can restore exactly this range later. Cheap no-op otherwise.
    fn dm_log_write(&self, index: usize, n: usize) {
        if DM_WRITE_LOG.with(|l| l.borrow().is_some()) {
            let old: Vec<u8> = (0..n).map(|i| self.dm_get(index + i).unwrap_or(0)).collect();
            DM_WRITE_LOG.with(|l| {
                if let Some(log) = l.borrow_mut().as_mut() {
                    log.push((index, old));
                }
            });
        }
    }

    /// Atomic compare-and-swap of a 32-bit **little-endian** integer at `index`,
    /// backing the `avm2.intrinsics.memory.casi32` intrinsic. Unlike
    /// [`atomic_compare_and_swap_int_at`](Self::atomic_compare_and_swap_int_at)
    /// (which respects the ByteArray's endianness, as the `ByteArray` AS3 method
    /// requires), domain memory is always little-endian — matching `si32`/`li32`.
    ///
    /// Returns the value that was at `index` before the call (whether or not the
    /// swap happened), or `None` if `[index, index + 4)` is out of bounds.
    pub fn dm_cas32(&mut self, index: usize, expected: i32, new: i32) -> Option<i32> {
        let expected_bytes = expected.to_le_bytes();
        let new_bytes = new.to_le_bytes();

        if let Some(shared) = self.shared.clone() {
            let old = shared.cas_bytes(index, &expected_bytes, &new_bytes)?;
            let old: [u8; 4] = old.as_slice().try_into().ok()?;
            // Keep the local mirror consistent with the post-CAS state.
            let resulting = if old == expected_bytes { new_bytes } else { old };
            let _ = self.write_at_nongrowing(&resulting, index);
            return Some(i32::from_le_bytes(old));
        }

        let cur = self.dm_read::<4>(index)?;
        let old = i32::from_le_bytes(cur);
        if old == expected {
            let _ = self.write_at_nongrowing(&new_bytes, index);
        }
        Some(old)
    }

    /// Write bytes at the next position in the ByteArray, growing if needed.
    #[inline]
    pub fn write_bytes(&mut self, buf: &[u8]) -> Result<(), ByteArrayError> {
        self.write_at(buf, self.position.get())?;
        self.position.set(self.position.get() + buf.len());
        Ok(())
    }

    #[inline]
    pub fn write_bytes_within(&mut self, start: usize, amnt: usize) -> Result<(), ByteArrayError> {
        self.write_at_within(start, amnt, self.position.get())?;
        self.position.set(self.position.get() + amnt);
        Ok(())
    }

    /// Reads any amount of bytes from the current position in the ByteArray
    #[inline]
    pub fn read_bytes(&mut self, amnt: usize) -> Result<&[u8], ByteArrayError> {
        let pos = self.position.get();
        let end = pos.checked_add(amnt).ok_or(ByteArrayError::EndOfFile)?;
        // Refresh the region from the shared buffer (see `read_at`).
        if self.shared.is_some() {
            if end > self.bytes.len() {
                self.bytes.resize(end, 0);
            }
            let shared = self.shared.clone().unwrap();
            if amnt > 0 && !shared.read(pos, &mut self.bytes[pos..end]) {
                return Err(ByteArrayError::EndOfFile);
            }
        }
        let bytes = self.bytes.get(pos..end).ok_or(ByteArrayError::EndOfFile)?;
        // `position` is a disjoint `Cell` field, so this does not clash with the
        // immutable borrow of `self.bytes` held by `bytes`.
        self.position.set(end);
        Ok(bytes)
    }

    /// Same as `read_bytes`, but:
    /// - cuts the result at the first null byte to recreate a bug in FP
    /// - strips off an optional UTF8 BOM at the beginning
    pub fn read_utf_bytes(&mut self, amnt: usize) -> Result<&[u8], ByteArrayError> {
        let mut bytes = self.read_bytes(amnt)?;
        if let Some(without_bom) = bytes.strip_prefix(&[0xEF, 0xBB, 0xBF]) {
            bytes = without_bom;
        }
        if let Some(null) = bytes.iter().position(|b| *b == b'\0') {
            bytes = &bytes[..null];
        }
        Ok(bytes)
    }

    /// Reads any amount of bytes at any offset in the ByteArray.
    ///
    /// Takes `&mut self` because, for a `shareable` ByteArray, the requested
    /// region is first refreshed from the arena-external shared buffer — this is
    /// how the DataInput read API (`readBytes`/`readUTFBytes`/...) observes writes
    /// made by *other* worker threads (the domain-memory `li*` opcodes already go
    /// straight to the shared buffer; the slice-returning API kept a stale mirror).
    #[inline]
    pub fn read_at(&mut self, amnt: usize, offset: usize) -> Result<&[u8], ByteArrayError> {
        if self.shared.is_some() {
            let end = offset
                .checked_add(amnt)
                .ok_or(ByteArrayError::EndOfFile)?;
            if end > self.bytes.len() {
                self.bytes.resize(end, 0);
            }
            let shared = self.shared.clone().unwrap();
            if amnt > 0 && !shared.read(offset, &mut self.bytes[offset..end]) {
                return Err(ByteArrayError::EndOfFile);
            }
        }
        self.bytes
            .get(offset..)
            .and_then(|bytes| bytes.get(..amnt))
            .ok_or(ByteArrayError::EndOfFile)
    }

    /// Write bytes at any offset in the ByteArray
    /// Will automatically grow the ByteArray to fit the new buffer
    pub fn write_at(&mut self, buf: &[u8], offset: usize) -> Result<(), ByteArrayError> {
        // DIAGNOSTIC (RUFFLE_DUMP_LUA_SRC): dump a ByteArray write that carries the
        // CrossBridge Lua sandbox source (`CModule.writeString` → `writeUTFBytes`),
        // so the bytes actually landing in domainMemory can be diffed against the
        // embedded `sandbox_env.lua` asset. The `len` gate short-circuits the common
        // small write before the (rarer) signature scan / env read.
        if buf.len() > 10_000
            && buf.windows(11).any(|w| w == b"sandbox_env")
            && std::env::var("RUFFLE_DUMP_LUA_SRC").is_ok()
        {
            let path = format!("/tmp/ram_lua_src_{offset}.bin");
            let _ = std::fs::write(&path, buf);
            tracing::error!(
                "RUFFLE_DUMP_LUA_SRC: wrote {} source bytes at offset {offset} -> {path}",
                buf.len()
            );
        }
        if offset.saturating_add(buf.len()) > u32::MAX as usize {
            return Err(ByteArrayError::IndexOutOfBounds);
        }

        // We know this is safe as we've already checked it's u32::MAX or lower
        let new_len = offset + buf.len();
        if self.len() < new_len {
            self.set_length(new_len);
        }
        match &self.shared {
            // Shareable: write straight to the shared buffer (the sole store);
            // `set_length` above already grew it. The read scratch is materialized
            // on demand and is NOT grown on write (the memory win).
            Some(s) => {
                s.write(offset, buf);
            }
            None => {
                if self.bytes.len() < new_len {
                    self.bytes.resize(new_len, 0);
                }
                self.bytes
                    .get_mut(offset..new_len)
                    .expect("ByteArray write out of bounds")
                    .copy_from_slice(buf);
            }
        }
        Ok(())
    }

    /// Write bytes at any offset in the ByteArray
    /// Will return an error if the new buffer does not fit the ByteArray
    pub fn write_at_nongrowing(&mut self, buf: &[u8], offset: usize) -> Result<(), ByteArrayError> {
        // Shareable: the store is the shared buffer, already written by the only
        // callers (`dm_write` / `atomic_compare_and_swap_int_at`); nothing to do
        // to the scratch (it's materialized on read). Bounds were validated by the
        // caller's `s.write`.
        if self.shared.is_some() {
            return Ok(());
        }
        self.bytes
            .get_mut(offset..)
            .and_then(|bytes| bytes.get_mut(..buf.len()))
            .ok_or(ByteArrayError::IndexOutOfBounds)?
            .copy_from_slice(buf);
        Ok(())
    }

    /// Write bytes at any offset in the ByteArray from within the current ByteArray using a memmove.
    /// Will automatically grow the ByteArray to fit the new buffer
    pub fn write_at_within(
        &mut self,
        start: usize,
        amnt: usize,
        offset: usize,
    ) -> Result<(), ByteArrayError> {
        // First verify that reading from `start` to `amnt` is valid
        let end = start
            .checked_add(amnt)
            .filter(|result| *result <= self.len())
            .ok_or(ByteArrayError::EndOfFile)?;

        // Second we resize our underlying buffer to ensure that writing `amnt` from `offset` is valid.
        if offset.saturating_add(amnt) > u32::MAX as usize {
            return Err(ByteArrayError::IndexOutOfBounds);
        }

        // We know this is safe as we've already checked it's u32::MAX or lower
        let new_len = offset + amnt;
        if self.len() < new_len {
            self.set_length(new_len);
        }
        match &self.shared {
            // Shareable: memmove within the shared buffer via a small transient
            // buffer (the store is shared; no scratch growth). Reading the source
            // range from shared also picks up another thread's writes.
            Some(s) => {
                if amnt > 0 {
                    let mut tmp = vec![0u8; amnt];
                    s.read(start, &mut tmp);
                    s.write(offset, &tmp);
                }
            }
            None => {
                if self.bytes.len() < new_len {
                    self.bytes.resize(new_len, 0);
                }
                self.bytes.copy_within(start..end, offset);
            }
        }
        Ok(())
    }

    /// Compress the ByteArray into a temporary buffer.
    pub fn compress(&mut self, algorithm: CompressionAlgorithm) -> Vec<u8> {
        self.materialize(); // refresh the read scratch (no-op when not shareable)
        let mut buffer = Vec::new();
        let error: Option<Box<dyn std::error::Error>> = match algorithm {
            CompressionAlgorithm::Zlib => {
                // Note: some content is sensitive to compression type
                // (as it's visible in the header)
                let mut encoder = ZlibEncoder::new(&*self.bytes, Compression::best());
                encoder.read_to_end(&mut buffer).err().map(|e| e.into())
            }
            CompressionAlgorithm::Deflate => {
                let mut encoder = DeflateEncoder::new(&*self.bytes, Compression::best());
                encoder.read_to_end(&mut buffer).err().map(|e| e.into())
            }
            #[cfg(feature = "lzma")]
            CompressionAlgorithm::Lzma => lzma_rs::lzma_compress(&mut &*self.bytes, &mut buffer)
                .err()
                .map(|e| e.into()),
            #[cfg(not(feature = "lzma"))]
            CompressionAlgorithm::Lzma => Some("Ruffle was not compiled with LZMA support".into()),
        };
        if let Some(error) = error {
            // On error, just return an empty buffer.
            tracing::warn!("ByteArray.compress: {}", error);
            buffer.clear();
        }
        buffer
    }

    /// Decompress the ByteArray into a temporary buffer.
    pub fn decompress(&mut self, algorithm: CompressionAlgorithm) -> Option<Vec<u8>> {
        self.materialize(); // refresh the read scratch (no-op when not shareable)
        let mut buffer = Vec::new();
        let error: Option<Box<dyn std::error::Error>> = match algorithm {
            CompressionAlgorithm::Zlib => {
                let mut decoder = ZlibDecoder::new(&*self.bytes);
                decoder.read_to_end(&mut buffer).err().map(|e| e.into())
            }
            CompressionAlgorithm::Deflate => {
                let mut decoder = DeflateDecoder::new(&*self.bytes);
                decoder.read_to_end(&mut buffer).err().map(|e| e.into())
            }
            #[cfg(feature = "lzma")]
            CompressionAlgorithm::Lzma => lzma_rs::lzma_decompress(&mut &*self.bytes, &mut buffer)
                .err()
                .map(|e| e.into()),
            #[cfg(not(feature = "lzma"))]
            CompressionAlgorithm::Lzma => Some("Ruffle was not compiled with LZMA support".into()),
        };
        if let Some(error) = error {
            tracing::warn!("ByteArray.decompress: {}", error);
            None
        } else {
            Some(buffer)
        }
    }

    pub fn read_utf(&mut self) -> Result<&[u8], ByteArrayError> {
        let len = self.read_unsigned_short()?;
        let val = self.read_utf_bytes(len.into())?;
        Ok(val)
    }

    pub fn write_boolean(&mut self, val: bool) -> Result<(), ByteArrayError> {
        self.write_bytes(&[val as u8; 1])
    }

    pub fn read_boolean(&mut self) -> Result<bool, ByteArrayError> {
        Ok(self.read_bytes(1)? != [0])
    }

    // Writes a UTF String into the buffer, with its length as a prefix
    pub fn write_utf(&mut self, utf_string: &str) -> Result<(), ByteArrayError> {
        if let Ok(str_size) = u16::try_from(utf_string.len()) {
            self.write_unsigned_short(str_size)?;
            self.write_bytes(utf_string.as_bytes())
        } else {
            Err(ByteArrayError::IndexOutOfBounds)
        }
    }

    #[inline]
    pub fn clear(&mut self) {
        self.bytes.clear();
        self.position.set(0);
        if let Some(s) = &self.shared {
            s.set_len(0);
        }
    }

    #[inline]
    pub fn shrink_to_fit(&mut self) {
        self.bytes.shrink_to_fit()
    }

    #[inline]
    pub fn set_length(&mut self, new_len: usize) {
        if let Some(s) = &self.shared {
            // Honor the requested length, shrink included. The old code refused to
            // shrink a shared buffer, to guard against a `new_len` *derived from a
            // stale-short local mirror* (another worker had grown the shared buffer
            // via FlasCC `sbrk`, so this arena's mirror lagged). The mirror is gone
            // now — `len()` returns the true shared length directly — so every
            // `new_len` here is authoritative: the internal grow paths pass a value
            // `>= len()`, and an explicit `ByteArray.length = X` is the app's own
            // resize (which must be able to shrink; otherwise the reported length
            // stays stuck high — the mops "float min" range bug). Cross-thread `sbrk`
            // growth still goes through `atomic_compare_and_swap_length`, not here.
            s.set_len(new_len);
            // Don't grow the scratch — it's materialized on read.
            self.position.set(self.position().min(new_len));
        } else {
            self.bytes.resize(new_len, 0);
            self.position.set(self.position().min(new_len));
        }
    }

    /// Atomically compares the 32-bit integer at `index` with `expected` and,
    /// if they are equal, replaces it with `new`. Returns the value that was at
    /// `index` before the call, regardless of whether the swap happened.
    ///
    /// The integer is read/written using the ByteArray's current endianness.
    /// Returns [`ByteArrayError::IndexOutOfBounds`] (a `RangeError` in AS3) if
    /// `[index, index + 4)` is not entirely within the ByteArray.
    ///
    /// In Ruffle's single-threaded cooperative model there is no real
    /// contention, so this is simply a read-compare-write; the semantics match
    /// `flash.utils.ByteArray.atomicCompareAndSwapIntAt`.
    pub fn atomic_compare_and_swap_int_at(
        &mut self,
        index: usize,
        expected: i32,
        new: i32,
    ) -> Result<i32, ByteArrayError> {
        let endian = self.endian;
        let decode = |bytes: [u8; 4]| match endian {
            Endian::Big => i32::from_be_bytes(bytes),
            Endian::Little => i32::from_le_bytes(bytes),
        };
        let encode = |value: i32| match endian {
            Endian::Big => value.to_be_bytes(),
            Endian::Little => value.to_le_bytes(),
        };

        // Shared ByteArray: the compare-and-swap must be atomic against other
        // worker threads, so it happens inside the shared buffer.
        if let Some(shared) = self.shared.clone() {
            let expected_bytes = encode(expected);
            let new_bytes = encode(new);
            let old_bytes = shared
                .cas_bytes(index, &expected_bytes, &new_bytes)
                .ok_or(ByteArrayError::IndexOutOfBounds)?;
            // Keep the local mirror consistent with the post-CAS state.
            let resulting = if old_bytes == expected_bytes {
                new_bytes
            } else {
                old_bytes.clone().try_into().unwrap()
            };
            let _ = self.write_at_nongrowing(&resulting, index);
            return Ok(decode(old_bytes.try_into().unwrap()));
        }

        if index.checked_add(4).is_none_or(|end| end > self.len()) {
            return Err(ByteArrayError::IndexOutOfBounds);
        }

        let old = self.read_int_at(index)?;
        if old == expected {
            self.write_at_nongrowing(&encode(new), index)?;
        }
        Ok(old)
    }

    /// Atomically compares the ByteArray's length with `expected` and, if they
    /// are equal, changes the length to `new`. Returns the length as it was
    /// before the call, regardless of whether the swap happened.
    ///
    /// This backs `flash.utils.ByteArray.atomicCompareAndSwapLength`, which
    /// FlasCC/CrossBridge uses to grow the shared "RAM" heap in `sbrk`.
    pub fn atomic_compare_and_swap_length(&mut self, expected: usize, new: usize) -> usize {
        if let Some(shared) = self.shared.clone() {
            let old = shared.cas_len(expected, new);
            // Don't grow the scratch — materialized on read.
            self.position.set(self.position().min(shared.len()));
            return old;
        }
        let old = self.len();
        if old == expected {
            self.set_length(new);
        }
        old
    }

    pub fn get(&self, pos: usize) -> Option<u8> {
        // Reads the shared buffer when `shareable`, so single-byte index reads
        // stay coherent with cross-thread writes.
        self.dm_get(pos)
    }

    pub fn set(&mut self, item: usize, value: u8) {
        if self.len() < (item + 1) {
            self.set_length(item + 1);
        }
        match &self.shared {
            Some(s) => {
                s.write(item, &[value]);
            }
            None => {
                *self.bytes.get_mut(item).unwrap() = value;
            }
        }
    }

    /// Swap all data stored in this bytearray with the passed `Vec<u8>`. This
    /// method sets the bytearray's `position` to 0.
    pub fn swap_storage_with(&mut self, new_data: &mut Vec<u8>) {
        self.position.set(0);
        // Materialize the current content into the scratch, then swap: the caller
        // receives the old content and we adopt theirs. For a shareable ByteArray
        // the new content is installed into the shared buffer (the store).
        self.materialize();
        std::mem::swap(&mut self.bytes, new_data);
        if let Some(s) = self.shared.clone() {
            s.set_len(self.bytes.len());
            s.write(0, &self.bytes);
            self.bytes = Vec::new(); // back to the on-demand scratch model
        }
    }

    /// Write a single byte at any offset in the bytearray, panicking if out of bounds.
    pub fn set_nongrowing(&mut self, item: usize, value: u8) {
        match &self.shared {
            Some(s) => {
                s.write(item, &[value]);
            }
            None => {
                self.bytes[item] = value;
            }
        }
    }

    pub fn delete(&mut self, item: usize) {
        match &self.shared {
            Some(s) => {
                if item < s.len() {
                    s.write(item, &[0]);
                }
            }
            None => {
                if let Some(i) = self.bytes.get_mut(item) {
                    *i = 0;
                }
            }
        }
    }

    /// The whole buffer as a slice. Takes `&mut self` because a shareable
    /// ByteArray must first materialize the shared buffer into the read scratch
    /// (there is no persistent mirror). For a non-shareable ByteArray this is the
    /// plain backing.
    #[inline]
    pub fn bytes(&mut self) -> &[u8] {
        self.materialize()
    }

    /// The whole buffer as a mutable slice, materialized from shared first. A
    /// mutation through this slice is written back to the shared buffer with
    /// [`Self::flush_scratch`] — call it after mutating (or use a write method).
    /// Only valid for wholesale in-place edits; prefer `write_at` for offset writes.
    #[inline]
    pub fn bytes_mut(&mut self) -> &mut [u8] {
        self.materialize();
        &mut self.bytes
    }

    /// Writes the read scratch back to the shared buffer (after a `bytes_mut`
    /// mutation). No-op for a non-shareable ByteArray.
    #[inline]
    pub fn flush_scratch(&mut self) {
        if let Some(s) = &self.shared {
            if !self.bytes.is_empty() {
                s.write(0, &self.bytes);
            }
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        // For a shareable ByteArray the authoritative length lives in the shared
        // buffer: another worker thread may have grown it (FlasCC `sbrk`) without
        // this arena's mirror seeing it. Returning the stale mirror length here
        // made the primordial's `sbrk`/malloc compute wrong lengths and try to
        // shrink the shared heap.
        match &self.shared {
            Some(s) => s.len(),
            None => self.bytes.len(),
        }
    }

    #[inline]
    pub fn position(&self) -> usize {
        self.position.get()
    }

    #[inline]
    pub fn set_position(&self, pos: usize) {
        self.position.set(pos);
    }

    #[inline]
    pub fn endian(&self) -> Endian {
        self.endian
    }

    #[inline]
    pub fn set_endian(&mut self, new_endian: Endian) {
        self.endian = new_endian;
    }

    #[inline]
    pub fn object_encoding(&self) -> ObjectEncoding {
        self.object_encoding
    }

    #[inline]
    pub fn set_object_encoding(&mut self, new_object_encoding: ObjectEncoding) {
        self.object_encoding = new_object_encoding;
    }

    #[inline]
    pub fn bytes_available(&self) -> usize {
        self.len().saturating_sub(self.position.get())
    }
}

impl Write for ByteArrayStorage {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.write_bytes(buf)
            .map_err(|_| io::Error::other("Failed to write to ByteArrayStorage"))?;

        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl Read for ByteArrayStorage {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let bytes = self
            .read_bytes(cmp::min(buf.len(), self.bytes_available()))
            .map_err(|_| io::Error::other("Failed to read from ByteArrayStorage"))?;
        buf[..bytes.len()].copy_from_slice(bytes);
        Ok(bytes.len())
    }
}

impl Seek for ByteArrayStorage {
    fn seek(&mut self, style: SeekFrom) -> io::Result<u64> {
        let (base_pos, offset) = match style {
            SeekFrom::Start(n) => {
                self.position.set(n as usize);
                return Ok(n);
            }
            SeekFrom::End(n) => (self.len(), n),
            SeekFrom::Current(n) => (self.position.get(), n),
        };

        let new_pos = if offset >= 0 {
            base_pos.checked_add(offset as usize)
        } else {
            base_pos.checked_sub((offset.wrapping_neg()) as usize)
        };

        match new_pos {
            Some(n) => {
                self.position.set(n);
                Ok(n as u64)
            }
            None => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "invalid seek to a negative or overflowing position",
            )),
        }
    }
}

macro_rules! impl_write{
    ($($method_name:ident $data_type:ty ), *)
    =>
    {
        impl ByteArrayStorage {
            $( pub fn $method_name (&mut self, val: $data_type) -> Result<(), ByteArrayError> {
                let val_bytes = match self.endian {
                    Endian::Big => val.to_be_bytes(),
                    Endian::Little => val.to_le_bytes(),
                };
                self.write_bytes(&val_bytes)
             } )*
        }
    }
}

macro_rules! impl_read{
    ($($method_name:ident $at_method_name:ident $size:expr; $data_type:ty ), *)
    =>
    {
        impl ByteArrayStorage {
            // Position-based typed reads route through `dm_read`, which reads from
            // the shared buffer when this ByteArray is `shareable`. This keeps
            // `readInt`/`readByte`/... coherent with cross-thread `si*` writes
            // (FlasCC's shared RAM is accessed via *both* the DataInput API and
            // domain-memory opcodes).
            $( pub fn $method_name (&self) -> Result<$data_type, ByteArrayError> {
                let pos = self.position.get();
                let bytes = self.dm_read::<$size>(pos).ok_or(ByteArrayError::EndOfFile)?;
                self.position.set(pos + $size);
                Ok(match self.endian {
                    Endian::Big => <$data_type>::from_be_bytes(bytes),
                    Endian::Little => <$data_type>::from_le_bytes(bytes),
                })
             } )*

             $( pub fn $at_method_name (&self, offset: usize) -> Result<$data_type, ByteArrayError> {
                let bytes = self.dm_read::<$size>(offset).ok_or(ByteArrayError::EndOfFile)?;
                Ok(match self.endian {
                    Endian::Big => <$data_type>::from_be_bytes(bytes),
                    Endian::Little => <$data_type>::from_le_bytes(bytes),
                })
             } )*
        }
    }
}

impl_write!(write_float f32, write_double f64, write_int i32, write_unsigned_int u32, write_short i16, write_unsigned_short u16);
impl_read!(read_float read_float_at 4; f32, read_double read_double_at 8; f64, read_int read_int_at 4; i32, read_unsigned_int read_unsigned_int_at 4; u32, read_short read_short_at 2; i16, read_unsigned_short read_unsigned_short_at 2; u16, read_byte read_byte_at 1; i8, read_unsigned_byte read_unsigned_byte_at 1; u8);
