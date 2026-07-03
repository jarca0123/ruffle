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

    /// When this ByteArray is `shareable`, its bytes are backed by an
    /// arena-external buffer shared by reference across worker threads (see
    /// [`SharedByteBuffer`]). `bytes` above then acts as a local write-through
    /// mirror; domain-memory (`si*`/`li*`) traffic and the atomic operations go
    /// straight to the shared buffer for cross-thread correctness.
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

    /// Mark this ByteArray as `shareable`, moving its current contents into a new
    /// arena-external shared buffer. Idempotent.
    pub fn make_shareable(&mut self) {
        if self.shared.is_none() {
            self.shared = Some(SharedByteBuffer::from_vec(self.bytes.clone()));
        }
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
    /// seed the local mirror from it and route future shared traffic through it.
    pub fn attach_shared(&mut self, buffer: SharedByteBuffer) {
        self.bytes = buffer.snapshot();
        self.shared = Some(buffer);
    }

    /// Refresh the local mirror from the shared buffer, making writes performed
    /// by other worker threads visible to the borrow-based read API.
    pub fn resync_shared(&mut self) {
        if let Some(shared) = &self.shared {
            self.bytes = shared.snapshot();
        }
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
            None => self.bytes.get(index..index + N).map(|s| s.try_into().unwrap()),
        }
    }

    /// Write domain-memory bytes at `index` (non-growing); also updates the
    /// local mirror.
    pub fn dm_write(&mut self, index: usize, data: &[u8]) -> Result<(), ByteArrayError> {
        if let Some(s) = &self.shared {
            if !s.write(index, data) {
                return Err(ByteArrayError::IndexOutOfBounds);
            }
        }
        self.write_at_nongrowing(data, index)
    }

    /// Write a single domain-memory byte at `index` (non-growing).
    pub fn dm_set(&mut self, index: usize, value: u8) {
        if let Some(s) = &self.shared {
            s.write(index, &[value]);
        }
        if index < self.bytes.len() {
            self.bytes[index] = value;
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
        if offset.saturating_add(buf.len()) > u32::MAX as usize {
            return Err(ByteArrayError::IndexOutOfBounds);
        }

        // We know this is safe as we've already checked it's u32::MAX or lower
        let new_len = offset + buf.len();
        if self.len() < new_len {
            self.set_length(new_len);
        }
        // The mirror can physically lag the (shared) logical length; grow it so
        // the indexed write below cannot panic.
        self.ensure_mirror_len();
        if self.bytes.len() < new_len {
            self.bytes.resize(new_len, 0);
        }
        self.bytes
            .get_mut(offset..new_len)
            .expect("ByteArray write out of bounds")
            .copy_from_slice(buf);
        // Write-through to the shared buffer so cross-thread readers (DataInput
        // *and* domain-memory) observe this write. `set_length` above already
        // grew the shared length when needed.
        if let Some(s) = &self.shared {
            s.write(offset, buf);
        }
        Ok(())
    }

    /// Write bytes at any offset in the ByteArray
    /// Will return an error if the new buffer does not fit the ByteArray
    pub fn write_at_nongrowing(&mut self, buf: &[u8], offset: usize) -> Result<(), ByteArrayError> {
        // Grow the mirror to the shared logical length so a write that fits the
        // (shared) array doesn't spuriously fail against a stale-short mirror.
        self.ensure_mirror_len();
        self.bytes
            .get_mut(offset..)
            .and_then(|bytes| bytes.get_mut(..buf.len()))
            .ok_or(ByteArrayError::IndexOutOfBounds)?
            .copy_from_slice(buf);
        // Note: shared write-through is intentionally *not* done here. The only
        // callers (`dm_write`, `atomic_compare_and_swap_int_at`) already push to
        // the shared buffer themselves; doing it here too would double-write.
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
        // Sync the mirror from the shared buffer so the memmove source/dest are
        // correct and in-bounds.
        self.ensure_mirror_len();
        if self.bytes.len() < new_len {
            self.bytes.resize(new_len, 0);
        }

        // `ensure_mirror_len` only refreshes the region *beyond* the old mirror
        // length; the source range may already be within the mirror yet stale
        // (another worker thread wrote it to the shared buffer without this
        // arena's mirror seeing it — e.g. FlasCC `getdirentries` filling a dirent
        // that C `memcpy` then moves via `ByteArray.writeBytes(this, ...)`).
        // Refresh the *source* range from shared before the memmove, or it copies
        // stale zeros.
        if let Some(s) = &self.shared {
            if amnt > 0 {
                s.read(start, &mut self.bytes[start..end]);
            }
        }

        self.bytes.copy_within(start..end, offset);
        if let Some(s) = &self.shared {
            s.write(offset, &self.bytes[offset..offset + amnt]);
        }
        Ok(())
    }

    /// Compress the ByteArray into a temporary buffer.
    pub fn compress(&mut self, algorithm: CompressionAlgorithm) -> Vec<u8> {
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
            // The shared buffer is process RAM shared by reference across worker
            // threads. Another thread may have grown it (FlasCC `sbrk` via
            // `atomicCompareAndSwapLength`) without this arena's mirror seeing it,
            // so `new_len` — derived from the stale-short mirror length — can be
            // *below* the true shared length. Never shrink the shared buffer here:
            // that would truncate another thread's live data (e.g. its stack / a
            // queued thunk request). Only grow; keep the mirror at the shared len.
            let shared_len = s.len();
            if new_len < shared_len {
                tracing::warn!(
                    "set_length t{:?}: refusing to shrink shared buffer {shared_len} -> {new_len}",
                    std::thread::current().id()
                );
            }
            let target = new_len.max(shared_len);
            s.set_len(target);
            self.bytes.resize(target, 0);
            self.position.set(self.position().min(target));
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
            // Mirror the resulting length locally.
            self.bytes.resize(shared.len(), 0);
            self.position.set(self.position().min(self.bytes.len()));
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
        self.ensure_mirror_len();

        *self.bytes.get_mut(item).unwrap() = value;
        if let Some(s) = &self.shared {
            s.write(item, &[value]);
        }
    }

    /// Swap all data stored in this bytearray with the passed `Vec<u8>`. This
    /// method sets the bytearray's `position` to 0.
    pub fn swap_storage_with(&mut self, new_data: &mut Vec<u8>) {
        self.position.set(0);
        std::mem::swap(&mut self.bytes, new_data);
    }

    /// Write a single byte at any offset in the bytearray, panicking if out of bounds.
    pub fn set_nongrowing(&mut self, item: usize, value: u8) {
        self.ensure_mirror_len();
        self.bytes[item] = value;
        if let Some(s) = &self.shared {
            s.write(item, &[value]);
        }
    }

    pub fn delete(&mut self, item: usize) {
        if let Some(i) = self.bytes.get_mut(item) {
            *i = 0;
            if let Some(s) = &self.shared {
                s.write(item, &[0]);
            }
        }
    }

    #[inline]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    #[inline]
    pub fn bytes_mut(&mut self) -> &mut [u8] {
        &mut self.bytes
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

    /// Grow the local mirror to cover the shared buffer's current length,
    /// seeding the new region from the shared buffer. No-op unless another thread
    /// grew the shared buffer past this arena's mirror.
    #[inline]
    fn ensure_mirror_len(&mut self) {
        if let Some(s) = &self.shared {
            let slen = s.len();
            let old = self.bytes.len();
            if old < slen {
                self.bytes.resize(slen, 0);
                s.read(old, &mut self.bytes[old..slen]);
            }
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
