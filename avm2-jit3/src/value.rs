//! `Value` ↔ raw-bits conversion.
//!
//! A `ruffle_core::avm2::Value<'gc>` is a plain NaN-boxed `u64`
//! (`{ bits: u64, _marker: PhantomData<Gc> }`, 8 bytes), so these transmutes are
//! sound. The JIT passes Values as raw `i64`/`u64` across the WASM boundary with no
//! translation. Within a frame the GC neither moves nor frees (collection runs only
//! between player ticks — see `AVM2_JIT_REDESIGN.md` §2), so an object Value's raw
//! pointer bits stay valid for the whole call.

use ruffle_core::avm2::Value;

/// The raw NaN-boxed bits of a `Value`.
#[inline]
pub fn to_bits(v: Value<'_>) -> u64 {
    // SAFETY: `Value<'gc>` is `#[repr(Rust)]` over a single `u64` (+ a ZST
    // `PhantomData`), so it has the same size/layout as `u64`.
    unsafe { core::mem::transmute::<Value<'_>, u64>(v) }
}

/// Reconstructs a `Value` from raw bits.
///
/// # Safety
/// `bits` must be a valid `Value` bit pattern produced this frame (an object
/// pointer it encodes must still be live — guaranteed for bits that round-tripped
/// through the JIT within one GC-quiescent frame).
#[inline]
pub unsafe fn from_bits<'gc>(bits: u64) -> Value<'gc> {
    unsafe { core::mem::transmute::<u64, Value<'gc>>(bits) }
}

/// Reinterprets a slice of raw `Value` bits (`&[i64]`, as the JIT wrote them into the outgoing
/// argument scratch) as `&[Value]` WITHOUT copying — `Value` and `i64` share layout (see
/// [`from_bits`]). Replaces a per-call `Vec<Value>` collect on the call helpers' hot path.
///
/// # Safety
/// Each element must be a valid `Value` bit pattern live this frame (as [`from_bits`] requires).
#[inline]
pub unsafe fn bits_as_values<'a, 'gc>(arg_bits: &'a [i64]) -> &'a [Value<'gc>] {
    // SAFETY: `Value` is layout-compatible with `i64` (8 bytes; sole non-ZST field is `u64`),
    // so the slice reinterprets element-for-element — the slice form of `from_bits`.
    unsafe { &*(arg_bits as *const [i64] as *const [Value<'gc>]) }
}
