//! The `Value` ⇄ `u64` NaN-box bridge, and the box-construction constants the
//! emitted WASM uses.
//!
//! `ruffle_core::avm2::Value` is a NaN-boxed `u64` (8 bytes) with a private
//! `bits` field, so — exactly as the reference crate does — we bridge to/from the
//! bit pattern by `transmute`. The encoding is mirrored from
//! `core/src/avm2/value.rs`; the `const` assertions below fail to compile if that
//! layout ever drifts, which is the tripwire that keeps this in sync.

use ruffle_core::avm2::Value;

/// Sign + all exponent bits + quiet-NaN bit: the marker that a word is a boxed
/// (non-`Number`) value rather than an inline `f64`. Mirrors `value.rs::BOX_MARK`.
pub const BOX_MARK: u64 = 0xFFF8_0000_0000_0000;
const TAG_SHIFT: u32 = 48;

// Variant tags (must match `value.rs`).
const TAG_UNDEFINED: u64 = 0;
const TAG_BOOL: u64 = 2;
const TAG_INT: u64 = 3;

/// The single canonical NaN bit pattern core stores for `Number(NaN)`. Distinct
/// from every box pattern (sign bit clear). Emitted code must canonicalize any
/// NaN it boxes to this, matching `Value::pack`'s `Number` arm.
pub const CANON_NAN: u64 = 0x7FF8_0000_0000_0000;

/// `Value::Undefined` bits — the result of `ReturnVoid`.
pub const UNDEFINED_BITS: u64 = BOX_MARK | (TAG_UNDEFINED << TAG_SHIFT);

/// High word OR'd with a zero-extended `i32` (as `u32`) to box an `Integer`:
/// `BOX_MARK | TAG_INT<<48`. So `Integer(i)` = `INT_BOX_HI | (i as u32 as u64)`.
pub const INT_BOX_HI: u64 = BOX_MARK | (TAG_INT << TAG_SHIFT);

/// High word OR'd with 0/1 to box a `Bool`: `Bool(b)` = `BOOL_BOX_HI | b`.
pub const BOOL_BOX_HI: u64 = BOX_MARK | (TAG_BOOL << TAG_SHIFT);

// The bridge is only valid while `Value` stays pointer-sized; assert it here so a
// layout change is a compile error, not silent corruption.
const _: () = assert!(size_of::<Value<'static>>() == 8);

/// The raw NaN-box bits of a `Value`.
///
/// SAFETY: `Value` is a `#[derive(Clone, Copy)]` NaN-boxed `u64` plus a
/// zero-sized `PhantomData`; reading its bits is total and side-effect-free.
#[inline]
pub fn to_bits(v: Value<'_>) -> u64 {
    unsafe { std::mem::transmute::<Value<'_>, u64>(v) }
}

/// Reconstruct a `Value` from bits the JIT produced.
///
/// SAFETY: `bits` must be a NaN-box pattern the emitted code produced from live,
/// rooted referents — for the step-1 numeric subset every result is `Integer`,
/// `Number`, `Bool`, or `Undefined`, none of which carry a pointer, so the
/// reconstruction can never fabricate a dangling `Gc`. The `'gc` brand is chosen
/// by the caller, which holds the arena lock.
#[inline]
pub unsafe fn from_bits<'gc>(bits: u64) -> Value<'gc> {
    unsafe { std::mem::transmute::<u64, Value<'gc>>(bits) }
}
