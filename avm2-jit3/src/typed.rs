//! Value-representation inference for jit3's boxed path — the sound, deopt-free
//! type-specialization the design brief (§1/§5) and avmplus/NanoJIT both rely on,
//! ported from the older `avm2-jit` crate's `typed.rs`.
//!
//! jit3 threads NaN-boxed `Value`s (i64) and, for every arithmetic op, runtime-checks
//! the operand tags and boxes/unboxes. When translation can *prove* an operand is a
//! canonical inline `Number`, the emitted arithmetic can skip that runtime guard (and
//! the throwing `binop` helper) entirely — a [`crate::emit::JitOp::BinOpNum`] — and a
//! `coerce`/`convert` to a type the value already has becomes a no-op.
//!
//! "Unsure" always resolves to [`Repr::Boxed`], so a wrong inference can only *miss*
//! an optimization, never enable an unsound one (design brief §5). The stronger facts
//! ([`Repr::Number`]) are what gate the unguarded `f64` fast path, so their producers
//! are deliberately narrow.

use ruffle_core::avm2::Class;

/// The proven representation of a value. "Unsure ⇒ `Boxed`" meet-semilattice.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Repr {
    /// Provably a `Number` box whose bits ARE a **canonical, inline** `f64` — never a
    /// heap-boxed colliding NaN (`TAG_BOXED_DOUBLE`). Safe to `f64.reinterpret_i64`
    /// and to re-box with NaN canonicalization: a sound unbox/box round-trip. This is
    /// the ONLY repr that gates the unguarded `f64` arithmetic (`BinOpNum`). Produced
    /// by `pushdouble` (`Value::pack` canonicalizes NaN — core `value.rs`), by native
    /// `f64` arithmetic results (which canonicalize), and by a real `coerce`/`convert`
    /// to `Number` (which runs `ToNumber`).
    Number,
    /// Provably numeric, but possibly a **heap-boxed double** (`number_lossless`, a
    /// colliding NaN stashed in a `Gc<f64>` by `lf32`/`lf64`) — f64-unboxable, but only
    /// via the box-safe deref, NEVER a blind `f64.reinterpret` (that would read the
    /// pointer as a double). Also the sound seed for a `Number`-typed **param** (a
    /// FlasCC caller can pass a colliding-NaN Number that the identity-coerce fast path
    /// raw-writes into the frame, so we may not assume canonical).
    NumberBoxed,
    /// Provably an **`int` box** — a `TAG_INT` value whose low 32 bits ARE the `i32` payload
    /// (`VALUE_INT_MARK | (i as u32)`). Safe to unbox to a native `i32` (`i32.wrap_i64`) and
    /// re-box (`i64.extend_i32_u | INT_MARK`) — the `i32`-register promotion target (Phase 4).
    /// Produced by `pushint`, `coerce_i`, `bitand/or/xor`, `lshift/rshift`, and the wrapping
    /// `*_i` ops. NOT `uint` (a `uint` ≥ 2^31 boxes as a `Number`) nor overflowing `add`/`sub`/
    /// `mul` (→ `Numeric`).
    Int,
    /// Provably numeric — an `int`/`uint` box OR a canonical `Number` — but the exact form is
    /// unknown: safe to feed `ToNumber` without a `valueOf` (so a `coerce_d` on it is
    /// guard-free), but NOT f64-unboxable as-is, NOT a provable `int` box. The result of
    /// overflowing `int` arithmetic / a `uint` value / a `coerce_u`.
    Numeric,
    /// Provably a **`Bool` box** — `VALUE_BOOL_MARK | (b as u1)` (low bit = 0/1). Unboxes to a
    /// native `i32` (0/1) and re-boxes (`i64.extend_i32_u | BOOL_MARK`) — an i32-register
    /// promotion target. NOT numeric (arithmetic on a `Bool` coerces via `ToNumber`). Produced
    /// by comparisons, `not`, `coerce_b`, `in`, `istypelate`, `hasnext`/`hasnext2`.
    Bool,
    /// Unknown — could be any `Value`.
    Boxed,
}

impl Repr {
    /// Provably numeric — an `int` box, a canonical `Number`, or a possibly-heap-boxed
    /// `Number`. Every numeric repr unboxes to `f64` without a `valueOf` coercion, so
    /// this gates dropping a runtime numeric check.
    pub(crate) fn is_numeric(self) -> bool {
        matches!(self, Repr::Int | Repr::Numeric | Repr::Number | Repr::NumberBoxed)
    }

    /// A provable `int` box — the `i32`-register promotion target (Phase 4).
    pub(crate) fn is_int(self) -> bool {
        matches!(self, Repr::Int)
    }

    /// A provable `Bool` box — an i32-register promotion target.
    pub(crate) fn is_bool(self) -> bool {
        matches!(self, Repr::Bool)
    }

    /// Unboxable to a raw `f64` (a genuine `Number`, inline or heap-boxed). An `int`
    /// box (`Numeric` only) is excluded — it needs `convert_i`, not the `f64` path.
    pub(crate) fn is_f64_unboxable(self) -> bool {
        matches!(self, Repr::Number | Repr::NumberBoxed)
    }

    /// A canonical, inline `f64` — the only repr safe for a blind `f64.reinterpret_i64`
    /// (the unguarded `BinOpNum` path). Excludes `NumberBoxed` (heap pointer bits).
    pub(crate) fn is_canonical_number(self) -> bool {
        matches!(self, Repr::Number)
    }

    /// Lattice meet, for merging a slot/local across control-flow predecessors (used by
    /// the cross-block fixpoint in later phases). `Number` survives only if `Number` on
    /// every path; any numeric mix degrades to the weakest still-sound numeric fact.
    pub(crate) fn meet(self, other: Repr) -> Repr {
        match (self, other) {
            (a, b) if a == b => a,
            (a, b)
                if a.is_f64_unboxable()
                    && b.is_f64_unboxable()
                    && (a == Repr::NumberBoxed || b == Repr::NumberBoxed) =>
            {
                Repr::NumberBoxed
            }
            (a, b) if a.is_numeric() && b.is_numeric() => Repr::Numeric,
            _ => Repr::Boxed,
        }
    }
}

/// The sound repr to seed a **parameter/local** of declared class `class`.
///
/// `try_enter` fills a typed param either by the fast path (`coerces_identically_to`, true for
/// a `Number` only when `is_f64()` — i.e. already a canonical inline `Number`) or by a real
/// `coerce_to_type` (which canonicalizes). So a **present** `Number` param is always a
/// canonical [`Repr::Number`]. The one hole is a *missing* param in an `unchecked` method
/// (→ `undefined`, not a `Number`); `canonical_params` is `false` for those, degrading a
/// `Number` param to the still-sound [`Repr::NumberBoxed`] (never used for the unguarded path,
/// and `undefined` is simply never treated as a `Number` because `NumberBoxed` isn't canonical
/// — a missing param would only *miss* an optimization). `int`/`uint` → [`Repr::Numeric`].
pub(crate) fn param_repr(class: Option<Class<'_>>, canonical_params: bool) -> Repr {
    match class {
        Some(c) if c.is_builtin_number() => {
            if canonical_params {
                Repr::Number
            } else {
                Repr::NumberBoxed
            }
        }
        // A CHECKED `int` param is always an `int` box (`ToInt32` → `i32` → Integer). `uint` is
        // `int`-or-`Number` (≥ 2^31 boxes as `Number`) → `Numeric`. An UNCHECKED method's missing
        // param is `undefined` (not numeric) → `Boxed`.
        Some(c) if c.is_builtin_int() => {
            if canonical_params {
                Repr::Int
            } else {
                Repr::Boxed
            }
        }
        Some(c) if c.is_builtin_uint() => {
            if canonical_params {
                Repr::Numeric
            } else {
                Repr::Boxed
            }
        }
        // A CHECKED `Boolean` param is always a `Bool` box; an unchecked missing one is `undefined`.
        Some(c) if c.is_builtin_boolean() => {
            if canonical_params {
                Repr::Bool
            } else {
                Repr::Boxed
            }
        }
        _ => Repr::Boxed,
    }
}

/// The repr of the value produced by a **real** `coerce`/`convert` to class `class`
/// (the helper actually runs, so a `Number` result is freshly canonicalized).
pub(crate) fn coerce_result_repr(class: Class<'_>) -> Repr {
    if class.is_builtin_number() {
        Repr::Number
    } else if class.is_builtin_int() {
        Repr::Int
    } else if class.is_builtin_uint() {
        Repr::Numeric
    } else if class.is_builtin_boolean() {
        Repr::Bool
    } else {
        Repr::Boxed
    }
}
