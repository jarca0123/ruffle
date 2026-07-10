//! The typed IR vocabulary shared by the frontend (analysis) and the backend
//! (WASM emit).
//!
//! Step 1 is deliberately small: a value is one of four machine types, each of
//! which lowers directly to a WASM value type and is register-allocated by the
//! host engine — the whole point of the design (see AVM2_JIT_DESIGN.md §1/§4).
//! There is no boxed, memory-resident frame: unboxed values live in WASM locals.

use wasm_encoder::ValType;

/// The machine type of an SSA/operand value.
///
/// `Boxed` is the escape hatch for a value whose AVM2 *tag* is only known at run
/// time — in step 1 that is exactly the result of a non-`*I` `Add`/`Subtract`/
/// `Multiply` on two `int`s, which the interpreter yields as `Integer` unless it
/// overflows `i32`, in which case it becomes `Number` (value.rs). Such a value is
/// materialized as its exact NaN-box bits in an `i64` local; step 1 only lets it
/// flow to a return or a same-typed local store (anything else declines), so we
/// never have to *unbox* one.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Ty {
    /// AS3 `int`/`uint`: an `i32` WASM local.
    Int,
    /// AS3 `Number`: an `f64` WASM local.
    Num,
    /// A `Number` loaded raw from domainMemory (`lf32`/`lf64`) whose exact bits
    /// (incl. a colliding-NaN payload) must be preserved — the interpreter uses
    /// `number_lossless`, not the NaN-canonicalizing `Value::Number` (§4/§10
    /// LESSON). It lives in an `f64` WASM local (bit-preserving), so it flows
    /// freely to stores/arithmetic/comparisons; **boxing** it (return/slot) would
    /// canonicalize a colliding NaN, so those sites decline (a lossless-box helper
    /// is a later step). Arithmetic on it yields an ordinary canonical `Num`.
    NumRaw,
    /// AS3 `Boolean`: an `i32` WASM local holding 0/1.
    Bool,
    /// A fully NaN-boxed `Value` in an `i64` local (see the type-level note).
    Boxed,
    /// A value with no machine representation that the numeric subset never
    /// inspects: `this`/an untyped local read only to feed the scope prologue
    /// (`getlocal0; pushscope`). It is never materialized into a WASM local and
    /// may only flow to `PushScope` or `Pop`; anything else declines.
    Opaque,
}

impl Ty {
    /// The WASM value type this lowers to. `Opaque` has none (it is never
    /// emitted); the arbitrary choice here is never used.
    pub fn val_type(self) -> ValType {
        match self {
            Ty::Int | Ty::Bool => ValType::I32,
            Ty::Num | Ty::NumRaw => ValType::F64,
            Ty::Boxed | Ty::Opaque => ValType::I64,
        }
    }
}

/// The generic (non-`*I`) binary arithmetic ops.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum BinArith {
    Add,
    Sub,
    Mul,
    Div,
}

/// How a generic binary arithmetic op lowers, chosen from its operand types.
/// Centralized here so the frontend's typing and the backend's codegen can never
/// disagree about which path a given op takes (a disagreement would be a silent
/// miscompile — AVM2_JIT_DESIGN.md §12).
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum ArithLower {
    /// Promote both operands to `f64`, do the `f64` op; result is `Num`.
    F64,
    /// Both operands are `int`: compute in `i64`, then pick `Integer` (fits `i32`)
    /// or `Number` (overflow) exactly as the interpreter does; result is `Boxed`.
    IntOverflow,
}

/// Decide how a generic `Add`/`Subtract`/`Multiply`/`Divide` lowers for the given
/// operand types, or `None` to decline the whole method.
///
/// Mirrors the interpreter (value.rs `add`, activation.rs `op_subtract`/
/// `op_multiply`/`op_divide`): `(int,int)` for `+`/`-`/`*` is `i32`-with-overflow
/// -to-`Number`; anything involving a `Number` coerces to `f64`; `/` is always
/// `Number`. `Bool`/`Boxed` operands are out of step-1 scope → decline.
pub fn arith_lower(op: BinArith, a: Ty, b: Ty) -> Option<(ArithLower, Ty)> {
    // A raw-loaded Number is numeric; arithmetic on it produces an ordinary
    // canonical `Num` (the op launders any colliding-NaN payload).
    let numeric = |t: Ty| matches!(t, Ty::Int | Ty::Num | Ty::NumRaw);
    if !numeric(a) || !numeric(b) {
        return None;
    }
    match op {
        // Division always coerces both sides to Number.
        BinArith::Div => Some((ArithLower::F64, Ty::Num)),
        BinArith::Add | BinArith::Sub | BinArith::Mul => {
            if a == Ty::Int && b == Ty::Int {
                Some((ArithLower::IntOverflow, Ty::Boxed))
            } else {
                // At least one Number ⇒ f64 arithmetic ⇒ Number.
                Some((ArithLower::F64, Ty::Num))
            }
        }
    }
}

/// How the emitted `ReturnValue` must coerce its operand before boxing, derived
/// from the method's declared return type. `Any` = no declared type (`*`): box
/// the operand as whatever it already is.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum ReturnCoerce {
    Any,
    Int,
    Number,
    Boolean,
}
