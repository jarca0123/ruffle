//! Direct-exec tier: tiny straight-line boxed methods run as a Rust match-loop
//! over their [`JitOp`]s — no wasmi (native) / browser instance at all.
//!
//! Motivation: a FlasCC accessor like `LI8` is `getlocal1; li8; returnvalue` —
//! three ops — yet the wasmi path pays a pool lookup, a register memcpy into the
//! instance memory, `TypedFunc::call` setup, wasmi opcode dispatch, and a
//! host-function crossing for the dm helper *per call*. Exception-heavy content
//! (the avmplus mops range tests: ~1.7M such calls) made that the dominant cost.
//! Here the same semantics is a handful of direct Rust calls into the very same
//! [`helpers`] the wasm imports bind, so per-call cost collapses to a small
//! match-loop.
//!
//! Semantics: every op is executed via its **helper** implementation (the
//! full-fidelity path the emitted code falls back to); the wasm-side inline fast
//! paths (int arithmetic, ToInt32 middle path, …) are pure-speed refinements of
//! those same helpers, so results are identical. After every helper-backed op the
//! pending error is checked, mirroring the emitted `perr` bail — on a pending
//! throw the run returns `undefined` bits and `try_run`'s existing
//! `take_pending_error` propagates the error.
//!
//! Eligibility ([`eligible`]) is decided once at compile time and is
//! deliberately narrow: at most [`MAX_OPS`] ops, straight-line (no branches, no
//! calls, no exception machinery, no local writes), every op in the supported
//! set, statically balanced operand stack, ending in a return op. Anything else
//! keeps the ordinary wasm path.

use crate::helpers;
use crate::lower::{
    JitOp, DECREMENT_I, H2_ADD_I, H2_MULTIPLY_I, H2_SUBTRACT_I, INCREMENT_I, TO_BOOLEAN,
    UNDEFINED_BITS, VALUE_BOOL_MARK,
};
use crate::lower::{COERCE_I, COERCE_RETURN, COERCE_S, COERCE_U, GET_PUSH_STRING};

/// Max op count for the direct tier. Covers the FlasCC accessor shapes (`LI8` =
/// 3 ops, `ADD_I` = 4, `SI8` = 4) with headroom for small getters; anything
/// larger is likely to have enough body to amortize the wasm path.
pub(crate) const MAX_OPS: usize = 10;

/// The arity-1 helper indices `translate` emits as `CallHelper(i)`:
/// increment/decrement/negate/bitnot/not, the dm loads, `coerce_u`/`coerce_i`,
/// the sign-extends, and the dm float loads. (The other `HELPERS` entries are
/// reached via dedicated ops — scope, strings, returns, exceptions — and are NOT
/// valid as a generic value-in/value-out `CallHelper`.)
fn call_helper_supported(i: u32) -> bool {
    matches!(i, 0..=4 | 8..=15 | 26 | 27)
}

/// The operand-stack effect `(pops, pushes)` of a supported op, or `None` if the
/// op is outside the direct tier. Keep in EXACT sync with [`run`]'s match.
fn effect(op: &JitOp) -> Option<(usize, usize)> {
    Some(match op {
        JitOp::Nop => (0, 0),
        JitOp::Pop => (1, 0),
        JitOp::DupValue => (1, 2),
        JitOp::SwapValue => (2, 2),
        JitOp::GetLocalValue(_)
        | JitOp::PushConst(_)
        | JitOp::PushIntValue(_)
        | JitOp::PushString(_)
        | JitOp::GetScriptGlobals(_) => (0, 1),
        JitOp::CallHelper(i) if call_helper_supported(*i) => (1, 1),
        JitOp::IncrementIBoxed
        | JitOp::DecrementIBoxed
        | JitOp::CoerceInt(_)
        | JitOp::CoerceBool
        | JitOp::CoerceString
        | JitOp::Coerce(_)
        | JitOp::GetProperty(_)
        | JitOp::GetSlot(_, _) => (1, 1),
        JitOp::CallHelper2(_)
        | JitOp::CmpNum(_)
        | JitOp::BitOpInt(_)
        | JitOp::ArithInt(_)
        | JitOp::ArithNum(_)
        | JitOp::AddIBoxed
        | JitOp::SubtractIBoxed
        | JitOp::MultiplyIBoxed
        | JitOp::GetPropertyFast(_, _) => (2, 1),
        JitOp::CallHelper3(_, _) => (2, 0),
        // Calls: `argc` spilled args + the receiver; the result is pushed for the
        // value form, dropped for the void form. Re-entrancy is fine — the callee
        // may re-enter the JIT (direct or wasm), each run's operand stack is local.
        JitOp::CallMethod(_, argc, push) | JitOp::CallProperty(_, argc, push) => {
            (*argc as usize + 1, *push as usize)
        }
        _ => return None,
    })
}

/// Whether `ops` qualifies for the direct tier: short, straight-line, every op
/// supported, operand stack statically balanced, and terminated by a return.
/// Decided once at compile time (cached in `Compiled::direct_ops`).
pub(crate) fn eligible(ops: &[JitOp]) -> bool {
    if ops.is_empty() || ops.len() > MAX_OPS {
        return false;
    }
    let (last, body) = ops.split_last().expect("non-empty");
    let mut depth = 0usize;
    for op in body {
        let Some((pops, pushes)) = effect(op) else {
            return false;
        };
        if depth < pops {
            return false;
        }
        depth = depth - pops + pushes;
    }
    match last {
        JitOp::ReturnVoidBoxed(_) => true,
        JitOp::ReturnValueBoxed | JitOp::ReturnValueCoerced => depth >= 1,
        _ => false,
    }
}

/// Executes an [`eligible`] method directly. MUST run inside
/// `helpers::with_run_ctx` (exactly like `runner::run` — the helpers reach the
/// activation and side tables through it). `regs` are the frame registers
/// (`Value` bits). Returns the result `Value`'s bits; a pending error is left
/// for `try_run`'s `take_pending_error`, mirroring the wasm path.
pub(crate) fn run(ops: &[JitOp], regs: &[u64]) -> Option<u64> {
    // Eligibility proved the max depth ≤ MAX_OPS statically.
    let mut stack: [i64; MAX_OPS] = [0; MAX_OPS];
    let mut sp = 0usize;
    macro_rules! push {
        ($v:expr) => {{
            stack[sp] = $v;
            sp += 1;
        }};
    }
    macro_rules! pop {
        () => {{
            sp -= 1;
            stack[sp]
        }};
    }
    // The emitted code bails after every throwing op (`perr`); checking after
    // every helper-backed op is the same observable behavior (only throwing
    // helpers can set it) at the cost of one thread-local flag read.
    macro_rules! perr_bail {
        () => {
            if helpers::pending_error() != 0 {
                return Some(UNDEFINED_BITS);
            }
        };
    }

    for op in ops {
        match *op {
            JitOp::Nop => {}
            JitOp::Pop => {
                let _ = pop!();
            }
            JitOp::DupValue => {
                let v = pop!();
                push!(v);
                push!(v);
            }
            JitOp::SwapValue => {
                let b = pop!();
                let a = pop!();
                push!(b);
                push!(a);
            }
            JitOp::GetLocalValue(i) => push!(*regs.get(i as usize)? as i64),
            JitOp::PushConst(bits) => push!(bits as i64),
            JitOp::PushIntValue(v) => {
                push!((crate::lower::VALUE_INT_MARK | (v as u32 as u64)) as i64)
            }
            JitOp::PushString(k) => {
                push!(helpers::HELPERS[GET_PUSH_STRING as usize](k as i64))
            }
            JitOp::GetScriptGlobals(k) => {
                push!(helpers::HELPERS[crate::lower::GET_SCRIPT_GLOBALS as usize](k as i64))
            }
            JitOp::CallHelper(i) => {
                let a = pop!();
                push!(helpers::HELPERS[i as usize](a));
                perr_bail!();
            }
            JitOp::CallHelper2(i)
            | JitOp::CmpNum(i)
            | JitOp::BitOpInt(i)
            | JitOp::ArithInt(i)
            | JitOp::ArithNum(i) => {
                let b = pop!();
                let a = pop!();
                push!(helpers::HELPERS2[i as usize](a, b));
                perr_bail!();
            }
            JitOp::AddIBoxed | JitOp::SubtractIBoxed | JitOp::MultiplyIBoxed => {
                let i = match op {
                    JitOp::AddIBoxed => H2_ADD_I,
                    JitOp::SubtractIBoxed => H2_SUBTRACT_I,
                    _ => H2_MULTIPLY_I,
                };
                let b = pop!();
                let a = pop!();
                push!(helpers::HELPERS2[i as usize](a, b));
            }
            JitOp::IncrementIBoxed => {
                let a = pop!();
                push!(helpers::HELPERS[INCREMENT_I as usize](a));
            }
            JitOp::DecrementIBoxed => {
                let a = pop!();
                push!(helpers::HELPERS[DECREMENT_I as usize](a));
            }
            JitOp::CoerceInt(signed) => {
                let a = pop!();
                let h = if signed { COERCE_I } else { COERCE_U };
                push!(helpers::HELPERS[h as usize](a));
                perr_bail!();
            }
            JitOp::CoerceBool => {
                let a = pop!();
                // `to_boolean` returns raw 0/1; box it as a Boolean `Value`,
                // matching the emitted `CoerceBool`.
                let bit = helpers::HELPERS[TO_BOOLEAN as usize](a) as u64 & 1;
                push!((VALUE_BOOL_MARK | bit) as i64);
            }
            JitOp::CoerceString => {
                let a = pop!();
                push!(helpers::HELPERS[COERCE_S as usize](a));
                perr_bail!();
            }
            JitOp::Coerce(k) => {
                let a = pop!();
                push!(helpers::coerce(a, k as i64));
                perr_bail!();
            }
            JitOp::CallHelper3(kind, imm) => {
                // Stack [first, second] + immediate; void (result dropped) —
                // matching the emitted arm (setslot family / `dm_store`).
                let second = pop!();
                let first = pop!();
                let _ = helpers::HELPERS3[kind as usize](first, second, imm as i64);
                perr_bail!();
            }
            JitOp::GetProperty(k) => {
                let recv = pop!();
                push!(helpers::get_property(recv, k as i64));
                perr_bail!();
            }
            JitOp::GetPropertyFast(k, _) => {
                let name = pop!();
                let recv = pop!();
                push!(helpers::get_property_fast(recv, name, k as i64));
                perr_bail!();
            }
            JitOp::GetSlot(slot_id, _) => {
                let recv = pop!();
                push!(helpers::get_slot(recv, slot_id as i64));
                perr_bail!();
            }
            JitOp::CallMethod(id, argc, push) | JitOp::CallProperty(id, argc, push) => {
                // Spill args top-first (matching the emitted `pca` order —
                // `drain_call_args` reverses), then the receiver is next on the
                // stack. Mirrors the emitted call arm incl. the perr bail.
                for _ in 0..argc {
                    helpers::push_call_arg(pop!());
                }
                let receiver = pop!();
                let r = if matches!(op, JitOp::CallMethod(..)) {
                    helpers::call_method(receiver, id as i64, argc as i64)
                } else {
                    helpers::call_property(receiver, id as i64, argc as i64)
                };
                perr_bail!();
                if push {
                    push!(r);
                }
            }
            JitOp::ReturnValueBoxed => return Some(pop!() as u64),
            JitOp::ReturnValueCoerced => {
                let v = pop!();
                let r = helpers::HELPERS[COERCE_RETURN as usize](v);
                // A failing `#1034` coercion stashes a pending error and returns
                // `undefined` — `try_run` propagates it either way.
                return Some(r as u64);
            }
            JitOp::ReturnVoidBoxed(bits) => return Some(bits),
            // Eligibility admits no other op.
            _ => unreachable!("direct tier ran an unsupported op: {op:?}"),
        }
    }
    unreachable!("direct-eligible method did not end in a return")
}
