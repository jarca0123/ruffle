//! The JIT's runtime helper surface — the narrow, deliberately Activation-light
//! ABI the emitted WASM calls back into (design §8).
//!
//! Step 3b introduces the first helper: `get_slot`. It is **pure** — given an
//! object's `Value` bits and a slot index it reads the slot and returns the
//! result's bits, touching nothing that needs a full `Activation` (`get_slot`
//! only borrows the object's own storage). This is the whole point of §8's
//! up-front design: a JITed leaf method depends on the smallest possible surface.

use std::cell::{Cell, RefCell, UnsafeCell};

use gc_arena::Mutation;
use ruffle_core::avm2::{Activation, Error, FunctionArgs, TObject, Value};

use crate::value_bits::{from_bits, to_bits, UNDEFINED_BITS};

/// Read slot `slot_id` of the object encoded in `receiver_bits`, returning the
/// result's `Value` bits. Imported into the emitted module as `env.gs`.
///
/// SAFETY / GC: `receiver_bits` is a `Value` the JIT produced from a
/// verifier-proven `Object` (the receiver of a resolved slot read). The object is
/// rooted by the live method frame, and gc_arena does not collect mid-method
/// (design §9), so the pointer stays valid for this synchronous call. The result
/// is reachable from that live object, so it too is alive until the caller uses
/// it. A non-object receiver can't occur for a proven slot read; we return
/// `undefined` defensively rather than fabricate a pointer.
pub(crate) fn get_slot(receiver_bits: u64, slot_id: u32) -> u64 {
    let receiver: Value<'_> = unsafe { from_bits(receiver_bits) };
    match receiver.as_object() {
        Some(obj) => to_bits(obj.get_slot(slot_id as usize)),
        None => UNDEFINED_BITS,
    }
}

/// ECMAScript `ToInt32` — mirrors `ruffle_core`'s `f64_to_wrapping_i32`
/// (`ecma_conversions.rs`): non-finite → 0, else `trunc(n) mod 2^32` as `i32`.
/// Imported as `env.ti` for `CoerceI`/int-return/`SetSlotCoerceI` on a `Number`
/// (the `|n| >= 2^63` wrap makes a correct inline sequence impractical).
pub(crate) fn to_int32(n: f64) -> i32 {
    if !n.is_finite() {
        0
    } else {
        n.trunc().rem_euclid(4294967296.0) as u32 as i32
    }
}

thread_local! {
    /// Type-erased pointer to the current GC `Mutation`, valid only for the
    /// duration of a [`with_mc`] scope. Slot WRITES need the write barrier, so —
    /// unlike the pure `get_slot` — they read the mutation from here. This is the
    /// deliberately tiny context bridge (§8): a single pointer, set around the
    /// synchronous call, never a full `Activation`.
    static MC: Cell<*const ()> = const { Cell::new(std::ptr::null()) };
}

/// Make `mc` available to slot-write helpers for the duration of `f` (the JIT
/// call). Restores the previous value so nested/re-entrant calls are safe.
pub(crate) fn with_mc<R>(mc: &Mutation<'_>, f: impl FnOnce() -> R) -> R {
    let prev = MC.with(|c| c.replace(mc as *const Mutation<'_> as *const ()));
    let r = f();
    MC.with(|c| c.set(prev));
    r
}

// ---- JIT→JIT calls (step 4): the full-`Activation` helper surface ----
//
// A call can run arbitrary AS3 (getters, the callee body) and can THROW, so —
// unlike the pure slot/int32 helpers — the call helper needs the live
// `Activation` and an error channel. Both are thread-locals set around the
// synchronous JIT run: `with_activation` stashes a type-erased `*mut Activation`,
// and `PENDING_ERROR` carries a thrown error out. The emitted code checks
// `pending_error` after each call and bails; `try_run` then propagates it.

thread_local! {
    static ACT: Cell<*mut ()> = const { Cell::new(std::ptr::null_mut()) };
    /// Args being marshalled for the pending call (bits), pushed by `push_call_arg`
    /// and drained by `call_method`.
    static CALL_ARGS: UnsafeCell<Vec<u64>> = const { UnsafeCell::new(Vec::new()) };
    /// A thrown error awaiting propagation by `try_run` (lifetime-erased).
    static PENDING_ERROR: RefCell<Option<Error<'static>>> = const { RefCell::new(None) };
}

/// Make `act` available to call helpers for the duration of `f` (the JIT call).
pub(crate) fn with_activation<R>(act: &mut Activation<'_, '_>, f: impl FnOnce() -> R) -> R {
    let prev = ACT.with(|c| c.replace(act as *mut Activation<'_, '_> as *mut ()));
    let r = f();
    ACT.with(|c| c.set(prev));
    r
}

/// Take any error a call helper stashed (cleared). `try_run` calls this after the
/// run to turn a JIT-executed throw back into a `Result::Err`.
pub(crate) fn take_pending_error<'gc>() -> Option<Error<'gc>> {
    // SAFETY: reverses the erasure in `stash`; the arena lock is held by the
    // caller, and no `'gc` reference outlives the surrounding `try_run` frame.
    PENDING_ERROR.with(|slot| {
        slot.borrow_mut()
            .take()
            .map(|e| unsafe { std::mem::transmute::<Error<'static>, Error<'gc>>(e) })
    })
}

fn stash_pending_error(e: Error<'_>) {
    // SAFETY: erased for storage; `take_pending_error` restores a fresh `'gc` and
    // it never escapes the JIT run.
    let erased = unsafe { std::mem::transmute::<Error<'_>, Error<'static>>(e) };
    PENDING_ERROR.with(|slot| *slot.borrow_mut() = Some(erased));
}

/// Push one argument (bits) for the pending call. Imported as `env.ca`.
pub(crate) fn push_call_arg(bits: u64) {
    // SAFETY: single-threaded, no re-entry between push and the matching drain in
    // `call_method` (args are drained before the callee runs).
    CALL_ARGS.with(|a| unsafe { (*a.get()).push(bits) });
}

/// `1` if a call has thrown (checked after each call by the emitted code).
/// Imported as `env.pe`.
pub(crate) fn pending_error() -> i32 {
    PENDING_ERROR.with(|slot| slot.borrow().is_some()) as i32
}

/// Invoke method `disp_id` on `receiver_bits` with the last `argc` pushed args,
/// mirroring `op_call_method` (null-check the receiver, vtable-dispatch). Returns
/// the result's bits, or stashes a thrown error and returns `undefined`. Imported
/// as `env.cm`.
pub(crate) fn call_method(receiver_bits: u64, disp_id: u32, argc: u32) -> u64 {
    // SAFETY: `ACT` holds a valid `Activation` for this synchronous call (set by
    // `with_activation`); the erased lifetimes are re-inferred here.
    let act = unsafe { &mut *(ACT.with(|c| c.get()) as *mut Activation<'_, '_>) };

    let arg_bits: Vec<u64> = CALL_ARGS.with(|a| {
        let v = unsafe { &mut *a.get() };
        let at = v.len() - argc as usize;
        v.split_off(at)
    });
    let args: Vec<Value<'_>> = arg_bits.iter().map(|&b| unsafe { from_bits(b) }).collect();
    let receiver: Value<'_> = unsafe { from_bits(receiver_bits) };

    let result = receiver.null_check(act, None).and_then(|r| {
        r.call_method_with_args(disp_id as usize, FunctionArgs::from_slice(&args), act)
    });
    match result {
        Ok(v) => to_bits(v),
        Err(e) => {
            stash_pending_error(e);
            UNDEFINED_BITS
        }
    }
}

/// Write `value_bits` into slot `slot_id` of the object in `receiver_bits`, using
/// the GC write barrier. Imported into the emitted module as `env.ss`.
///
/// SAFETY / GC: as for [`get_slot`], the receiver and value are live for this
/// synchronous call (rooted, no mid-method collection). `MC` holds a valid
/// `Mutation` pointer because this only runs inside [`with_mc`].
pub(crate) fn set_slot(receiver_bits: u64, value_bits: u64, slot_id: u32) {
    let mc_ptr = MC.with(|c| c.get());
    debug_assert!(!mc_ptr.is_null(), "set_slot outside with_mc");
    let mc: &Mutation<'_> = unsafe { &*(mc_ptr as *const Mutation<'_>) };
    let receiver: Value<'_> = unsafe { from_bits(receiver_bits) };
    if let Some(obj) = receiver.as_object() {
        obj.set_slot_no_coerce(slot_id as usize, unsafe { from_bits(value_bits) }, mc);
    }
}
