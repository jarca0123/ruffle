//! `avm2.intrinsics.memory` — CrossBridge/Alchemy domain-memory intrinsics.
//!
//! `si*`/`li*`/`sf*`/`lf*`/`sxi*` compile straight to AVM2 memory opcodes, so
//! they never need a native function. There is no compare-and-swap opcode,
//! however, so CrossBridge emits `casi32` as a real method call that the player
//! implements natively: an atomic CAS on the current
//! `ApplicationDomain.domainMemory`. FlasCC's `avm2_cmpSwapUns` uses it for
//! every thread-arbiter lock, so a threaded (Worker) build hits it immediately.

use crate::avm2::error::make_error_1506;
use crate::avm2::parameters::ParametersExt;
use crate::avm2::{Activation, Error, Value};

/// Implements `avm2.intrinsics.memory.casi32`.
///
/// `casi32(address, expectedValue, newValue)` atomically compares the 32-bit
/// little-endian integer at `address` in domain memory with `expectedValue` and,
/// if equal, stores `newValue`. Returns the value that was at `address` before
/// the call (whether or not the swap happened).
pub fn casi32<'gc>(
    activation: &mut Activation<'_, 'gc>,
    _this: Value<'gc>,
    args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    let address = args.get_i32(0) as usize;
    let expected = args.get_i32(1);
    let new = args.get_i32(2);

    // `casi32` operates on the *caller's* domain memory (i.e.
    // `ApplicationDomain.currentDomain.domainMemory`), not this native function's
    // own (builtin) domain — mirroring how the `si*`/`li*` opcodes resolve domain
    // memory from the executing method's scope.
    let domain = activation
        .caller_domain()
        .unwrap_or_else(|| activation.domain());
    let old = domain.domain_memory().storage_mut().dm_cas32(address, expected, new);

    match old {
        Some(old) => Ok(old.into()),
        // Out-of-bounds domain-memory access raises the same error as `li*`/`si*`.
        None => Err(make_error_1506(activation)),
    }
}

/// Implements `avm2.intrinsics.memory.mfence`.
///
/// A full memory fence (CrossBridge emits this for GCC's `__sync_synchronize`).
/// The cross-thread shared buffers are already `Mutex`-guarded, so this only has
/// to enforce ordering for the calling thread.
pub fn mfence<'gc>(
    _activation: &mut Activation<'_, 'gc>,
    _this: Value<'gc>,
    _args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
    Ok(Value::Undefined)
}
