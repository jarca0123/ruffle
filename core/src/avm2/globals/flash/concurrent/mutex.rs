//! `flash.concurrent.Mutex` native methods

use crate::avm2::Error;
use crate::avm2::activation::Activation;
use crate::avm2::object::MutexObject;
use crate::avm2::value::Value;

// The instance allocator is resolved by the playerglobal codegen at this module
// path (`<class package>::<class>_allocator`).
pub use crate::avm2::object::mutex_allocator;

fn this_mutex<'gc>(this: Value<'gc>) -> MutexObject<'gc> {
    this.as_object()
        .and_then(|o| o.as_mutex_object())
        .expect("Mutex native called on non-Mutex object")
}

/// Implements `Mutex.isSupported`
pub fn get_is_supported<'gc>(
    activation: &mut Activation<'_, 'gc>,
    _this: Value<'gc>,
    _args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    Ok(activation.context.worker_host.is_supported().into())
}

/// Implements `Mutex.lock`
pub fn lock<'gc>(
    _activation: &mut Activation<'_, 'gc>,
    this: Value<'gc>,
    _args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    // `lock` returns `false` if the worker was interrupted (terminate requested)
    // while parked — the mutex is not held. Raise an uncatchable error so the
    // AS3/FlasCC stack unwinds out to the worker's run loop (avmplus's interrupt).
    if !this_mutex(this).shared_mutex().lock() {
        return Err(Error::rust_error("worker terminated".into()));
    }
    Ok(Value::Undefined)
}

/// Implements `Mutex.tryLock`
pub fn try_lock<'gc>(
    _activation: &mut Activation<'_, 'gc>,
    this: Value<'gc>,
    _args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    Ok(this_mutex(this).shared_mutex().try_lock().into())
}

/// Implements `Mutex.unlock`
pub fn unlock<'gc>(
    _activation: &mut Activation<'_, 'gc>,
    this: Value<'gc>,
    _args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    this_mutex(this).shared_mutex().unlock();
    Ok(Value::Undefined)
}
