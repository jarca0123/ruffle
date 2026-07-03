//! `flash.concurrent.Condition` native methods

use crate::avm2::Error;
use crate::avm2::activation::Activation;
use crate::avm2::object::ConditionObject;
use crate::avm2::parameters::ParametersExt;
use crate::avm2::value::Value;

// The instance allocator is resolved by the playerglobal codegen at this module
// path (`<class package>::<class>_allocator`).
pub use crate::avm2::object::condition_allocator;

fn this_condition<'gc>(this: Value<'gc>) -> ConditionObject<'gc> {
    this.as_object()
        .and_then(|o| o.as_condition_object())
        .expect("Condition native called on non-Condition object")
}

/// Implements `Condition.isSupported`
pub fn get_is_supported<'gc>(
    activation: &mut Activation<'_, 'gc>,
    _this: Value<'gc>,
    _args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    Ok(activation.context.worker_host.is_supported().into())
}

/// Constructor helper: bind this condition to its `Mutex`.
pub fn init<'gc>(
    activation: &mut Activation<'_, 'gc>,
    this: Value<'gc>,
    args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    let mutex = args
        .get_object(activation, 0, "mutex")?
        .as_mutex_object()
        .expect("Condition constructed with a non-Mutex")
        .shared_mutex();
    this_condition(this).bind(mutex);
    Ok(Value::Undefined)
}

/// Implements `Condition.wait`
pub fn wait<'gc>(
    activation: &mut Activation<'_, 'gc>,
    this: Value<'gc>,
    args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    let timeout = args.get_f64(0);
    // Flash uses a huge sentinel for "no timeout"; treat anything negative or
    // ludicrously large as an indefinite wait.
    let timeout_ms = if timeout < 0.0 || timeout >= u32::MAX as f64 {
        None
    } else {
        Some(timeout as u64)
    };

    match this_condition(this).shared_condition() {
        Some(cond) => Ok(cond.wait(timeout_ms).into()),
        None => Ok(false.into()),
    }
}

/// Implements `Condition.notify`
pub fn notify<'gc>(
    _activation: &mut Activation<'_, 'gc>,
    this: Value<'gc>,
    _args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    if let Some(cond) = this_condition(this).shared_condition() {
        cond.notify();
    }
    Ok(Value::Undefined)
}

/// Implements `Condition.notifyAll`
pub fn notify_all<'gc>(
    _activation: &mut Activation<'_, 'gc>,
    this: Value<'gc>,
    _args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    if let Some(cond) = this_condition(this).shared_condition() {
        cond.notify_all();
    }
    Ok(Value::Undefined)
}
