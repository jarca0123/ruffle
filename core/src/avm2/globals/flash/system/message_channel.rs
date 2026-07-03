//! `flash.system.MessageChannel` native methods

use crate::avm2::Error;
use crate::avm2::activation::Activation;
use crate::avm2::globals::flash::system::worker::{shared_to_value, value_to_shared};
use crate::avm2::object::MessageChannelObject;
use crate::avm2::parameters::ParametersExt;
use crate::avm2::value::Value;

fn this_channel<'gc>(this: Value<'gc>) -> MessageChannelObject<'gc> {
    this.as_object()
        .and_then(|o| o.as_message_channel_object())
        .expect("MessageChannel native called on non-MessageChannel object")
}

/// Implements `MessageChannel.send`
pub fn send<'gc>(
    _activation: &mut Activation<'_, 'gc>,
    this: Value<'gc>,
    args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    let value = args.get_value(0);
    if let Some(shared) = value_to_shared(value) {
        this_channel(this).channel().send(shared);
    }
    Ok(Value::Undefined)
}

/// Implements `MessageChannel.receive`
pub fn receive<'gc>(
    activation: &mut Activation<'_, 'gc>,
    this: Value<'gc>,
    _args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    // `blockUntilReceived` is ignored: blocking the receiver's thread here would
    // deadlock the cooperative fallback; callers poll `messageAvailable` instead.
    Ok(match this_channel(this).channel().receive() {
        Some(shared) => shared_to_value(activation, shared),
        None => Value::Null,
    })
}

/// Implements `MessageChannel.messageAvailable`
pub fn get_message_available<'gc>(
    _activation: &mut Activation<'_, 'gc>,
    this: Value<'gc>,
    _args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    Ok(this_channel(this).channel().message_available().into())
}
