use crate::avm2::Error;
use crate::avm2::activation::Activation;
use crate::avm2::value::Value;

pub use crate::avm2::object::dictionary_allocator;

/// Implements `Dictionary.setWeakKeys`, called from the AS3 constructor when
/// `weakKeys` is `true`.
pub fn set_weak_keys<'gc>(
    _activation: &mut Activation<'_, 'gc>,
    this: Value<'gc>,
    _args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    if let Some(dictionary) = this.as_object().and_then(|o| o.as_dictionary_object()) {
        dictionary.set_weak_keys();
    }

    Ok(Value::Undefined)
}
