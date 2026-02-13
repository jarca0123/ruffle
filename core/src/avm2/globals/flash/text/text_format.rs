use crate::avm2::Error;
use crate::avm2::activation::Activation;
use crate::avm2::error::{make_error_2007, make_error_2008};
use crate::avm2::object::ArrayObject;
use crate::avm2::parameters::ParametersExt;
use crate::avm2::value::Value;
use crate::ecma_conversions::round_to_even;
use crate::html::TextDisplay;
use crate::string::{AvmString, WStr};

pub use crate::avm2::object::textformat_allocator as text_format_allocator;

use ruffle_macros::istr;

pub fn get_align<'gc>(
    activation: &mut Activation<'_, 'gc>,
    this: Value<'gc>,
    _args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    let this = this.as_object().unwrap();

    if let Some(text_format) = this.as_text_format() {
        return Ok(text_format
            .align
            .as_ref()
            .map_or(Value::NULL, |align| match align {
                swf::TextAlign::Left => istr!("left").into(),
                swf::TextAlign::Center => istr!("center").into(),
                swf::TextAlign::Right => istr!("right").into(),
                swf::TextAlign::Justify => istr!("justify").into(),
            }));
    }

    Ok(Value::UNDEFINED)
}

pub fn set_align<'gc>(
    activation: &mut Activation<'_, 'gc>,
    this: Value<'gc>,
    args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    let this = this.as_object().unwrap();

    if let Some(mut text_format) = this.as_text_format_mut() {
        let value = args.try_get_string(0);
        let value = match value {
            Some(value) => value,
            None => {
                text_format.align = None;
                return Ok(Value::UNDEFINED);
            }
        };

        text_format.align = if value == WStr::from_units(b"left") {
            Some(swf::TextAlign::Left)
        } else if value == WStr::from_units(b"center") {
            Some(swf::TextAlign::Center)
        } else if value == WStr::from_units(b"right") {
            Some(swf::TextAlign::Right)
        } else if value == WStr::from_units(b"justify") {
            Some(swf::TextAlign::Justify)
        } else {
            return Err(make_error_2008(activation, "align"));
        };
    }

    Ok(Value::UNDEFINED)
}

pub fn get_block_indent<'gc>(
    _activation: &mut Activation<'_, 'gc>,
    this: Value<'gc>,
    _args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    let this = this.as_object().unwrap();

    if let Some(text_format) = this.as_text_format() {
        return Ok(text_format
            .block_indent
            .as_ref()
            .map_or(Value::NULL, |&block_indent| block_indent.into()));
    }

    Ok(Value::UNDEFINED)
}

pub fn set_block_indent<'gc>(
    activation: &mut Activation<'_, 'gc>,
    this: Value<'gc>,
    args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    let this = this.as_object().unwrap();

    if let Some(mut text_format) = this.as_text_format_mut() {
        let value = args.get_value(0);
        text_format.block_indent = if value.is_null() {
                None
            } else {
                Some(round_to_even(value.coerce_to_number(activation)?).into())
            };
    }

    Ok(Value::UNDEFINED)
}

pub fn get_bold<'gc>(
    _activation: &mut Activation<'_, 'gc>,
    this: Value<'gc>,
    _args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    let this = this.as_object().unwrap();

    if let Some(text_format) = this.as_text_format() {
        return Ok(text_format
            .bold
            .as_ref()
            .map_or(Value::NULL, |&bold| bold.into()));
    }

    Ok(Value::UNDEFINED)
}

pub fn set_bold<'gc>(
    _activation: &mut Activation<'_, 'gc>,
    this: Value<'gc>,
    args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    let this = this.as_object().unwrap();

    if let Some(mut text_format) = this.as_text_format_mut() {
        let value = args.get_value(0);
        text_format.bold = if value.is_null() {
                None
            } else {
                Some(value.coerce_to_boolean())
            };
    }

    Ok(Value::UNDEFINED)
}

pub fn get_bullet<'gc>(
    _activation: &mut Activation<'_, 'gc>,
    this: Value<'gc>,
    _args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    let this = this.as_object().unwrap();

    if let Some(text_format) = this.as_text_format() {
        return Ok(text_format
            .bullet
            .as_ref()
            .map_or(Value::NULL, |&bullet| bullet.into()));
    }

    Ok(Value::UNDEFINED)
}

pub fn set_bullet<'gc>(
    _activation: &mut Activation<'_, 'gc>,
    this: Value<'gc>,
    args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    let this = this.as_object().unwrap();

    if let Some(mut text_format) = this.as_text_format_mut() {
        let value = args.get_value(0);
        text_format.bullet = if value.is_null() {
                None
            } else {
                Some(value.coerce_to_boolean())
            };
    }

    Ok(Value::UNDEFINED)
}

pub fn get_color<'gc>(
    _activation: &mut Activation<'_, 'gc>,
    this: Value<'gc>,
    _args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    let this = this.as_object().unwrap();

    if let Some(text_format) = this.as_text_format() {
        return Ok(text_format
            .color
            .as_ref()
            .map_or(Value::NULL, |color| (color.to_rgba() as i32).into()));
    }

    Ok(Value::UNDEFINED)
}

pub fn set_color<'gc>(
    activation: &mut Activation<'_, 'gc>,
    this: Value<'gc>,
    args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    let this = this.as_object().unwrap();

    if let Some(mut text_format) = this.as_text_format_mut() {
        let value = args.get_value(0);
        text_format.color = if value.is_null() {
                None
            } else {
                Some(swf::Color::from_rgba(value.coerce_to_u32(activation)?))
            };
    }

    Ok(Value::UNDEFINED)
}

pub fn get_display<'gc>(
    activation: &mut Activation<'_, 'gc>,
    this: Value<'gc>,
    _args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    let this = this.as_object().unwrap();

    if let Some(text_format) = this.as_text_format() {
        return Ok(text_format
            .display
            .as_ref()
            .map_or(Value::NULL, |display| match display {
                TextDisplay::Block => istr!("block").into(),
                TextDisplay::Inline => istr!("inline").into(),
                TextDisplay::None => istr!("none").into(),
            }));
    }

    Ok(Value::UNDEFINED)
}

pub fn set_display<'gc>(
    activation: &mut Activation<'_, 'gc>,
    this: Value<'gc>,
    args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    let this = this.as_object().unwrap();

    if let Some(mut text_format) = this.as_text_format_mut() {
        let value = args.get_value(0);
        if value.is_null() {
            return Err(make_error_2007(activation, "display"));
        }
        let value = value.coerce_to_string(activation)?;

        text_format.display = if &value == b"block" {
            Some(TextDisplay::Block)
        } else if &value == b"inline" {
            Some(TextDisplay::Inline)
        } else if &value == b"none" {
            Some(TextDisplay::None)
        } else {
            // No error message for this, silently set it to None/null
            None
        };
    }

    Ok(Value::UNDEFINED)
}

pub fn get_font<'gc>(
    activation: &mut Activation<'_, 'gc>,
    this: Value<'gc>,
    _args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    let this = this.as_object().unwrap();

    if let Some(text_format) = this.as_text_format() {
        return Ok(text_format.font.as_ref().map_or(Value::NULL, |font| {
            AvmString::new(activation.gc(), font.as_wstr()).into()
        }));
    }

    Ok(Value::UNDEFINED)
}

pub fn set_font<'gc>(
    activation: &mut Activation<'_, 'gc>,
    this: Value<'gc>,
    args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    let this = this.as_object().unwrap();

    if let Some(mut text_format) = this.as_text_format_mut() {
        let value = args.get_value(0);
        text_format.font = if value.is_null() {
            None
        } else {
            let font = value.coerce_to_string(activation)?;
            Some(font.slice(0..64).unwrap_or(&font).to_owned())
        };
    }

    Ok(Value::UNDEFINED)
}

pub fn get_indent<'gc>(
    _activation: &mut Activation<'_, 'gc>,
    this: Value<'gc>,
    _args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    let this = this.as_object().unwrap();

    if let Some(text_format) = this.as_text_format() {
        return Ok(text_format
            .indent
            .as_ref()
            .map_or(Value::NULL, |&indent| indent.into()));
    }

    Ok(Value::UNDEFINED)
}

pub fn set_indent<'gc>(
    activation: &mut Activation<'_, 'gc>,
    this: Value<'gc>,
    args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    let this = this.as_object().unwrap();

    if let Some(mut text_format) = this.as_text_format_mut() {
        let value = args.get_value(0);
        text_format.indent = if value.is_null() {
                None
            } else {
                Some(round_to_even(value.coerce_to_number(activation)?).into())
            };
    }

    Ok(Value::UNDEFINED)
}

pub fn get_italic<'gc>(
    _activation: &mut Activation<'_, 'gc>,
    this: Value<'gc>,
    _args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    let this = this.as_object().unwrap();

    if let Some(text_format) = this.as_text_format() {
        return Ok(text_format
            .italic
            .as_ref()
            .map_or(Value::NULL, |&italic| italic.into()));
    }

    Ok(Value::UNDEFINED)
}

pub fn set_italic<'gc>(
    _activation: &mut Activation<'_, 'gc>,
    this: Value<'gc>,
    args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    let this = this.as_object().unwrap();

    if let Some(mut text_format) = this.as_text_format_mut() {
        let value = args.get_value(0);
        text_format.italic = if value.is_null() {
                None
            } else {
                Some(value.coerce_to_boolean())
            };
    }

    Ok(Value::UNDEFINED)
}

pub fn get_kerning<'gc>(
    _activation: &mut Activation<'_, 'gc>,
    this: Value<'gc>,
    _args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    let this = this.as_object().unwrap();

    if let Some(text_format) = this.as_text_format() {
        return Ok(text_format
            .kerning
            .as_ref()
            .map_or(Value::NULL, |&kerning| kerning.into()));
    }

    Ok(Value::UNDEFINED)
}

pub fn set_kerning<'gc>(
    _activation: &mut Activation<'_, 'gc>,
    this: Value<'gc>,
    args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    let this = this.as_object().unwrap();

    if let Some(mut text_format) = this.as_text_format_mut() {
        let value = args.get_value(0);
        text_format.kerning = if value.is_null() {
                None
            } else {
                Some(value.coerce_to_boolean())
            };
    }

    Ok(Value::UNDEFINED)
}

pub fn get_leading<'gc>(
    _activation: &mut Activation<'_, 'gc>,
    this: Value<'gc>,
    _args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    let this = this.as_object().unwrap();

    if let Some(text_format) = this.as_text_format() {
        return Ok(text_format
            .leading
            .as_ref()
            .map_or(Value::NULL, |&leading| leading.into()));
    }

    Ok(Value::UNDEFINED)
}

pub fn set_leading<'gc>(
    activation: &mut Activation<'_, 'gc>,
    this: Value<'gc>,
    args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    let this = this.as_object().unwrap();

    if let Some(mut text_format) = this.as_text_format_mut() {
        let value = args.get_value(0);
        text_format.leading = if value.is_null() {
                None
            } else {
                Some(round_to_even(value.coerce_to_number(activation)?).into())
            };
    }

    Ok(Value::UNDEFINED)
}

pub fn get_left_margin<'gc>(
    _activation: &mut Activation<'_, 'gc>,
    this: Value<'gc>,
    _args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    let this = this.as_object().unwrap();

    if let Some(text_format) = this.as_text_format() {
        return Ok(text_format
            .left_margin
            .as_ref()
            .map_or(Value::NULL, |&left_margin| left_margin.into()));
    }

    Ok(Value::UNDEFINED)
}

pub fn set_left_margin<'gc>(
    activation: &mut Activation<'_, 'gc>,
    this: Value<'gc>,
    args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    let this = this.as_object().unwrap();

    if let Some(mut text_format) = this.as_text_format_mut() {
        let value = args.get_value(0);
        text_format.left_margin = if value.is_null() {
                None
            } else {
                Some(round_to_even(value.coerce_to_number(activation)?).into())
            };
    }

    Ok(Value::UNDEFINED)
}

pub fn get_letter_spacing<'gc>(
    _activation: &mut Activation<'_, 'gc>,
    this: Value<'gc>,
    _args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    let this = this.as_object().unwrap();

    if let Some(text_format) = this.as_text_format() {
        return Ok(text_format
            .letter_spacing
            .as_ref()
            .map_or(Value::NULL, |&letter_spacing| letter_spacing.into()));
    }

    Ok(Value::UNDEFINED)
}

pub fn set_letter_spacing<'gc>(
    activation: &mut Activation<'_, 'gc>,
    this: Value<'gc>,
    args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    let this = this.as_object().unwrap();

    if let Some(mut text_format) = this.as_text_format_mut() {
        let value = args.get_value(0);
        text_format.letter_spacing = if value.is_null() {
            None
        } else {
            Some(value.coerce_to_number(activation)?)
        };
    }

    Ok(Value::UNDEFINED)
}

pub fn get_right_margin<'gc>(
    _activation: &mut Activation<'_, 'gc>,
    this: Value<'gc>,
    _args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    let this = this.as_object().unwrap();

    if let Some(text_format) = this.as_text_format() {
        return Ok(text_format
            .right_margin
            .as_ref()
            .map_or(Value::NULL, |&right_margin| right_margin.into()));
    }

    Ok(Value::UNDEFINED)
}

pub fn set_right_margin<'gc>(
    activation: &mut Activation<'_, 'gc>,
    this: Value<'gc>,
    args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    let this = this.as_object().unwrap();

    if let Some(mut text_format) = this.as_text_format_mut() {
        let value = args.get_value(0);
        text_format.right_margin = if value.is_null() {
                None
            } else {
                Some(round_to_even(value.coerce_to_number(activation)?).into())
            };
    }

    Ok(Value::UNDEFINED)
}

pub fn get_size<'gc>(
    _activation: &mut Activation<'_, 'gc>,
    this: Value<'gc>,
    _args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    let this = this.as_object().unwrap();

    if let Some(text_format) = this.as_text_format() {
        return Ok(text_format
            .size
            .as_ref()
            .map_or(Value::NULL, |&size| size.into()));
    }

    Ok(Value::UNDEFINED)
}

pub fn set_size<'gc>(
    activation: &mut Activation<'_, 'gc>,
    this: Value<'gc>,
    args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    let this = this.as_object().unwrap();

    if let Some(mut text_format) = this.as_text_format_mut() {
        let value = args.get_value(0);
        text_format.size = if value.is_null() {
                None
            } else {
                Some(round_to_even(value.coerce_to_number(activation)?).into())
            };
    }

    Ok(Value::UNDEFINED)
}

pub fn get_tab_stops<'gc>(
    activation: &mut Activation<'_, 'gc>,
    this: Value<'gc>,
    _args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    let this = this.as_object().unwrap();

    if let Some(text_format) = this.as_text_format() {
        return text_format
            .tab_stops
            .as_ref()
            .map_or(Ok(Value::NULL), |tab_stops| {
                let tab_stop_storage = tab_stops.iter().copied().collect();
                Ok(ArrayObject::from_storage(activation.context, tab_stop_storage).into())
            });
    }

    Ok(Value::UNDEFINED)
}

pub fn set_tab_stops<'gc>(
    activation: &mut Activation<'_, 'gc>,
    this: Value<'gc>,
    args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    let this = this.as_object().unwrap();

    if let Some(mut text_format) = this.as_text_format_mut() {
        let value = args.try_get_object(0);
        text_format.tab_stops = match value {
            Some(obj) => {
                let array_storage = obj.as_array_storage().unwrap();
                let length = array_storage.length();

                let tab_stops: Result<Vec<_>, Error<'gc>> = (0..length)
                    .map(|i| {
                        let value = array_storage.get(i).unwrap_or(Value::from_f64(0.0));

                        Ok(round_to_even(value.coerce_to_number(activation)?).into())
                    })
                    .collect();
                Some(tab_stops?)
            }
            None => None,
        };
    }

    Ok(Value::UNDEFINED)
}

pub fn get_target<'gc>(
    activation: &mut Activation<'_, 'gc>,
    this: Value<'gc>,
    _args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    let this = this.as_object().unwrap();

    if let Some(text_format) = this.as_text_format() {
        return Ok(text_format.target.as_ref().map_or(Value::NULL, |target| {
            AvmString::new(activation.gc(), target.as_wstr()).into()
        }));
    }

    Ok(Value::UNDEFINED)
}

pub fn set_target<'gc>(
    _activation: &mut Activation<'_, 'gc>,
    this: Value<'gc>,
    args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    let this = this.as_object().unwrap();

    if let Some(mut text_format) = this.as_text_format_mut() {
        let target = args.try_get_string(0);
        text_format.target = target.map(|t| t.as_wstr().into());
    }

    Ok(Value::UNDEFINED)
}

pub fn get_underline<'gc>(
    _activation: &mut Activation<'_, 'gc>,
    this: Value<'gc>,
    _args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    let this = this.as_object().unwrap();

    if let Some(text_format) = this.as_text_format() {
        return Ok(text_format
            .underline
            .as_ref()
            .map_or(Value::NULL, |&underline| underline.into()));
    }

    Ok(Value::UNDEFINED)
}

pub fn set_underline<'gc>(
    _activation: &mut Activation<'_, 'gc>,
    this: Value<'gc>,
    args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    let this = this.as_object().unwrap();

    if let Some(mut text_format) = this.as_text_format_mut() {
        let value = args.get_value(0);
        text_format.underline = if value.is_null() {
                None
            } else {
                Some(value.coerce_to_boolean())
            };
    }

    Ok(Value::UNDEFINED)
}

pub fn get_url<'gc>(
    activation: &mut Activation<'_, 'gc>,
    this: Value<'gc>,
    _args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    let this = this.as_object().unwrap();

    if let Some(text_format) = this.as_text_format() {
        return Ok(text_format.url.as_ref().map_or(Value::NULL, |url| {
            AvmString::new(activation.gc(), url.as_wstr()).into()
        }));
    }

    Ok(Value::UNDEFINED)
}

pub fn set_url<'gc>(
    _activation: &mut Activation<'_, 'gc>,
    this: Value<'gc>,
    args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    let this = this.as_object().unwrap();

    if let Some(mut text_format) = this.as_text_format_mut() {
        let url = args.try_get_string(0);
        text_format.url = url.map(|u| u.as_wstr().into());
    }

    Ok(Value::UNDEFINED)
}
