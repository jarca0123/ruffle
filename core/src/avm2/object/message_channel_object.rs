use crate::avm2::activation::Activation;
use crate::avm2::object::TObject;
use crate::avm2::object::kind;
use crate::avm2::object::script_object::ScriptObjectData;
use crate::avm2::worker_shared::SharedMessageChannel;
use core::fmt;
use gc_arena::{Collect, Gc};
use ruffle_common::utils::HasPrefixField;

#[derive(Clone, Collect, Copy)]
#[collect(no_drop)]
pub struct MessageChannelObject<'gc>(pub Gc<'gc, MessageChannelObjectData<'gc>>);

impl fmt::Debug for MessageChannelObject<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MessageChannelObject")
            .field("ptr", &Gc::as_ptr(self.0))
            .finish()
    }
}

#[derive(Collect, HasPrefixField)]
#[collect(no_drop)]
#[repr(C, align(8))]
pub struct MessageChannelObjectData<'gc> {
    /// Base script object
    base: ScriptObjectData<'gc, kind::MessageChannelObject>,

    /// The shared, arena-external message queue. Both endpoints (both workers)
    /// hold a clone of the same channel.
    #[collect(require_static)]
    channel: SharedMessageChannel,
}

impl<'gc> TObject<'gc> for MessageChannelObject<'gc> {
    fn gc_base(&self) -> Gc<'gc, ScriptObjectData<'gc>> {
        ScriptObjectData::erase_kind(HasPrefixField::as_prefix_gc(self.0))
    }
}

impl<'gc> MessageChannelObject<'gc> {
    pub fn new(activation: &mut Activation<'_, 'gc>) -> Self {
        Self::from_shared(activation, SharedMessageChannel::new())
    }

    /// Build a `MessageChannel` object wrapping an existing shared channel (used
    /// when a channel crosses the worker boundary by reference).
    pub fn from_shared(
        activation: &mut Activation<'_, 'gc>,
        channel: SharedMessageChannel,
    ) -> Self {
        let class = activation.avm2().classes().messagechannel;
        let base = ScriptObjectData::new(class);
        MessageChannelObject(Gc::new(
            activation.gc(),
            MessageChannelObjectData { base, channel },
        ))
    }

    pub fn channel(self) -> SharedMessageChannel {
        self.0.channel.clone()
    }
}
