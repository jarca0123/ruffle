//! Object representation for `flash.concurrent.Condition`.

use crate::avm2::Error;
use crate::avm2::activation::Activation;
use crate::avm2::object::kind;
use crate::avm2::object::script_object::ScriptObjectData;
use crate::avm2::object::{ClassObject, Object, TObject};
use crate::avm2::worker_shared::{SharedCondition, SharedMutex};
use core::fmt;
use gc_arena::{Collect, Gc};
use ruffle_common::utils::HasPrefixField;
use std::cell::RefCell;

/// Allocator for `flash.concurrent.Condition`. The associated shared condition
/// is created in the constructor (native `init`), once the `Mutex` argument is
/// known.
pub fn condition_allocator<'gc>(
    class: ClassObject<'gc>,
    activation: &mut Activation<'_, 'gc>,
) -> Result<Object<'gc>, Error<'gc>> {
    let base = ScriptObjectData::new(class);
    Ok(ConditionObject(Gc::new(
        activation.gc(),
        ConditionObjectData {
            base,
            condition: RefCell::new(None),
        },
    ))
    .into())
}

#[derive(Clone, Collect, Copy)]
#[collect(no_drop)]
pub struct ConditionObject<'gc>(pub Gc<'gc, ConditionObjectData<'gc>>);

impl fmt::Debug for ConditionObject<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ConditionObject")
            .field("ptr", &Gc::as_ptr(self.0))
            .finish()
    }
}

#[derive(Collect, HasPrefixField)]
#[collect(no_drop)]
#[repr(C, align(8))]
pub struct ConditionObjectData<'gc> {
    /// Base script object
    base: ScriptObjectData<'gc, kind::ConditionObject>,

    /// The shared, arena-external condition. Set by the constructor (`init`).
    #[collect(require_static)]
    condition: RefCell<Option<SharedCondition>>,
}

impl<'gc> TObject<'gc> for ConditionObject<'gc> {
    fn gc_base(&self) -> Gc<'gc, ScriptObjectData<'gc>> {
        ScriptObjectData::erase_kind(HasPrefixField::as_prefix_gc(self.0))
    }
}

impl<'gc> ConditionObject<'gc> {
    /// Build a `Condition` object wrapping an existing shared condition (used
    /// when a `Condition` crosses the worker boundary by reference).
    pub fn from_shared(activation: &mut Activation<'_, 'gc>, condition: SharedCondition) -> Self {
        let class = activation.avm2().classes().condition;
        let base = ScriptObjectData::new(class);
        ConditionObject(Gc::new(
            activation.gc(),
            ConditionObjectData {
                base,
                condition: RefCell::new(Some(condition)),
            },
        ))
    }

    /// Bind this condition to `mutex` (constructor). No-op if already bound.
    pub fn bind(self, mutex: SharedMutex) {
        let mut slot = self.0.condition.borrow_mut();
        if slot.is_none() {
            *slot = Some(SharedCondition::new(mutex));
        }
    }

    pub fn shared_condition(self) -> Option<SharedCondition> {
        self.0.condition.borrow().clone()
    }
}
