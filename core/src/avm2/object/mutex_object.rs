//! Object representation for `flash.concurrent.Mutex`.

use crate::avm2::Error;
use crate::avm2::activation::Activation;
use crate::avm2::object::kind;
use crate::avm2::object::script_object::ScriptObjectData;
use crate::avm2::object::{ClassObject, Object, TObject};
use crate::avm2::worker_shared::SharedMutex;
use core::fmt;
use gc_arena::{Collect, Gc};
use ruffle_common::utils::HasPrefixField;

/// Allocator for `flash.concurrent.Mutex`. Each instance gets a fresh shared
/// mutex, held by reference so it can cross worker threads via
/// `setSharedProperty`.
pub fn mutex_allocator<'gc>(
    class: ClassObject<'gc>,
    activation: &mut Activation<'_, 'gc>,
) -> Result<Object<'gc>, Error<'gc>> {
    let base = ScriptObjectData::new(class);
    Ok(MutexObject(Gc::new(
        activation.gc(),
        MutexObjectData {
            base,
            mutex: SharedMutex::new(),
        },
    ))
    .into())
}

#[derive(Clone, Collect, Copy)]
#[collect(no_drop)]
pub struct MutexObject<'gc>(pub Gc<'gc, MutexObjectData<'gc>>);

impl fmt::Debug for MutexObject<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MutexObject")
            .field("ptr", &Gc::as_ptr(self.0))
            .finish()
    }
}

#[derive(Collect, HasPrefixField)]
#[collect(no_drop)]
#[repr(C, align(8))]
pub struct MutexObjectData<'gc> {
    /// Base script object
    base: ScriptObjectData<'gc, kind::MutexObject>,

    /// The shared, arena-external lock.
    #[collect(require_static)]
    mutex: SharedMutex,
}

impl<'gc> TObject<'gc> for MutexObject<'gc> {
    fn gc_base(&self) -> Gc<'gc, ScriptObjectData<'gc>> {
        ScriptObjectData::erase_kind(HasPrefixField::as_prefix_gc(self.0))
    }
}

impl<'gc> MutexObject<'gc> {
    /// Build a `Mutex` object wrapping an existing shared lock (used when a
    /// `Mutex` crosses the worker boundary by reference).
    pub fn from_shared(activation: &mut Activation<'_, 'gc>, mutex: SharedMutex) -> Self {
        let class = activation.avm2().classes().mutex;
        let base = ScriptObjectData::new(class);
        MutexObject(Gc::new(activation.gc(), MutexObjectData { base, mutex }))
    }

    pub fn shared_mutex(self) -> SharedMutex {
        self.0.mutex.clone()
    }
}
