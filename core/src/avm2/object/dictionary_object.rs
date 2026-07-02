//! Object representation for `flash.utils.Dictionary`

use crate::avm2::Error;
use crate::avm2::activation::Activation;
use crate::avm2::dynamic_map::{DynamicKey, DynamicMap};
use crate::avm2::object::kind;
use crate::avm2::object::script_object::ScriptObjectData;
use crate::avm2::object::{ClassObject, Object, TObject, WeakObject};
use crate::avm2::value::Value;
use crate::string::AvmString;
use core::fmt;
use gc_arena::barrier::unlock;
use gc_arena::lock::RefLock;
use gc_arena::{Collect, Gc, Mutation};
use ruffle_common::utils::HasPrefixField;
use std::cell::{Cell, Ref, RefMut};
use std::hash::{Hash, Hasher};

/// Enumerant indices for the weak object-space are tagged with this flag, to
/// distinguish them from the regular (base) enumerant indices.
///
/// AVM2 enumeration (`op_has_next_2`) coerces the index to `i32` and stops as
/// soon as it sees a negative value, so the tag must keep the index
/// non-negative — hence bit 30 rather than the sign bit. Base indices are small
/// sequential integers and never approach this, so the bit is always free.
const WEAK_INDEX_FLAG: u32 = 0x4000_0000;

/// A class instance allocator that allocates Dictionary objects.
pub fn dictionary_allocator<'gc>(
    class: ClassObject<'gc>,
    activation: &mut Activation<'_, 'gc>,
) -> Result<Object<'gc>, Error<'gc>> {
    let base = ScriptObjectData::new(class);

    Ok(DictionaryObject(Gc::new(
        activation.gc(),
        DictionaryObjectData {
            base,
            weak_keys: Cell::new(false),
            weak_space: RefLock::new(DynamicMap::new()),
            last_prune_len: Cell::new(0),
        },
    ))
    .into())
}

/// A weak reference to an object used as a key in a weak-keyed `Dictionary`.
///
/// Equality and hashing are by identity (pointer); a `GcWeak` keeps its
/// allocation alive, so the pointer stays stable even after the referenced
/// object has been collected.
#[derive(Collect, Clone, Copy)]
#[collect(no_drop)]
struct WeakObjectKey<'gc>(WeakObject<'gc>);

impl PartialEq for WeakObjectKey<'_> {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self.0.as_ptr(), other.0.as_ptr())
    }
}

impl Eq for WeakObjectKey<'_> {}

impl Hash for WeakObjectKey<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (self.0.as_ptr() as usize).hash(state);
    }
}

/// An object that allows associations between objects and values.
///
/// This is implemented by way of "object space", parallel to the property
/// space that ordinary properties live in. This space has no namespaces, and
/// keys are objects instead of strings.
///
/// When constructed with `weakKeys`, object keys are held weakly (in a separate
/// `weak_space`) so that an entry no longer keeps its key alive, matching
/// Flash's `Dictionary(true)` semantics. Dead entries are pruned lazily.
#[derive(Clone, Collect, Copy)]
#[collect(no_drop)]
pub struct DictionaryObject<'gc>(pub Gc<'gc, DictionaryObjectData<'gc>>);

impl fmt::Debug for DictionaryObject<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DictionaryObject")
            .field("ptr", &Gc::as_ptr(self.0))
            .finish()
    }
}

#[derive(Clone, Collect, HasPrefixField)]
#[collect(no_drop)]
#[repr(C, align(8))]
pub struct DictionaryObjectData<'gc> {
    /// Base script object
    base: ScriptObjectData<'gc, kind::DictionaryObject>,

    /// Whether object keys are held weakly (constructed via `Dictionary(true)`).
    #[collect(require_static)]
    weak_keys: Cell<bool>,

    /// Object-space for weak-keyed dictionaries: weak key -> (strong) value.
    /// Only used when `weak_keys` is set; otherwise object keys live in `base`.
    weak_space: RefLock<DynamicMap<WeakObjectKey<'gc>, Value<'gc>>>,

    /// Size of `weak_space` after the last prune, used to amortize pruning.
    #[collect(require_static)]
    last_prune_len: Cell<usize>,
}

impl<'gc> DictionaryObject<'gc> {
    /// Mark this dictionary as using weak keys. Called from the AS3 constructor
    /// when `weakKeys` is `true`.
    pub fn set_weak_keys(self) {
        self.0.weak_keys.set(true);
    }

    fn is_weak(self) -> bool {
        self.0.weak_keys.get()
    }

    fn weak_space(&self) -> Ref<'_, DynamicMap<WeakObjectKey<'gc>, Value<'gc>>> {
        self.0.weak_space.borrow()
    }

    fn weak_space_mut(
        &self,
        mc: &Mutation<'gc>,
    ) -> RefMut<'_, DynamicMap<WeakObjectKey<'gc>, Value<'gc>>> {
        unlock!(Gc::write(mc, self.0), DictionaryObjectData, weak_space).borrow_mut()
    }

    /// Drop entries whose weak key has been collected.
    fn prune_dead_weak_keys(self, mc: &Mutation<'gc>) {
        let dead: Vec<WeakObjectKey<'gc>> = self
            .weak_space()
            .keys()
            .filter(|key| key.0.upgrade(mc).is_none())
            .copied()
            .collect();

        if !dead.is_empty() {
            let mut space = self.weak_space_mut(mc);
            for key in &dead {
                space.remove(key);
            }
        }

        self.0.last_prune_len.set(self.weak_space().len());
    }

    /// Prune dead entries once the map has grown noticeably since the last
    /// prune. This keeps pruning amortized to O(1) per insertion.
    fn maybe_prune(self, mc: &Mutation<'gc>) {
        let threshold = self.0.last_prune_len.get().saturating_mul(2).max(16);
        if self.weak_space().len() > threshold {
            self.prune_dead_weak_keys(mc);
        }
    }

    /// Retrieve a value in the dictionary's object space.
    pub fn get_property_by_object(self, name: Object<'gc>) -> Value<'gc> {
        if self.is_weak() {
            self.weak_space()
                .get(&WeakObjectKey(name.downgrade()))
                .map(|v| v.value)
                .unwrap_or(Value::Undefined)
        } else {
            self.base()
                .values()
                .get(&DynamicKey::Object(name))
                .map(|v| v.value)
                .unwrap_or(Value::Undefined)
        }
    }

    /// Set a value in the dictionary's object space.
    pub fn set_property_by_object(self, name: Object<'gc>, value: Value<'gc>, mc: &Mutation<'gc>) {
        if self.is_weak() {
            self.weak_space_mut(mc)
                .insert(WeakObjectKey(name.downgrade()), value);
            self.maybe_prune(mc);
        } else {
            self.base()
                .values_mut(mc)
                .insert(DynamicKey::Object(name), value);
        }
    }

    /// Delete a value from the dictionary's object space.
    pub fn delete_property_by_object(self, name: Object<'gc>, mc: &Mutation<'gc>) {
        if self.is_weak() {
            self.weak_space_mut(mc)
                .remove(&WeakObjectKey(name.downgrade()));
        } else {
            self.base().values_mut(mc).remove(&DynamicKey::Object(name));
        }
    }

    pub fn has_property_by_object(self, name: Object<'gc>) -> bool {
        if self.is_weak() {
            self.weak_space()
                .contains_key(&WeakObjectKey(name.downgrade()))
        } else {
            self.base().values().contains_key(&DynamicKey::Object(name))
        }
    }
}

impl<'gc> TObject<'gc> for DictionaryObject<'gc> {
    fn gc_base(&self) -> Gc<'gc, ScriptObjectData<'gc>> {
        ScriptObjectData::erase_kind(HasPrefixField::as_prefix_gc(self.0))
    }

    // Calling `setPropertyIsEnumerable` on a `Dictionary` has no effect -
    // stringified properties are always enumerable.
    fn set_local_property_is_enumerable(
        &self,
        _mc: &Mutation<'gc>,
        _name: AvmString<'gc>,
        _is_enumerable: bool,
    ) {
    }

    fn get_next_enumerant(
        self,
        last_index: u32,
        activation: &mut Activation<'_, 'gc>,
    ) -> Result<u32, Error<'gc>> {
        // While still enumerating the base (string/uint) keys, defer to it.
        if !self.is_weak() || (last_index & WEAK_INDEX_FLAG) == 0 {
            let next = self.base().get_next_enumerant(last_index);
            if next != 0 {
                return Ok(next);
            }

            // Base exhausted: continue into the weak object-space (if any).
            if self.is_weak() {
                self.prune_dead_weak_keys(activation.gc());
                if let Some(weak_next) = self.weak_space().next(0) {
                    return Ok(WEAK_INDEX_FLAG | weak_next as u32);
                }
            }

            Ok(0)
        } else {
            match self
                .weak_space()
                .next((last_index & !WEAK_INDEX_FLAG) as usize)
            {
                Some(weak_next) => Ok(WEAK_INDEX_FLAG | weak_next as u32),
                None => Ok(0),
            }
        }
    }

    fn get_enumerant_name(
        self,
        index: u32,
        activation: &mut Activation<'_, 'gc>,
    ) -> Result<Value<'gc>, Error<'gc>> {
        if (index & WEAK_INDEX_FLAG) == 0 {
            Ok(self.base().get_enumerant_name(index).unwrap_or(Value::Null))
        } else {
            let key = self
                .weak_space()
                .key_at((index & !WEAK_INDEX_FLAG) as usize)
                .and_then(|key| key.0.upgrade(activation.gc()));
            Ok(key.map(Value::Object).unwrap_or(Value::Null))
        }
    }

    fn get_enumerant_value(
        self,
        index: u32,
        _activation: &mut Activation<'_, 'gc>,
    ) -> Result<Value<'gc>, Error<'gc>> {
        if (index & WEAK_INDEX_FLAG) == 0 {
            Ok(*self
                .base()
                .values()
                .value_at(index as usize)
                .unwrap_or(&Value::Undefined))
        } else {
            Ok(self
                .weak_space()
                .value_at((index & !WEAK_INDEX_FLAG) as usize)
                .copied()
                .unwrap_or(Value::Undefined))
        }
    }
}
