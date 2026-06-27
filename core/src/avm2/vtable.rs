use crate::avm2::activation::Activation;
use crate::avm2::error::{Error, make_error_1107};
use crate::avm2::metadata::Metadata;
use crate::avm2::method::Method;
use crate::avm2::object::{ClassObject, FunctionObject};
use crate::avm2::property::{Property, PropertyClass};
use crate::avm2::property_map::PropertyMap;
use crate::avm2::scope::ScopeChain;
use crate::avm2::traits::{Trait, TraitKind};
use crate::avm2::value::Value;
use crate::avm2::{Class, Multiname, Namespace, QName};
use crate::context::UpdateContext;
use crate::string::{AvmString, StringContext};
use gc_arena::barrier::{field, unlock};
use gc_arena::lock::Lock;
use gc_arena::{Collect, Gc, Mutation};
use std::collections::HashMap;

#[derive(Collect, Clone, Copy)]
#[collect(no_drop)]
pub struct VTable<'gc>(Gc<'gc, VTableData<'gc>>);

#[derive(Collect, Default)]
#[collect(no_drop)]
struct VTableData<'gc> {
    scope: Option<ScopeChain<'gc>>,

    protected_namespace: Option<Namespace<'gc>>,

    /// Copy-on-write parent. Inherited traits are *not* copied into this
    /// vtable; instead they are resolved by walking up the `parent` chain.
    /// This avoids cloning the (potentially huge) superclass trait map for
    /// every subclass — see `init_vtable`.
    parent: Option<VTable<'gc>>,

    /// Only the traits *declared or overridden* by the defining class.
    /// Use `VTable::get_trait` / `all_resolved_traits` to get the full,
    /// inheritance-flattened view that walks the `parent` chain.
    resolved_traits: PropertyMap<'gc, Property>,

    /// Use hashmaps for the metadata tables because metadata will rarely be present on traits
    slot_metadata_table: HashMap<usize, Box<[Metadata<'gc>]>>,

    disp_metadata_table: HashMap<usize, Box<[Metadata<'gc>]>>,

    /// slot_table is indexed by `slot_id`
    slot_table: Box<[SlotInfo<'gc>]>,

    /// The number of methods owned by the `parent` chain. `disp_id`s below this
    /// value are inherited and resolved by walking `parent` (unless overridden,
    /// see `method_overrides`); `disp_id`s at or above it index `own_methods`.
    method_base: usize,

    /// Methods *declared* by this class, indexed by `disp_id - method_base`.
    /// Inherited methods are NOT copied here — they stay shared via `parent`.
    own_methods: Box<[ClassBoundMethod<'gc>]>,

    /// Methods this class *overrides* from an ancestor, keyed by the inherited
    /// `disp_id` (which is `< method_base`). Kept as a small slice rather than a
    /// `HashMap` so the entries can be mutated in place through the gc barrier
    /// (see `replace_scopes_with`). Overrides are rare, so linear scan is fine.
    method_overrides: Box<[MethodOverride<'gc>]>,
}

#[derive(Collect, Clone)]
#[collect(no_drop)]
struct MethodOverride<'gc> {
    disp_id: usize,
    method: ClassBoundMethod<'gc>,
}

impl PartialEq for VTable<'_> {
    fn eq(&self, other: &Self) -> bool {
        Gc::ptr_eq(self.0, other.0)
    }
}

#[derive(Collect, Clone)]
#[collect(no_drop)]
pub struct SlotInfo<'gc> {
    property_class: Lock<PropertyClass<'gc>>,
    pub default_value: Value<'gc>,
}

// TODO: it would be nice to remove the Option-ness from `scope` field for this
// to be more intuitive and cheaper
#[derive(Collect, Clone)]
#[collect(no_drop)]
pub struct ClassBoundMethod<'gc> {
    pub super_class_obj: Option<ClassObject<'gc>>,
    scope: Lock<Option<ScopeChain<'gc>>>,
    pub method: Method<'gc>,
}

impl<'gc> ClassBoundMethod<'gc> {
    pub fn scope(&self) -> ScopeChain<'gc> {
        self.scope.get().expect("Scope should exists here")
    }
}

impl<'gc> VTable<'gc> {
    pub fn empty(mc: &Mutation<'gc>) -> Self {
        VTable(Gc::new(mc, VTableData::default()))
    }

    /// Builds a new vtable by calculating the flattened list of instance traits
    /// that this class maintains.
    pub fn new(
        defining_class_def: Class<'gc>,
        super_class_obj: Option<ClassObject<'gc>>,
        scope: Option<ScopeChain<'gc>>,
        superclass_vtable: Option<Self>,
        mc: &Mutation<'gc>,
    ) -> Result<Self, VTableInitError> {
        let this = Self::init_vtable(
            defining_class_def,
            super_class_obj,
            scope,
            superclass_vtable,
        )?;
        Ok(VTable(Gc::new(mc, this)))
    }

    /// Like `VTable::new`, but also copies properties from the defining class' interfaces.
    pub fn new_with_interface_properties(
        defining_class_def: Class<'gc>,
        super_class_obj: Option<ClassObject<'gc>>,
        scope: Option<ScopeChain<'gc>>,
        superclass_vtable: Option<Self>,
        context: &UpdateContext<'gc>,
    ) -> Result<Self, VTableInitError> {
        let mut this = Self::init_vtable(
            defining_class_def,
            super_class_obj,
            scope,
            superclass_vtable,
        )?;
        Self::copy_interface_properties(&mut this, defining_class_def, context);

        Ok(VTable(Gc::new(context.gc(), this)))
    }

    /// The traits declared/overridden by *this* class only (not inherited).
    /// For the full inheritance-flattened view, use [`Self::all_resolved_traits`].
    pub fn resolved_traits(self) -> &'gc PropertyMap<'gc, Property> {
        &Gc::as_ref(self.0).resolved_traits
    }

    /// Look up a single trait by exact `QName`, walking the COW parent chain.
    /// Child traits shadow inherited ones with the same name+namespace.
    fn get_trait_by_qname(self, name: QName<'gc>) -> Option<Property> {
        let mut current = Some(self);
        while let Some(vt) = current {
            if let Some(prop) = vt.0.resolved_traits.get(name) {
                return Some(*prop);
            }
            current = vt.0.parent;
        }
        None
    }

    /// The full, inheritance-flattened list of resolved traits, walking the
    /// COW parent chain. A child entry shadows any inherited entry with the
    /// same local name + namespace, so each property is yielded exactly once.
    pub fn all_resolved_traits(self) -> Vec<(AvmString<'gc>, Namespace<'gc>, &'gc Property)> {
        let mut out: Vec<(AvmString<'gc>, Namespace<'gc>, &'gc Property)> = Vec::new();
        let mut current = Some(self);
        while let Some(vt) = current {
            for (name, ns, prop) in Gc::as_ref(vt.0).resolved_traits.iter() {
                if out
                    .iter()
                    .any(|(n, s, _)| *n == name && s.exact_version_match(ns))
                {
                    continue;
                }
                out.push((name, ns, prop));
            }
            current = vt.0.parent;
        }
        out
    }

    pub fn get_metadata_for_slot(self, slot_id: usize) -> Option<&'gc [Metadata<'gc>]> {
        Gc::as_ref(self.0)
            .slot_metadata_table
            .get(&slot_id)
            .map(|v| &**v)
    }

    pub fn get_metadata_for_disp(self, disp_id: usize) -> Option<&'gc [Metadata<'gc>]> {
        Gc::as_ref(self.0)
            .disp_metadata_table
            .get(&disp_id)
            .map(|v| &**v)
    }

    pub fn slot_class_name(
        self,
        context: &mut StringContext<'gc>,
        slot_id: usize,
    ) -> AvmString<'gc> {
        self.0
            .slot_table
            .get(slot_id)
            .expect("Invalid slot ID")
            .property_class
            .get()
            .get_name(context)
    }

    pub fn get_trait(self, name: &Multiname<'gc>) -> Option<Property> {
        if name.is_attribute() {
            return None;
        }

        let mut current = Some(self);
        while let Some(vt) = current {
            if let Some(prop) = vt.0.resolved_traits.get_for_multiname(name) {
                return Some(*prop);
            }
            current = vt.0.parent;
        }
        None
    }

    pub fn get_trait_with_ns(self, name: &Multiname<'gc>) -> Option<(Namespace<'gc>, Property)> {
        if name.is_attribute() {
            return None;
        }

        let mut current = Some(self);
        while let Some(vt) = current {
            if let Some((ns, p)) = vt.0.resolved_traits.get_with_ns_for_multiname(name) {
                return Some((ns, *p));
            }
            current = vt.0.parent;
        }
        None
    }

    /// Coerces `value` to the type of the slot with id `slot_id`
    pub fn coerce_trait_value(
        self,
        slot_id: usize,
        value: Value<'gc>,
        activation: &mut Activation<'_, 'gc>,
    ) -> Result<Value<'gc>, Error<'gc>> {
        let mut slot_class = self.0.slot_table[slot_id].property_class.get();

        let (value, changed) = slot_class.coerce(activation, value)?;

        // Calling coerce modified `PropertyClass` to cache the class lookup,
        // so store the new value back in the vtable.
        if changed {
            self.set_slot_class(activation.gc(), slot_id, slot_class);
        }
        Ok(value)
    }

    pub fn has_trait(self, name: &Multiname<'gc>) -> bool {
        self.get_trait(name).is_some()
    }

    /// The total number of methods reachable from this vtable, i.e. the number
    /// of valid `disp_id`s. Used to compute the `method_base` of a subclass.
    fn method_total(self) -> usize {
        self.0.method_base + self.0.own_methods.len()
    }

    pub fn get_method(self, disp_id: usize) -> Option<Method<'gc>> {
        self.get_full_method(disp_id).map(|m| m.method)
    }

    pub fn get_full_method(self, disp_id: usize) -> Option<&'gc ClassBoundMethod<'gc>> {
        // Walk the copy-on-write parent chain: methods owned by this class live
        // in `own_methods`, overridden inherited methods in `method_overrides`,
        // and everything else is resolved from the superclass vtable.
        let mut vt = self;
        loop {
            let data = Gc::as_ref(vt.0);
            if disp_id >= data.method_base {
                return data.own_methods.get(disp_id - data.method_base);
            }
            if let Some(o) = data.method_overrides.iter().find(|o| o.disp_id == disp_id) {
                return Some(&o.method);
            }
            match data.parent {
                Some(p) => vt = p,
                None => return None,
            }
        }
    }

    pub fn slot_count(self) -> usize {
        self.0.slot_table.len()
    }

    pub fn slot_table(&self) -> &[SlotInfo<'gc>] {
        &self.0.slot_table
    }

    pub fn slot_class(self, slot_id: usize) -> Option<PropertyClass<'gc>> {
        self.0
            .slot_table
            .get(slot_id)
            .map(|e| e.property_class.get())
    }

    pub fn set_slot_class(self, mc: &Mutation<'gc>, slot_id: usize, value: PropertyClass<'gc>) {
        let slots = field!(Gc::write(mc, self.0), VTableData, slot_table).as_deref();
        let slot = &slots[slot_id];
        let slot = unlock!(slot, SlotInfo, property_class);

        slot.set(value);
    }

    pub fn replace_scopes_with(self, mc: &Mutation<'gc>, new_scope: ScopeChain<'gc>) {
        // NOTE: with the copy-on-write method table this only rewrites the
        // scopes of methods *owned/overridden by this vtable*, not inherited
        // ones (those are shared from the parent and rewriting them would leak
        // into sibling subclasses). The sole caller is a bootstrap hack on the
        // `Object` class vtable, whose own methods are what matter.
        let write = Gc::write(mc, self.0);
        let own = field!(write, VTableData, own_methods).as_deref();
        for i in 0..own.len() {
            unlock!(&own[i], ClassBoundMethod, scope).set(Some(new_scope));
        }
        let overrides = field!(write, VTableData, method_overrides).as_deref();
        for i in 0..overrides.len() {
            let method = field!(&overrides[i], MethodOverride, method);
            unlock!(method, ClassBoundMethod, scope).set(Some(new_scope));
        }
    }

    fn init_vtable(
        defining_class_def: Class<'gc>,
        super_class_obj: Option<ClassObject<'gc>>,
        scope: Option<ScopeChain<'gc>>,
        superclass_vtable: Option<Self>,
    ) -> Result<VTableData<'gc>, VTableInitError> {
        // Let's talk about slot_ids and disp_ids.
        // Specification is one thing, but reality is another.

        // disp_id in FP:
        // It appears that FP completely ignores it and assigns values on its own.
        // Any attempt to use `callmethod` opcode to observe the disp_id fails
        // with VerifyError.
        //
        // disp_id in Ruffle:
        // Let's just do the same. We could go the easy way and always-increment,
        // but reusing same disp_id for overriding virtual methods is a nice idea,
        // both for space savings and lets us still use call_method() internally
        // for virtual dispatch when it's safe to do so.
        // And let's error on every `callmethod` opcode and hope it never ever happens.

        // slot_id in FP:
        // It's a bit more complex here.
        //
        // If class and superclass come from the same ABC (constant pool) or superclass has no slots,
        // then slot_ids are respected; conflicts result in VerifyError.
        // You are only allowed to call `getslot` on the object if calling method,
        // callee's class and all subclasses come from the same ABC (constant pool).
        // (or class has no slots, but then `getslot` fails verification anyway as it's out-of-range)
        //
        // If class and superclass come from different ABC (constant pool) and superclass has slots,
        // then subclass's slot_ids are ignored and assigned automatically.
        // ignored, as in: even if trait's slot_id conflicts, it's not verified at all.
        //
        // In practice, this all means that compiler is allowed to use `getslot`
        // or affect/observe slots in any other way only on classes
        // it had 100% control over slot layout of, on the entire class hierarchy.
        //
        // (*in particular, trying to use `getslot` in script initializer
        //   on class defined in same script also throws VerifyError;
        //   not sure why it's treated as "different constant pool")

        // slot_id in Ruffle:
        // Currently we don't really have ability to "compare abc between
        // methods/activations/traits/etc", so let's do something simpler.
        // We try to respect slot_id whenever possible, but if a conflict arises,
        // let's just auto-assign a higher one.
        // The logic is that if we ever see a conflict, either it's a class that
        // wouldn't have passed verification in the first place, or trying to observe
        // such slot with `getslot` wouldn't have passed verification in the first place.
        // So such SWFs shouldn't be encountered in the wild.
        //
        // Worst-case is that someone can hand-craft such an SWF specifically for Ruffle
        // and be able to access private class members with `getslot/setslot,
        // so long-term it's still something we should verify.
        // (and it's far from the only verification check we lack anyway)

        let mut resolved_traits = PropertyMap::new();
        let mut slot_metadata_table = HashMap::new();
        let mut disp_metadata_table = HashMap::new();
        let mut slot_table = Vec::new();

        // Copy-on-write method table: `own_methods` holds only methods declared
        // by this class (global `disp_id = method_base + index`), inherited
        // methods are resolved via the parent chain, and `method_overrides`
        // holds entries for inherited `disp_id`s this class overrides.
        let mut own_methods: Vec<ClassBoundMethod<'gc>> = Vec::new();
        let mut method_overrides: Vec<MethodOverride<'gc>> = Vec::new();
        let method_base = superclass_vtable.map(|sv| sv.method_total()).unwrap_or(0);

        // Subclasses cannot "override" slots in superclasses, so we only
        // maintain the list of slots that were declared by the subclass. At the
        // end of this method, we will append this list to `slot_table`.
        let mut new_slots = Vec::new();
        let mut first_slot_offset = 0;
        let mut force_auto_assign_slots = false;

        if let Some(superclass_vtable) = superclass_vtable {
            // NOTE: `resolved_traits` is intentionally NOT cloned from the
            // superclass. Inherited traits stay shared through the `parent`
            // link (set when building `VTableData` below) and are resolved by
            // walking the chain in `get_trait` / `all_resolved_traits`. This
            // copy-on-write scheme is the whole point: it avoids cloning the
            // (potentially huge) flattened superclass trait map per subclass.
            //
            // The metadata tables are still cloned, but they are almost always
            // empty (metadata is rare on traits), so the clone is essentially
            // free. The dense `slot_table` is still copied so slots can be
            // indexed directly by slot_id; `method_table` is now copy-on-write
            // (see `own_methods` / `method_overrides` / `method_base`).
            slot_metadata_table = superclass_vtable.0.slot_metadata_table.clone();
            disp_metadata_table = superclass_vtable.0.disp_metadata_table.clone();
            slot_table.extend_from_slice(&superclass_vtable.0.slot_table);

            first_slot_offset = superclass_vtable.slot_count();

            if let Some(protected_namespace) = defining_class_def.protected_namespace()
                && let Some(super_protected_namespace) = superclass_vtable.0.protected_namespace
            {
                // Copy all protected traits from the whole superclass hierarchy
                // but with this class's protected namespace. These are stored as
                // *owned* traits of this vtable (they use a new namespace, so
                // they never collide with the inherited ones).
                for (local_name, ns, prop) in superclass_vtable.all_resolved_traits() {
                    if ns.exact_version_match(super_protected_namespace) {
                        let new_name = QName::new(protected_namespace, local_name);
                        resolved_traits.insert(new_name, *prop);
                    }
                }
            }
        }

        if let Some(defining_tunit) = defining_class_def.translation_unit() {
            let mut current_super_class = defining_class_def.super_class();
            while let Some(super_class) = current_super_class {
                if let Some(super_tunit) = super_class.translation_unit()
                    && !defining_tunit.same_abc(super_tunit)
                    && super_class.vtable().slot_count() != 0
                {
                    // If the superclass of this class comes from a
                    // different ABC, and it has a non-zero slot count,
                    // the slots for this vtable are auto-assigned.
                    force_auto_assign_slots = true;
                    break;
                }
                current_super_class = super_class.super_class();
            }
        }

        for trait_data in defining_class_def.traits() {
            match trait_data.kind() {
                TraitKind::Method { method, .. } => {
                    let entry = ClassBoundMethod {
                        super_class_obj,
                        scope: Lock::new(scope),
                        method: *method,
                    };
                    // Resolve any inherited trait with the same name by walking
                    // the COW parent chain (it is no longer copied into our map).
                    let existing = resolved_traits.get(trait_data.name()).copied().or_else(|| {
                        superclass_vtable.and_then(|sv| sv.get_trait_by_qname(trait_data.name()))
                    });
                    match existing {
                        Some(Property::Method { disp_id }) => {
                            if let Some(metadata) = trait_data.metadata() {
                                disp_metadata_table.insert(disp_id, metadata);
                            }

                            install_method(
                                &mut own_methods,
                                &mut method_overrides,
                                method_base,
                                disp_id,
                                entry,
                            );
                        }
                        // note: ideally overwriting other property types
                        // should be a VerifyError
                        None => {
                            let disp_id = method_base + own_methods.len();
                            own_methods.push(entry);
                            resolved_traits
                                .insert(trait_data.name(), Property::new_method(disp_id));

                            if let Some(metadata) = trait_data.metadata() {
                                disp_metadata_table.insert(disp_id, metadata);
                            }
                        }
                        _ => unreachable!(
                            "`Class::validate_class` ensures overridden trait is correct"
                        ),
                    }
                }
                TraitKind::Getter { method, .. } => {
                    let entry = ClassBoundMethod {
                        super_class_obj,
                        scope: Lock::new(scope),
                        method: *method,
                    };
                    // If a matching virtual property is inherited (only in the
                    // parent chain), copy it down into our own map (copy-on-write)
                    // so the `get_mut` below can complete it with this accessor.
                    if resolved_traits.get(trait_data.name()).is_none()
                        && let Some(inherited) = superclass_vtable
                            .and_then(|sv| sv.get_trait_by_qname(trait_data.name()))
                    {
                        resolved_traits.insert(trait_data.name(), inherited);
                    }
                    match resolved_traits.get_mut(trait_data.name()) {
                        Some(Property::Virtual {
                            get: Some(disp_id), ..
                        }) => {
                            if let Some(metadata) = trait_data.metadata() {
                                disp_metadata_table.insert(*disp_id, metadata);
                            }

                            install_method(
                                &mut own_methods,
                                &mut method_overrides,
                                method_base,
                                *disp_id,
                                entry,
                            );
                        }
                        Some(Property::Virtual { get, .. }) => {
                            let disp_id = method_base + own_methods.len();
                            *get = Some(disp_id);
                            own_methods.push(entry);

                            if let Some(metadata) = trait_data.metadata() {
                                disp_metadata_table.insert(disp_id, metadata);
                            }
                        }
                        None => {
                            let disp_id = method_base + own_methods.len();
                            own_methods.push(entry);
                            resolved_traits
                                .insert(trait_data.name(), Property::new_getter(disp_id));

                            if let Some(metadata) = trait_data.metadata() {
                                disp_metadata_table.insert(disp_id, metadata);
                            }
                        }
                        _ => unreachable!(
                            "`Class::validate_class` ensures overridden trait is correct"
                        ),
                    }
                }
                TraitKind::Setter { method, .. } => {
                    let entry = ClassBoundMethod {
                        super_class_obj,
                        scope: Lock::new(scope),
                        method: *method,
                    };
                    // See the getter arm: copy an inherited virtual down into our
                    // own map (copy-on-write) before completing it with this setter.
                    if resolved_traits.get(trait_data.name()).is_none()
                        && let Some(inherited) = superclass_vtable
                            .and_then(|sv| sv.get_trait_by_qname(trait_data.name()))
                    {
                        resolved_traits.insert(trait_data.name(), inherited);
                    }
                    match resolved_traits.get_mut(trait_data.name()) {
                        Some(Property::Virtual {
                            set: Some(disp_id), ..
                        }) => {
                            if let Some(metadata) = trait_data.metadata() {
                                disp_metadata_table.insert(*disp_id, metadata);
                            }

                            install_method(
                                &mut own_methods,
                                &mut method_overrides,
                                method_base,
                                *disp_id,
                                entry,
                            );
                        }
                        Some(Property::Virtual { set, .. }) => {
                            let disp_id = method_base + own_methods.len();
                            own_methods.push(entry);
                            *set = Some(disp_id);

                            if let Some(metadata) = trait_data.metadata() {
                                disp_metadata_table.insert(disp_id, metadata);
                            }
                        }
                        None => {
                            let disp_id = method_base + own_methods.len();
                            own_methods.push(entry);
                            resolved_traits
                                .insert(trait_data.name(), Property::new_setter(disp_id));

                            if let Some(metadata) = trait_data.metadata() {
                                disp_metadata_table.insert(disp_id, metadata);
                            }
                        }
                        _ => unreachable!(
                            "`Class::validate_class` ensures overridden trait is correct"
                        ),
                    }
                }
                TraitKind::Slot { slot_id, .. }
                | TraitKind::Const { slot_id, .. }
                | TraitKind::Class { slot_id, .. } => {
                    let slot_id = *slot_id;

                    let default_value = trait_to_default_value(trait_data);

                    let prop_class = match trait_data.kind() {
                        TraitKind::Slot { slot_type, .. } | TraitKind::Const { slot_type, .. } => {
                            *slot_type
                        }
                        TraitKind::Class { class, .. } => PropertyClass::Class(
                            class.c_class().expect("Trait should hold an i_class"),
                        ),
                        _ => unreachable!(),
                    };

                    let slot_info = SlotInfo {
                        property_class: Lock::new(prop_class),
                        default_value,
                    };

                    let new_slot_id = if slot_id == 0 || force_auto_assign_slots {
                        new_slots.push(Some(slot_info));
                        new_slots.len() - 1
                    } else {
                        // it's non-zero, so let's turn it from 1-based to 0-based.
                        let slot_id = slot_id - 1;
                        if slot_id < first_slot_offset {
                            // Conflict: subclass attempted to use slot id of
                            // slot in superclass
                            return Err(VTableInitError::SlotConflict);
                        }

                        // now make the slot id relative to the start of this
                        // subclass's slots
                        let slot_id = slot_id - first_slot_offset;

                        if let Some(Some(_)) = new_slots.get(slot_id) {
                            // Conflict: subclass attempted to use slot id that
                            // it already used
                            return Err(VTableInitError::SlotConflict);
                        } else {
                            if slot_id >= new_slots.len() {
                                new_slots.resize(slot_id + 1, None);
                            }
                            new_slots[slot_id] = Some(slot_info);

                            slot_id
                        }
                    };

                    // Convert new_slot_id to an absolute slot id
                    let new_slot_id = new_slot_id + first_slot_offset;

                    if let Some(metadata) = trait_data.metadata() {
                        slot_metadata_table.insert(new_slot_id, metadata);
                    }

                    let property = match trait_data.kind() {
                        TraitKind::Slot { .. } => Property::new_slot(new_slot_id),
                        TraitKind::Const { .. } | TraitKind::Class { .. } => {
                            Property::new_const_slot(new_slot_id)
                        }
                        _ => unreachable!(),
                    };

                    resolved_traits.insert(trait_data.name(), property);
                }
            }
        }

        // Append the new slots to the slot table now.
        for slot in new_slots {
            if let Some(slot) = slot {
                slot_table.push(slot);
            } else {
                // Gaps in the slot numbering are filled with a default
                // `*`-typed slot, matching avmplus behavior.
                slot_table.push(SlotInfo {
                    property_class: Lock::new(PropertyClass::Any),
                    default_value: Value::Undefined,
                });
            }
        }

        Ok(VTableData {
            scope,
            protected_namespace: defining_class_def.protected_namespace(),
            parent: superclass_vtable,
            resolved_traits,
            slot_metadata_table,
            disp_metadata_table,
            slot_table: slot_table.into_boxed_slice(),
            method_base,
            own_methods: own_methods.into_boxed_slice(),
            method_overrides: method_overrides.into_boxed_slice(),
        })
    }

    fn copy_interface_properties(
        this: &mut VTableData<'gc>,
        class: Class<'gc>,
        context: &UpdateContext<'gc>,
    ) {
        // FIXME - we should only be copying properties for newly-implemented
        // interfaces (i.e. those that were not already implemented by the superclass)
        // Otherwise, our behavior diverges from Flash Player in certain cases.
        // See the ignored test 'tests/tests/swfs/avm2/weird_superinterface_properties/'
        let internal_ns = context.avm2.namespaces.public_vm_internal();
        for interface in class.all_interfaces() {
            for interface_trait in interface.traits() {
                let interface_name = interface_trait.name();
                if !interface_name.namespace().is_public() {
                    let public_name = QName::new(internal_ns, interface_name.local_name());
                    // The implementing trait may be inherited, so consult the
                    // COW parent chain in addition to this class's own traits.
                    let prop = this
                        .resolved_traits
                        .get(public_name)
                        .copied()
                        .or_else(|| this.parent.and_then(|p| p.get_trait_by_qname(public_name)));
                    if let Some(prop) = prop {
                        this.resolved_traits.insert(interface_name, prop);
                    }
                }
            }
        }
    }

    /// Retrieve a bound instance method suitable for use as a value.
    ///
    /// This returns the bound method object itself, as well as its dispatch
    /// ID. You will need the additional properties in order to install the
    /// method into your object.
    ///
    /// You should only call this method once per receiver/name pair, and cache
    /// the result. Otherwise, code that relies on bound methods having stable
    /// object identitities (e.g. `EventDispatcher.removeEventListener`) will
    /// fail.
    ///
    /// It is the caller's responsibility to ensure that the `receiver` passed
    /// to this method is not Value::Null or Value::Undefined.
    pub fn make_bound_method(
        self,
        context: &mut UpdateContext<'gc>,
        receiver: Value<'gc>,
        disp_id: usize,
    ) -> Option<FunctionObject<'gc>> {
        self.get_full_method(disp_id)
            .map(|method| Self::bind_method(context, receiver, method))
    }

    /// Bind an instance method to a receiver, allowing it to be used as a value. See `VTable::make_bound_method`
    ///
    /// It is the caller's responsibility to ensure that the `receiver` passed
    /// to this method is not Value::Null or Value::Undefined.
    pub fn bind_method(
        context: &mut UpdateContext<'gc>,
        receiver: Value<'gc>,
        method: &ClassBoundMethod<'gc>,
    ) -> FunctionObject<'gc> {
        FunctionObject::from_method(
            context,
            method.method,
            method.scope(),
            Some(receiver),
            method.super_class_obj,
        )
    }

    pub fn public_properties(self) -> impl Iterator<Item = (AvmString<'gc>, Property)> {
        self.all_resolved_traits()
            .into_iter()
            .filter(|(_, ns, _)| ns.is_public())
            .map(|(name, _, prop)| (name, *prop))
    }
}

#[derive(Debug)]
pub enum VTableInitError {
    SlotConflict,
}

impl VTableInitError {
    pub fn into_avm<'gc>(self, activation: &mut Activation<'_, 'gc>) -> Error<'gc> {
        match self {
            VTableInitError::SlotConflict => make_error_1107(activation),
        }
    }
}

/// Install a method into the copy-on-write method table being built by
/// `init_vtable`. A brand-new `disp_id` (>= `method_base`, i.e. a method this
/// class declares) re-uses or extends `own_methods`; overriding an inherited
/// `disp_id` (< `method_base`) records the override in `method_overrides`.
fn install_method<'gc>(
    own_methods: &mut [ClassBoundMethod<'gc>],
    method_overrides: &mut Vec<MethodOverride<'gc>>,
    method_base: usize,
    disp_id: usize,
    entry: ClassBoundMethod<'gc>,
) {
    if disp_id >= method_base {
        own_methods[disp_id - method_base] = entry;
    } else if let Some(o) = method_overrides.iter_mut().find(|o| o.disp_id == disp_id) {
        o.method = entry;
    } else {
        method_overrides.push(MethodOverride {
            disp_id,
            method: entry,
        });
    }
}

fn trait_to_default_value<'gc>(trait_data: &Trait<'gc>) -> Value<'gc> {
    match trait_data.kind() {
        TraitKind::Slot { default_value, .. } => *default_value,
        TraitKind::Const { default_value, .. } => *default_value,
        TraitKind::Class { .. } => Value::Null,
        _ => unreachable!(),
    }
}
