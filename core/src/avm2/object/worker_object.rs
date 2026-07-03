use crate::avm2::Error;
use crate::avm2::activation::Activation;
use crate::avm2::error::make_error_3732;
use crate::avm2::object::TObject;
use crate::avm2::object::kind;
use crate::avm2::object::script_object::ScriptObjectData;
use crate::avm2::value::Value;
use crate::avm2::worker_shared::SharedProperties;
use crate::context::UpdateContext;
use crate::string::AvmString;
use core::fmt;
use fnv::FnvHashMap;
use gc_arena::barrier::unlock;
use gc_arena::lock::RefLock;
use gc_arena::{Collect, Gc, Mutation};
use ruffle_common::utils::HasPrefixField;
use ruffle_macros::Avm2Enum;
use std::cell::Cell;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

#[derive(Clone, Collect, Copy)]
#[collect(no_drop)]
pub struct WorkerObject<'gc>(pub Gc<'gc, WorkerObjectData<'gc>>);

impl fmt::Debug for WorkerObject<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WorkerObject")
            .field("ptr", &Gc::as_ptr(self.0))
            .finish()
    }
}

#[derive(Collect, HasPrefixField)]
#[collect(no_drop)]
#[repr(C, align(8))]
pub struct WorkerObjectData<'gc> {
    /// Base script object
    base: ScriptObjectData<'gc, kind::WorkerObject>,

    kind: WorkerKind,

    /// Values set via `Worker.setSharedProperty`, retrievable with
    /// `getSharedProperty`. In real Flash these cross the worker boundary (by
    /// AMF copy, or by reference for shareable `ByteArray`/`Mutex`/`Condition`);
    /// in Ruffle's single-context cooperative model they are simply stored on
    /// the target worker.
    shared_properties: RefLock<FnvHashMap<AvmString<'gc>, Value<'gc>>>,

    /// `Send` mirror of the shared properties, so a spawned worker thread (with
    /// its own arena) can read what the creator set. Populated for values that
    /// can cross the thread boundary (shareable ByteArray by reference; other
    /// values by AMF copy).
    #[collect(require_static)]
    shared_send: SharedProperties,

    /// Set to request this worker's runtime stop; carried into the worker's
    /// [`WorkerConfig`](crate::worker_runtime::WorkerConfig).
    #[collect(require_static)]
    terminate: Arc<AtomicBool>,
}

impl<'gc> TObject<'gc> for WorkerObject<'gc> {
    fn gc_base(&self) -> Gc<'gc, ScriptObjectData<'gc>> {
        ScriptObjectData::erase_kind(HasPrefixField::as_prefix_gc(self.0))
    }
}

/// Distinguishes the primordial worker (main SWF runtime) from workers created
/// via `WorkerDomain.createWorker`. Only `Spawned` carries lifecycle state;
/// the primordial worker is permanently `Running`.
#[derive(Collect)]
#[collect(require_static)]
pub enum WorkerKind {
    Primordial,
    Spawned { state: Cell<WorkerState> },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Avm2Enum)]
pub enum WorkerState {
    #[avm2_variant("new")]
    New,
    #[avm2_variant("running")]
    Running,
    #[avm2_variant("terminated")]
    Terminated,
}

impl<'gc> WorkerObject<'gc> {
    pub fn new_regular(activation: &mut Activation<'_, 'gc>) -> Self {
        Self::new(
            activation.context,
            WorkerKind::Spawned {
                state: Cell::new(WorkerState::New),
            },
        )
    }

    pub fn new_primordial(context: &mut UpdateContext<'gc>) -> Self {
        Self::new(context, WorkerKind::Primordial)
    }

    /// The `Worker.current` object for a spawned worker's own runtime: already
    /// `Running`, and backed by the `Send` shared-property store the creating
    /// thread populated.
    pub fn new_worker_runtime(
        context: &mut UpdateContext<'gc>,
        shared_send: SharedProperties,
    ) -> Self {
        let class = context.avm2.classes().worker;
        let base = ScriptObjectData::new(class);
        Self(Gc::new(
            context.gc(),
            WorkerObjectData {
                base,
                kind: WorkerKind::Spawned {
                    state: Cell::new(WorkerState::Running),
                },
                shared_properties: RefLock::new(FnvHashMap::default()),
                shared_send,
                terminate: Arc::default(),
            },
        ))
    }

    fn new(context: &mut UpdateContext<'gc>, kind: WorkerKind) -> Self {
        let class = context.avm2.classes().worker;
        let base = ScriptObjectData::new(class);

        Self(Gc::new(
            context.gc(),
            WorkerObjectData {
                base,
                kind,
                shared_properties: RefLock::new(FnvHashMap::default()),
                shared_send: Arc::default(),
                terminate: Arc::default(),
            },
        ))
    }

    /// Stable identity of this worker (its GC pointer), used as a worker id
    /// across the thread boundary.
    pub fn id(self) -> u64 {
        Gc::as_ptr(self.0) as u64
    }

    /// The `Send` shared-property store shared with this worker's thread.
    pub fn shared_send(self) -> SharedProperties {
        self.0.shared_send.clone()
    }

    /// The termination flag handed to this worker's runtime.
    pub fn terminate_flag(self) -> Arc<AtomicBool> {
        self.0.terminate.clone()
    }

    /// Retrieves a value stored with `setSharedProperty`, or `undefined` if the
    /// key was never set.
    pub fn get_shared_property(self, key: AvmString<'gc>) -> Value<'gc> {
        self.0
            .shared_properties
            .borrow()
            .get(&key)
            .copied()
            .unwrap_or(Value::Undefined)
    }

    /// Stores a value under `key`, retrievable with `getSharedProperty`.
    pub fn set_shared_property(self, mc: &Mutation<'gc>, key: AvmString<'gc>, value: Value<'gc>) {
        unlock!(Gc::write(mc, self.0), WorkerObjectData, shared_properties)
            .borrow_mut()
            .insert(key, value);
    }

    pub fn is_primordial(self) -> bool {
        matches!(self.0.kind, WorkerKind::Primordial)
    }

    pub fn state(self) -> WorkerState {
        match &self.0.kind {
            WorkerKind::Primordial => WorkerState::Running,
            WorkerKind::Spawned { state } => state.get(),
        }
    }

    /// Transition from `New` to `Running`. Returns `true` if the state
    /// changed. Always returns `false` for the primordial worker, which is
    /// already `Running`.
    pub fn start(self) -> bool {
        let WorkerKind::Spawned { state } = &self.0.kind else {
            return false;
        };

        let mut changed = false;

        state.update(|s| match s {
            WorkerState::New => {
                changed = true;
                WorkerState::Running
            }
            s => s,
        });

        changed
    }

    /// Attempt to transition `Running` → `Terminated`. Returns `true` if the state
    /// changed. Matches Flash: a `New` worker that was never started cannot
    /// be terminated, and the primordial worker throws `Error #3732`.
    pub fn terminate(self, activation: &mut Activation<'_, 'gc>) -> Result<bool, Error<'gc>> {
        let state = match &self.0.kind {
            WorkerKind::Primordial => return Err(make_error_3732(activation)),
            WorkerKind::Spawned { state } => state,
        };

        let mut changed = false;

        state.update(|s| match s {
            WorkerState::Running => {
                changed = true;
                WorkerState::Terminated
            }
            s => s,
        });

        Ok(changed)
    }
}
