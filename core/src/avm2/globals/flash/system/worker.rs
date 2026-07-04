//! `flash.system.Worker` native methods

use crate::avm2::Avm2;
use crate::avm2::Avm2StrRepresentable;
use crate::avm2::Error;
use crate::avm2::activation::Activation;
use crate::avm2::bytearray::ByteArrayStorage;
use crate::avm2::object::{
    ByteArrayObject, ConditionObject, EventObject, MessageChannelObject, MutexObject, WorkerObject,
};
use crate::avm2::parameters::ParametersExt;
use crate::avm2::value::Value;
use crate::avm2::worker_shared::SharedValue;
use crate::avm2::ValueEnum;
use crate::avm2_stub_method;
use crate::string::AvmString;
use crate::worker_runtime::{WorkerConfig, run_worker};
use std::sync::atomic::Ordering;

fn this_worker<'gc>(this: Value<'gc>) -> WorkerObject<'gc> {
    this.as_object()
        .and_then(|o| o.as_worker_object())
        .expect("Worker native called on non-Worker object")
}

/// Implements `Worker.state`
pub fn get_state<'gc>(
    activation: &mut Activation<'_, 'gc>,
    this: Value<'gc>,
    _args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    let state = this_worker(this).state().as_avm2_str(activation);

    Ok(state.into())
}

/// Implements `Worker.isPrimordial`
pub fn get_is_primordial<'gc>(
    _activation: &mut Activation<'_, 'gc>,
    this: Value<'gc>,
    _args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    Ok(this_worker(this).is_primordial().into())
}

/// Implements `Worker.createMessageChannel`
pub fn create_message_channel<'gc>(
    activation: &mut Activation<'_, 'gc>,
    _this: Value<'gc>,
    args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    avm2_stub_method!(activation, "flash.system.Worker", "createMessageChannel");

    let _receiver = args.get_object(activation, 0, "receiver")?;

    let message_channel = MessageChannelObject::new(activation);

    Ok(message_channel.into())
}

/// Short label for tracing a shared value's kind.
fn shared_kind(v: &SharedValue) -> &'static str {
    match v {
        SharedValue::Undefined => "undefined",
        SharedValue::Null => "null",
        SharedValue::Bool(_) => "bool",
        SharedValue::Int(_) => "int",
        SharedValue::Number(_) => "number",
        SharedValue::Str(_) => "string",
        SharedValue::ByteArrayCopy(_) => "byteArrayCopy",
        SharedValue::ByteBuffer(_) => "shareableByteArray",
        SharedValue::Mutex(_) => "mutex",
        SharedValue::Condition(_) => "condition",
        SharedValue::MessageChannel(_) => "messageChannel",
    }
}

/// Convert an AS3 value into the `Send` form stored for cross-thread sharing.
/// Returns `None` for values that cannot currently cross the boundary.
pub(crate) fn value_to_shared(value: Value<'_>) -> Option<SharedValue> {
    match value.unpack() {
        ValueEnum::Undefined => Some(SharedValue::Undefined),
        ValueEnum::Null => Some(SharedValue::Null),
        ValueEnum::Bool(b) => Some(SharedValue::Bool(b)),
        ValueEnum::Integer(i) => Some(SharedValue::Int(i)),
        ValueEnum::Number(n) => Some(SharedValue::Number(n)),
        ValueEnum::String(s) => Some(SharedValue::Str(s.to_string())),
        ValueEnum::Object(o) => {
            if let Some(ba) = o.as_bytearray() {
                // A `shareable` ByteArray crosses by reference; otherwise it is
                // copied by value.
                match ba.shared_buffer() {
                    Some(buffer) => Some(SharedValue::ByteBuffer(buffer)),
                    None => Some(SharedValue::ByteArrayCopy(ba.bytes().to_vec())),
                }
            } else if let Some(mutex) = o.as_mutex_object() {
                Some(SharedValue::Mutex(mutex.shared_mutex()))
            } else if let Some(cond) = o.as_condition_object() {
                cond.shared_condition().map(SharedValue::Condition)
            } else if let Some(mc) = o.as_message_channel_object() {
                Some(SharedValue::MessageChannel(mc.channel()))
            } else {
                None
            }
        }
    }
}

/// Reconstruct an AS3 value in the current arena from a `Send` shared value.
pub(crate) fn shared_to_value<'gc>(
    activation: &mut Activation<'_, 'gc>,
    shared: SharedValue,
) -> Value<'gc> {
    match shared {
        SharedValue::Undefined => Value::Undefined,
        SharedValue::Null => Value::Null,
        SharedValue::Bool(b) => b.into(),
        SharedValue::Int(i) => i.into(),
        SharedValue::Number(n) => n.into(),
        SharedValue::Str(s) => AvmString::new_utf8(activation.gc(), &s).into(),
        SharedValue::ByteArrayCopy(bytes) => {
            let storage = ByteArrayStorage::from_vec(activation.context, bytes);
            ByteArrayObject::from_storage(activation.context, storage).into()
        }
        SharedValue::ByteBuffer(buffer) => {
            let mut storage = ByteArrayStorage::new(activation.context);
            storage.attach_shared(buffer);
            ByteArrayObject::from_storage(activation.context, storage).into()
        }
        SharedValue::Mutex(mutex) => MutexObject::from_shared(activation, mutex).into(),
        SharedValue::Condition(cond) => ConditionObject::from_shared(activation, cond).into(),
        SharedValue::MessageChannel(channel) => {
            MessageChannelObject::from_shared(activation, channel).into()
        }
    }
}

/// Implements `Worker.setSharedProperty`
pub fn set_shared_property<'gc>(
    activation: &mut Activation<'_, 'gc>,
    this: Value<'gc>,
    args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    let key = args.get_string(activation, 0);
    let value = args.get_value(1);
    let worker = this_worker(this);

    // Same-arena store (keeps object identity for the owning worker).
    worker.set_shared_property(activation.gc(), key, value);

    // `Send` mirror, so a spawned worker thread can read the value.
    if let Some(shared) = value_to_shared(value) {
        // Skip the thousands of `flascc.sect.*` linker symbols to keep the log
        // (and its formatting cost) sane.
        if !key.to_utf8_lossy().starts_with("flascc.sect") {
            tracing::info!(
                "setSharedProperty(t{:?}, worker {:#x}) {key} = {}",
                std::thread::current().id(),
                worker.id(),
                shared_kind(&shared),
            );
        }
        worker
            .shared_send()
            .lock()
            .expect("shared property store poisoned")
            .insert(key.to_string(), shared);
    }

    Ok(Value::Undefined)
}

/// Implements `Worker.getSharedProperty`
pub fn get_shared_property<'gc>(
    activation: &mut Activation<'_, 'gc>,
    this: Value<'gc>,
    args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    let key = args.get_string(activation, 0);
    let worker = this_worker(this);

    // Prefer the same-arena value (preserves object identity for the owner).
    let local = worker.get_shared_property(key);
    if !matches!(local.unpack(), ValueEnum::Undefined) {
        return Ok(local);
    }

    // Otherwise reconstruct from the `Send` store (spawned-worker side). Look up
    // by borrowed `&str` to avoid allocating a `String` for every lookup — the
    // FlasCC linker probes thousands of long C++ symbol keys.
    let key_str = key.to_utf8_lossy();
    // Milestone marker: the worker branch (`AlcWorkerSprite.run`) reads these,
    // so seeing them on a worker thread means it's about to `callI(thread_entry)`.
    if key_str.contains("thread_entry") || key_str.contains("thread_args") {
        tracing::info!(
            "worker reached run(): getSharedProperty(t{:?}) {key}",
            std::thread::current().id()
        );
    }
    let shared = worker
        .shared_send()
        .lock()
        .expect("shared property store poisoned")
        .get(key_str.as_ref())
        .cloned();

    Ok(match shared {
        Some(shared) => shared_to_value(activation, shared),
        None => Value::Undefined,
    })
}

/// Implements `Worker.isSupported`
pub fn get_is_supported<'gc>(
    activation: &mut Activation<'_, 'gc>,
    _this: Value<'gc>,
    _args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    Ok(activation.context.worker_host.is_supported().into())
}

/// Implements `Worker.start`
pub fn start<'gc>(
    activation: &mut Activation<'_, 'gc>,
    this: Value<'gc>,
    _args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    let worker = this_worker(this);

    if worker.start() {
        // Launch the worker's headless runtime on a background thread. The SWF
        // re-runs there and takes its worker branch (e.g. FlasCC's
        // `AlcWorkerSprite`).
        // Run the SWF whose code started the worker (e.g. FlasCC's game SWF,
        // which may be a `Loader.loadBytes` child), not the outermost player
        // root — that can be a tiny loader stub.
        let movie = (*activation.caller_movie_or_root()).clone();
        tracing::info!(
            "Worker.start: worker {:#x}, movie {} bytes",
            worker.id(),
            movie.data().len()
        );
        let config = WorkerConfig {
            movie,
            worker_id: worker.id(),
            shared_properties: worker.shared_send(),
            terminate: worker.terminate_flag(),
        };
        activation
            .context
            .worker_host
            .spawn(Box::new(move || run_worker(config)));

        let event = EventObject::bare_default_event(activation.context, "workerState");
        Avm2::dispatch_event(activation.context, event, worker.into());
    }

    Ok(Value::Undefined)
}

/// Implements `Worker.terminate`
pub fn terminate<'gc>(
    activation: &mut Activation<'_, 'gc>,
    this: Value<'gc>,
    _args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    avm2_stub_method!(activation, "flash.system.Worker", "terminate");

    let worker = this_worker(this);
    let changed = worker.terminate(activation)?;

    if changed {
        // Ask the worker's runtime thread to stop.
        worker.terminate_flag().store(true, Ordering::Relaxed);


        let event = EventObject::bare_default_event(activation.context, "workerState");

        Avm2::dispatch_event(activation.context, event, worker.into());
    }

    Ok(changed.into())
}

/// Implements `Worker.current`
pub fn get_current<'gc>(
    activation: &mut Activation<'_, 'gc>,
    _this: Value<'gc>,
    _args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    Ok(activation.avm2().current_worker().into())
}
