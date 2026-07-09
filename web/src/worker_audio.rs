//! Audio backend for the player-in-worker path.
//!
//! A Web Worker has no `AudioContext`, so audio is split across the two threads.
//! The worker owns the [`AudioMixer`]: every AVM-facing call (register a sound,
//! start/stop an instance, set volume) runs there, synchronously, like any other
//! backend. Playback — the part that needs an `AudioContext` — runs on the main
//! thread via an [`AudioMixerProxy`], which shares the mixer's instance list,
//! master volume and output buffers through `Arc`s. Because the worker and the
//! main thread share one `WebAssembly.Memory`, those `Arc`s are valid on both
//! sides, so no samples cross the thread boundary: the main thread pulls mixed
//! audio straight out of the shared instance list.
//!
//! Wiring (all set up on the main thread in `ruffle_prepare_worker_player`):
//! [`create_worker_audio`] builds the mixer, keeps its proxy to drive a
//! [`WorkerAudioDriver`] (the `AudioContext` playback loop), and hands the mixer
//! to `WorkerInit` for the worker to wrap in [`WebWorkerAudioBackend`].

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use ruffle_core::backend::audio::{
    AudioBackend, AudioMixer, AudioMixerProxy, DecodeError, RegisterError, SoundHandle,
    SoundInstanceHandle, SoundStreamInfo, SoundTransform, swf,
};
use ruffle_core::impl_audio_mixer_backend;
use ruffle_web_common::JsResult;
use wasm_bindgen::prelude::*;
use web_sys::{AudioContext, AudioContextOptions, AudioScheduledSourceNode};

/// SWF audio tops out at 44.1 kHz, so pin both the mixer and the `AudioContext`
/// there (the same reasoning as the main-thread `WebAudioBackend`).
const SAMPLE_RATE: u32 = 44_100;

/// Fixed playback buffer, in frames (L/R pairs). ~46 ms at 44.1 kHz — a balance
/// between latency and refill headroom. (The main-thread backend adapts this at
/// runtime; the worker driver keeps it fixed for simplicity, revisit if underruns
/// show up.)
const BUFFER_FRAMES: u32 = 2048;

// ---- worker side -----------------------------------------------------------

/// The worker's [`AudioBackend`]: owns the [`AudioMixer`]. All the AVM-facing
/// work happens here; the main thread only pulls samples through the proxy.
pub struct WebWorkerAudioBackend {
    mixer: AudioMixer,
}

impl WebWorkerAudioBackend {
    /// Wraps a mixer created on the main thread (so its proxy can start driving
    /// the `AudioContext` before the worker even exists).
    pub fn from_mixer(mixer: AudioMixer) -> Self {
        Self { mixer }
    }
}

impl AudioBackend for WebWorkerAudioBackend {
    impl_audio_mixer_backend!(mixer);

    // The `AudioContext` lives on the main thread; resuming/suspending it on a
    // user gesture is handled there (see [`WorkerAudioDriver`]). The mixer keeps
    // producing samples regardless — silence when nothing is playing — so these
    // are no-ops here.
    fn play(&mut self) {}
    fn pause(&mut self) {}
}

// ---- main-thread side ------------------------------------------------------

/// Creates the mixer (main thread) plus a driver that plays its output through an
/// `AudioContext`. Hand the returned [`AudioMixer`] to the worker (via
/// `WorkerInit`) and keep the [`WorkerAudioDriver`] alive for as long as the
/// player runs.
pub fn create_worker_audio() -> Result<(AudioMixer, WorkerAudioDriver), JsError> {
    let mixer = AudioMixer::new(2, SAMPLE_RATE);
    let driver = WorkerAudioDriver::new(&mixer)?;
    Ok((mixer, driver))
}

/// Owns the `AudioContext` and the ping-pong playback buffers on the main thread,
/// each pulling from the shared mixer via an [`AudioMixerProxy`].
pub struct WorkerAudioDriver {
    context: AudioContext,
    // The buffers reschedule themselves via their `onended` closures; we only
    // keep them alive (and drop them, unhooking the closures) with the driver.
    _buffers: Vec<Rc<RefCell<DriverBuffer>>>,
}

impl WorkerAudioDriver {
    fn new(mixer: &AudioMixer) -> Result<Self, JsError> {
        // Pin to 44.1 kHz to match the mixer (see `SAMPLE_RATE`). Per Web Audio
        // §1.2.1 the UA resamples to the device rate on output.
        let opts = AudioContextOptions::new();
        opts.set_sample_rate(SAMPLE_RATE as f32);
        let context = AudioContext::new_with_context_options(&opts).into_js_result()?;

        // Shared playout cursor (seconds on the context clock) the two buffers
        // hand back and forth so they schedule back-to-back without gaps.
        let time = Rc::new(Cell::new(0.0));

        let mut buffers = Vec::with_capacity(2);
        for _ in 0..2 {
            let buffer = DriverBuffer::new(&context, mixer.proxy(), time.clone())?;
            buffer.borrow_mut().play()?;
            buffers.push(buffer);
        }

        Ok(Self {
            context,
            _buffers: buffers,
        })
    }

    /// A clone of the underlying `AudioContext` (a JS handle), for wiring a
    /// resume-on-gesture listener without moving the driver. Browsers start the
    /// context suspended until a user gesture, so the caller resumes it there.
    pub fn audio_context(&self) -> AudioContext {
        self.context.clone()
    }
}

impl Drop for WorkerAudioDriver {
    fn drop(&mut self) {
        let _ = self.context.close();
    }
}

/// One of the two ping-pong playback buffers. On `play()` it mixes a fresh chunk
/// from the proxy, deinterleaves it into an `AudioBuffer`, schedules it, and
/// arms `onended` to refill itself.
struct DriverBuffer {
    context: AudioContext,
    proxy: AudioMixerProxy,
    js_buffer: web_sys::AudioBuffer,
    /// Mixer output, 2-channel interleaved (`BUFFER_FRAMES` frames).
    interleaved: Vec<f32>,
    time: Rc<Cell<f64>>,
    audio_node: Option<web_sys::AudioBufferSourceNode>,
    on_ended: Closure<dyn FnMut()>,
}

impl DriverBuffer {
    fn new(
        context: &AudioContext,
        proxy: AudioMixerProxy,
        time: Rc<Cell<f64>>,
    ) -> Result<Rc<RefCell<Self>>, JsError> {
        let js_buffer = context
            .create_buffer(2, BUFFER_FRAMES, SAMPLE_RATE as f32)
            .into_js_result()?;
        let buffer = Rc::new(RefCell::new(Self {
            context: context.clone(),
            proxy,
            js_buffer,
            interleaved: vec![0.0; 2 * BUFFER_FRAMES as usize],
            time,
            audio_node: None,
            on_ended: Closure::new(|| {}),
        }));

        // Refill-and-reschedule when this buffer finishes playing.
        let handle = buffer.clone();
        buffer.borrow_mut().on_ended = Closure::new(move || {
            let _ = handle.borrow_mut().play();
        });

        Ok(buffer)
    }

    fn play(&mut self) -> Result<(), JsError> {
        // Pull mixed audio (interleaved L/R) from the shared instance list.
        self.proxy.mix(&mut self.interleaved);

        // Copy into the `AudioBuffer`. We can't use `AudioBuffer.copyToChannel`
        // here: in the threaded build the wasm memory is a `SharedArrayBuffer`,
        // and `copyToChannel` rejects a view backed by one. The JS helper instead
        // writes element-wise into the buffer's own (non-shared) channel arrays —
        // *reading* from our shared-memory view is allowed, only the argument
        // can't be shared. Deinterleaving happens on the JS side.
        copy_to_audio_buffer_interleaved(&self.js_buffer, &self.interleaved);

        // A fresh source node per buffer (they're single-use).
        let node = self.context.create_buffer_source().into_js_result()?;
        node.set_buffer(Some(&self.js_buffer));
        node.connect_with_audio_node(&self.context.destination())
            .into_js_result()?;
        let scheduled: &AudioScheduledSourceNode = &node;
        scheduled.set_onended(Some(self.on_ended.as_ref().unchecked_ref()));

        // Never schedule in the past — an underrun would otherwise stack nodes at
        // t=0 and play them all at once.
        let start = f64::max(self.time.get(), self.context.current_time());
        node.start_with_when(start).into_js_result()?;
        self.time
            .set(start + f64::from(BUFFER_FRAMES) / f64::from(SAMPLE_RATE));

        self.audio_node = Some(node);
        Ok(())
    }
}

impl Drop for DriverBuffer {
    fn drop(&mut self) {
        if let Some(node) = self.audio_node.take() {
            // Detach the closure so a late `onended` can't call into a dropped buffer.
            let scheduled: &AudioScheduledSourceNode = &node;
            scheduled.set_onended(None);
        }
    }
}

#[wasm_bindgen(raw_module = "./ruffle-imports")]
unsafe extern "C" {
    // Copies interleaved stereo data into an `AudioBuffer` element-wise on the JS
    // side (shared-memory-safe; see the call site). Same import the main-thread
    // `WebAudioBackend` uses.
    #[wasm_bindgen(js_name = "copyToAudioBufferInterleaved")]
    fn copy_to_audio_buffer_interleaved(
        audio_buffer: &web_sys::AudioBuffer,
        interleaved_data: &[f32],
    );
}
