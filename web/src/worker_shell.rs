//! Main-thread shell for the player-in-worker path (see [`crate::worker_player`]).
//!
//! This is the thin half that stays on the UI thread. It never runs the player;
//! it builds the shared [`WorkerBridge`] + [`WorkerInit`], forwards DOM input +
//! viewport changes into the bridge, and fulfils the primordial's nested-worker
//! spawns. The *worker itself* is created in JS (`start-worker-player.ts`), so
//! the bundler (Vite) can package it — this returns the [`WorkerInit`] pointer
//! for JS to hand across; JS also does `transferControlToOffscreen` and the
//! `postMessage`.
//!
//! Deliberately a *separate* entry rather than a rewrite of the normal
//! `RuffleInstance`, so the standard main-thread player path is untouched.

use std::sync::Arc;

use ruffle_core::events::{MouseButton, MouseWheelDelta, PlayerEvent};
use ruffle_core::tag_utils::SwfMovie;
use ruffle_render::backend::ViewportDimensions;
use ruffle_render::quality::StageQuality;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{Event, HtmlCanvasElement, KeyboardEvent, PointerEvent, WheelEvent};

use crate::input::web_input_to_ruffle_key_descriptor;
use crate::worker_bridge::WorkerBridge;
use crate::worker_player::WorkerInit;

/// Live handle to a worker-hosted player. Dropping it (or calling
/// [`Self::destroy`]) asks the worker to stop. JS reads [`Self::init_ptr`] to
/// hand the [`WorkerInit`] to the worker. (Input listeners are `forget()`-leaked,
/// not held here, so dropping the handle can't leave dangling callbacks.)
#[wasm_bindgen]
pub struct WorkerPlayerHandle {
    bridge: Arc<WorkerBridge>,
    device_pixel_ratio: f64,
    init_ptr: u32,
}

/// Gracefully terminates every spawned `flash.system.Worker` (the avmplus
/// interrupt model): sets each worker's terminate flag and wakes anything parked
/// in `Condition.wait` / `Mutex.lock`, so each unwinds its AS3/FlasCC stack and
/// exits its run loop — releasing the shared allocator/mutex locks on the way
/// out. JS calls this on SWF change / teardown *before* any force `terminate()`,
/// so a hard kill never lands on a thread mid-`malloc`.
#[wasm_bindgen]
pub fn ruffle_terminate_all_workers() {
    ruffle_core::avm2::worker_shared::terminate_all_workers();
}

/// The compiled wasm module, for JS to hand to the worker so it `initSync`s over
/// the *same* module + memory and shares this thread's linear memory.
#[wasm_bindgen]
pub fn ruffle_wasm_module() -> JsValue {
    wasm_bindgen::module()
}

/// The shared linear memory (see [`ruffle_wasm_module`]).
#[wasm_bindgen]
pub fn ruffle_wasm_memory() -> JsValue {
    wasm_bindgen::memory()
}

/// Prepares a worker-hosted player for `canvas`: builds the shared bridge +
/// `WorkerInit`, wires DOM input into the bridge, and starts the main-thread
/// spawn pump. Does NOT create the worker or transfer the canvas — JS does that
/// (so the bundler can package the worker) using [`WorkerPlayerHandle::init_ptr`].
#[wasm_bindgen]
pub fn ruffle_prepare_worker_player(
    canvas: HtmlCanvasElement,
    swf_data: &[u8],
    movie_url: String,
) -> Result<WorkerPlayerHandle, JsValue> {
    let window = web_sys::window().ok_or_else(|| JsValue::from_str("no window"))?;
    let device_pixel_ratio = window.device_pixel_ratio();

    // Device-pixel viewport from the canvas' CSS size.
    let viewport = ViewportDimensions {
        width: (f64::from(canvas.client_width()) * device_pixel_ratio) as u32,
        height: (f64::from(canvas.client_height()) * device_pixel_ratio) as u32,
        scale_factor: device_pixel_ratio,
    };

    let movie = SwfMovie::from_data(swf_data, movie_url, false, None)
        .map_err(|e| JsValue::from_str(&format!("bad SWF: {e}")))?;

    // Shared control block: the main thread keeps `bridge`, the worker gets a clone.
    let bridge = WorkerBridge::new(viewport);
    let init = Box::new(WorkerInit {
        bridge: bridge.clone(),
        movie,
        viewport,
        // Do NOT force a frame rate: forcing sets `forced_frame_rate`, which makes
        // the game's own `stage.frameRate = X` a no-op — so the game ran (and
        // rendered/pumped) at 60 FPS instead of its intended rate. `None` lets the
        // document class set the real rate at startup (frame 1 runs at t=0), and the
        // worker's frame loop paces to `Player::frame_rate()` so it follows it.
        frame_rate: None,
        quality: StageQuality::High,
    });
    let init_ptr = init.into_shared_ptr();

    // Forward DOM input into the bridge. Coordinates are in device pixels to
    // match the worker's viewport. (The canvas element still receives DOM events
    // after JS transfers its rendering control to the worker.) The listener
    // closures are `forget()`-leaked, not stored: the handle may be dropped while
    // the movie keeps running, and a dropped-but-still-registered closure would
    // crash ("closure invoked after being dropped") on the next event. Pushing to
    // a shut-down bridge afterwards is harmless.
    let target = canvas.clone();

    // Pointer move / down / up.
    for (name, kind) in [("pointermove", 0u8), ("pointerdown", 1), ("pointerup", 2)] {
        let bridge = bridge.clone();
        let cb = Closure::wrap(Box::new(move |event: Event| {
            let Ok(e) = event.dyn_into::<PointerEvent>() else {
                return;
            };
            let x = e.offset_x() as f64 * device_pixel_ratio;
            let y = e.offset_y() as f64 * device_pixel_ratio;
            let player_event = match kind {
                0 => PlayerEvent::MouseMove { x, y },
                _ => {
                    let button = match e.button() {
                        0 => MouseButton::Left,
                        1 => MouseButton::Middle,
                        2 => MouseButton::Right,
                        _ => MouseButton::Unknown,
                    };
                    if kind == 1 {
                        PlayerEvent::MouseDown {
                            x,
                            y,
                            button,
                            index: None,
                        }
                    } else {
                        PlayerEvent::MouseUp { x, y, button }
                    }
                }
            };
            bridge.push_event(player_event);
        }) as Box<dyn FnMut(Event)>);
        target
            .add_event_listener_with_callback(name, cb.as_ref().unchecked_ref())?;
        cb.forget();
    }

    // Wheel.
    {
        let bridge = bridge.clone();
        let cb = Closure::wrap(Box::new(move |event: Event| {
            let Ok(e) = event.dyn_into::<WheelEvent>() else {
                return;
            };
            bridge.push_event(PlayerEvent::MouseWheel {
                delta: MouseWheelDelta::Pixels(-e.delta_y()),
            });
        }) as Box<dyn FnMut(Event)>);
        target.add_event_listener_with_callback("wheel", cb.as_ref().unchecked_ref())?;
        cb.forget();
    }

    // Keyboard (on the window, so focus isn't required on the canvas).
    for (name, down) in [("keydown", true), ("keyup", false)] {
        let bridge = bridge.clone();
        let cb = Closure::wrap(Box::new(move |event: Event| {
            let Ok(e) = event.dyn_into::<KeyboardEvent>() else {
                return;
            };
            let key = web_input_to_ruffle_key_descriptor(&e);
            bridge.push_event(if down {
                PlayerEvent::KeyDown { key }
            } else {
                PlayerEvent::KeyUp { key }
            });
        }) as Box<dyn FnMut(Event)>);
        window.add_event_listener_with_callback(name, cb.as_ref().unchecked_ref())?;
        cb.forget();
    }

    Ok(WorkerPlayerHandle {
        bridge,
        device_pixel_ratio,
        init_ptr,
    })
}

/// Runs a queued `flash.system.Worker` entry (`run_worker`) on the calling
/// worker thread. JS calls this from a fresh (bundler-packaged) worker after
/// `initSync`ing over the shared module+memory — because `wasm_thread`'s own
/// worker bootstrap does not run in the bundled build, but our Vite-packaged
/// workers do.
///
/// # Safety
/// `ptr` must be a `Box<Box<dyn FnOnce() + Send>>` produced by
/// [`BridgeWorkerHost`](crate::worker_bridge::BridgeWorkerHost) and drained via
/// [`WorkerPlayerHandle::take_spawn_requests`], run exactly once.
#[wasm_bindgen]
pub fn ruffle_run_worker_entry(ptr: u32) {
    // SAFETY: delegated to the caller (see above).
    let entry: Box<Box<dyn FnOnce() + Send + 'static>> =
        unsafe { Box::from_raw(ptr as usize as *mut Box<dyn FnOnce() + Send + 'static>) };
    entry();
}

#[wasm_bindgen]
impl WorkerPlayerHandle {
    /// The [`WorkerInit`] pointer for JS to `postMessage` to the worker.
    pub fn init_ptr(&self) -> u32 {
        self.init_ptr
    }

    /// Drains pending `Worker.start()` spawn requests (pointers to leaked
    /// `run_worker` closures). JS polls this and launches a bundler-packaged
    /// worker per pointer (running it via [`ruffle_run_worker_entry`]).
    pub fn take_spawn_requests(&self) -> Vec<u32> {
        let mut requests = Vec::new();
        self.bridge.drain_spawn_requests(&mut requests);
        requests.into_iter().map(|p| p as u32).collect()
    }

    /// Pushes a new viewport (call from a ResizeObserver / resize handler). Width
    /// and height are CSS pixels; they are scaled by the device pixel ratio.
    pub fn set_viewport(&self, css_width: f64, css_height: f64) {
        self.bridge.set_viewport(ViewportDimensions {
            width: (css_width * self.device_pixel_ratio) as u32,
            height: (css_height * self.device_pixel_ratio) as u32,
            scale_factor: self.device_pixel_ratio,
        });
    }

    /// Asks the worker's run loop to stop.
    pub fn destroy(&self) {
        self.bridge.request_shutdown();
    }
}
