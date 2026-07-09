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

use std::cell::Cell;
use std::sync::Arc;

use ruffle_core::events::{MouseButton, MouseWheelDelta, PlayerEvent, TextControlCode};
use ruffle_core::tag_utils::SwfMovie;
use ruffle_render::backend::ViewportDimensions;
use ruffle_render::quality::StageQuality;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{Event, HtmlCanvasElement, KeyboardEvent, PointerEvent, WheelEvent};

use crate::input::{web_input_to_ruffle_key_descriptor, web_to_ruffle_text_control};
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
    /// The page canvas, for applying the worker's cursor / fullscreen requests
    /// (the worker owns neither the element nor the DOM). See [`Self::service_ui`].
    canvas: HtmlCanvasElement,
    /// Last cursor code applied to the canvas, so we only touch CSS on a change.
    last_cursor: Cell<u8>,
    /// Main-thread audio playback (`AudioContext` + buffers) driven off the
    /// worker mixer's proxy. Held only to keep it alive for the player's lifetime.
    _audio_driver: crate::worker_audio::WorkerAudioDriver,
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

    // Audio: the mixer goes to the worker (all AVM-facing sound calls), while its
    // proxy drives an `AudioContext` here on the main thread — a worker has none.
    let (audio_mixer, audio_driver) = crate::worker_audio::create_worker_audio()?;

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
        // Snapshot `localStorage` here (main thread) — the worker can't reach it.
        storage: read_local_storage_snapshot(),
        // No credentialed hosts by default (matches the config default). The
        // worker fetches `SameOrigin`, like the main-thread backend does for any
        // host outside its `credentialAllowList`; thread the embed's list here if
        // a game needs cross-origin cookies.
        credential_allow_list: Vec::new(),
        audio_mixer,
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
    //
    // Keydown must also drive editable `TextField`s: those insert characters from
    // `TextInput` and handle editing keys (backspace/arrows/…) via `TextControl`,
    // NOT from `KeyDown` — so forwarding `KeyDown` alone leaves text boxes dead.
    // Mirror the normal player (see `lib.rs`): emit `KeyDown`, then a `TextControl`
    // for a recognised editing key, else a `TextInput` for the typed character.
    // `Paste` is the exception: the clipboard text isn't available until the DOM
    // `paste` event fires, so we DON'T emit it here (and don't `preventDefault`),
    // letting the `paste` listener below fill the buffer and then drive it.
    {
        let bridge = bridge.clone();
        let cb = Closure::wrap(Box::new(move |event: Event| {
            let Ok(e) = event.dyn_into::<KeyboardEvent>() else {
                return;
            };
            let key = web_input_to_ruffle_key_descriptor(&e);
            bridge.push_event(PlayerEvent::KeyDown { key });

            let is_ctrl_cmd = e.ctrl_key() || e.meta_key();
            if let Some(code) =
                web_to_ruffle_text_control(&e.key(), &e.code(), is_ctrl_cmd, e.shift_key())
            {
                if code != TextControlCode::Paste {
                    bridge.push_event(PlayerEvent::TextControl { code });
                }
            } else if let Some(codepoint) = key.logical_key.character() {
                bridge.push_event(PlayerEvent::TextInput { codepoint });
            }
        }) as Box<dyn FnMut(Event)>);
        window.add_event_listener_with_callback("keydown", cb.as_ref().unchecked_ref())?;
        cb.forget();
    }

    // Clipboard paste: the DOM `paste` event is the only place the clipboard text
    // is available. Stash it on the bridge (the worker's UI backend serves
    // `clipboard_content()` from it), then push the `Paste` `TextControl` so the
    // AVM text field inserts it — order matters, buffer before the event.
    {
        let bridge = bridge.clone();
        let cb = Closure::wrap(Box::new(move |event: Event| {
            let Ok(e) = event.dyn_into::<web_sys::ClipboardEvent>() else {
                return;
            };
            let text = e
                .clipboard_data()
                .and_then(|d| d.get_data("text/plain").ok())
                .unwrap_or_default();
            bridge.set_clipboard(text);
            bridge.push_event(PlayerEvent::TextControl {
                code: TextControlCode::Paste,
            });
            e.prevent_default();
        }) as Box<dyn FnMut(Event)>);
        window.add_event_listener_with_callback("paste", cb.as_ref().unchecked_ref())?;
        cb.forget();
    }
    {
        let bridge = bridge.clone();
        let cb = Closure::wrap(Box::new(move |event: Event| {
            let Ok(e) = event.dyn_into::<KeyboardEvent>() else {
                return;
            };
            let key = web_input_to_ruffle_key_descriptor(&e);
            bridge.push_event(PlayerEvent::KeyUp { key });
        }) as Box<dyn FnMut(Event)>);
        window.add_event_listener_with_callback("keyup", cb.as_ref().unchecked_ref())?;
        cb.forget();
    }

    // Browsers start an `AudioContext` suspended until a user gesture. Resume it
    // on the first pointer/key event so game audio actually plays. These fire on
    // every event (resume is idempotent/cheap once running); leaked like the
    // input listeners above so a dropped handle leaves no dangling callback.
    {
        let ctx = audio_driver.audio_context();
        let resume = move || {
            // Guard on state: these listeners are leaked, so after a teardown one
            // can fire against a context a later start already `close()`d, and
            // `resume()` on a closed context throws an uncaught DOMException.
            if ctx.state() != web_sys::AudioContextState::Closed {
                let _ = ctx.resume();
            }
        };
        for event in ["pointerdown", "keydown"] {
            let ctx = resume.clone();
            let cb =
                Closure::wrap(Box::new(move |_: Event| ctx()) as Box<dyn FnMut(Event)>);
            window.add_event_listener_with_callback(event, cb.as_ref().unchecked_ref())?;
            cb.forget();
        }
    }

    Ok(WorkerPlayerHandle {
        bridge,
        device_pixel_ratio,
        init_ptr,
        canvas,
        last_cursor: Cell::new(crate::worker_ui::CURSOR_ARROW),
        // Keep the audio driver (AudioContext + playback buffers) alive for the
        // player's lifetime; dropping the handle tears it down.
        _audio_driver: audio_driver,
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

    /// Persists the worker's pending SharedObject writes to `localStorage` (call
    /// from the main-thread pump, alongside `take_spawn_requests`). The worker
    /// can't reach `localStorage`, so it queues writes over the bridge; this
    /// drains and applies them (base64 values, matching `LocalStorageBackend`).
    pub fn service_storage(&self) {
        let mut writes = Vec::new();
        self.bridge.drain_storage_writes(&mut writes);
        if writes.is_empty() {
            return;
        }
        let Some(storage) = web_sys::window().and_then(|w| w.local_storage().ok().flatten()) else {
            return;
        };
        for w in writes {
            match w.value {
                Some(v) => {
                    let _ = storage.set(&w.name, &v);
                }
                None => {
                    let _ = storage.delete(&w.name);
                }
            }
        }
    }

    /// Applies the worker's pending UI requests to the DOM (call from the
    /// main-thread pump, alongside `service_storage`): the mouse cursor and any
    /// fullscreen transition on the canvas, plus copy requests written to the
    /// system clipboard. A worker owns none of these, so it routes them here.
    pub fn service_ui(&self) {
        // Mouse cursor — only touch CSS when it actually changed.
        let cursor = self.bridge.cursor();
        if cursor != self.last_cursor.get() {
            self.last_cursor.set(cursor);
            let _ = self
                .canvas
                .style()
                .set_property("cursor", crate::worker_ui::cursor_css(cursor));
        }

        // Copy → system clipboard. Only the most recent copy matters; write it
        // via `navigator.clipboard` (async — swallow the result so a rejected
        // permission doesn't surface as an uncaught promise).
        let mut writes = Vec::new();
        self.bridge.drain_clipboard_writes(&mut writes);
        if let Some(text) = writes.pop() {
            if let Some(win) = web_sys::window() {
                let promise = win.navigator().clipboard().write_text(&text);
                wasm_bindgen_futures::spawn_local(async move {
                    let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
                });
            }
        }

        // Fullscreen transition on the canvas.
        if let Some(enter) = self.bridge.take_fullscreen_request() {
            if enter {
                let _ = self.canvas.request_fullscreen();
            } else if let Some(doc) = web_sys::window().and_then(|w| w.document()) {
                doc.exit_fullscreen();
            }
        }
    }
}

/// Reads all of `localStorage` (main thread) into a `name -> value` snapshot the
/// worker's storage backend serves reads from. Values are the raw stored strings
/// (base64), matching [`LocalStorageBackend`](crate::storage).
fn read_local_storage_snapshot() -> std::collections::HashMap<String, String> {
    let mut map = std::collections::HashMap::new();
    let Some(storage) = web_sys::window().and_then(|w| w.local_storage().ok().flatten()) else {
        return map;
    };
    let len = storage.length().unwrap_or(0);
    for i in 0..len {
        if let Ok(Some(key)) = storage.key(i)
            && let Ok(Some(val)) = storage.get_item(&key)
        {
            map.insert(key, val);
        }
    }
    map
}
