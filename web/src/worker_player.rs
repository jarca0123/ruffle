//! The primordial Ruffle player running on a dedicated worker (player-in-worker).
//!
//! Built for the browser OpenTTD case: the primordial cannot run on the main
//! thread because Ruffle's Flash sync primitives park on the wasm futex
//! (`Atomics.wait`), which traps on the UI thread. Here — on a real worker —
//! blocking is allowed, matching the desktop threading model.
//!
//! The main thread transfers an `OffscreenCanvas` (from
//! `canvas.transferControlToOffscreen()`) and a pointer to a [`WorkerInit`]; this
//! entry builds a [`Player`] rendering straight to that canvas via the GL
//! backend and runs the frame loop, taking input and viewport changes from the
//! shared [`WorkerBridge`]. DOM-bound backends (audio/ui/navigator) are stubbed
//! for now — a first "it renders" milestone; forwarding them to the main thread
//! is later work.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Duration;

use wasm_bindgen::closure::Closure;
use wasm_bindgen::JsCast;

use ruffle_core::limits::ExecutionLimit;
use ruffle_core::loader::LoadBehavior;
use ruffle_core::tag_utils::SwfMovie;
use ruffle_core::config::Letterbox;
use ruffle_core::{FloatDuration, PlayerBuilder, PlayerEvent, StageAlign, StageScaleMode};
use ruffle_render::backend::ViewportDimensions;
use ruffle_render::quality::StageQuality;
use ruffle_render_gl::GlRenderBackend;
use wasm_bindgen::prelude::*;
use web_sys::OffscreenCanvas;
use web_time::Instant;

use crate::worker_bridge::WorkerBridge;

/// Install the AVM2 JIT on the worker player (primordial + spawned Flash workers).
/// Flip to `false` to run the worker path on the pure interpreter — a bisection
/// switch for JIT-vs-core bugs (e.g. deciding whether a `#1506` domainMemory OOB
/// originates in JIT'd code or in CrossBridge startup itself).
const WORKER_PLAYER_JIT: bool = true;

/// One-shot init payload moved from the main thread into the worker via a leaked
/// `Box` pointer (a wasm32 address into the shared `WebAssembly.Memory`). Every
/// field is `Send` (no `Gc`, no JS objects — the `OffscreenCanvas` is transferred
/// separately over `postMessage`).
pub struct WorkerInit {
    /// The worker's handle on the shared control block (its own `Arc` clone).
    pub bridge: Arc<WorkerBridge>,
    /// The root movie, already loaded on the main thread.
    pub movie: SwfMovie,
    /// Initial viewport.
    pub viewport: ViewportDimensions,
    /// Frame rate override, if the embed set one.
    pub frame_rate: Option<f64>,
    /// Render quality.
    pub quality: StageQuality,
    /// Startup snapshot of `localStorage` (`name -> base64`), read on the main
    /// thread — the worker can't reach `localStorage`, so SharedObject reads are
    /// served from this and writes are pushed back over the bridge to persist.
    pub storage: std::collections::HashMap<String, String>,
    /// Hosts (`scheme://host`) allowed credentialed (`credentials: include`)
    /// fetches; mirrors the embed's `credentialAllowList`. Everything else is
    /// `SameOrigin` (see `WebWorkerNavigatorBackend`).
    pub credential_allow_list: Vec<String>,
    /// The audio mixer, created on the main thread so its proxy can drive the
    /// `AudioContext` there before the worker exists. The worker owns it (all the
    /// AVM-facing sound calls); see `worker_audio`.
    pub audio_mixer: ruffle_core::backend::audio::AudioMixer,
}

impl WorkerInit {
    /// Leaks `self` to a raw pointer to hand across `postMessage` as a `u32`.
    /// Reclaim exactly once with [`Self::from_shared_ptr`].
    pub fn into_shared_ptr(self: Box<Self>) -> u32 {
        Box::into_raw(self) as usize as u32
    }

    /// Reclaims the payload on the worker.
    ///
    /// # Safety
    /// `ptr` must come from [`Self::into_shared_ptr`] on the same shared memory,
    /// reclaimed exactly once.
    pub unsafe fn from_shared_ptr(ptr: u32) -> Box<Self> {
        // SAFETY: delegated to the caller (see above).
        unsafe { Box::from_raw(ptr as usize as *mut Self) }
    }
}

/// Which renderer the worker uses. `Some(true)` = wgpu over **WebGPU**
/// (needs `navigator.gpu` on the worker — Chromium; rendered BLACK on the
/// OffscreenCanvas in testing, kept for experiments), `Some(false)` = wgpu over
/// WebGL2 (what upstream Ruffle ships on the web), `None` = the glow GL backend.
///
/// NOTE: a canvas accepts only ONE context type ever, and the worker's
/// transferred `OffscreenCanvas` cannot be recreated — so once a wgpu attempt
/// touches it (`create_surface`), a glow-GL fallback on the same canvas is
/// impossible ("Couldn't create GL context"). The WebGPU availability check
/// therefore happens BEFORE the canvas is touched; a failure past that point is
/// terminal for the worker player.
const FORCE_WGPU: Option<bool> = None;

/// Worker entry point (called from the worker's JS bootstrap once the wasm
/// module is initialised over the shared memory). Runs until the bridge signals
/// shutdown; never returns to the JS event loop while ticking, which is exactly
/// why it must be a worker.
#[wasm_bindgen]
pub async fn ruffle_worker_player_entry(offscreen: OffscreenCanvas, init_ptr: u32) {
    // SAFETY: the main thread passed a pointer from `WorkerInit::into_shared_ptr`.
    let init = *unsafe { WorkerInit::from_shared_ptr(init_ptr) };
    if let Err(e) = run(offscreen, init).await {
        tracing::error!("primordial worker player failed: {e}");
    }
}

/// Builds the worker's renderer per [`FORCE_WGPU`]. The only safe fallback
/// point is BEFORE the canvas is touched (see [`FORCE_WGPU`]'s note): WebGPU
/// unavailability is detected up front and falls back to glow GL; a wgpu
/// failure after `create_surface` is terminal (the canvas is already claimed).
async fn create_worker_renderer(
    offscreen: &OffscreenCanvas,
    quality: StageQuality,
) -> Result<Box<dyn ruffle_render::backend::RenderBackend>, Box<dyn std::error::Error>> {
    if let Some(webgpu) = FORCE_WGPU {
        // WebGPU needs `navigator.gpu` on the worker scope — check WITHOUT
        // touching the canvas, so the glow fallback still can claim it.
        let webgpu_available = webgpu
            && js_sys::Reflect::get(&js_sys::global(), &"navigator".into())
                .ok()
                .map(|nav| {
                    js_sys::Reflect::has(&nav, &wasm_bindgen::JsValue::from_str("gpu"))
                        .unwrap_or(false)
                })
                .unwrap_or(false);
        if !webgpu || webgpu_available {
            let renderer = ruffle_render_wgpu::backend::WgpuRenderBackend::for_offscreen_canvas(
                offscreen.clone(),
                webgpu,
            )
            .await
            .map_err(|e| format!("worker wgpu renderer failed: {e} (canvas is already claimed — no GL fallback possible)"))?;
            tracing::info!(
                "worker renderer: wgpu ({})",
                if webgpu { "WebGPU" } else { "WebGL2" }
            );
            return Ok(Box::new(renderer));
        }
        tracing::warn!("worker WebGPU unavailable (no navigator.gpu); using glow GL");
    }
    tracing::info!("worker renderer: glow GL");
    Ok(Box::new(GlRenderBackend::new_for_webgl_offscreen(
        offscreen, false, quality,
    )?))
}

async fn run(offscreen: OffscreenCanvas, init: WorkerInit) -> Result<(), Box<dyn std::error::Error>> {
    // Ensure spawned `flash.system.Worker`s (which carry the bulk of the AVM2
    // work — see the profile) also get the JIT. `ruffle_core` can't reference the
    // JIT crate, so register the factory here, before the SWF spawns any worker.
    // NOTE: `WasmJit::shared_verified()` (the differential diagnostic) is
    // UNSOUND for event/timer-driven content: it re-runs call-bearing methods
    // 4x and rolls back only slots + domainMemory, so native side effects
    // (setTimeout, addEventListener, requestContext3D, …) multiply — a Starling
    // app spirals into exponential timer storms. Use it only on compute-heavy
    // (FlasCC-style) content.
    if WORKER_PLAYER_JIT {
        ruffle_core::worker_runtime::set_worker_jit_factory(|| ruffle_avm2_jit::WasmJit::shared());
    }

    let renderer = create_worker_renderer(&offscreen, init.quality).await?;

    // Base for relative URLs / credentialed fetch (the movie's own URL).
    let movie_url = init.movie.url().to_string();

    let player = PlayerBuilder::new()
        .with_boxed_renderer(renderer)
        // Networking is worker-native (`WorkerGlobalScope.fetch`); credentialed
        // fetches follow `credential_allow_list` (default `SameOrigin`, like the
        // main-thread backend). Audio owns the mixer here; the `AudioContext`
        // playback runs on the main thread off its proxy (see `worker_audio`).
        // Only UI stays stubbed — it needs a DOM bridge, unlike the rest.
        .with_audio(crate::worker_audio::WebWorkerAudioBackend::from_mixer(
            init.audio_mixer,
        ))
        .with_navigator(crate::worker_navigator::WebWorkerNavigatorBackend::new(
            Some(movie_url),
            init.credential_allow_list,
        ))
        .with_storage(Box::new(crate::worker_storage::WebWorkerStorageBackend::new(
            init.storage,
            init.bridge.clone(),
        )))
        // UI is bridged to the main thread (clipboard/cursor/fullscreen need the
        // DOM the worker lacks); see `worker_ui`.
        .with_ui(crate::worker_ui::WebWorkerUiBackend::new(init.bridge.clone()))
        // Flash workers (FlasCC's compute thread) can't spawn nested from here;
        // route each spawn to the main thread via the bridge.
        .with_worker_host(crate::worker_bridge::BridgeWorkerHost::new(
            init.bridge.clone(),
        ))
        .with_movie(init.movie)
        .with_autoplay(true)
        .with_viewport_dimensions(
            init.viewport.width,
            init.viewport.height,
            init.viewport.scale_factor,
        )
        .with_quality(init.quality)
        // Center + letterbox the movie in the viewport (matches the demo's
        // `letterbox: On`, forced scale/align); otherwise it sits at the top-left.
        .with_letterbox(Letterbox::On)
        .with_scale_mode(StageScaleMode::ShowAll, true)
        .with_align(StageAlign::default(), true)
        // Preload fully before running, or `run_frame` returns early each tick
        // and the document class never constructs.
        .with_load_behavior(LoadBehavior::Blocking)
        .with_frame_rate(init.frame_rate)
        // The primordial runs a long-lived FlasCC loop; disable the per-script
        // execution guard that would otherwise kill it.
        .with_max_execution_duration(Duration::from_secs(60 * 60 * 24 * 365))
        .build();

    // Install the JIT, consistent with the main-thread build.
    if WORKER_PLAYER_JIT {
        player
            .lock()
            .expect("worker player poisoned")
            .set_avm2_jit_backend(ruffle_avm2_jit::WasmJit::shared());
    }

    // Drive preload to completion up front.
    {
        let mut guard = player.lock().expect("worker player poisoned");
        guard.set_is_playing(true);
        let mut passes = 0u32;
        while !guard.preload(&mut ExecutionLimit::none()) {
            passes += 1;
            if passes > 100_000 {
                tracing::warn!("primordial worker: preload did not finish");
                break;
            }
        }
    }

    let bridge = init.bridge;

    // Drive the frame loop with `requestAnimationFrame` on the *worker's* global
    // scope. A dedicated worker that owns a transferred `OffscreenCanvas` gets a
    // real rAF tied to the **display's vsync** (Chrome/Firefox), so frames are
    // produced and presented in lockstep with the compositor. That kills the judder
    // of `setTimeout`/`sleep` pacing (which isn't vsync-aligned) *and* the
    // timer-throttle "2x faster while profiling" artifact (rAF is compositor-driven,
    // not timer-driven). The movie still advances at its own rate — `tick` gets the
    // real elapsed dt and gates enterFrame internally — so rendering at vsync just
    // presents the latest state smoothly. (rAF also presents the OffscreenCanvas, so
    // no separate event-loop yield is needed.)
    let global: web_sys::DedicatedWorkerGlobalScope = js_sys::global().unchecked_into();
    let raf_global = global.clone();

    let mut last_viewport_gen = bridge.viewport_generation();
    let mut last_tick = Instant::now();
    let mut events: Vec<PlayerEvent> = Vec::new();

    // Rolling per-frame cost breakdown (script `tick` vs `render`), logged every
    // ~2s — the aggregate profile can't show what a *peak* frame is made of,
    // which is exactly what decides a benchmark's fps threshold.
    let mut stat_frames = 0u32;
    let mut stat_tick_ms = 0f64;
    let mut stat_render_ms = 0f64;
    let mut stat_max_tick_ms = 0f64;
    let mut stat_max_render_ms = 0f64;
    let mut stat_window_start = Instant::now();

    let callback: Rc<RefCell<Option<Closure<dyn FnMut(f64)>>>> = Rc::new(RefCell::new(None));
    let reschedule = callback.clone();

    *callback.borrow_mut() = Some(Closure::wrap(Box::new(move |_timestamp: f64| {
        if bridge.is_shutdown() {
            tracing::info!("primordial worker player shut down");
            return;
        }

        events.clear();
        bridge.drain_events(&mut events);

        let vp_gen = bridge.viewport_generation();
        let viewport_changed = vp_gen != last_viewport_gen;
        last_viewport_gen = vp_gen;

        {
            let mut guard = player.lock().expect("worker player poisoned");
            for event in events.drain(..) {
                guard.handle_event(event);
            }
            if viewport_changed {
                guard.set_viewport_dimensions(bridge.viewport());
            }

            let now = Instant::now();
            let dt_ms = now.duration_since(last_tick).as_secs_f64() * 1000.0;
            last_tick = now;
            guard.tick(FloatDuration::from_millis(dt_ms));
            let after_tick = Instant::now();

            if guard.needs_render() || viewport_changed {
                guard.render();
            }
            let after_render = Instant::now();

            // Accumulate the frame-cost breakdown; report every ~2s.
            let tick_ms = after_tick.duration_since(now).as_secs_f64() * 1000.0;
            let render_ms = after_render.duration_since(after_tick).as_secs_f64() * 1000.0;
            stat_frames += 1;
            stat_tick_ms += tick_ms;
            stat_render_ms += render_ms;
            stat_max_tick_ms = stat_max_tick_ms.max(tick_ms);
            stat_max_render_ms = stat_max_render_ms.max(render_ms);
            let window_s = after_render.duration_since(stat_window_start).as_secs_f64();
            if window_s >= 2.0 {
                tracing::info!(
                    "frame cost: {:.0} fps | tick avg {:.2}ms max {:.2}ms | render avg {:.2}ms max {:.2}ms",
                    stat_frames as f64 / window_s,
                    stat_tick_ms / stat_frames as f64,
                    stat_max_tick_ms,
                    stat_render_ms / stat_frames as f64,
                    stat_max_render_ms,
                );
                stat_frames = 0;
                stat_tick_ms = 0.0;
                stat_render_ms = 0.0;
                stat_max_tick_ms = 0.0;
                stat_max_render_ms = 0.0;
                stat_window_start = after_render;
            }
        }

        // Schedule the next frame at the next vsync.
        if let Some(cb) = reschedule.borrow().as_ref() {
            let _ = raf_global.request_animation_frame(cb.as_ref().unchecked_ref());
        }
    }) as Box<dyn FnMut(f64)>));

    let _ = global.request_animation_frame(
        callback
            .borrow()
            .as_ref()
            .expect("callback set above")
            .as_ref()
            .unchecked_ref(),
    );
    // The event loop owns the callback from here on.
    std::mem::forget(callback);
    Ok(())
}
