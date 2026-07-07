//! Headless runtime for a spawned `flash.system.Worker` (real-threads "Path A").
//!
//! Runs on a background thread (native `std::thread` / web Web Worker via the
//! [`WorkerHost`](crate::worker_host::WorkerHost)). It builds a self-contained
//! headless [`Player`](crate::Player) for the worker's SWF and ticks it until
//! the worker is terminated. Everything shared with the creating thread crosses
//! by reference through the `Send` `Arc` handles in [`WorkerConfig`] — never a
//! `Gc`/arena object, which is `!Send`.

use crate::FloatDuration;
use crate::avm2::JitBackend;
use crate::avm2::worker_shared::SharedProperties;
use crate::limits::ExecutionLimit;
use crate::loader::LoadBehavior;
use crate::player::PlayerBuilder;
use crate::tag_utils::SwfMovie;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use web_time::Instant;
use crate::display_object::TDisplayObject;

/// Optional factory for the AVM2 JIT backend installed on spawned-worker players.
///
/// `ruffle_core` can't depend on a JIT crate (layering), so the embedder registers
/// one here via [`set_worker_jit_factory`]. A bare `fn` pointer is `Send + Sync`
/// and — since all worker threads share one wasm linear memory — valid on the
/// worker thread, where the factory is invoked to build a fresh (`!Send`) backend.
/// Without a factory, workers keep the default `NullJit`.
static WORKER_JIT_FACTORY: OnceLock<fn() -> Rc<dyn JitBackend>> = OnceLock::new();

/// Registers the JIT backend factory for spawned-worker players (see
/// [`WORKER_JIT_FACTORY`]). Call once at startup, before any worker spawns; later
/// calls are ignored.
pub fn set_worker_jit_factory(factory: fn() -> Rc<dyn JitBackend>) {
    let _ = WORKER_JIT_FACTORY.set(factory);
}

/// `Send` configuration handed across the thread boundary to a spawned worker.
pub struct WorkerConfig {
    /// The worker's movie (FlasCC workers re-run the primordial SWF). `SwfMovie`
    /// is `Clone + Send + 'static`, so the creating thread hands over a clone
    /// directly — no re-parsing.
    pub movie: SwfMovie,
    /// Stable id of this worker (matches its primordial-side `WorkerObject`).
    pub worker_id: u64,
    /// The worker's shared-property store, shared with the creating thread; how
    /// `Worker.current.getSharedProperty(...)` inside the worker sees the values
    /// the creator set on it.
    pub shared_properties: SharedProperties,
    /// Set by `Worker.terminate()` to ask this runtime to stop.
    pub terminate: Arc<AtomicBool>,
}

/// Worker thread entry point. Consumes the config and does not return until the
/// worker terminates.
pub fn run_worker(config: WorkerConfig) {
    // Bind this thread to its terminate flag so a blocked `Condition.wait` /
    // `Mutex.lock` reports interruption (avmplus interrupt model → the native
    // raises an uncatchable error that unwinds back here), and register it so a
    // page teardown can stop every worker at once.
    crate::avm2::worker_shared::register_worker_flag(&config.terminate);
    crate::avm2::worker_shared::bind_worker_terminate(config.terminate.clone());

    // Headless: default Null render/audio/ui/etc. backends. The worker branch of
    // the SWF (e.g. FlasCC's `AlcWorkerSprite`) performs no rendering.
    //
    // The per-script execution guard (which keeps a *UI* thread responsive by
    // killing long-running scripts) must be effectively disabled here: a worker
    // deliberately runs a long-lived loop (FlasCC's C `thread_entry`), which is
    // exactly what that guard would otherwise terminate.
    let data_len = config.movie.data().len();
    let num_frames = config.movie.num_frames();
    let player = PlayerBuilder::new()
        .with_movie(config.movie)
        .with_autoplay(true)
        // A headless player still needs a non-zero viewport, or the frame loop
        // does nothing and the document class never constructs.
        .with_viewport_dimensions(1024, 768, 1.0)
        // Do NOT force a frame rate: `with_frame_rate(Some(..))` sets
        // `forced_frame_rate`, which makes the game's own `stage.frameRate = X`
        // setter a no-op (`set_frame_rate` guards on it). FlasCC games set their
        // intended rate at startup; forcing 60 ignored it and ran the sim too fast.
        // The SWF header's absurd ~0.12 FPS only applies until the document class
        // sets the real rate (frame 1 runs at t=0 regardless), and the loop below
        // paces to the *current* rate via `time_til_next_frame`.
        // Fully preload before running: otherwise `run_frame` returns early each
        // tick (streaming) and the document class — which takes the worker
        // branch — never constructs.
        .with_load_behavior(LoadBehavior::Blocking)
        .with_max_execution_duration(Duration::from_secs(60 * 60 * 24 * 365))
        .build();

    // Install the AVM2 JIT on the worker's player, if the embedder registered a
    // factory. A spawned worker carries the bulk of the AVM2 work (the primordial
    // player mostly renders), so this is where the JIT actually pays off.
    if let Some(factory) = WORKER_JIT_FACTORY.get() {
        player
            .lock()
            .expect("worker player poisoned")
            .set_avm2_jit_backend(factory());
    }

    // Drive the movie's preload to completion up front so the document class is
    // defined before the first frame runs.
    {
        let mut guard = player.lock().expect("worker player poisoned");
        guard.set_is_playing(true);
        let mut passes = 0u32;
        while !guard.preload(&mut ExecutionLimit::none()) {
            passes += 1;
            if passes > 100_000 {
                tracing::warn!("worker {}: preload did not finish", config.worker_id);
                break;
            }
        }
        tracing::info!(
            "worker {}: movie {data_len} bytes, {num_frames} frames; preload passes={passes}, playing={}",
            config.worker_id,
            guard.is_playing()
        );
    }

    // Install this worker's identity so `Worker.current` inside the worker is a
    // spawned worker backed by `config.shared_properties`. FlasCC then reads the
    // shared "RAM" ByteArray via `getSharedProperty` and points its own
    // `ApplicationDomain.domainMemory` at it, so `si*`/`li*` hit the shared
    // buffer — no extra Rust-side domainMemory wiring needed.
    player
        .lock()
        .expect("worker player poisoned")
        .install_worker_context(config.worker_id, config.shared_properties);

    // Advance by the **real** elapsed time each tick (not a fixed `1/60`, which
    // decoupled game-time from wall-clock and ran the sim fast/non-uniformly), and
    // pace to the player's *own* frame timing — which the game controls via
    // `stage.frameRate` now that the rate isn't forced. No hardcoded 60 anywhere.
    let mut last_tick = Instant::now();
    let mut ticks: u64 = 0;
    while !config.terminate.load(Ordering::Relaxed) {
        tracing::debug!("worker {}: tick {ticks} begin", config.worker_id);
        let now = Instant::now();
        let dt = now.duration_since(last_tick);
        last_tick = now;
        let sleep_dur = {
            let mut guard = player.lock().expect("worker player poisoned");
            guard.tick(FloatDuration::from_secs(dt.as_secs_f64()));
            guard.time_til_next_frame()
        };
        if ticks < 3 {
            player
                .lock()
                .expect("worker player poisoned")
                .mutate_with_update_context(|context| {
                    use crate::display_object::TDisplayObjectContainer;
                    tracing::info!(
                        "worker stage after tick {ticks}: as3={}, stage_children={}, root_children={}",
                        context.stage.movie().is_action_script_3(),
                        context.stage.num_children(),
                        context
                            .stage
                            .child_by_index(0)
                            .and_then(|c| c.as_container())
                            .map(|c| c.num_children())
                            .unwrap_or(usize::MAX),
                    );
                });
        }
        tracing::debug!("worker {}: tick {ticks} end", config.worker_id);
        ticks += 1;
        // Sleep until the player's next frame is due (at the game's own frame rate).
        // `time_til_next_frame` is `0` when a frame is overdue, so a slow tick just
        // proceeds without sleeping. Cap the sleep so a transient absurd startup rate
        // (the SWF header's ~0.12 FPS, before the document class sets the real rate)
        // can't stall the loop for seconds — it stays responsive to `terminate`.
        // A real worker would instead park on its Condition/MessageChannel.
        let sleep_dur = sleep_dur.min(Duration::from_millis(250));
        if !sleep_dur.is_zero() {
            std::thread::sleep(sleep_dur);
        }
    }

    tracing::info!("worker {} terminated after {ticks} ticks", config.worker_id);
}
