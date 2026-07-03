//! Headless runtime for a spawned `flash.system.Worker` (real-threads "Path A").
//!
//! Runs on a background thread (native `std::thread` / web Web Worker via the
//! [`WorkerHost`](crate::worker_host::WorkerHost)). It builds a self-contained
//! headless [`Player`](crate::Player) for the worker's SWF and ticks it until
//! the worker is terminated. Everything shared with the creating thread crosses
//! by reference through the `Send` `Arc` handles in [`WorkerConfig`] — never a
//! `Gc`/arena object, which is `!Send`.

use crate::FloatDuration;
use crate::avm2::worker_shared::SharedProperties;
use crate::limits::ExecutionLimit;
use crate::loader::LoadBehavior;
use crate::player::PlayerBuilder;
use crate::tag_utils::SwfMovie;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use crate::display_object::TDisplayObject;

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
        // FlasCC SWFs declare an absurdly low frame rate (~0.12 FPS ≈ 8.5s/frame)
        // because they drive execution via timers, not Flash frames. A worker is
        // frame-driven, so force a high rate — otherwise the document class (and
        // every subsequent frame-gated step) is 8.5s apart.
        .with_frame_rate(Some(60.0))
        // Fully preload before running: otherwise `run_frame` returns early each
        // tick (streaming) and the document class — which takes the worker
        // branch — never constructs.
        .with_load_behavior(LoadBehavior::Blocking)
        .with_max_execution_duration(Duration::from_secs(60 * 60 * 24 * 365))
        .build();

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

    let frame = FloatDuration::from_secs(1.0 / 60.0);
    let mut ticks: u64 = 0;
    while !config.terminate.load(Ordering::Relaxed) {
        tracing::debug!("worker {}: tick {ticks} begin", config.worker_id);
        player.lock().expect("worker player poisoned").tick(frame);
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
        // A real worker parks on its Condition / MessageChannel between ticks;
        // this fixed sleep is a placeholder cooperative pacing.
        std::thread::sleep(Duration::from_millis(16));
    }

    tracing::info!("worker {} terminated after {ticks} ticks", config.worker_id);
}
