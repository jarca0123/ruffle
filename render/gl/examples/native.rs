//! Minimal native harness for the glow GL backend.
//!
//! Opens a winit window with a glutin OpenGL context and plays an SWF through a
//! `ruffle_core::Player` driven by `GlRenderBackend`. This exists to exercise
//! and visually verify the native GL path without touching the wgpu-based
//! desktop GUI.
//!
//! Usage: `cargo +stable run -p ruffle_render_gl --example native -- path/to.swf`

use std::ffi::CString;
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use glutin::config::{Config, ConfigTemplateBuilder, GlConfig};
use glutin::context::{ContextAttributesBuilder, NotCurrentGlContext, PossiblyCurrentContext};
use glutin::display::{GetGlDisplay, GlDisplay};
use glutin::surface::{GlSurface, Surface, SwapInterval, WindowSurface};
use glutin_winit::{DisplayBuilder, GlWindow};
use raw_window_handle::HasWindowHandle;

use ruffle_core::events::{
    KeyDescriptor, KeyLocation, LogicalKey, MouseButton as RuffleMouseButton, MouseWheelDelta,
    NamedKey as RuffleNamedKey, PhysicalKey, PlayerEvent,
};
use ruffle_core::compatibility_rules::CompatibilityRules;
use ruffle_core::limits::ExecutionLimit;
use ruffle_core::tag_utils::movie_from_path;
use ruffle_core::{FloatDuration, Player, PlayerBuilder};
use ruffle_render::backend::ViewportDimensions;
use ruffle_render::quality::StageQuality;
use ruffle_render_gl::GlRenderBackend;

use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{
    Key, KeyCode as WinitKeyCode, KeyLocation as WinitKeyLocation, NamedKey,
    PhysicalKey as WinitPhysicalKey,
};
use winit::window::{Window, WindowId};

struct State {
    window: Window,
    gl_surface: Surface<WindowSurface>,
    gl_context: PossiblyCurrentContext,
    player: Arc<Mutex<Player>>,
    last_time: Instant,
    mouse_pos: (f64, f64),
    preloaded: bool,
}

struct App {
    swf_path: PathBuf,
    state: Option<State>,
}

fn pick_config(configs: Box<dyn Iterator<Item = Config> + '_>) -> Config {
    // Prefer a single-sample config with a stencil buffer (we do our own MSAA
    // in an FBO; on-screen masking needs a stencil).
    configs
        .reduce(|best, cfg| {
            let score = |c: &Config| (c.stencil_size() >= 8) as u8 + (c.num_samples() == 0) as u8;
            if score(&cfg) > score(&best) {
                cfg
            } else {
                best
            }
        })
        .expect("no GL configs available")
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }

        let window_attributes = Window::default_attributes()
            .with_title("Ruffle — glow GL backend")
            .with_inner_size(winit::dpi::LogicalSize::new(640.0, 480.0));

        let template = ConfigTemplateBuilder::new()
            .with_alpha_size(8)
            .with_stencil_size(8);

        let (window, gl_config) = DisplayBuilder::new()
            .with_window_attributes(Some(window_attributes))
            .build(event_loop, template, pick_config)
            .expect("failed to create window / GL config");
        let window = window.expect("winit window was not created");

        let gl_display = gl_config.display();
        let raw_window_handle = window.window_handle().expect("window handle").as_raw();

        let context_attributes = ContextAttributesBuilder::new().build(Some(raw_window_handle));
        let not_current = unsafe {
            gl_display
                .create_context(&gl_config, &context_attributes)
                .expect("failed to create GL context")
        };

        let surface_attributes = window
            .build_surface_attributes(Default::default())
            .expect("failed to build surface attributes");
        let gl_surface = unsafe {
            gl_display
                .create_window_surface(&gl_config, &surface_attributes)
                .expect("failed to create window surface")
        };

        let gl_context = not_current
            .make_current(&gl_surface)
            .expect("failed to make GL context current");

        // Enable vsync (ignore failure on platforms that don't support it).
        let _ = gl_surface
            .set_swap_interval(&gl_context, SwapInterval::Wait(NonZeroU32::new(1).unwrap()));

        let backend = unsafe {
            GlRenderBackend::new_from_loader_function(
                |symbol| {
                    let cstring = CString::new(symbol).unwrap();
                    gl_display.get_proc_address(cstring.as_c_str())
                },
                false,
                StageQuality::High,
            )
        }
        .expect("failed to create GL render backend");

        let movie = movie_from_path(&self.swf_path, None).expect("failed to load SWF");
        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);

        // Spoofed URL (e.g. RUFFLE_SPOOF_URL=https://www.maxgames.com/) lets
        // sitelocked games past their sponsor check, and compatibility rules
        // handle common preloader/sponsor APIs — without these many games hang
        // on their splash screen (this is game logic, not rendering).
        let spoof_url = std::env::var("RUFFLE_SPOOF_URL").ok();
        let player = PlayerBuilder::new()
            .with_renderer(backend)
            .with_movie(movie)
            .with_viewport_dimensions(width, height, 1.0)
            .with_autoplay(true)
            .with_compatibility_rules(CompatibilityRules::default())
            .with_spoofed_url(spoof_url)
            .build();

        self.state = Some(State {
            window,
            gl_surface,
            gl_context,
            player,
            last_time: Instant::now(),
            mouse_pos: (0.0, 0.0),
            preloaded: false,
        });
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = self.state.as_mut() else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                let width = size.width.max(1);
                let height = size.height.max(1);
                state.gl_surface.resize(
                    &state.gl_context,
                    NonZeroU32::new(width).unwrap(),
                    NonZeroU32::new(height).unwrap(),
                );
                state
                    .player
                    .lock()
                    .unwrap()
                    .set_viewport_dimensions(ViewportDimensions {
                        width,
                        height,
                        scale_factor: 1.0,
                    });
                state.window.request_redraw();
            }
            WindowEvent::CursorMoved { position, .. } => {
                state.mouse_pos = (position.x, position.y);
                state
                    .player
                    .lock()
                    .unwrap()
                    .handle_event(PlayerEvent::MouseMove {
                        x: position.x,
                        y: position.y,
                    });
            }
            WindowEvent::MouseInput {
                button,
                state: el_state,
                ..
            } => {
                let (x, y) = state.mouse_pos;
                let button = match button {
                    MouseButton::Left => RuffleMouseButton::Left,
                    MouseButton::Right => RuffleMouseButton::Right,
                    MouseButton::Middle => RuffleMouseButton::Middle,
                    _ => RuffleMouseButton::Unknown,
                };
                let event = match el_state {
                    ElementState::Pressed => PlayerEvent::MouseDown {
                        x,
                        y,
                        button,
                        index: None,
                    },
                    ElementState::Released => PlayerEvent::MouseUp { x, y, button },
                };
                state.player.lock().unwrap().handle_event(event);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let delta = match delta {
                    MouseScrollDelta::LineDelta(_, dy) => MouseWheelDelta::Lines(dy.into()),
                    MouseScrollDelta::PixelDelta(pos) => MouseWheelDelta::Pixels(pos.y),
                };
                state
                    .player
                    .lock()
                    .unwrap()
                    .handle_event(PlayerEvent::MouseWheel { delta });
            }
            WindowEvent::CursorLeft { .. } => {
                state
                    .player
                    .lock()
                    .unwrap()
                    .handle_event(PlayerEvent::MouseLeave);
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let key = winit_input_to_ruffle_key_descriptor(&event);
                let mut player = state.player.lock().unwrap();
                match event.state {
                    ElementState::Pressed => {
                        player.handle_event(PlayerEvent::KeyDown { key });
                        if let Some(text) = &event.text {
                            for codepoint in text.chars() {
                                player.handle_event(PlayerEvent::TextInput { codepoint });
                            }
                        }
                    }
                    ElementState::Released => {
                        player.handle_event(PlayerEvent::KeyUp { key });
                    }
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let Some(state) = self.state.as_mut() else {
            return;
        };

        // Drive the player entirely here (timer-paced via `ControlFlow::WaitUntil`),
        // including rendering — do NOT rely on `RedrawRequested`, which Wayland
        // compositors deliver irregularly (causing a frozen screen with the CPU
        // idle between timer wake-ups).
        let now = Instant::now();
        let dt = FloatDuration::from_std(now.saturating_duration_since(state.last_time));
        let ticked = dt.as_millis() > 0.0;

        let til = {
            let mut player = state.player.lock().unwrap();

            // A directly-supplied movie isn't streamed, so we preload it
            // ourselves. Do it incrementally with a per-frame time budget so the
            // window stays responsive (and any preloader animates) instead of
            // blocking on one giant up-front load.
            if !state.preloaded {
                state.preloaded = player.preload(&mut ExecutionLimit::with_max_ops_and_time(
                    100_000,
                    std::time::Duration::from_millis(10),
                ));
            }

            if ticked {
                state.last_time = now;
                player.tick(dt);
                player.render();
            }
            player.time_til_next_frame()
        };

        if ticked {
            state
                .gl_surface
                .swap_buffers(&state.gl_context)
                .expect("failed to swap buffers");
        }

        // While still preloading, keep waking promptly to make load progress.
        let wait = if state.preloaded {
            til
        } else {
            til.min(std::time::Duration::from_millis(5))
        };
        event_loop.set_control_flow(ControlFlow::WaitUntil(now + wait));
    }
}

fn main() {
    env_logger::init();

    let swf_path = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .expect("usage: native <path-to.swf>");

    let event_loop = EventLoop::new().expect("failed to create event loop");
    let mut app = App {
        swf_path,
        state: None,
    };
    event_loop.run_app(&mut app).expect("event loop error");
}

// --- winit -> ruffle key mapping (adapted from desktop/src/util.rs) ---

fn winit_input_to_ruffle_key_descriptor(event: &KeyEvent) -> KeyDescriptor {
    KeyDescriptor {
        physical_key: map_physical_key(event),
        logical_key: map_logical_key(event),
        key_location: map_key_location(event),
    }
}

fn map_physical_key(event: &KeyEvent) -> PhysicalKey {
    let WinitPhysicalKey::Code(key_code) = event.physical_key else {
        return PhysicalKey::Unknown;
    };
    match key_code {
        WinitKeyCode::Backquote => PhysicalKey::Backquote,
        WinitKeyCode::Backslash => PhysicalKey::Backslash,
        WinitKeyCode::BracketLeft => PhysicalKey::BracketLeft,
        WinitKeyCode::BracketRight => PhysicalKey::BracketRight,
        WinitKeyCode::Comma => PhysicalKey::Comma,
        WinitKeyCode::Digit0 => PhysicalKey::Digit0,
        WinitKeyCode::Digit1 => PhysicalKey::Digit1,
        WinitKeyCode::Digit2 => PhysicalKey::Digit2,
        WinitKeyCode::Digit3 => PhysicalKey::Digit3,
        WinitKeyCode::Digit4 => PhysicalKey::Digit4,
        WinitKeyCode::Digit5 => PhysicalKey::Digit5,
        WinitKeyCode::Digit6 => PhysicalKey::Digit6,
        WinitKeyCode::Digit7 => PhysicalKey::Digit7,
        WinitKeyCode::Digit8 => PhysicalKey::Digit8,
        WinitKeyCode::Digit9 => PhysicalKey::Digit9,
        WinitKeyCode::Equal => PhysicalKey::Equal,
        WinitKeyCode::IntlBackslash => PhysicalKey::IntlBackslash,
        WinitKeyCode::IntlRo => PhysicalKey::IntlRo,
        WinitKeyCode::IntlYen => PhysicalKey::IntlYen,
        WinitKeyCode::KeyA => PhysicalKey::KeyA,
        WinitKeyCode::KeyB => PhysicalKey::KeyB,
        WinitKeyCode::KeyC => PhysicalKey::KeyC,
        WinitKeyCode::KeyD => PhysicalKey::KeyD,
        WinitKeyCode::KeyE => PhysicalKey::KeyE,
        WinitKeyCode::KeyF => PhysicalKey::KeyF,
        WinitKeyCode::KeyG => PhysicalKey::KeyG,
        WinitKeyCode::KeyH => PhysicalKey::KeyH,
        WinitKeyCode::KeyI => PhysicalKey::KeyI,
        WinitKeyCode::KeyJ => PhysicalKey::KeyJ,
        WinitKeyCode::KeyK => PhysicalKey::KeyK,
        WinitKeyCode::KeyL => PhysicalKey::KeyL,
        WinitKeyCode::KeyM => PhysicalKey::KeyM,
        WinitKeyCode::KeyN => PhysicalKey::KeyN,
        WinitKeyCode::KeyO => PhysicalKey::KeyO,
        WinitKeyCode::KeyP => PhysicalKey::KeyP,
        WinitKeyCode::KeyQ => PhysicalKey::KeyQ,
        WinitKeyCode::KeyR => PhysicalKey::KeyR,
        WinitKeyCode::KeyS => PhysicalKey::KeyS,
        WinitKeyCode::KeyT => PhysicalKey::KeyT,
        WinitKeyCode::KeyU => PhysicalKey::KeyU,
        WinitKeyCode::KeyV => PhysicalKey::KeyV,
        WinitKeyCode::KeyW => PhysicalKey::KeyW,
        WinitKeyCode::KeyX => PhysicalKey::KeyX,
        WinitKeyCode::KeyY => PhysicalKey::KeyY,
        WinitKeyCode::KeyZ => PhysicalKey::KeyZ,
        WinitKeyCode::Minus => PhysicalKey::Minus,
        WinitKeyCode::Period => PhysicalKey::Period,
        WinitKeyCode::Quote => PhysicalKey::Quote,
        WinitKeyCode::Semicolon => PhysicalKey::Semicolon,
        WinitKeyCode::Slash => PhysicalKey::Slash,
        WinitKeyCode::AltLeft => PhysicalKey::AltLeft,
        WinitKeyCode::AltRight => PhysicalKey::AltRight,
        WinitKeyCode::Backspace => PhysicalKey::Backspace,
        WinitKeyCode::CapsLock => PhysicalKey::CapsLock,
        WinitKeyCode::ContextMenu => PhysicalKey::ContextMenu,
        WinitKeyCode::ControlLeft => PhysicalKey::ControlLeft,
        WinitKeyCode::ControlRight => PhysicalKey::ControlRight,
        WinitKeyCode::Enter => PhysicalKey::Enter,
        WinitKeyCode::SuperLeft => PhysicalKey::SuperLeft,
        WinitKeyCode::SuperRight => PhysicalKey::SuperRight,
        WinitKeyCode::ShiftLeft => PhysicalKey::ShiftLeft,
        WinitKeyCode::ShiftRight => PhysicalKey::ShiftRight,
        WinitKeyCode::Space => PhysicalKey::Space,
        WinitKeyCode::Tab => PhysicalKey::Tab,
        WinitKeyCode::Delete => PhysicalKey::Delete,
        WinitKeyCode::End => PhysicalKey::End,
        WinitKeyCode::Home => PhysicalKey::Home,
        WinitKeyCode::Insert => PhysicalKey::Insert,
        WinitKeyCode::PageDown => PhysicalKey::PageDown,
        WinitKeyCode::PageUp => PhysicalKey::PageUp,
        WinitKeyCode::ArrowDown => PhysicalKey::ArrowDown,
        WinitKeyCode::ArrowLeft => PhysicalKey::ArrowLeft,
        WinitKeyCode::ArrowRight => PhysicalKey::ArrowRight,
        WinitKeyCode::ArrowUp => PhysicalKey::ArrowUp,
        WinitKeyCode::NumLock => PhysicalKey::NumLock,
        WinitKeyCode::Numpad0 => PhysicalKey::Numpad0,
        WinitKeyCode::Numpad1 => PhysicalKey::Numpad1,
        WinitKeyCode::Numpad2 => PhysicalKey::Numpad2,
        WinitKeyCode::Numpad3 => PhysicalKey::Numpad3,
        WinitKeyCode::Numpad4 => PhysicalKey::Numpad4,
        WinitKeyCode::Numpad5 => PhysicalKey::Numpad5,
        WinitKeyCode::Numpad6 => PhysicalKey::Numpad6,
        WinitKeyCode::Numpad7 => PhysicalKey::Numpad7,
        WinitKeyCode::Numpad8 => PhysicalKey::Numpad8,
        WinitKeyCode::Numpad9 => PhysicalKey::Numpad9,
        WinitKeyCode::NumpadAdd => PhysicalKey::NumpadAdd,
        WinitKeyCode::NumpadComma => PhysicalKey::NumpadComma,
        WinitKeyCode::NumpadDecimal => PhysicalKey::NumpadDecimal,
        WinitKeyCode::NumpadDivide => PhysicalKey::NumpadDivide,
        WinitKeyCode::NumpadEnter => PhysicalKey::NumpadEnter,
        WinitKeyCode::NumpadMultiply => PhysicalKey::NumpadMultiply,
        WinitKeyCode::NumpadSubtract => PhysicalKey::NumpadSubtract,
        WinitKeyCode::Escape => PhysicalKey::Escape,
        WinitKeyCode::Fn => PhysicalKey::Fn,
        WinitKeyCode::FnLock => PhysicalKey::FnLock,
        WinitKeyCode::PrintScreen => PhysicalKey::PrintScreen,
        WinitKeyCode::ScrollLock => PhysicalKey::ScrollLock,
        WinitKeyCode::Pause => PhysicalKey::Pause,
        WinitKeyCode::F1 => PhysicalKey::F1,
        WinitKeyCode::F2 => PhysicalKey::F2,
        WinitKeyCode::F3 => PhysicalKey::F3,
        WinitKeyCode::F4 => PhysicalKey::F4,
        WinitKeyCode::F5 => PhysicalKey::F5,
        WinitKeyCode::F6 => PhysicalKey::F6,
        WinitKeyCode::F7 => PhysicalKey::F7,
        WinitKeyCode::F8 => PhysicalKey::F8,
        WinitKeyCode::F9 => PhysicalKey::F9,
        WinitKeyCode::F10 => PhysicalKey::F10,
        WinitKeyCode::F11 => PhysicalKey::F11,
        WinitKeyCode::F12 => PhysicalKey::F12,
        _ => PhysicalKey::Unknown,
    }
}

fn map_logical_key(event: &KeyEvent) -> LogicalKey {
    match event.logical_key.as_ref() {
        Key::Named(NamedKey::Alt) => LogicalKey::Named(RuffleNamedKey::Alt),
        Key::Named(NamedKey::AltGraph) => LogicalKey::Named(RuffleNamedKey::AltGraph),
        Key::Named(NamedKey::CapsLock) => LogicalKey::Named(RuffleNamedKey::CapsLock),
        Key::Named(NamedKey::Control) => LogicalKey::Named(RuffleNamedKey::Control),
        Key::Named(NamedKey::Fn) => LogicalKey::Named(RuffleNamedKey::Fn),
        Key::Named(NamedKey::NumLock) => LogicalKey::Named(RuffleNamedKey::NumLock),
        Key::Named(NamedKey::ScrollLock) => LogicalKey::Named(RuffleNamedKey::ScrollLock),
        Key::Named(NamedKey::Shift) => LogicalKey::Named(RuffleNamedKey::Shift),
        Key::Named(NamedKey::Super) => LogicalKey::Named(RuffleNamedKey::Super),
        Key::Named(NamedKey::Enter) => LogicalKey::Named(RuffleNamedKey::Enter),
        Key::Named(NamedKey::Tab) => LogicalKey::Named(RuffleNamedKey::Tab),
        Key::Named(NamedKey::Space) => LogicalKey::Character(' '),
        Key::Named(NamedKey::ArrowDown) => LogicalKey::Named(RuffleNamedKey::ArrowDown),
        Key::Named(NamedKey::ArrowLeft) => LogicalKey::Named(RuffleNamedKey::ArrowLeft),
        Key::Named(NamedKey::ArrowRight) => LogicalKey::Named(RuffleNamedKey::ArrowRight),
        Key::Named(NamedKey::ArrowUp) => LogicalKey::Named(RuffleNamedKey::ArrowUp),
        Key::Named(NamedKey::End) => LogicalKey::Named(RuffleNamedKey::End),
        Key::Named(NamedKey::Home) => LogicalKey::Named(RuffleNamedKey::Home),
        Key::Named(NamedKey::PageDown) => LogicalKey::Named(RuffleNamedKey::PageDown),
        Key::Named(NamedKey::PageUp) => LogicalKey::Named(RuffleNamedKey::PageUp),
        Key::Named(NamedKey::Backspace) => LogicalKey::Named(RuffleNamedKey::Backspace),
        Key::Named(NamedKey::Delete) => LogicalKey::Named(RuffleNamedKey::Delete),
        Key::Named(NamedKey::Insert) => LogicalKey::Named(RuffleNamedKey::Insert),
        Key::Named(NamedKey::ContextMenu) => LogicalKey::Named(RuffleNamedKey::ContextMenu),
        Key::Named(NamedKey::Escape) => LogicalKey::Named(RuffleNamedKey::Escape),
        Key::Named(NamedKey::Pause) => LogicalKey::Named(RuffleNamedKey::Pause),
        Key::Named(NamedKey::F1) => LogicalKey::Named(RuffleNamedKey::F1),
        Key::Named(NamedKey::F2) => LogicalKey::Named(RuffleNamedKey::F2),
        Key::Named(NamedKey::F3) => LogicalKey::Named(RuffleNamedKey::F3),
        Key::Named(NamedKey::F4) => LogicalKey::Named(RuffleNamedKey::F4),
        Key::Named(NamedKey::F5) => LogicalKey::Named(RuffleNamedKey::F5),
        Key::Named(NamedKey::F6) => LogicalKey::Named(RuffleNamedKey::F6),
        Key::Named(NamedKey::F7) => LogicalKey::Named(RuffleNamedKey::F7),
        Key::Named(NamedKey::F8) => LogicalKey::Named(RuffleNamedKey::F8),
        Key::Named(NamedKey::F9) => LogicalKey::Named(RuffleNamedKey::F9),
        Key::Named(NamedKey::F10) => LogicalKey::Named(RuffleNamedKey::F10),
        Key::Named(NamedKey::F11) => LogicalKey::Named(RuffleNamedKey::F11),
        Key::Named(NamedKey::F12) => LogicalKey::Named(RuffleNamedKey::F12),
        Key::Character(ch) => ch
            .chars()
            .next_back()
            .map(LogicalKey::Character)
            .unwrap_or(LogicalKey::Unknown),
        _ => LogicalKey::Unknown,
    }
}

fn map_key_location(event: &KeyEvent) -> KeyLocation {
    match event.location {
        WinitKeyLocation::Standard => KeyLocation::Standard,
        WinitKeyLocation::Left => KeyLocation::Left,
        WinitKeyLocation::Right => KeyLocation::Right,
        WinitKeyLocation::Numpad => KeyLocation::Numpad,
    }
}
