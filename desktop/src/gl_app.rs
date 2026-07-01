//! Native OpenGL desktop path (experimental, `--gl-native`).
//!
//! Plays an SWF through `ruffle_core::Player` driven by the glow-based
//! `GlRenderBackend`, in a glutin/winit window, with the full desktop egui UI
//! (menu bar + dialogs) rendered via `egui_glow` on top. This bypasses the
//! wgpu `GuiController` but reuses `RuffleGui`, the navigator, and audio.
//!
//! egui and the Flash renderer don't share a glow wrapper (the backend uses
//! `Rc<glow::Context>`, egui_glow wants `Arc`), but they DO share the same
//! underlying GL context — we just hand each its own glow function table built
//! from the same loader.

use std::collections::HashSet;
use std::ffi::CString;
use std::fs::File;
use std::io;
use std::num::NonZeroU32;
use std::path::Path;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{anyhow, Error};
use egui::ViewportId;
use glutin::config::{Config, ConfigTemplateBuilder, GlConfig};
use glutin::context::{
    ContextApi, ContextAttributesBuilder, NotCurrentGlContext, PossiblyCurrentContext, Version,
};
use glutin::display::{GetGlDisplay, GlDisplay};
use glutin::surface::{GlSurface, Surface, SwapInterval, WindowSurface};
use glutin_winit::{DisplayBuilder, GlWindow};
use raw_window_handle::HasWindowHandle;

use ruffle_core::backend::navigator::SocketMode;
use ruffle_core::backend::ui::{
    DialogResultFuture, FileFilter, FontDefinition, FullscreenError, LanguageIdentifier,
    MouseCursor, MultiDialogResultFuture, NullUiBackend, UiBackend,
};
use ruffle_core::compatibility_rules::CompatibilityRules;
use ruffle_core::events::{MouseButton as RuffleMouseButton, MouseWheelDelta, PlayerEvent};
use ruffle_core::font::{DefaultFont, FontFileData, FontQuery};
use ruffle_core::limits::ExecutionLimit;
use ruffle_core::tag_utils::SwfMovie;
use ruffle_core::{FloatDuration, Player, PlayerBuilder};
use ruffle_frontend_utils::backends::audio::CpalAudioBackend;
use ruffle_frontend_utils::backends::navigator::{ExternalNavigatorBackend, NavigatorInterface};
use ruffle_frontend_utils::content::{ContentDescriptor, PlayingContent};
use ruffle_render::backend::ViewportDimensions;
use ruffle_render::quality::StageQuality;
use ruffle_render_gl::GlRenderBackend;
use url::Url;

use winit::application::ApplicationHandler;
use winit::event::{ElementState, Modifiers, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy};
use winit::window::{Window, WindowId};

use crate::custom_event::RuffleEvent;
use crate::gui::{RuffleGui, MENU_HEIGHT};
use crate::player::{LaunchOptions, WinitExecutor};
use crate::preferences::GlobalPreferences;

/// Minimal navigator interface for the GL path: open files locally, open links
/// in the browser, and allow every socket connection without prompting.
#[derive(Clone)]
struct NullNavigatorInterface;

impl NavigatorInterface for NullNavigatorInterface {
    fn navigate_to_website(&self, url: Url) {
        let _ = webbrowser::open(url.as_str());
    }

    fn open_file(&self, path: &Path) -> impl std::future::Future<Output = io::Result<File>> + Send {
        let path = path.to_owned();
        async move { File::open(path) }
    }

    fn confirm_socket(
        &self,
        _host: &str,
        _port: u16,
    ) -> impl std::future::Future<Output = bool> + Send {
        async { true }
    }
}

// Field order matters: Rust drops fields top-to-bottom, and teardown has a
// dependency chain. `player`'s `GlRenderBackend` (and `egui_painter`) delete GL
// objects in `Drop`, so they must drop while the context is still current.
// `gl_surface`/`gl_context` are EGL objects bound to the `window`, so they must
// drop before it. Hence: GL users first, then surface/context, then window.
struct State {
    player: Arc<Mutex<Player>>,
    egui_painter: egui_glow::Painter,
    egui_winit: egui_winit::State,
    gui: RuffleGui,
    mouse_pos: (f64, f64),
    modifiers: Modifiers,
    /// Physical-pixel height of the egui menu bar. The movie is rendered into the
    /// area below it, and mouse coordinates are shifted up by this amount.
    menu_offset: u32,
    /// Set when egui requests a repaint (menu interaction); forces a redraw even
    /// when the movie itself is unchanged.
    egui_dirty: bool,
    last_time: Instant,
    preloaded: bool,
    gl_surface: Surface<WindowSurface>,
    gl_context: PossiblyCurrentContext,
    window: Arc<Window>,
}

struct GlApp {
    movie_url: Url,
    spoof_url: Option<String>,
    preferences: GlobalPreferences,
    proxy: EventLoopProxy<RuffleEvent>,
    state: Option<State>,
}

/// Prefer a single-sample config with a stencil buffer (we do our own MSAA in
/// an FBO; on-screen masking needs a stencil).
fn pick_config(configs: Box<dyn Iterator<Item = Config> + '_>) -> Config {
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

impl GlApp {
    fn build_player(
        &self,
        width: u32,
        height: u32,
        loader: impl Fn(&str) -> *const std::ffi::c_void,
    ) -> Result<Arc<Mutex<Player>>, Error> {
        let backend = unsafe {
            GlRenderBackend::new_from_loader_function(loader, false, StageQuality::High)
        }
        .map_err(|e| anyhow!("failed to create GL render backend: {e}"))?;

        let descriptor = if self.movie_url.scheme() == "file" {
            let path = self
                .movie_url
                .to_file_path()
                .map_err(|_| anyhow!("invalid file URL"))?;
            ContentDescriptor::new_local(&path, None).ok_or_else(|| anyhow!("invalid SWF path"))?
        } else {
            ContentDescriptor::new_remote(self.movie_url.clone())
        };
        let content = Rc::new(PlayingContent::DirectFile(descriptor));
        let base_url = content.initial_swf_url().clone();

        let navigator = ExternalNavigatorBackend::new(
            base_url,
            None,
            None,
            WinitExecutor::new(self.proxy.clone()),
            None,
            false,
            HashSet::new(),
            SocketMode::Allow,
            content,
            NullNavigatorInterface,
        );

        // A shared system-font database, used both for the eager default-font
        // registration below and for lazy `load_device_font` lookups when a movie
        // names a specific device font (e.g. Starling's "Verdana" text fields).
        let mut font_database = fontdb::Database::new();
        font_database.load_system_fonts();
        let font_database = Rc::new(font_database);

        let mut builder = PlayerBuilder::new()
            .with_renderer(backend)
            .with_navigator(navigator)
            .with_ui(GlUiBackend::new(font_database.clone()))
            .with_viewport_dimensions(width, height, 1.0)
            .with_autoplay(true)
            .with_compatibility_rules(CompatibilityRules::default())
            .with_spoofed_url(self.spoof_url.clone());

        match CpalAudioBackend::new(None) {
            Ok(audio) => builder = builder.with_audio(audio),
            Err(e) => tracing::error!("Unable to create audio device: {e}"),
        }

        let player = builder.build();
        {
            let mut player = player.lock().expect("player lock");
            register_device_fonts(&mut player, &font_database);
            player.fetch_root_movie(self.movie_url.to_string(), Vec::new(), Box::new(|_| {}));
        }
        Ok(player)
    }
}

impl ApplicationHandler<RuffleEvent> for GlApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }

        let window_attributes = Window::default_attributes()
            .with_title("Ruffle — native GL backend")
            .with_inner_size(winit::dpi::LogicalSize::new(640.0, 480.0));

        let template = ConfigTemplateBuilder::new()
            .with_alpha_size(8)
            .with_stencil_size(8);

        let (window, gl_config) = DisplayBuilder::new()
            .with_window_attributes(Some(window_attributes))
            .build(event_loop, template, pick_config)
            .expect("failed to create window / GL config");
        let window = Arc::new(window.expect("winit window was not created"));

        let gl_display = gl_config.display();
        let raw_window_handle = window.window_handle().expect("window handle").as_raw();

        // Request desktop GL 3.3 so the backend's FBO MSAA path is available
        // (it needs GL >= 3.0); otherwise a driver default of GLES2/GL2 would
        // silently disable antialiasing.
        let context_attributes = ContextAttributesBuilder::new()
            .with_context_api(ContextApi::OpenGl(Some(Version::new(3, 3))))
            .build(Some(raw_window_handle));
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

        let _ = gl_surface
            .set_swap_interval(&gl_context, SwapInterval::Wait(NonZeroU32::new(1).unwrap()));

        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);
        // Render the movie below the egui menu bar, not behind it.
        let menu_offset = (MENU_HEIGHT as f64 * window.scale_factor()).round() as u32;
        let movie_height = height.saturating_sub(menu_offset).max(1);

        let player = self
            .build_player(width, movie_height, |symbol| {
                let cstring = CString::new(symbol).unwrap();
                gl_display.get_proc_address(cstring.as_c_str())
            })
            .expect("failed to build GL player");

        // egui gets its own glow function table over the same GL context.
        let egui_gl = Arc::new(unsafe {
            egui_glow::glow::Context::from_loader_function(|symbol| {
                let cstring = CString::new(symbol).unwrap();
                gl_display.get_proc_address(cstring.as_c_str()) as *const _
            })
        });
        let egui_painter = egui_glow::Painter::new(egui_gl, "", None, false)
            .expect("failed to create egui_glow painter");

        let egui_ctx = egui::Context::default();
        egui_extras::install_image_loaders(&egui_ctx);
        let egui_winit = egui_winit::State::new(
            egui_ctx,
            ViewportId::ROOT,
            window.as_ref(),
            None,
            None,
            None,
        );

        let content_desc = ContentDescriptor::new_remote(self.movie_url.clone());
        let launch_options = LaunchOptions::from(&self.preferences);
        let mut gui = RuffleGui::new(
            Arc::downgrade(&window),
            self.proxy.clone(),
            Some(content_desc.clone()),
            launch_options.clone(),
            self.preferences.clone(),
        );
        // Tell the GUI a movie is open, so the menu's reload (CloseFile + Open)
        // re-opens rather than just closing, and the reload/close items enable.
        gui.on_player_created(
            launch_options,
            content_desc,
            player.lock().expect("player lock"),
        );

        self.state = Some(State {
            window,
            gl_surface,
            gl_context,
            player,
            mouse_pos: (0.0, 0.0),
            modifiers: Modifiers::default(),
            menu_offset,
            egui_dirty: true,
            egui_winit,
            egui_painter,
            gui,
            last_time: Instant::now(),
            preloaded: false,
        });
    }

    fn user_event(&mut self, event_loop: &ActiveEventLoop, event: RuffleEvent) {
        match event {
            RuffleEvent::TaskPoll(task) => task.run(),
            RuffleEvent::ExitRequested => event_loop.exit(),
            RuffleEvent::CloseFile => {
                // Unload the current movie (blank stage). A plain Close just
                // unloads; Reload sends CloseFile followed by Open, so it
                // reloads.
                if let Some(state) = self.state.as_mut() {
                    state
                        .player
                        .lock()
                        .unwrap()
                        .mutate_with_update_context(|context| {
                            // Version is irrelevant for a code-less blank stage.
                            context.set_root_movie(SwfMovie::empty(32, None));
                        });
                }
            }
            RuffleEvent::Open(content, opts) => {
                if let Some(state) = self.state.as_mut() {
                    let url = content.url.to_string();
                    state
                        .player
                        .lock()
                        .unwrap()
                        .fetch_root_movie(url, Vec::new(), Box::new(|_| {}));
                    // Re-arm reload: the menu took `currently_opened` when it
                    // sent this Open, so re-register the movie with the GUI.
                    state.gui.on_player_created(
                        *opts,
                        content,
                        state.player.lock().expect("player lock"),
                    );
                }
            }
            // Dialogs, metadata, fullscreen, etc. — not wired in the GL path yet.
            _ => {}
        }
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

        // Let egui consume UI input first; only forward to the player otherwise.
        let response = state.egui_winit.on_window_event(&state.window, &event);
        // Only genuine input dirties egui. Reacting to `RedrawRequested` (or
        // calling `request_redraw` here) forms a feedback loop that pins the
        // frame rate to vsync — real input events already wake the loop.
        if response.repaint && !matches!(event, WindowEvent::RedrawRequested) {
            state.egui_dirty = true;
        }
        let consumed = response.consumed;

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
                // Keep the movie below the menu bar (scale factor may have changed).
                state.menu_offset =
                    (MENU_HEIGHT as f64 * state.window.scale_factor()).round() as u32;
                let movie_height = height.saturating_sub(state.menu_offset).max(1);
                state
                    .player
                    .lock()
                    .unwrap()
                    .set_viewport_dimensions(ViewportDimensions {
                        width,
                        height: movie_height,
                        scale_factor: 1.0,
                    });
                state.window.request_redraw();
            }
            _ if consumed => {}
            WindowEvent::CursorMoved { position, .. } => {
                // Shift into movie space (the movie sits below the menu bar).
                let y = position.y - state.menu_offset as f64;
                state.mouse_pos = (position.x, y);
                state
                    .player
                    .lock()
                    .unwrap()
                    .handle_event(PlayerEvent::MouseMove { x: position.x, y });
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
                let event = if el_state == ElementState::Pressed {
                    PlayerEvent::MouseDown {
                        x,
                        y,
                        button,
                        index: None,
                    }
                } else {
                    PlayerEvent::MouseUp { x, y, button }
                };
                state.player.lock().unwrap().handle_event(event);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let delta = match delta {
                    MouseScrollDelta::LineDelta(_, dy) => MouseWheelDelta::Lines(dy.into()),
                    MouseScrollDelta::PixelDelta(p) => MouseWheelDelta::Pixels(p.y),
                };
                state
                    .player
                    .lock()
                    .unwrap()
                    .handle_event(PlayerEvent::MouseWheel { delta });
            }
            WindowEvent::CursorLeft { .. } => {
                state.player.lock().unwrap().handle_event(PlayerEvent::MouseLeave);
            }
            WindowEvent::ModifiersChanged(new_modifiers) => {
                state.modifiers = new_modifiers;
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let key = crate::util::winit_input_to_ruffle_key_descriptor(&event);
                let modifiers = state.modifiers;
                let mut player = state.player.lock().unwrap();
                match event.state {
                    ElementState::Pressed => {
                        player.handle_event(PlayerEvent::KeyDown { key });
                        // Cursor movement, selection and clipboard shortcuts in
                        // EditText come through as TextControl; only fall back to
                        // TextInput for keys that produce actual text.
                        if let Some(control_code) =
                            crate::util::winit_to_ruffle_text_control(&event, modifiers)
                        {
                            player.handle_event(PlayerEvent::TextControl { code: control_code });
                        } else if let Some(text) = &event.text {
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

        let now = Instant::now();
        let dt = FloatDuration::from_std(now.saturating_duration_since(state.last_time));
        let ticked = dt.as_millis() > 0.0;

        // 1) Drive the Flash content. Rendering is gated on `needs_render` below
        //    so we don't redraw frames the movie didn't actually advance (the
        //    SWF runs at its own rate, not the display's).
        let (needs_render, til) = {
            let mut player = state.player.lock().unwrap();
            if !state.preloaded {
                state.preloaded = player.preload(&mut ExecutionLimit::with_max_ops_and_time(
                    100_000,
                    Duration::from_millis(10),
                ));
            }
            if ticked {
                state.last_time = now;
                player.tick(dt);
            }
            (player.needs_render(), player.time_til_next_frame())
        };

        // 2) Render the movie + egui overlay only when something changed.
        if ticked && (needs_render || state.egui_dirty) {
            state.player.lock().unwrap().render();
            let raw_input = state.egui_winit.take_egui_input(&state.window);
            let egui_ctx = state.egui_winit.egui_ctx().clone();
            let gui = &mut state.gui;
            let player = &state.player;
            let full_output = egui_ctx.run(raw_input, |ctx| {
                let mut player = player.lock().unwrap();
                gui.update(ctx, true, Some(&mut player), 0.0);
            });
            state
                .egui_winit
                .handle_platform_output(&state.window, full_output.platform_output);
            let ppp = egui_ctx.pixels_per_point();
            let primitives = egui_ctx.tessellate(full_output.shapes, ppp);
            let size = state.window.inner_size();
            state.egui_painter.paint_and_update_textures(
                [size.width.max(1), size.height.max(1)],
                ppp,
                &primitives,
                &full_output.textures_delta,
            );

            // Match the movie viewport to the *actual* egui menu-bar height so the
            // movie sits flush below it with no transparent strip. `available_rect`
            // top is the menu height in points; convert to physical pixels.
            let menu_phys = (egui_ctx.available_rect().top() * ppp).round().max(0.0) as u32;
            if menu_phys != state.menu_offset {
                state.menu_offset = menu_phys;
                let movie_height = size.height.max(1).saturating_sub(menu_phys).max(1);
                state
                    .player
                    .lock()
                    .unwrap()
                    .set_viewport_dimensions(ViewportDimensions {
                        width: size.width.max(1),
                        height: movie_height,
                        scale_factor: 1.0,
                    });
            }

            state
                .gl_surface
                .swap_buffers(&state.gl_context)
                .expect("failed to swap buffers");
            state.egui_dirty = false;
        }

        let wait = if state.preloaded {
            til
        } else {
            til.min(Duration::from_millis(5))
        };
        event_loop.set_control_flow(ControlFlow::WaitUntil(now + wait));
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        // Free egui's GL resources while the context is still current, so the
        // painter doesn't leak (and warn) on drop.
        if let Some(state) = self.state.as_mut() {
            state.egui_painter.destroy();
        }
    }
}

/// Minimal UI backend for the native-GL path. Behaves like [`NullUiBackend`]
/// for everything except device-font loading: when a movie names a device font
/// (e.g. Starling renders its `TextField`s with "Verdana"), core calls
/// `load_device_font` lazily, and we resolve it against the system font
/// database — falling back to a generic sans-serif face when the exact name is
/// not installed, so text still renders instead of vanishing.
struct GlUiBackend {
    inner: NullUiBackend,
    font_database: Rc<fontdb::Database>,
}

impl GlUiBackend {
    fn new(font_database: Rc<fontdb::Database>) -> Self {
        Self {
            inner: NullUiBackend::new(),
            font_database,
        }
    }
}

impl UiBackend for GlUiBackend {
    fn load_device_font(&self, query: &FontQuery, register: &mut dyn FnMut(FontDefinition)) {
        // Try the exact requested face first, then fall back to concrete
        // sans-serif families that are actually installed on the system. We
        // avoid `fontdb::Family::SansSerif` because `fontdb` does not resolve
        // the generic families after `load_system_fonts()` (that needs
        // fontconfig), so it would silently return nothing — leaving named
        // fonts like "Verdana" (uninstalled on most Linux boxes) as empty text.
        for candidate in std::iter::once(query.name.as_str()).chain(
            [
                "Arial",
                "Liberation Sans",
                "DejaVu Sans",
                "Noto Sans",
                "FreeSans",
            ]
            .into_iter(),
        ) {
            if let Some(mut font) =
                query_device_font(&self.font_database, candidate, query.is_bold, query.is_italic)
            {
                // Register under the *requested* name so the movie's lookup hits,
                // even when we substituted a different family.
                if let FontDefinition::FontFile { name, .. } = &mut font {
                    *name = query.name.to_string();
                }
                tracing::info!(
                    "GL: device font \"{}\" (bold: {}, italic: {}) resolved via \"{candidate}\"",
                    query.name,
                    query.is_bold,
                    query.is_italic,
                );
                register(font);
                return;
            }
        }
        tracing::warn!("GL: no device font found for \"{}\"", query.name);
    }

    fn mouse_visible(&self) -> bool {
        self.inner.mouse_visible()
    }
    fn set_mouse_visible(&mut self, visible: bool) {
        self.inner.set_mouse_visible(visible)
    }
    fn set_mouse_cursor(&mut self, cursor: MouseCursor) {
        self.inner.set_mouse_cursor(cursor)
    }
    fn clipboard_content(&mut self) -> String {
        self.inner.clipboard_content()
    }
    fn set_clipboard_content(&mut self, content: String) {
        self.inner.set_clipboard_content(content)
    }
    fn set_fullscreen(&mut self, is_full: bool) -> Result<(), FullscreenError> {
        self.inner.set_fullscreen(is_full)
    }
    fn display_root_movie_download_failed_message(&self, invalid_swf: bool, fetch_error: String) {
        self.inner
            .display_root_movie_download_failed_message(invalid_swf, fetch_error)
    }
    fn message(&self, message: &str) {
        self.inner.message(message)
    }
    fn open_virtual_keyboard(&self) {
        self.inner.open_virtual_keyboard()
    }
    fn close_virtual_keyboard(&self) {
        self.inner.close_virtual_keyboard()
    }
    fn language(&self) -> LanguageIdentifier {
        self.inner.language()
    }
    fn display_unsupported_video(&self, url: Url) {
        self.inner.display_unsupported_video(url)
    }
    fn sort_device_fonts(
        &self,
        query: &FontQuery,
        register: &mut dyn FnMut(FontDefinition),
    ) -> Vec<FontQuery> {
        self.inner.sort_device_fonts(query, register)
    }
    fn display_file_open_dialog(&mut self, filters: Vec<FileFilter>) -> Option<DialogResultFuture> {
        self.inner.display_file_open_dialog(filters)
    }
    fn display_file_open_dialog_multiple(
        &mut self,
        filters: Vec<FileFilter>,
    ) -> Option<MultiDialogResultFuture> {
        self.inner.display_file_open_dialog_multiple(filters)
    }
    fn display_file_save_dialog(
        &mut self,
        file_name: String,
        title: String,
    ) -> Option<DialogResultFuture> {
        self.inner.display_file_save_dialog(file_name, title)
    }
    fn close_file_dialog(&mut self) {
        self.inner.close_file_dialog()
    }
}

/// Eagerly register system device fonts so HTML/TLF text renders.
fn register_device_fonts(player: &mut Player, db: &fontdb::Database) {
    let fallbacks: &[(DefaultFont, &[&str])] = &[
        (
            DefaultFont::Sans,
            &["Arial", "Arimo", "Liberation Sans", "DejaVu Sans", "Noto Sans"],
        ),
        (
            DefaultFont::Serif,
            &[
                "Times New Roman",
                "Tinos",
                "Liberation Serif",
                "DejaVu Serif",
                "Noto Serif",
            ],
        ),
        (
            DefaultFont::Typewriter,
            &[
                "Courier New",
                "Cousine",
                "Liberation Mono",
                "DejaVu Sans Mono",
                "Noto Sans Mono",
            ],
        ),
    ];

    let mut registered: HashSet<String> = HashSet::new();
    for (default, names) in fallbacks {
        player.set_default_font(*default, names.iter().map(|s| s.to_string()).collect());
        for name in *names {
            if !registered.insert(name.to_string()) {
                continue;
            }
            for (bold, italic) in [(false, false), (true, false), (false, true), (true, true)] {
                if let Some(font) = query_device_font(db, name, bold, italic) {
                    player.register_device_font(font);
                }
            }
        }
    }
}

fn query_device_font(
    db: &fontdb::Database,
    name: &str,
    bold: bool,
    italic: bool,
) -> Option<FontDefinition<'static>> {
    let query = fontdb::Query {
        families: &[fontdb::Family::Name(name)],
        weight: if bold {
            fontdb::Weight::BOLD
        } else {
            fontdb::Weight::NORMAL
        },
        style: if italic {
            fontdb::Style::Italic
        } else {
            fontdb::Style::Normal
        },
        ..Default::default()
    };
    let id = db.query(&query)?;
    let face = db.face(id)?;
    let data = match &face.source {
        fontdb::Source::File(path) => FontFileData::new(std::fs::read(path).ok()?),
        fontdb::Source::Binary(bin) | fontdb::Source::SharedFile(_, bin) => {
            FontFileData::new_shared(bin.clone())
        }
    };
    Some(FontDefinition::FontFile {
        name: name.to_string(),
        is_bold: bold,
        is_italic: italic,
        data,
        index: face.index,
    })
}

/// Run the native GL desktop path with the given movie (local file or remote URL).
pub fn run(
    movie_url: Url,
    spoof_url: Option<String>,
    preferences: GlobalPreferences,
) -> Result<(), Error> {
    let runtime = tokio::runtime::Runtime::new()?;
    let _enter = runtime.enter();

    let event_loop = EventLoop::<RuffleEvent>::with_user_event().build()?;
    let proxy = event_loop.create_proxy();
    let mut app = GlApp {
        movie_url,
        spoof_url,
        preferences,
        proxy,
        state: None,
    };
    event_loop.run_app(&mut app)?;
    Ok(())
}
