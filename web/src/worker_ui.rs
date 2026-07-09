//! UI backend for the player-in-worker path.
//!
//! Mirrors [`WebUiBackend`](crate::ui::WebUiBackend), but a worker has no DOM:
//! no canvas element to style, no `document`/`navigator.clipboard`, no
//! `JavascriptPlayer`. So the DOM-bound operations are bridged to the main
//! thread over the [`WorkerBridge`] and applied there (see
//! `WorkerPlayerHandle::service_ui`):
//!
//! * **Clipboard** — paste text is captured by the main thread's DOM `paste`
//!   event (the only place it's available) and read back here; copy requests are
//!   queued for the main thread to write via `navigator.clipboard`.
//! * **Mouse cursor** — the worker records the desired cursor; the main thread
//!   applies the CSS `cursor` to the canvas it owns.
//! * **Fullscreen** — requested here, performed on the main thread's canvas.
//!
//! The rest either work on the worker directly (`language` via the worker's
//! `navigator`) or aren't meaningful on the bare-canvas worker embed (no
//! `JavascriptPlayer` for virtual keyboard / modal messages) and log instead.

use std::sync::Arc;

use ruffle_core::backend::ui::{
    DialogResultFuture, FileDialogResult, FileFilter, FontDefinition, FullscreenError,
    LanguageIdentifier, MouseCursor, MultiDialogResultFuture, MultiFileDialogResult, UiBackend,
    US_ENGLISH,
};
use ruffle_core::font::FontQuery;
use url::Url;
use wasm_bindgen::JsCast;

use crate::worker_bridge::WorkerBridge;

// Cursor codes exchanged over the bridge (main thread maps them to CSS `cursor`).
pub const CURSOR_HIDDEN: u8 = 0;
pub const CURSOR_ARROW: u8 = 1;
pub const CURSOR_HAND: u8 = 2;
pub const CURSOR_IBEAM: u8 = 3;
pub const CURSOR_GRAB: u8 = 4;

/// Maps a cursor code back to a CSS `cursor` keyword (used by the main thread).
pub fn cursor_css(code: u8) -> &'static str {
    match code {
        CURSOR_HIDDEN => "none",
        CURSOR_HAND => "pointer",
        CURSOR_IBEAM => "text",
        CURSOR_GRAB => "grab",
        _ => "auto",
    }
}

pub struct WebWorkerUiBackend {
    bridge: Arc<WorkerBridge>,
    cursor_visible: bool,
    cursor: MouseCursor,
    language: LanguageIdentifier,
}

impl WebWorkerUiBackend {
    pub fn new(bridge: Arc<WorkerBridge>) -> Self {
        Self {
            bridge,
            cursor_visible: true,
            cursor: MouseCursor::Arrow,
            language: worker_language().unwrap_or_else(|| US_ENGLISH.clone()),
        }
    }

    /// Pushes the current (visibility + shape) cursor state to the main thread.
    fn push_cursor(&self) {
        let code = if !self.cursor_visible {
            CURSOR_HIDDEN
        } else {
            match self.cursor {
                MouseCursor::Arrow => CURSOR_ARROW,
                MouseCursor::Hand => CURSOR_HAND,
                MouseCursor::IBeam => CURSOR_IBEAM,
                MouseCursor::Grab => CURSOR_GRAB,
            }
        };
        self.bridge.set_cursor(code);
    }
}

impl UiBackend for WebWorkerUiBackend {
    fn mouse_visible(&self) -> bool {
        self.cursor_visible
    }

    fn set_mouse_visible(&mut self, visible: bool) {
        self.cursor_visible = visible;
        self.push_cursor();
    }

    fn set_mouse_cursor(&mut self, cursor: MouseCursor) {
        self.cursor = cursor;
        self.push_cursor();
    }

    fn clipboard_content(&mut self) -> String {
        // Served from the bridge, which the main thread fills on the DOM `paste`
        // event (a worker can't read the clipboard directly).
        self.bridge.clipboard()
    }

    fn clipboard_available(&mut self) -> bool {
        // Assume available, like `WebUiBackend`: pasting works via the `paste` event.
        true
    }

    fn set_clipboard_content(&mut self, content: String) {
        // Keep the shared buffer current so an in-app paste right after a copy
        // sees it, and queue the system-clipboard write for the main thread.
        self.bridge.set_clipboard(content.clone());
        self.bridge.push_clipboard_write(content);
    }

    fn set_fullscreen(&mut self, is_full: bool) -> Result<(), FullscreenError> {
        self.bridge.request_fullscreen(is_full);
        Ok(())
    }

    fn display_root_movie_download_failed_message(&self, invalid_swf: bool, fetch_error: String) {
        tracing::error!(
            "worker ui: root movie download failed (invalid_swf={invalid_swf}): {fetch_error}"
        );
    }

    fn message(&self, message: &str) {
        tracing::info!("worker ui message: {message}");
    }

    fn open_virtual_keyboard(&self) {
        // Needs a `JavascriptPlayer`-managed hidden input on the main thread,
        // which the bare-canvas worker embed doesn't have.
        tracing::debug!("worker ui: open_virtual_keyboard unsupported on this path");
    }

    fn close_virtual_keyboard(&self) {
        tracing::debug!("worker ui: close_virtual_keyboard unsupported on this path");
    }

    fn language(&self) -> LanguageIdentifier {
        self.language.clone()
    }

    fn display_unsupported_video(&self, url: Url) {
        tracing::warn!("worker ui: unsupported video: {url}");
    }

    fn load_device_font(&self, _query: &FontQuery, _register: &mut dyn FnMut(FontDefinition)) {
        // Device fonts are provided upfront at Player creation on this path (as
        // when the main backend runs without the canvas font renderer).
    }

    fn sort_device_fonts(
        &self,
        _query: &FontQuery,
        _register: &mut dyn FnMut(FontDefinition),
    ) -> Vec<FontQuery> {
        Vec::new()
    }

    fn display_file_open_dialog(&mut self, _filters: Vec<FileFilter>) -> Option<DialogResultFuture> {
        // File dialogs need the DOM (`rfd`'s `<input type=file>`); unsupported on
        // a worker. Resolve as canceled rather than hang the AVM future.
        Some(Box::pin(async move { Ok(FileDialogResult::Canceled) }))
    }

    fn display_file_open_dialog_multiple(
        &mut self,
        _filters: Vec<FileFilter>,
    ) -> Option<MultiDialogResultFuture> {
        Some(Box::pin(async move { Ok(MultiFileDialogResult::Canceled) }))
    }

    fn close_file_dialog(&mut self) {}

    fn display_file_save_dialog(
        &mut self,
        _file_name: String,
        _title: String,
    ) -> Option<DialogResultFuture> {
        None
    }
}

/// Reads the browser UI language from the worker's `navigator` (a worker has a
/// `WorkerNavigator` with `language`, just no `window`).
fn worker_language() -> Option<LanguageIdentifier> {
    let scope: web_sys::WorkerGlobalScope = js_sys::global().unchecked_into();
    scope.navigator().language().and_then(|l| l.parse().ok())
}
