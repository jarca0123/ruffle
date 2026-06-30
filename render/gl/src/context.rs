//! Platform-specific GL context creation and capability detection.
//!
//! This is the only module that needs `cfg(target_family = "wasm")` branching:
//! on the web we wrap a `web_sys` WebGL1/WebGL2 context, on native we wrap a GL
//! context obtained from a loader function (e.g. from glutin). Everything past
//! construction talks to a portable [`glow::Context`].

use crate::error::Error;
use glow::HasContext as _;
use ruffle_render::quality::StageQuality;
use std::rc::Rc;

#[cfg(target_family = "wasm")]
use ruffle_web_common::JsResult;
#[cfg(target_family = "wasm")]
use wasm_bindgen::{JsCast, JsValue};
#[cfg(target_family = "wasm")]
use web_sys::HtmlCanvasElement;

/// A reference-counted handle to the shared GL context.
///
/// `glow`'s object handles (`NativeBuffer`, `NativeTexture`, ...) are plain
/// `Copy` values with no `Drop`, so each resource owner keeps a clone of this
/// and deletes its objects explicitly in `Drop`. This makes the backend
/// `!Send`, matching the single-threaded "context is current here" model.
pub(crate) type GlContext = Rc<glow::Context>;

/// Capabilities of the active GL context, detected once at construction.
pub(crate) struct Capabilities {
    /// `true` on WebGL2 / desktop GL >= 3.0 / GLES3. Enables MSAA renderbuffers
    /// and `blit_framebuffer`.
    pub is_gles3_or_webgl2: bool,

    /// `true` for WebGL or native OpenGL ES contexts (picks the GLSL ES shader
    /// preamble); `false` for desktop OpenGL (picks the GLSL 130 preamble).
    pub is_embedded: bool,

    /// Whether `REPEAT` wrapping works on non-power-of-two textures. False only
    /// on WebGL1 / GLES2.
    pub supports_npot_repeat: bool,

    /// Maximum MSAA sample count, or 0 if MSAA is unavailable.
    pub max_samples: u32,
}

/// The result of platform context creation, handed to
/// `GlRenderBackend::finish_construction`.
pub(crate) struct CreatedContext {
    pub gl: GlContext,
    pub caps: Capabilities,
    pub msaa_sample_count: u32,

    /// The original web context, kept on wasm to query things glow doesn't
    /// expose (drawing-buffer size, unmasked renderer string).
    #[cfg(target_family = "wasm")]
    pub web_context: WebContext,
}

fn detect_capabilities(gl: &glow::Context) -> Capabilities {
    let version = gl.version();
    let is_embedded = version.is_embedded;
    let is_gles3_or_webgl2 = version.major >= 3;
    // Desktop GL (>= 2.0, non-embedded) and WebGL2/GLES3 support NPOT REPEAT;
    // WebGL1 / GLES2 do not.
    let supports_npot_repeat = !is_embedded || is_gles3_or_webgl2;
    let max_samples = if is_gles3_or_webgl2 {
        // SAFETY: a context is current; MAX_SAMPLES is valid on GLES3/GL3+.
        unsafe { gl.get_parameter_i32(glow::MAX_SAMPLES).max(0) as u32 }
    } else {
        0
    };
    Capabilities {
        is_gles3_or_webgl2,
        is_embedded,
        supports_npot_repeat,
        max_samples,
    }
}

pub(crate) fn recommended_msaa_samples(caps: &Capabilities, quality: StageQuality) -> u32 {
    if caps.is_gles3_or_webgl2 {
        let want = quality.sample_count().min(4);
        if caps.max_samples > 0 {
            want.min(caps.max_samples)
        } else {
            want
        }
    } else {
        // No MSAA on WebGL1 / GLES2 (these rely on context-level `antialias`).
        1
    }
}

/// Creates a GL context from a native loader function (e.g. glutin's
/// `get_proc_address`).
///
/// # Safety
/// The caller must have made a GL context current on the calling thread before
/// invoking this, and `loader` must return valid function pointers.
#[cfg(not(target_family = "wasm"))]
pub(crate) unsafe fn create_from_loader<F>(
    loader: F,
    quality: StageQuality,
) -> Result<CreatedContext, Error>
where
    F: FnMut(&str) -> *const std::ffi::c_void,
{
    // SAFETY: forwarded to the caller's contract above.
    let glow_ctx = unsafe { glow::Context::from_loader_function(loader) };
    let gl: GlContext = Rc::new(glow_ctx);
    let caps = detect_capabilities(&gl);
    let msaa = recommended_msaa_samples(&caps, quality);
    Ok(CreatedContext {
        gl,
        caps,
        msaa_sample_count: msaa,
    })
}

/// Creates a GL context from an HTML canvas, preferring WebGL2 and falling back
/// to a fully-featured WebGL1 path.
#[cfg(target_family = "wasm")]
pub(crate) fn create_for_webgl(
    canvas: &HtmlCanvasElement,
    is_transparent: bool,
    quality: StageQuality,
) -> Result<CreatedContext, Error> {
    let options = [
        ("stencil", JsValue::TRUE),
        ("alpha", JsValue::from_bool(is_transparent)),
        ("antialias", JsValue::FALSE),
        ("depth", JsValue::FALSE),
        ("failIfMajorPerformanceCaveat", JsValue::TRUE), // fail if no GPU available
        ("premultipliedAlpha", JsValue::TRUE),
    ];
    let context_options = js_sys::Object::new();
    for (name, value) in options.into_iter() {
        js_sys::Reflect::set(&context_options, &JsValue::from(name), &value).warn_on_error();
    }

    // Attempt a WebGL2 context, falling back to WebGL1 if unavailable.
    if let Ok(Some(context)) = canvas.get_context_with_context_options("webgl2", &context_options) {
        log::info!("Creating WebGL2 context.");
        let gl2 = context
            .dyn_into::<web_sys::WebGl2RenderingContext>()
            .map_err(|_| Error::CantCreateGLContext)?;
        let gl: GlContext = Rc::new(glow::Context::from_webgl2_context(gl2.clone()));
        let caps = detect_capabilities(&gl);
        let msaa = recommended_msaa_samples(&caps, quality);
        Ok(CreatedContext {
            gl,
            caps,
            msaa_sample_count: msaa,
            web_context: WebContext::WebGl2(gl2),
        })
    } else {
        // Request antialiasing on WebGL1, because there isn't general MSAA support.
        js_sys::Reflect::set(
            &context_options,
            &JsValue::from("antialias"),
            &JsValue::TRUE,
        )
        .warn_on_error();

        log::info!("Falling back to WebGL1.");
        let context = canvas
            .get_context_with_context_options("webgl", &context_options)
            .into_js_result()?
            .ok_or(Error::CantCreateGLContext)?;
        let gl1 = context
            .dyn_into::<web_sys::WebGlRenderingContext>()
            .map_err(|_| Error::CantCreateGLContext)?;
        let gl: GlContext = Rc::new(glow::Context::from_webgl1_context(gl1.clone()));

        // u32 index buffers require OES_element_index_uint on WebGL1. glow probes
        // it (and OES_vertex_array_object) during `from_webgl1_context`, so this
        // also enables it for the context.
        if !gl.supported_extensions().contains("OES_element_index_uint") {
            return Err(Error::OESExtensionNotFound);
        }

        let caps = detect_capabilities(&gl);
        let msaa = recommended_msaa_samples(&caps, quality);
        Ok(CreatedContext {
            gl,
            caps,
            msaa_sample_count: msaa,
            web_context: WebContext::WebGl1(gl1),
        })
    }
}

/// The original `web_sys` context, retained for queries glow doesn't expose.
#[cfg(target_family = "wasm")]
pub(crate) enum WebContext {
    WebGl1(web_sys::WebGlRenderingContext),
    WebGl2(web_sys::WebGl2RenderingContext),
}

#[cfg(target_family = "wasm")]
impl WebContext {
    /// The actual drawing-buffer size in device pixels. Returns zero when the
    /// context is lost (callers clamp viewport dimensions to this).
    pub fn drawing_buffer_size(&self) -> (i32, i32) {
        match self {
            WebContext::WebGl1(gl) => (gl.drawing_buffer_width(), gl.drawing_buffer_height()),
            WebContext::WebGl2(gl) => (gl.drawing_buffer_width(), gl.drawing_buffer_height()),
        }
    }
}
