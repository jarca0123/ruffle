#![deny(clippy::unwrap_used)]
// BitmapHandle/ShapeHandle wrap their payload in `Arc`, but our payload holds an
// `Rc<glow::Context>` and is therefore `!Send`/`!Sync`. This mirrors the old
// webgl backend and is sound because the backend is single-threaded.
#![allow(clippy::arc_with_non_send_sync)]

//! A portable OpenGL render backend built on [`glow`].
//!
//! This supersedes the `ruffle_render_webgl` backend: the same code compiles to
//! WebGL1/WebGL2 on wasm and to native OpenGL on the desktop (via a loader
//! function, e.g. from glutin). WebGL1 is a complete first-class path — WebGL2,
//! GLES3 and desktop GL >= 3.0 only add MSAA on top.

mod agal;
mod context;
mod context3d;
mod error;
mod filters;
mod pixelbender;
mod pool;
mod shader;

pub use error::Error;

use bytemuck::{Pod, Zeroable};
use context::{Capabilities, CreatedContext, GlContext};
use glow::HasContext as _;
use ruffle_render::backend::{
    BitmapCacheEntry, Context3D, Context3DProfile, PixelBenderOutput, PixelBenderTarget,
    RenderBackend, RenderOffscreenBatches, ShapeHandle, ShapeHandleImpl, ViewportDimensions,
};
use ruffle_render::bitmap::{
    Bitmap, BitmapFormat, BitmapHandle, BitmapHandleImpl, BitmapSource, PixelRegion, PixelSnapping,
    RgbaBufRead, SyncHandle,
};
use ruffle_render::commands::{Command, CommandHandler, CommandList, RenderBlendMode};
use ruffle_render::pixel_bender::{
    PixelBenderShader, PixelBenderShaderHandle, PixelBenderType,
};
use ruffle_render::pixel_bender_support::{ImageInputTexture, PixelBenderShaderArgument};
use ruffle_render::error::Error as BitmapError;
use ruffle_render::filters::Filter;
use ruffle_render::matrix::Matrix;
use ruffle_render::quality::StageQuality;
use ruffle_render::shape_utils::{DistilledShape, GradientType};
use ruffle_render::tessellator::{
    Gradient as TessGradient, ShapeTessellator, Vertex as TessVertex,
};
use ruffle_render::transform::Transform;
use shader::{ShaderProgram, ShaderUniform};
use std::any::Any;
use std::borrow::Cow;
use std::fmt;
use std::num::NonZeroU32;
use std::sync::Arc;
use swf::{BlendMode, Color, Rectangle, Twips};

const COLOR_VERTEX_GLSL: &str = include_str!("../shaders/color.vert");
const COLOR_FRAGMENT_GLSL: &str = include_str!("../shaders/color.frag");
const BATCH_COLOR_VERTEX_GLSL: &str = include_str!("../shaders/batch_color.vert");
const BATCH_COLOR_FRAGMENT_GLSL: &str = include_str!("../shaders/batch_color.frag");
const BATCH_BITMAP_VERTEX_GLSL: &str = include_str!("../shaders/batch_bitmap.vert");
const COPY_FRAGMENT_GLSL: &str = include_str!("../shaders/copy.frag");
// Fullscreen NDC quad for PixelBender; the fragment stage reads gl_FragCoord.
const PIXELBENDER_VERTEX_GLSL: &str =
    "attribute vec2 position;\nvoid main() {\n    gl_Position = vec4(position, 0.0, 1.0);\n}\n";

/// A compiled PixelBender shader: the GLSL program plus the metadata needed to
/// bind parameters and image inputs at run time.
struct GlPixelBenderShader {
    gl: GlContext,
    program: glow::Program,
    shader: PixelBenderShader,
    float_slots: usize,
    int_slots: usize,
    param_slots: Vec<Option<pixelbender::ParamSlot>>,
    output_channels: usize,
    position_loc: Option<u32>,
    u_float_params: Option<glow::UniformLocation>,
    u_int_params: Option<glow::UniformLocation>,
    u_zeroed: Option<glow::UniformLocation>,
    u_out_size: Option<glow::UniformLocation>,
    // (input index, sampler location, size location, nearest filtering)
    u_inputs: Vec<(
        u8,
        Option<glow::UniformLocation>,
        Option<glow::UniformLocation>,
        bool,
    )>,
}

impl std::fmt::Debug for GlPixelBenderShader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GlPixelBenderShader")
            .field("name", &self.shader.name)
            .finish()
    }
}

impl ruffle_render::pixel_bender::PixelBenderShaderImpl for GlPixelBenderShader {
    fn parsed_shader(&self) -> &PixelBenderShader {
        &self.shader
    }
}

impl Drop for GlPixelBenderShader {
    fn drop(&mut self) {
        unsafe { self.gl.delete_program(self.program) };
    }
}

/// Writes a PixelBender parameter value into the float/int uniform-array data at
/// its assigned slot (matching the naga port's packing).
fn write_param_value(
    value: &PixelBenderType,
    slot: pixelbender::ParamSlot,
    float_data: &mut [f32],
    int_data: &mut [i32],
) {
    use PixelBenderType::*;
    let putf = |d: &mut [f32], off: usize, v: [f32; 4]| {
        if off + 4 <= d.len() {
            d[off..off + 4].copy_from_slice(&v);
        }
    };
    let puti = |d: &mut [i32], off: usize, v: [i32; 4]| {
        if off + 4 <= d.len() {
            d[off..off + 4].copy_from_slice(&v);
        }
    };
    let b = slot.offset * 4;
    match value {
        TFloat(a) => putf(float_data, b, [*a, 0.0, 0.0, 0.0]),
        TFloat2(a, c) => putf(float_data, b, [*a, *c, 0.0, 0.0]),
        TFloat3(a, c, e) => putf(float_data, b, [*a, *c, *e, 0.0]),
        TFloat4(a, c, e, f) => putf(float_data, b, [*a, *c, *e, *f]),
        TFloat2x2(arr) => putf(float_data, b, *arr),
        TFloat3x3(arr) => {
            for c in 0..3 {
                putf(
                    float_data,
                    b + c * 4,
                    [arr[c * 3], arr[c * 3 + 1], arr[c * 3 + 2], 0.0],
                );
            }
        }
        TFloat4x4(arr) => {
            for c in 0..4 {
                putf(
                    float_data,
                    b + c * 4,
                    [arr[c * 4], arr[c * 4 + 1], arr[c * 4 + 2], arr[c * 4 + 3]],
                );
            }
        }
        TInt(a) | TBool(a) => puti(int_data, b, [*a as i32, 0, 0, 0]),
        TInt2(a, c) | TBool2(a, c) => puti(int_data, b, [*a as i32, *c as i32, 0, 0]),
        TInt3(a, c, e) | TBool3(a, c, e) => {
            puti(int_data, b, [*a as i32, *c as i32, *e as i32, 0])
        }
        TInt4(a, c, e, f) | TBool4(a, c, e, f) => {
            puti(int_data, b, [*a as i32, *c as i32, *e as i32, *f as i32])
        }
        TString(_) => {}
    }
}
const TEXTURE_VERTEX_GLSL: &str = include_str!("../shaders/texture.vert");
const GRADIENT_FRAGMENT_GLSL: &str = include_str!("../shaders/gradient.frag");
const BITMAP_FRAGMENT_GLSL: &str = include_str!("../shaders/bitmap.frag");
const PERSPECTIVE_BITMAP_VERTEX_GLSL: &str = include_str!("../shaders/perspective_bitmap.vert");
const PERSPECTIVE_BITMAP_FRAGMENT_GLSL: &str = include_str!("../shaders/perspective_bitmap.frag");

const NUM_VERTEX_ATTRIBUTES: u32 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MaskState {
    NoMask,
    DrawMaskStencil,
    DrawMaskedContent,
    ClearMaskStencil,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Vertex {
    position: [f32; 2],
    color: u32,
}

/// Vertex for the bitmap batcher: world-transformed position + texture UV.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct BitmapVertex {
    position: [f32; 2],
    uv: [f32; 2],
}

/// Vertex for perspective-correct textured triangles: shape-space position (px)
/// plus `(u, v, t = 1/w)`. The vertex shader forms `(u*t, v*t, t)` and the fragment
/// shader divides, giving perspective-correct sampling with no CPU subdivision.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct PerspectiveVertex {
    position: [f32; 2],
    uvt: [f32; 3],
}

impl From<ruffle_render::shape_utils::PerspectiveVertex> for PerspectiveVertex {
    fn from(v: ruffle_render::shape_utils::PerspectiveVertex) -> Self {
        Self {
            position: [v.x, v.y],
            uvt: [v.u, v.v, v.t],
        }
    }
}

/// State a run of batched bitmaps must share. A change flushes the batch.
#[derive(Copy, Clone, PartialEq)]
struct BitmapBatchKey {
    texture: glow::Texture,
    smoothing: bool,
    mult: [f32; 4],
    add: [f32; 4],
}

impl From<TessVertex> for Vertex {
    fn from(vertex: TessVertex) -> Self {
        Self {
            position: [vertex.x, vertex.y],
            color: u32::from_le_bytes([
                vertex.color.r,
                vertex.color.g,
                vertex.color.b,
                vertex.color.a,
            ]),
        }
    }
}

pub struct GlRenderBackend {
    gl: GlContext,
    caps: Capabilities,

    // Kept on the web to query things glow can't (drawing-buffer size).
    #[cfg(target_family = "wasm")]
    web_context: context::WebContext,

    // The frame buffers used for resolving MSAA.
    msaa_buffers: Option<MsaaBuffers>,
    msaa_sample_count: u32,

    // When presenting to a real window (desktop), the MSAA-resolve present quad
    // must flip vertically: the scene is drawn top-down (Flash-top at framebuffer
    // row 0, so headless `read_framebuffer` needs no flip), but a window scans the
    // default framebuffer with row 0 at the *bottom*, which would show the movie
    // upside down. Headless capture leaves this false.
    present_flip: bool,

    color_program: ShaderProgram,
    bitmap_program: ShaderProgram,
    gradient_program: ShaderProgram,
    /// Shared 256x1 RGBA8 ramp texture, re-uploaded per gradient draw. Matches
    /// wgpu, which bakes each gradient into a 256-texel texture and samples it
    /// with hardware linear filtering (see `Gradient::ramp`).
    gradient_texture: glow::Texture,
    batch_color_program: ShaderProgram,
    /// Raw texture passthrough used to seed offscreen MSAA buffers.
    copy_program: ShaderProgram,
    /// Static NDC fullscreen quad ([-1,1]) for PixelBender passes, plus a VAO
    /// (required on core profiles) reconfigured per pass for the shader's
    /// `position` attribute location.
    pb_quad_vbo: glow::Buffer,
    pb_vao: glow::VertexArray,

    // Shared dynamic buffers for the solid-color draw batcher.
    batch_vao: glow::VertexArray,
    batch_vbo: glow::Buffer,
    batch_ibo: glow::Buffer,
    batch_vertices: Vec<Vertex>,
    batch_indices: Vec<u32>,

    // Bitmap batcher: consecutive bitmaps that share a texture, smoothing, and
    // color transform merge into one draw. Only one of the color/bitmap batches
    // is ever pending at a time (the other is flushed first) to preserve draw
    // order.
    batch_bitmap_program: ShaderProgram,
    batch_bitmap_vao: glow::VertexArray,
    batch_bitmap_vbo: glow::Buffer,
    batch_bitmap_ibo: glow::Buffer,
    // Perspective-correct textured triangles (Graphics.drawTriangles + 3-comp uvt).
    perspective_bitmap_program: ShaderProgram,
    perspective_uvt_location: Option<u32>,
    batch_bitmap_vertices: Vec<BitmapVertex>,
    batch_bitmap_indices: Vec<u32>,
    batch_bitmap_key: Option<BitmapBatchKey>,
    // Keeps the current bitmap batch's texture alive until it flushes. The batch
    // stores only the raw `glow::Texture` name (in the `Copy` key), so without
    // holding the handle the caller could drop its last `BitmapHandle` before the
    // flush, `RegistryData::drop` would delete the GL texture, and the flush would
    // bind a freed name (GL_INVALID_OPERATION). One texture per pending batch.
    batch_bitmap_handle: Option<BitmapHandle>,
    // Blend func (eq+factors) currently active for non-batched draws, and the one
    // the pending batch was filled under. A run of same-func draws (e.g. 500
    // Multiply puffs) accumulates into one draw; the batch is flushed under
    // `batch_blend` and only when the func actually changes.
    active_hw_blend: [u32; 6],
    batch_blend: [u32; 6],

    shape_tessellator: ShapeTessellator,

    // Lazily-built GPU filter programs (see `filters` module).
    filters: Option<filters::Filters>,

    // Recycled RGBA8 textures for filter/offscreen passes (see `pool` module).
    pool: pool::TexturePool,

    // Color-only framebuffer reused for filter copy-backs, pixel copies and
    // readback; an offscreen framebuffer (with a resized-on-demand stencil
    // renderbuffer) reused for render-to-texture. Both created once.
    scratch_fbo: glow::Framebuffer,
    offscreen_fbo: glow::Framebuffer,
    offscreen_stencil: glow::Renderbuffer,
    offscreen_stencil_dims: (i32, i32),

    // Multisampled offscreen target for antialiasing a complex blend's source
    // content (which would otherwise render single-sample and look jagged). The
    // multisampled result is resolved into the caller's single-sample texture.
    blend_msaa_fbo: glow::Framebuffer,
    blend_msaa_color: glow::Renderbuffer,
    blend_msaa_stencil: glow::Renderbuffer,
    blend_msaa_dims: (i32, i32),
    // Single-sample resolve target for `blend_msaa`, so a complex blend nested
    // inside an MSAA offscreen pass (e.g. an Overlay inside a cacheAsBitmap) can
    // read its multisampled parent back as a regular texture.
    blend_msaa_resolve_fbo: glow::Framebuffer,
    blend_msaa_resolve_color: glow::Texture,
    // Whether the offscreen pass currently being rendered uses `blend_msaa`
    // (true) or the single-sample `offscreen_fbo` (false). Lets a nested complex
    // blend know how to read the parent.
    offscreen_msaa: bool,
    // Color texture of the current single-sample offscreen pass (the parent
    // content lives here directly, no resolve needed). `None` for MSAA/screen.
    offscreen_color: Option<glow::Texture>,
    // Testing knob (RUFFLE_GL_FORCE_OLDEST): also drop GL_MIN/MAX so Darken/
    // Lighten take the GLES2 fallback path.
    force_oldest: bool,

    // Single-sample offscreen framebuffer for `Layer` blends: children render
    // into it so that Alpha/Erase (and nested complex blends) composite against
    // the layer's own transparent content rather than the opaque stage.
    layer_fbo: glow::Framebuffer,
    layer_stencil: glow::Renderbuffer,
    layer_stencil_dims: (i32, i32),

    // The current complex-blend draw target. `None` = the screen (MSAA/default);
    // `Some` = an active `Layer` offscreen. `target_origin` is the stage-pixel
    // top-left of the target's local space (so a blend region in stage pixels can
    // be made target-local).
    target_fbo: Option<glow::Framebuffer>,
    target_texture: Option<glow::Texture>,
    target_origin: (i32, i32),

    color_quad_draws: Vec<Draw>,
    bitmap_quad_draws: Vec<Draw>,

    mask_state: MaskState,
    num_masks: u32,
    mask_state_dirty: bool,
    is_transparent: bool,

    /// True while rendering into an offscreen texture (cacheAsBitmap, a complex
    /// blend's source, or `render_offscreen`). Complex blends read and composite
    /// against the screen framebuffer, so they must not run nested inside an
    /// offscreen pass — they would corrupt the screen.
    in_offscreen: bool,

    active_program: *const ShaderProgram,
    blend_modes: Vec<RenderBlendMode>,
    mult_color: Option<[f32; 4]>,
    add_color: Option<[f32; 4]>,

    renderbuffer_width: i32,
    renderbuffer_height: i32,
    view_matrix: [[f32; 4]; 4],

    // Currently unused except to expose via `viewport_dimensions`.
    viewport_scale_factor: f64,
}

pub(crate) struct RegistryData {
    pub(crate) gl: GlContext,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) texture: glow::Texture,
}

impl fmt::Debug for RegistryData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RegistryData")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("texture", &self.texture)
            .finish()
    }
}

impl Drop for RegistryData {
    fn drop(&mut self) {
        unsafe { self.gl.delete_texture(self.texture) };
    }
}

impl BitmapHandleImpl for RegistryData {}

fn as_registry_data(handle: &BitmapHandle) -> &RegistryData {
    <dyn Any>::downcast_ref(&*handle.0).expect("Bitmap handle must be gl RegistryData")
}

/// Maps a glow object-creation failure to a `BitmapError`. These essentially
/// never fail in practice (context loss / OOM only).
#[cold]
fn bitmap_gl_error(message: String) -> BitmapError {
    log::error!("GL bitmap error: {message}");
    #[cfg(target_family = "wasm")]
    {
        BitmapError::JavascriptError(wasm_bindgen::JsValue::from_str(&message))
    }
    #[cfg(not(target_family = "wasm"))]
    {
        BitmapError::Unimplemented(Cow::Owned(format!("GL error: {message}")))
    }
}

fn color_to_rgba(c: Color) -> [f32; 4] {
    [
        c.r as f32 / 255.0,
        c.g as f32 / 255.0,
        c.b as f32 / 255.0,
        c.a as f32 / 255.0,
    ]
}

/// `round(c * a / 255)` for `u8` operands, with integers. Exact for all 256×256
/// inputs and bit-identical to `(c as f32 * a as f32 / 255.0).round()` (the
/// product `c*a` never lands on an exact half), but with no per-channel `roundf`.
#[inline]
fn mul_div255_round(c: u8, a: u8) -> u8 {
    let t = c as u32 * a as u32 + 128;
    ((t + (t >> 8)) >> 8) as u8
}

/// Like [`color_to_rgba`], but premultiplied by alpha. Bevel composites in
/// premultiplied space (matching wgpu's bevel filter).
fn premultiplied_rgba(c: Color) -> [f32; 4] {
    let a = c.a as f32 / 255.0;
    [
        c.r as f32 / 255.0 * a,
        c.g as f32 / 255.0 * a,
        c.b as f32 / 255.0 * a,
        a,
    ]
}

/// GL convolution uniforms from a `swf::ConvolutionFilter`, or `None` if the
/// kernel is empty or larger than the shader supports.
#[allow(clippy::type_complexity)]
fn convolution_params(
    f: &swf::ConvolutionFilter,
) -> Option<(
    [f32; filters::MAX_CONVOLUTION_TAPS],
    f32,
    f32,
    f32,
    f32,
    [f32; 4],
    bool,
    bool,
)> {
    let cols = f.num_matrix_cols as usize;
    let rows = f.num_matrix_rows as usize;
    let count = cols * rows;
    if count == 0 || count > filters::MAX_CONVOLUTION_TAPS || f.matrix.len() < count {
        return None;
    }
    let mut kernel = [0.0f32; filters::MAX_CONVOLUTION_TAPS];
    kernel[..count].copy_from_slice(&f.matrix[..count]);
    // A divisor of 0 is ignored (uses 1) per the Flash spec.
    let divisor = if f.divisor == 0.0 { 1.0 } else { f.divisor };
    Some((
        kernel,
        cols as f32,
        rows as f32,
        divisor,
        f.bias / 255.0,
        premultiplied_rgba(f.default_color),
        f.is_clamped(),
        f.is_preserve_alpha(),
    ))
}

/// Displacement mode index matching the shader (0 wrap, 1 clamp, 2 ignore, 3 color).
fn displacement_mode(mode: ruffle_render::filters::DisplacementMapFilterMode) -> f32 {
    use ruffle_render::filters::DisplacementMapFilterMode as M;
    match mode {
        M::Wrap => 0.0,
        M::Clamp => 1.0,
        M::Ignore => 2.0,
        M::Color => 3.0,
    }
}

fn lerp_color(a: Color, b: Color, t: f32) -> Color {
    let lerp = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t).round() as u8;
    Color {
        r: lerp(a.r, b.r),
        g: lerp(a.g, b.g),
        b: lerp(a.b, b.b),
        a: lerp(a.a, b.a),
    }
}

/// Samples a gradient (records sorted by ratio) at a 0-255 position.
fn sample_gradient(colors: &[swf::GradientRecord], ratio: u8) -> Color {
    let first = &colors[0];
    let last = &colors[colors.len() - 1];
    if ratio <= first.ratio {
        return first.color;
    }
    if ratio >= last.ratio {
        return last.color;
    }
    for w in colors.windows(2) {
        if ratio >= w[0].ratio && ratio <= w[1].ratio {
            let span = (w[1].ratio - w[0].ratio) as f32;
            let t = if span > 0.0 {
                (ratio - w[0].ratio) as f32 / span
            } else {
                0.0
            };
            return lerp_color(w[0].color, w[1].color, t);
        }
    }
    last.color
}

/// Builds the 256-entry premultiplied RGBA ramp for a gradient glow/bevel.
fn build_gradient_ramp(colors: &[swf::GradientRecord]) -> [u8; filters::GRADIENT_RAMP_SIZE * 4] {
    let mut ramp = [0u8; filters::GRADIENT_RAMP_SIZE * 4];
    if colors.is_empty() {
        return ramp;
    }
    for i in 0..filters::GRADIENT_RAMP_SIZE {
        let c = sample_gradient(colors, i as u8);
        let a = c.a as u32;
        ramp[i * 4] = (c.r as u32 * a / 255) as u8;
        ramp[i * 4 + 1] = (c.g as u32 * a / 255) as u8;
        ramp[i * 4 + 2] = (c.b as u32 * a / 255) as u8;
        ramp[i * 4 + 3] = c.a;
    }
    ramp
}

fn check_error(gl: &glow::Context, msg: &'static str) -> Result<(), Error> {
    let error = unsafe { gl.get_error() };
    if error == glow::NO_ERROR {
        Ok(())
    } else {
        Err(Error::GLError(msg, error))
    }
}

impl GlRenderBackend {
    /// Creates the backend from an HTML canvas, preferring WebGL2 and falling
    /// back to a complete WebGL1 path.
    #[cfg(target_family = "wasm")]
    pub fn new_for_webgl(
        canvas: &web_sys::HtmlCanvasElement,
        is_transparent: bool,
        quality: StageQuality,
    ) -> Result<Self, Error> {
        let created =
            context::create_for_webgl(context::WebGlCanvas::Html(canvas), is_transparent, quality)?;
        Self::finish_construction(created, is_transparent)
    }

    /// Creates the backend from an [`OffscreenCanvas`], for running the renderer
    /// on a worker thread (the player-in-worker path). Otherwise identical to
    /// [`Self::new_for_webgl`].
    ///
    /// [`OffscreenCanvas`]: web_sys::OffscreenCanvas
    #[cfg(target_family = "wasm")]
    pub fn new_for_webgl_offscreen(
        canvas: &web_sys::OffscreenCanvas,
        is_transparent: bool,
        quality: StageQuality,
    ) -> Result<Self, Error> {
        let created = context::create_for_webgl(
            context::WebGlCanvas::Offscreen(canvas),
            is_transparent,
            quality,
        )?;
        let mut backend = Self::finish_construction(created, is_transparent)?;
        // Presentation from an `OffscreenCanvas` (`transferControlToOffscreen`) is
        // vertically mirrored versus a normal on-page canvas, so flip the present
        // quad to show the movie upright.
        backend.set_present_flipped(true);
        Ok(backend)
    }

    /// Creates the backend from a native GL loader function (e.g. glutin's
    /// `get_proc_address`).
    ///
    /// # Safety
    /// A GL context must be current on the calling thread, and `loader` must
    /// return valid GL function pointers.
    #[cfg(not(target_family = "wasm"))]
    pub unsafe fn new_from_loader_function<F>(
        loader: F,
        is_transparent: bool,
        quality: StageQuality,
    ) -> Result<Self, Error>
    where
        F: FnMut(&str) -> *const std::ffi::c_void,
    {
        // SAFETY: forwarded to the caller's contract above.
        let created = unsafe { context::create_from_loader(loader, quality)? };
        Self::finish_construction(created, is_transparent)
    }

    fn finish_construction(created: CreatedContext, is_transparent: bool) -> Result<Self, Error> {
        let gl = created.gl;
        let mut caps = created.caps;
        let mut msaa_sample_count = created.msaa_sample_count;

        // Testing knob: pretend we're on the oldest GL (WebGL1/GLES2 feature set)
        // even on a modern desktop context — disables MSAA and the GL_MIN/MAX
        // blend equations, exercising the single-sample offscreen / fallback code
        // paths. Keeps the desktop shader dialect so shaders still compile.
        let force_oldest = std::env::var_os("RUFFLE_GL_FORCE_OLDEST").is_some();
        if force_oldest {
            caps.is_gles3_or_webgl2 = false;
            msaa_sample_count = 1;
            log::warn!("RUFFLE_GL_FORCE_OLDEST: forcing oldest-GL feature set (no MSAA, no MIN/MAX)");
        }

        let is_embedded = caps.is_embedded;

        if log::log_enabled!(log::Level::Info) {
            let renderer = unsafe { gl.get_parameter_string(glow::RENDERER) };
            let version = unsafe { gl.get_parameter_string(glow::VERSION) };
            log::info!("OpenGL graphics driver: {renderer}");
            log::info!(
                "OpenGL version: {version} (gles3_or_webgl2={}, embedded={}, msaa_samples={})",
                caps.is_gles3_or_webgl2,
                caps.is_embedded,
                msaa_sample_count,
            );
        }

        let color_vertex =
            shader::compile_shader(&gl, is_embedded, glow::VERTEX_SHADER, COLOR_VERTEX_GLSL)?;
        let texture_vertex =
            shader::compile_shader(&gl, is_embedded, glow::VERTEX_SHADER, TEXTURE_VERTEX_GLSL)?;
        let color_fragment =
            shader::compile_shader(&gl, is_embedded, glow::FRAGMENT_SHADER, COLOR_FRAGMENT_GLSL)?;
        let bitmap_fragment = shader::compile_shader(
            &gl,
            is_embedded,
            glow::FRAGMENT_SHADER,
            BITMAP_FRAGMENT_GLSL,
        )?;
        let gradient_fragment = shader::compile_shader(
            &gl,
            is_embedded,
            glow::FRAGMENT_SHADER,
            GRADIENT_FRAGMENT_GLSL,
        )?;

        let copy_fragment =
            shader::compile_shader(&gl, is_embedded, glow::FRAGMENT_SHADER, COPY_FRAGMENT_GLSL)?;

        let color_program = ShaderProgram::new(&gl, color_vertex, color_fragment)?;
        let bitmap_program = ShaderProgram::new(&gl, texture_vertex, bitmap_fragment)?;
        let gradient_program = ShaderProgram::new(&gl, texture_vertex, gradient_fragment)?;
        let copy_program = ShaderProgram::new(&gl, texture_vertex, copy_fragment)?;

        // Shared ramp texture for gradient draws (data uploaded per draw).
        let gradient_texture =
            unsafe { gl.create_texture() }.map_err(Error::UnableToCreateTexture)?;

        let batch_color_vertex =
            shader::compile_shader(&gl, is_embedded, glow::VERTEX_SHADER, BATCH_COLOR_VERTEX_GLSL)?;
        let batch_color_fragment = shader::compile_shader(
            &gl,
            is_embedded,
            glow::FRAGMENT_SHADER,
            BATCH_COLOR_FRAGMENT_GLSL,
        )?;
        let batch_color_program =
            ShaderProgram::new(&gl, batch_color_vertex, batch_color_fragment)?;

        // The batch VAO points its position/color attributes at the shared
        // dynamic vertex buffer (re-uploaded each flush).
        let batch_vao = unsafe { gl.create_vertex_array() }.map_err(Error::UnableToCreateVAO)?;
        let batch_vbo = unsafe { gl.create_buffer() }.map_err(Error::UnableToCreateBuffer)?;
        let batch_ibo = unsafe { gl.create_buffer() }.map_err(Error::UnableToCreateBuffer)?;
        unsafe {
            gl.bind_vertex_array(Some(batch_vao));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(batch_vbo));
            gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(batch_ibo));
            if let Some(loc) = batch_color_program.vertex_position_location {
                gl.vertex_attrib_pointer_f32(loc, 2, glow::FLOAT, false, 12, 0);
                gl.enable_vertex_attrib_array(loc);
            }
            if let Some(loc) = batch_color_program.vertex_color_location {
                gl.vertex_attrib_pointer_f32(loc, 4, glow::UNSIGNED_BYTE, true, 12, 8);
                gl.enable_vertex_attrib_array(loc);
            }
            gl.bind_vertex_array(None);
        }

        // Bitmap batcher: reuses bitmap.frag (color transform + premult round-trip
        // via per-batch uniforms); a dedicated vertex shader takes a precomputed
        // `uv` attribute instead of deriving it from a texture matrix.
        let batch_bitmap_vertex = shader::compile_shader(
            &gl,
            is_embedded,
            glow::VERTEX_SHADER,
            BATCH_BITMAP_VERTEX_GLSL,
        )?;
        let batch_bitmap_program =
            ShaderProgram::new(&gl, batch_bitmap_vertex, bitmap_fragment)?;
        let batch_bitmap_uv_location =
            unsafe { gl.get_attrib_location(batch_bitmap_program.program, "uv") };
        let batch_bitmap_vao =
            unsafe { gl.create_vertex_array() }.map_err(Error::UnableToCreateVAO)?;
        let batch_bitmap_vbo = unsafe { gl.create_buffer() }.map_err(Error::UnableToCreateBuffer)?;
        let batch_bitmap_ibo = unsafe { gl.create_buffer() }.map_err(Error::UnableToCreateBuffer)?;
        unsafe {
            gl.bind_vertex_array(Some(batch_bitmap_vao));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(batch_bitmap_vbo));
            gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(batch_bitmap_ibo));
            if let Some(loc) = batch_bitmap_program.vertex_position_location {
                gl.vertex_attrib_pointer_f32(loc, 2, glow::FLOAT, false, 16, 0);
                gl.enable_vertex_attrib_array(loc);
            }
            if let Some(loc) = batch_bitmap_uv_location {
                gl.vertex_attrib_pointer_f32(loc, 2, glow::FLOAT, false, 16, 8);
                gl.enable_vertex_attrib_array(loc);
            }
            gl.bind_vertex_array(None);
        }

        // Perspective-correct textured triangles: a dedicated vertex shader takes a
        // per-vertex `(u, v, t)` attribute; the fragment shader does the divide.
        let perspective_bitmap_vertex = shader::compile_shader(
            &gl,
            is_embedded,
            glow::VERTEX_SHADER,
            PERSPECTIVE_BITMAP_VERTEX_GLSL,
        )?;
        let perspective_bitmap_fragment = shader::compile_shader(
            &gl,
            is_embedded,
            glow::FRAGMENT_SHADER,
            PERSPECTIVE_BITMAP_FRAGMENT_GLSL,
        )?;
        let perspective_bitmap_program =
            ShaderProgram::new(&gl, perspective_bitmap_vertex, perspective_bitmap_fragment)?;
        let perspective_uvt_location =
            unsafe { gl.get_attrib_location(perspective_bitmap_program.program, "uvt") };

        unsafe {
            gl.enable(glow::BLEND);
            // Initialise the blend func to Normal so it matches `active_hw_blend`
            // from the first draw (otherwise it's GL's default GL_ONE/GL_ZERO).
            gl.blend_equation_separate(glow::FUNC_ADD, glow::FUNC_ADD);
            gl.blend_func_separate(
                glow::ONE,
                glow::ONE_MINUS_SRC_ALPHA,
                glow::ONE,
                glow::ONE_MINUS_SRC_ALPHA,
            );
            // Necessary to load RGB textures (alignment defaults to 4).
            gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);
        }

        let scratch_fbo =
            unsafe { gl.create_framebuffer() }.map_err(Error::UnableToCreateFrameBuffer)?;
        let offscreen_fbo =
            unsafe { gl.create_framebuffer() }.map_err(Error::UnableToCreateFrameBuffer)?;
        let offscreen_stencil =
            unsafe { gl.create_renderbuffer() }.map_err(Error::UnableToCreateRenderBuffer)?;
        let blend_msaa_fbo =
            unsafe { gl.create_framebuffer() }.map_err(Error::UnableToCreateFrameBuffer)?;
        let blend_msaa_color =
            unsafe { gl.create_renderbuffer() }.map_err(Error::UnableToCreateRenderBuffer)?;
        let blend_msaa_stencil =
            unsafe { gl.create_renderbuffer() }.map_err(Error::UnableToCreateRenderBuffer)?;
        let blend_msaa_resolve_fbo =
            unsafe { gl.create_framebuffer() }.map_err(Error::UnableToCreateFrameBuffer)?;
        let blend_msaa_resolve_color =
            unsafe { gl.create_texture() }.map_err(Error::UnableToCreateTexture)?;
        let layer_fbo =
            unsafe { gl.create_framebuffer() }.map_err(Error::UnableToCreateFrameBuffer)?;
        let layer_stencil =
            unsafe { gl.create_renderbuffer() }.map_err(Error::UnableToCreateRenderBuffer)?;

        let pb_quad_vbo = unsafe { gl.create_buffer() }.map_err(Error::UnableToCreateBuffer)?;
        let pb_vao = unsafe { gl.create_vertex_array() }.map_err(Error::UnableToCreateVAO)?;
        unsafe {
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(pb_quad_vbo));
            // A single oversized triangle covering the [-1,1] viewport, rather
            // than a two-triangle quad. A quad's internal (diagonal) edge leaves a
            // faint seam in gl_FragCoord-derived math on some drivers; a single
            // triangle has no internal edge, so PixelBender output is seam-free.
            let tri: [f32; 6] = [-1.0, -1.0, 3.0, -1.0, -1.0, 3.0];
            gl.buffer_data_u8_slice(
                glow::ARRAY_BUFFER,
                bytemuck::cast_slice(&tri),
                glow::STATIC_DRAW,
            );
        }

        let mut renderer = Self {
            gl: gl.clone(),
            caps,
            #[cfg(target_family = "wasm")]
            web_context: created.web_context,

            msaa_buffers: None,
            msaa_sample_count,
            present_flip: false,

            color_program,
            gradient_program,
            gradient_texture,
            bitmap_program,
            batch_color_program,
            copy_program,
            pb_quad_vbo,
            pb_vao,
            batch_vao,
            batch_vbo,
            batch_ibo,
            batch_vertices: Vec::new(),
            batch_indices: Vec::new(),
            batch_bitmap_program,
            perspective_bitmap_program,
            perspective_uvt_location,
            batch_bitmap_vao,
            batch_bitmap_vbo,
            batch_bitmap_ibo,
            batch_bitmap_vertices: Vec::new(),
            batch_bitmap_indices: Vec::new(),
            batch_bitmap_key: None,
            batch_bitmap_handle: None,
            active_hw_blend: NORMAL_BLEND_KEY,
            batch_blend: NORMAL_BLEND_KEY,

            shape_tessellator: ShapeTessellator::new(),

            filters: None,
            pool: pool::TexturePool::new(gl),
            scratch_fbo,
            offscreen_fbo,
            offscreen_stencil,
            offscreen_stencil_dims: (0, 0),
            blend_msaa_fbo,
            blend_msaa_color,
            blend_msaa_stencil,
            blend_msaa_dims: (0, 0),
            blend_msaa_resolve_fbo,
            blend_msaa_resolve_color,
            offscreen_msaa: false,
            offscreen_color: None,
            force_oldest,
            layer_fbo,
            layer_stencil,
            layer_stencil_dims: (0, 0),
            target_fbo: None,
            target_texture: None,
            target_origin: (0, 0),

            color_quad_draws: vec![],
            bitmap_quad_draws: vec![],
            renderbuffer_width: 1,
            renderbuffer_height: 1,
            view_matrix: [[0.0; 4]; 4],

            mask_state: MaskState::NoMask,
            num_masks: 0,
            mask_state_dirty: true,
            is_transparent,
            in_offscreen: false,

            active_program: std::ptr::null(),
            blend_modes: vec![],
            mult_color: None,
            add_color: None,

            viewport_scale_factor: 1.0,
        };

        renderer.push_blend_mode(RenderBlendMode::Builtin(BlendMode::Normal));

        let mut color_quad_mesh = renderer.build_quad_mesh(&renderer.color_program)?;
        let mut bitmap_quad_mesh = renderer.build_quad_mesh(&renderer.bitmap_program)?;
        renderer.color_quad_draws.append(&mut color_quad_mesh);
        renderer.bitmap_quad_draws.append(&mut bitmap_quad_mesh);

        renderer.set_viewport_dimensions(ViewportDimensions {
            width: 1,
            height: 1,
            scale_factor: 1.0,
        });

        Ok(renderer)
    }

    fn build_quad_mesh(&self, program: &ShaderProgram) -> Result<Vec<Draw>, Error> {
        let vao = self.create_vertex_array()?;

        let vertex_buffer =
            unsafe { self.gl.create_buffer() }.map_err(Error::UnableToCreateBuffer)?;
        let index_buffer =
            unsafe { self.gl.create_buffer() }.map_err(Error::UnableToCreateBuffer)?;

        unsafe {
            self.gl.bind_buffer(glow::ARRAY_BUFFER, Some(vertex_buffer));
            self.gl.buffer_data_u8_slice(
                glow::ARRAY_BUFFER,
                bytemuck::cast_slice(&[
                    Vertex {
                        position: [0.0, 0.0],
                        color: 0xffff_ffff,
                    },
                    Vertex {
                        position: [1.0, 0.0],
                        color: 0xffff_ffff,
                    },
                    Vertex {
                        position: [1.0, 1.0],
                        color: 0xffff_ffff,
                    },
                    Vertex {
                        position: [0.0, 1.0],
                        color: 0xffff_ffff,
                    },
                ]),
                glow::STATIC_DRAW,
            );

            self.gl
                .bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(index_buffer));
            self.gl.buffer_data_u8_slice(
                glow::ELEMENT_ARRAY_BUFFER,
                bytemuck::cast_slice(&[0u32, 1, 2, 3]),
                glow::STATIC_DRAW,
            );

            if let Some(loc) = program.vertex_position_location {
                self.gl
                    .vertex_attrib_pointer_f32(loc, 2, glow::FLOAT, false, 12, 0);
                self.gl.enable_vertex_attrib_array(loc);
            }
            if let Some(loc) = program.vertex_color_location {
                self.gl
                    .vertex_attrib_pointer_f32(loc, 4, glow::UNSIGNED_BYTE, true, 12, 8);
                self.gl.enable_vertex_attrib_array(loc);
            }
        }

        self.bind_vertex_array(None);
        for i in program.num_vertex_attributes..NUM_VERTEX_ATTRIBUTES {
            unsafe { self.gl.disable_vertex_attrib_array(i) };
        }

        let draw = Draw {
            draw_type: if program.program == self.bitmap_program.program {
                DrawType::Bitmap(BitmapDraw {
                    matrix: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    handle: None,
                    is_smoothed: true,
                    is_repeating: false,
                })
            } else {
                DrawType::Color
            },
            vao,
            vertex_buffer: Buffer {
                gl: self.gl.clone(),
                buffer: vertex_buffer,
            },
            index_buffer: Buffer {
                gl: self.gl.clone(),
                buffer: index_buffer,
            },
            num_indices: 4,
            num_mask_indices: 4,
            color_cpu: None,
        };
        Ok(vec![draw])
    }

    fn build_msaa_buffers(&mut self) {
        // These offscreen scene buffers serve two independent purposes:
        //   1. Multisample resolve when the quality asks for MSAA (samples > 1).
        //   2. Upright presentation to a real window (`present_flip`): the scene is
        //      rendered top-down (Flash-top at framebuffer row 0), but a window
        //      scans row 0 at the bottom, so the frame must be rendered offscreen
        //      and blitted through the flipping present quad in `end_frame`.
        // Purpose 2 means the present buffer must ALWAYS exist when `present_flip`
        // is set — even with MSAA off, and even if a specific MSAA sample count is
        // rejected by the driver — otherwise the movie renders straight to the
        // backbuffer with no flip and shows upside-down (e.g. a game dropping to
        // `stage.quality = "medium"` mid-play). So we fall back to single-sample.
        let multisample = self.msaa_sample_count > 1;
        let want_buffers = self.caps.is_gles3_or_webgl2 && (multisample || self.present_flip);
        if !want_buffers {
            // Drop any stale buffers so `end_frame`/`begin_frame` fall back to the
            // default framebuffer (headless capture reads it directly, no flip).
            self.msaa_buffers = None;
            unsafe {
                self.gl.bind_framebuffer(glow::FRAMEBUFFER, None);
                self.gl.bind_renderbuffer(glow::RENDERBUFFER, None);
            }
            return;
        }

        // Drain any GL error left by earlier commands so the `check_error` calls in
        // the build don't misattribute it and abort — which would drop the present
        // buffer and flip the window. Report the first one: a stray error here means
        // an earlier operation failed silently and is worth chasing down.
        let mut stray = glow::NO_ERROR;
        loop {
            let e = unsafe { self.gl.get_error() };
            if e == glow::NO_ERROR {
                break;
            }
            if stray == glow::NO_ERROR {
                stray = e;
            }
        }
        if stray != glow::NO_ERROR {
            log::warn!("GL: draining stray error 0x{stray:04x} before building scene buffers");
        }

        // Delete previous buffers first (Drop frees the GL objects).
        self.msaa_buffers = None;

        // Try the requested sample count; if the driver rejects it (some accept
        // only certain counts — e.g. 4 but not 2), fall back to a single-sample
        // present buffer so the flip still works, just without antialiasing.
        let samples = self.msaa_sample_count.max(1);
        let built = match self.build_scene_buffers(samples) {
            Ok(b) => Some(b),
            Err(e) if samples > 1 => {
                log::warn!(
                    "GL: MSAA x{samples} scene buffer failed ({e:?}); using a single-sample present buffer"
                );
                while unsafe { self.gl.get_error() } != glow::NO_ERROR {}
                match self.build_scene_buffers(1) {
                    Ok(b) => Some(b),
                    Err(e) => {
                        log::error!("GL: single-sample present buffer also failed: {e:?}");
                        None
                    }
                }
            }
            Err(e) => {
                log::error!("GL: failed to build scene buffers: {e:?}");
                None
            }
        };
        unsafe {
            self.gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            self.gl.bind_renderbuffer(glow::RENDERBUFFER, None);
        }
        self.msaa_buffers = built;
    }

    /// Builds the offscreen scene framebuffer set: a `render_framebuffer` the frame
    /// is drawn into (multisample when `samples > 1`) plus a single-sample
    /// `color_framebuffer`/`framebuffer_texture` that `end_frame` resolves into and
    /// presents. `samples <= 1` builds a plain single-sample target — the resolve
    /// blit then becomes a same-size copy, so `end_frame` works unchanged.
    fn build_scene_buffers(&self, samples: u32) -> Result<MsaaBuffers, Error> {
        let multisample = samples > 1;
        let gl = self.gl.clone();
        let buffers = unsafe {
            let render_framebuffer = gl
                .create_framebuffer()
                .map_err(Error::UnableToCreateFrameBuffer)?;
            let color_framebuffer = gl
                .create_framebuffer()
                .map_err(Error::UnableToCreateFrameBuffer)?;

            let color_renderbuffer = gl
                .create_renderbuffer()
                .map_err(Error::UnableToCreateRenderBuffer)?;
            gl.bind_renderbuffer(glow::RENDERBUFFER, Some(color_renderbuffer));
            if multisample {
                gl.renderbuffer_storage_multisample(
                    glow::RENDERBUFFER,
                    samples as i32,
                    glow::RGBA8,
                    self.renderbuffer_width,
                    self.renderbuffer_height,
                );
            } else {
                gl.renderbuffer_storage(
                    glow::RENDERBUFFER,
                    glow::RGBA8,
                    self.renderbuffer_width,
                    self.renderbuffer_height,
                );
            }
            check_error(&gl, "renderbuffer_storage (color)")?;

            let stencil_renderbuffer = gl
                .create_renderbuffer()
                .map_err(Error::UnableToCreateRenderBuffer)?;
            gl.bind_renderbuffer(glow::RENDERBUFFER, Some(stencil_renderbuffer));
            if multisample {
                gl.renderbuffer_storage_multisample(
                    glow::RENDERBUFFER,
                    samples as i32,
                    glow::STENCIL_INDEX8,
                    self.renderbuffer_width,
                    self.renderbuffer_height,
                );
            } else {
                gl.renderbuffer_storage(
                    glow::RENDERBUFFER,
                    glow::STENCIL_INDEX8,
                    self.renderbuffer_width,
                    self.renderbuffer_height,
                );
            }
            check_error(&gl, "renderbuffer_storage (stencil)")?;

            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(render_framebuffer));
            gl.framebuffer_renderbuffer(
                glow::FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                glow::RENDERBUFFER,
                Some(color_renderbuffer),
            );
            gl.framebuffer_renderbuffer(
                glow::FRAMEBUFFER,
                glow::STENCIL_ATTACHMENT,
                glow::RENDERBUFFER,
                Some(stencil_renderbuffer),
            );

            let framebuffer_texture = gl.create_texture().map_err(Error::UnableToCreateTexture)?;
            gl.bind_texture(glow::TEXTURE_2D, Some(framebuffer_texture));
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MAG_FILTER,
                glow::NEAREST as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MIN_FILTER,
                glow::NEAREST as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_WRAP_S,
                glow::CLAMP_TO_EDGE as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_WRAP_T,
                glow::CLAMP_TO_EDGE as i32,
            );
            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::RGBA as i32,
                self.renderbuffer_width,
                self.renderbuffer_height,
                0,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(None),
            );
            check_error(&gl, "tex_image_2d (scene resolve texture)")?;
            gl.bind_texture(glow::TEXTURE_2D, None);

            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(color_framebuffer));
            gl.framebuffer_texture_2d(
                glow::FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_2D,
                Some(framebuffer_texture),
                0,
            );
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);

            MsaaBuffers {
                gl: self.gl.clone(),
                color_renderbuffer,
                stencil_renderbuffer,
                render_framebuffer,
                color_framebuffer,
                framebuffer_texture,
            }
        };

        Ok(buffers)
    }

    fn register_shape_internal(
        &mut self,
        shape: DistilledShape,
        bitmap_source: &dyn BitmapSource,
        scale: f32,
    ) -> Result<Vec<Draw>, Error> {
        use ruffle_render::tessellator::DrawType as TessDrawType;

        let lyon_mesh =
            self.shape_tessellator
                .tessellate_shape_with_scale(shape, bitmap_source, scale);

        let mut draws = Vec::with_capacity(lyon_mesh.draws.len());
        for draw in lyon_mesh.draws {
            // Perspective triangles carry their per-vertex `(u, v, t)` in the draw
            // type, not the lyon vertex buffer — build them with a stride-20 VBO
            // (position + uvt) and draw non-indexed.
            if matches!(draw.draw_type, TessDrawType::PerspectiveBitmap(_)) {
                let TessDrawType::PerspectiveBitmap(pb) = draw.draw_type else {
                    unreachable!()
                };
                let vertices: Vec<PerspectiveVertex> =
                    pb.vertices.into_iter().map(PerspectiveVertex::from).collect();
                let vertex_count = vertices.len() as i32;

                let vao = self.create_vertex_array()?;
                let vertex_buffer =
                    unsafe { self.gl.create_buffer() }.map_err(Error::UnableToCreateBuffer)?;
                let index_buffer =
                    unsafe { self.gl.create_buffer() }.map_err(Error::UnableToCreateBuffer)?;
                let program = &self.perspective_bitmap_program;
                unsafe {
                    self.gl.bind_buffer(glow::ARRAY_BUFFER, Some(vertex_buffer));
                    self.gl.buffer_data_u8_slice(
                        glow::ARRAY_BUFFER,
                        bytemuck::cast_slice(&vertices),
                        glow::STATIC_DRAW,
                    );
                    if let Some(loc) = program.vertex_position_location {
                        self.gl
                            .vertex_attrib_pointer_f32(loc, 2, glow::FLOAT, false, 20, 0);
                        self.gl.enable_vertex_attrib_array(loc);
                    }
                    if let Some(loc) = self.perspective_uvt_location {
                        self.gl
                            .vertex_attrib_pointer_f32(loc, 3, glow::FLOAT, false, 20, 8);
                        self.gl.enable_vertex_attrib_array(loc);
                    }
                }
                self.bind_vertex_array(None);
                draws.push(Draw {
                    draw_type: DrawType::PerspectiveBitmap(PerspectiveBitmapDraw {
                        handle: bitmap_source.bitmap_handle(pb.bitmap_id, self),
                        is_repeating: pb.is_repeating,
                        is_smoothed: pb.is_smoothed,
                    }),
                    vao,
                    vertex_buffer: Buffer {
                        gl: self.gl.clone(),
                        buffer: vertex_buffer,
                    },
                    index_buffer: Buffer {
                        gl: self.gl.clone(),
                        buffer: index_buffer,
                    },
                    // Vertex count for `draw_arrays`; masks omit it (0).
                    num_indices: vertex_count,
                    num_mask_indices: 0,
                    color_cpu: None,
                });
                continue;
            }

            let num_indices = draw.indices.len() as i32;
            let num_mask_indices = draw.mask_index_count as i32;

            let vao = self.create_vertex_array()?;
            let vertex_buffer =
                unsafe { self.gl.create_buffer() }.map_err(Error::UnableToCreateBuffer)?;
            let index_buffer =
                unsafe { self.gl.create_buffer() }.map_err(Error::UnableToCreateBuffer)?;

            let vertices: Vec<Vertex> = draw.vertices.into_iter().map(Vertex::from).collect();

            let program = match draw.draw_type {
                TessDrawType::Color => &self.color_program,
                TessDrawType::Gradient { .. } => &self.gradient_program,
                TessDrawType::Bitmap(_) => &self.bitmap_program,
                TessDrawType::PerspectiveBitmap(_) => unreachable!("handled above"),
            };

            unsafe {
                self.gl.bind_buffer(glow::ARRAY_BUFFER, Some(vertex_buffer));
                self.gl.buffer_data_u8_slice(
                    glow::ARRAY_BUFFER,
                    bytemuck::cast_slice(&vertices),
                    glow::STATIC_DRAW,
                );

                self.gl
                    .bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(index_buffer));
                self.gl.buffer_data_u8_slice(
                    glow::ELEMENT_ARRAY_BUFFER,
                    bytemuck::cast_slice(&draw.indices),
                    glow::STATIC_DRAW,
                );

                if let Some(loc) = program.vertex_position_location {
                    self.gl
                        .vertex_attrib_pointer_f32(loc, 2, glow::FLOAT, false, 12, 0);
                    self.gl.enable_vertex_attrib_array(loc);
                }
                if let Some(loc) = program.vertex_color_location {
                    self.gl
                        .vertex_attrib_pointer_f32(loc, 4, glow::UNSIGNED_BYTE, true, 12, 8);
                    self.gl.enable_vertex_attrib_array(loc);
                }
            }

            let num_vertex_attributes = program.num_vertex_attributes;

            let vertex_buffer = Buffer {
                gl: self.gl.clone(),
                buffer: vertex_buffer,
            };
            let index_buffer = Buffer {
                gl: self.gl.clone(),
                buffer: index_buffer,
            };

            // Keep color geometry CPU-side so the batcher can transform and merge
            // it into one draw call.
            let color_cpu = if matches!(draw.draw_type, TessDrawType::Color) {
                Some(ColorGeometry {
                    vertices: vertices.clone(),
                    indices: draw.indices.clone(),
                })
            } else {
                None
            };

            draws.push(match draw.draw_type {
                TessDrawType::Color => Draw {
                    draw_type: DrawType::Color,
                    vao,
                    vertex_buffer,
                    index_buffer,
                    num_indices,
                    num_mask_indices,
                    color_cpu,
                },
                TessDrawType::Gradient { matrix, gradient } => Draw {
                    draw_type: DrawType::Gradient(Box::new(Gradient::new(
                        lyon_mesh.gradients[gradient].clone(), // TODO: Gradient deduplication
                        matrix,
                    ))),
                    vao,
                    vertex_buffer,
                    index_buffer,
                    num_indices,
                    num_mask_indices,
                    color_cpu: None,
                },
                TessDrawType::Bitmap(bitmap) => Draw {
                    draw_type: DrawType::Bitmap(BitmapDraw {
                        matrix: bitmap.matrix,
                        handle: bitmap_source.bitmap_handle(bitmap.bitmap_id, self),
                        is_smoothed: bitmap.is_smoothed,
                        is_repeating: bitmap.is_repeating,
                    }),
                    vao,
                    vertex_buffer,
                    index_buffer,
                    num_indices,
                    num_mask_indices,
                    color_cpu: None,
                },
                TessDrawType::PerspectiveBitmap(_) => unreachable!("handled above"),
            });

            self.bind_vertex_array(None);

            // Don't use 'program' here in order to satisfy the borrow checker.
            for i in num_vertex_attributes..NUM_VERTEX_ATTRIBUTES {
                unsafe { self.gl.disable_vertex_attrib_array(i) };
            }
        }

        Ok(draws)
    }

    /// Creates and binds a new VAO. glow handles WebGL1's
    /// `OES_vertex_array_object` extension internally, so no branching needed.
    fn create_vertex_array(&self) -> Result<glow::VertexArray, Error> {
        unsafe {
            let vao = self
                .gl
                .create_vertex_array()
                .map_err(Error::UnableToCreateVAO)?;
            self.gl.bind_vertex_array(Some(vao));
            Ok(vao)
        }
    }

    fn bind_vertex_array(&self, vao: Option<glow::VertexArray>) {
        unsafe { self.gl.bind_vertex_array(vao) };
    }

    fn set_stencil_state(&self) {
        // Set stencil state for masking, if necessary.
        if self.mask_state_dirty {
            unsafe {
                match self.mask_state {
                    MaskState::NoMask => {
                        self.gl.disable(glow::STENCIL_TEST);
                        self.gl.color_mask(true, true, true, true);
                    }
                    MaskState::DrawMaskStencil => {
                        self.gl.enable(glow::STENCIL_TEST);
                        self.gl
                            .stencil_func(glow::EQUAL, (self.num_masks - 1) as i32, 0xff);
                        self.gl.stencil_op(glow::KEEP, glow::KEEP, glow::INCR);
                        self.gl.color_mask(false, false, false, false);
                    }
                    MaskState::DrawMaskedContent => {
                        self.gl.enable(glow::STENCIL_TEST);
                        self.gl
                            .stencil_func(glow::EQUAL, self.num_masks as i32, 0xff);
                        self.gl.stencil_op(glow::KEEP, glow::KEEP, glow::KEEP);
                        self.gl.color_mask(true, true, true, true);
                    }
                    MaskState::ClearMaskStencil => {
                        self.gl.enable(glow::STENCIL_TEST);
                        self.gl
                            .stencil_func(glow::EQUAL, self.num_masks as i32, 0xff);
                        self.gl.stencil_op(glow::KEEP, glow::KEEP, glow::DECR);
                        self.gl.color_mask(false, false, false, false);
                    }
                }
            }
        }
    }

    fn apply_blend_mode(&mut self, mode: RenderBlendMode) {
        let minmax_ok = self.minmax_ok();
        let key = blend_key(&mode, minmax_ok);
        self.active_hw_blend = key;
        self.apply_hw_blend(key);
    }

    /// When rendering to a real window (desktop), flips the final present quad
    /// vertically so the top-down scene appears upright on-screen. Leave unset for
    /// headless capture, which reads the framebuffer directly.
    pub fn set_present_flipped(&mut self, flipped: bool) {
        if self.present_flip != flipped {
            self.present_flip = flipped;
            // Presentation now needs (or no longer needs) an offscreen scene buffer
            // to flip through — even with MSAA off — so (re)build accordingly.
            self.build_msaa_buffers();
        }
    }

    /// Reads the default framebuffer back as top-down RGBA8 bytes (used by the
    /// headless test harness to capture a rendered frame). The scene is rendered
    /// top-down, so glReadPixels row 0 is already Flash-top — no flip needed.
    pub fn read_framebuffer(&self) -> (u32, u32, Vec<u8>) {
        let w = self.renderbuffer_width.max(0) as u32;
        let h = self.renderbuffer_height.max(0) as u32;
        let mut pixels = vec![0u8; (w * h * 4) as usize];
        unsafe {
            self.gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            self.gl.read_pixels(
                0,
                0,
                w as i32,
                h as i32,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelPackData::Slice(Some(&mut pixels)),
            );
        }
        // Top-down render: glReadPixels row 0 is already Flash-top, no flip.
        (w, h, pixels)
    }

    /// Whether GL_MIN/GL_MAX blend equations are available (desktop GL, or
    /// GLES3/WebGL2). Used to gate the Darken/Lighten fast path.
    fn minmax_ok(&self) -> bool {
        (!self.caps.is_embedded || self.caps.is_gles3_or_webgl2) && !self.force_oldest
    }

    /// Sets GL blend state from a key (`[rgb_eq, alpha_eq, rgb_src, rgb_dst,
    /// alpha_src, alpha_dst]`) without touching `active_hw_blend`.
    fn apply_hw_blend(&self, key: [u32; 6]) {
        unsafe {
            self.gl.blend_equation_separate(key[0], key[1]);
            self.gl
                .blend_func_separate(key[2], key[3], key[4], key[5]);
        }
    }

    fn begin_frame(&mut self, clear: Color) {
        self.flush_batch();

        // Flash colors are already sRGB and we write them verbatim. egui's glow
        // painter enables GL_FRAMEBUFFER_SRGB and doesn't restore it, which would
        // make the driver sRGB-encode our present a second time (lifting darks —
        // a washed-out look most visible on dark textures). Keep it off. Desktop
        // GL only; GLES handles framebuffer sRGB via the surface format instead.
        if !self.caps.is_embedded {
            unsafe { self.gl.disable(glow::FRAMEBUFFER_SRGB) };
        }

        // Start each frame from a known Normal blend func (matches the base mode).
        self.active_hw_blend = NORMAL_BLEND_KEY;
        self.batch_blend = NORMAL_BLEND_KEY;
        self.apply_hw_blend(NORMAL_BLEND_KEY);

        self.active_program = std::ptr::null();
        self.mask_state = MaskState::NoMask;
        self.num_masks = 0;
        self.mask_state_dirty = true;

        self.mult_color = None;
        self.add_color = None;

        unsafe {
            // Bind to MSAA render buffer if using MSAA.
            if let Some(msaa_buffers) = &self.msaa_buffers {
                self.gl
                    .bind_framebuffer(glow::FRAMEBUFFER, Some(msaa_buffers.render_framebuffer));
            }

            self.gl
                .viewport(0, 0, self.renderbuffer_width, self.renderbuffer_height);

            self.set_stencil_state();
            if self.is_transparent {
                self.gl.clear_color(0.0, 0.0, 0.0, 0.0);
            } else {
                self.gl.clear_color(
                    clear.r as f32 / 255.0,
                    clear.g as f32 / 255.0,
                    clear.b as f32 / 255.0,
                    clear.a as f32 / 255.0,
                );
            }
            self.gl.stencil_mask(0xff);
            self.gl
                .clear(glow::COLOR_BUFFER_BIT | glow::STENCIL_BUFFER_BIT);
        }
    }

    fn end_frame(&self) {
        // Resolve MSAA, if we're using it (WebGL2 / GLES3 / desktop GL >= 3.0).
        let Some(msaa_buffers) = &self.msaa_buffers else {
            return;
        };
        if !self.caps.is_gles3_or_webgl2 {
            return;
        }

        unsafe {
            // Disable any remaining masking state.
            self.gl.disable(glow::STENCIL_TEST);
            self.gl.color_mask(true, true, true, true);

            // Resolve the MSAA in the render buffer.
            self.gl.bind_framebuffer(
                glow::READ_FRAMEBUFFER,
                Some(msaa_buffers.render_framebuffer),
            );
            self.gl
                .bind_framebuffer(glow::DRAW_FRAMEBUFFER, Some(msaa_buffers.color_framebuffer));
            self.gl.blit_framebuffer(
                0,
                0,
                self.renderbuffer_width,
                self.renderbuffer_height,
                0,
                0,
                self.renderbuffer_width,
                self.renderbuffer_height,
                glow::COLOR_BUFFER_BIT,
                glow::NEAREST,
            );

            // Render the resolved framebuffer texture to a quad on the screen.
            self.gl.bind_framebuffer(glow::FRAMEBUFFER, None);

            #[cfg(target_family = "wasm")]
            let (vw, vh) = self.web_context.drawing_buffer_size();
            #[cfg(not(target_family = "wasm"))]
            let (vw, vh) = (self.renderbuffer_width, self.renderbuffer_height);
            self.gl.viewport(0, 0, vw, vh);

            let program = &self.bitmap_program;
            self.gl.use_program(Some(program.program));

            // Scale to fill screen.
            program.uniform_matrix4fv(
                &self.gl,
                ShaderUniform::WorldMatrix,
                &[
                    [2.0, 0.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [-1.0, -1.0, 0.0, 1.0],
                ],
            );
            program.uniform_matrix4fv(
                &self.gl,
                ShaderUniform::ViewMatrix,
                &[
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            );
            program.uniform4fv(&self.gl, ShaderUniform::MultColor, &[1.0, 1.0, 1.0, 1.0]);
            program.uniform4fv(&self.gl, ShaderUniform::AddColor, &[0.0, 0.0, 0.0, 0.0]);
            // The scene is rendered top-down (Flash-top at framebuffer row 0), and
            // the resolve texture inherits that (texel row 0 = Flash-top). Headless
            // capture reads the framebuffer directly, so present straight (keep
            // Flash-top at row 0). A real window scans row 0 at the bottom, so flip
            // V there to present the movie upright (`present_flip`).
            let texture_matrix = if self.present_flip {
                [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 1.0, 1.0]]
            } else {
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            };
            program.uniform_matrix3fv(&self.gl, ShaderUniform::TextureMatrix, &texture_matrix);

            // Bind the framebuffer texture.
            self.gl.active_texture(glow::TEXTURE0);
            self.gl
                .bind_texture(glow::TEXTURE_2D, Some(msaa_buffers.framebuffer_texture));
            program.uniform1i(&self.gl, ShaderUniform::BitmapTexture, 0);

            // Render the quad.
            let quad = &self.bitmap_quad_draws;
            self.bind_vertex_array(Some(quad[0].vao));
            self.gl.draw_elements(
                glow::TRIANGLE_FAN,
                quad[0].num_indices,
                glow::UNSIGNED_INT,
                0,
            );
        }
    }

    /// Flushes any pending batched draws (color and bitmap). Must be called
    /// before any GL state the batched draws depend on changes (blend mode,
    /// mask/stencil, program, view matrix, render target). At most one of the two
    /// batches is ever pending at a time, so order between them is moot here.
    fn flush_batch(&mut self) {
        self.flush_color_batch();
        self.flush_bitmap_batch();
    }

    /// Uploads and draws the accumulated solid-color batch, then clears it.
    fn flush_color_batch(&mut self) {
        if self.batch_indices.is_empty() {
            return;
        }
        // Draw under the blend func the batch was filled with, then restore the
        // currently-active func for subsequent (non-batched) draws.
        self.apply_hw_blend(self.batch_blend);
        let gl = self.gl.clone();
        let program = &self.batch_color_program;
        unsafe {
            gl.bind_vertex_array(Some(self.batch_vao));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.batch_vbo));
            gl.buffer_data_u8_slice(
                glow::ARRAY_BUFFER,
                bytemuck::cast_slice(&self.batch_vertices),
                glow::DYNAMIC_DRAW,
            );
            gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(self.batch_ibo));
            gl.buffer_data_u8_slice(
                glow::ELEMENT_ARRAY_BUFFER,
                bytemuck::cast_slice(&self.batch_indices),
                glow::DYNAMIC_DRAW,
            );
            gl.use_program(Some(program.program));
            program.uniform_matrix4fv(&gl, ShaderUniform::ViewMatrix, &self.view_matrix);
            gl.draw_elements(
                glow::TRIANGLES,
                self.batch_indices.len() as i32,
                glow::UNSIGNED_INT,
                0,
            );
            gl.bind_vertex_array(None);
        }
        self.apply_hw_blend(self.active_hw_blend);
        self.batch_vertices.clear();
        self.batch_indices.clear();
        // We bound our own program/VAO; invalidate the cached active program.
        self.active_program = std::ptr::null();
    }

    /// Uploads and draws the accumulated bitmap batch (one texture/transform),
    /// then clears it.
    fn flush_bitmap_batch(&mut self) {
        if self.batch_bitmap_indices.is_empty() {
            return;
        }
        let Some(key) = self.batch_bitmap_key else {
            self.batch_bitmap_vertices.clear();
            self.batch_bitmap_indices.clear();
            self.batch_bitmap_handle = None;
            return;
        };
        self.apply_hw_blend(self.batch_blend);
        let gl = self.gl.clone();
        let program = &self.batch_bitmap_program;
        unsafe {
            gl.bind_vertex_array(Some(self.batch_bitmap_vao));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.batch_bitmap_vbo));
            gl.buffer_data_u8_slice(
                glow::ARRAY_BUFFER,
                bytemuck::cast_slice(&self.batch_bitmap_vertices),
                glow::DYNAMIC_DRAW,
            );
            gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(self.batch_bitmap_ibo));
            gl.buffer_data_u8_slice(
                glow::ELEMENT_ARRAY_BUFFER,
                bytemuck::cast_slice(&self.batch_bitmap_indices),
                glow::DYNAMIC_DRAW,
            );
            gl.use_program(Some(program.program));
            program.uniform_matrix4fv(&gl, ShaderUniform::ViewMatrix, &self.view_matrix);
            program.uniform4fv(&gl, ShaderUniform::MultColor, &key.mult);
            program.uniform4fv(&gl, ShaderUniform::AddColor, &key.add);
            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, Some(key.texture));
            program.uniform1i(&gl, ShaderUniform::BitmapTexture, 0);
            let filter = if key.smoothing {
                glow::LINEAR
            } else {
                glow::NEAREST
            } as i32;
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, filter);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, filter);
            let wrap = glow::CLAMP_TO_EDGE as i32;
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, wrap);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, wrap);
            gl.draw_elements(
                glow::TRIANGLES,
                self.batch_bitmap_indices.len() as i32,
                glow::UNSIGNED_INT,
                0,
            );
            gl.bind_vertex_array(None);
        }
        self.apply_hw_blend(self.active_hw_blend);
        self.batch_bitmap_vertices.clear();
        self.batch_bitmap_indices.clear();
        self.batch_bitmap_key = None;
        // The texture has been bound and drawn; the handle can be released now.
        self.batch_bitmap_handle = None;
        self.active_program = std::ptr::null();
        self.mult_color = None;
        self.add_color = None;
    }

    /// Appends a bitmap quad to the bitmap batch, flushing first if the colour
    /// batch is pending (draw order) or the batch key changes.
    fn append_bitmap_draw(&mut self, key: BitmapBatchKey, matrix: Matrix, handle: BitmapHandle) {
        self.flush_color_batch();
        if !self.batch_bitmap_indices.is_empty()
            && (self.batch_bitmap_key != Some(key) || self.batch_blend != self.active_hw_blend)
        {
            self.flush_bitmap_batch();
        }
        self.batch_blend = self.active_hw_blend;
        self.batch_bitmap_key = Some(key);
        // Hold the handle for this batch's texture (same texture for the whole
        // pending batch, so one handle suffices) until it flushes.
        self.batch_bitmap_handle = Some(handle);

        // The bitmap quad is [0,1]² with an identity texture matrix, so UV equals
        // the corner. Transform each corner by the world matrix on the CPU.
        const CORNERS: [[f32; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let base = self.batch_bitmap_vertices.len() as u32;
        let tx = matrix.tx.to_pixels() as f32;
        let ty = matrix.ty.to_pixels() as f32;
        for &[qx, qy] in &CORNERS {
            self.batch_bitmap_vertices.push(BitmapVertex {
                position: [
                    matrix.a * qx + matrix.c * qy + tx,
                    matrix.b * qx + matrix.d * qy + ty,
                ],
                uv: [qx, qy],
            });
        }
        for &i in &[0u32, 1, 2, 0, 2, 3] {
            self.batch_bitmap_indices.push(base + i);
        }
    }

    /// Transforms a solid-color draw's geometry by `matrix`, bakes the color
    /// transform (premultiplied), and appends it to the batch. `num_indices`
    /// selects the full or mask-only index range.
    fn append_color_draw(
        &mut self,
        geom: &ColorGeometry,
        num_indices: usize,
        matrix: Matrix,
        mult: [f32; 4],
        add: [f32; 4],
    ) {
        // A pending bitmap batch was drawn before these colour shapes, so flush
        // it first to keep paint order. Also flush if the colour batch so far was
        // filled under a different blend func.
        self.flush_bitmap_batch();
        if !self.batch_indices.is_empty() && self.batch_blend != self.active_hw_blend {
            self.flush_color_batch();
        }
        self.batch_blend = self.active_hw_blend;
        let base = self.batch_vertices.len() as u32;
        let tx = matrix.tx.to_pixels() as f32;
        let ty = matrix.ty.to_pixels() as f32;
        let (a, b, c, d) = (matrix.a, matrix.b, matrix.c, matrix.d);

        // The common case is no color transform (mult identity, add zero), where
        // baking reduces to premultiplying the source color — skip the per-channel
        // mult/add/clamp entirely. Both branches build the vertices with a single
        // `extend` over a `TrustedLen` slice-map: one capacity check for the whole
        // draw (vs one per `push`), and a store loop LLVM can vectorize.
        let identity_color = mult == [1.0, 1.0, 1.0, 1.0] && add == [0.0, 0.0, 0.0, 0.0];
        if identity_color {
            // A tessellated fill is usually one solid colour across all its
            // vertices, so memoize the last raw->premultiplied mapping and skip the
            // premultiply for runs of the same colour.
            let mut cache: Option<(u32, u32)> = None;
            self.batch_vertices
                .extend(geom.vertices.iter().map(|v| {
                    let x = v.position[0];
                    let y = v.position[1];
                    let color = match cache {
                        Some((raw, premul)) if raw == v.color => premul,
                        _ => {
                            let [cr, cg, cb, ca] = v.color.to_le_bytes();
                            let premul = u32::from_le_bytes([
                                mul_div255_round(cr, ca),
                                mul_div255_round(cg, ca),
                                mul_div255_round(cb, ca),
                                ca,
                            ]);
                            cache = Some((v.color, premul));
                            premul
                        }
                    };
                    Vertex {
                        position: [a * x + c * y + tx, b * x + d * y + ty],
                        color,
                    }
                }));
        } else {
            self.batch_vertices
                .extend(geom.vertices.iter().map(|v| {
                    let x = v.position[0];
                    let y = v.position[1];
                    let [cr, cg, cb, ca] = v.color.to_le_bytes();
                    // frag_color = clamp(color * mult + add), then premultiply
                    // (matches the per-draw color shader, done here on the CPU).
                    let r = (cr as f32 / 255.0 * mult[0] + add[0]).clamp(0.0, 1.0);
                    let g = (cg as f32 / 255.0 * mult[1] + add[1]).clamp(0.0, 1.0);
                    let bl = (cb as f32 / 255.0 * mult[2] + add[2]).clamp(0.0, 1.0);
                    let al = (ca as f32 / 255.0 * mult[3] + add[3]).clamp(0.0, 1.0);
                    let color = u32::from_le_bytes([
                        (r * al * 255.0).round() as u8,
                        (g * al * 255.0).round() as u8,
                        (bl * al * 255.0).round() as u8,
                        (al * 255.0).round() as u8,
                    ]);
                    Vertex {
                        position: [a * x + c * y + tx, b * x + d * y + ty],
                        color,
                    }
                }));
        }
        self.batch_indices.reserve(num_indices);
        self.batch_indices
            .extend(geom.indices[..num_indices].iter().map(|&i| base + i));
    }

    fn push_blend_mode(&mut self, blend: RenderBlendMode) {
        // No flush here: a run of same-func draws batches across push/pop. The
        // batch records the func it was filled under (`batch_blend`) and is
        // flushed lazily when an append sees a different active func.
        if !same_blend_mode(self.blend_modes.last(), &blend) {
            self.apply_blend_mode(blend.clone());
        }
        self.blend_modes.push(blend);
    }

    fn pop_blend_mode(&mut self) {
        let old = self.blend_modes.pop();
        // We never pop our base 'BlendMode::Normal'.
        let current = self
            .blend_modes
            .last()
            .cloned()
            .unwrap_or(RenderBlendMode::Builtin(BlendMode::Normal));
        if !same_blend_mode(old.as_ref(), &current) {
            self.apply_blend_mode(current);
        }
    }

    fn draw_quad<const MODE: u32, const COUNT: i32>(&mut self, color: Color, matrix: Matrix) {
        self.flush_batch();
        let world_matrix = [
            [matrix.a, matrix.b, 0.0, 0.0],
            [matrix.c, matrix.d, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [
                matrix.tx.to_pixels() as f32,
                matrix.ty.to_pixels() as f32,
                0.0,
                1.0,
            ],
        ];

        let mult_color = [
            color.r as f32 * 255.0,
            color.g as f32 * 255.0,
            color.b as f32 * 255.0,
            color.a as f32 * 255.0,
        ];
        let add_color = [0.0; 4];

        self.set_stencil_state();

        let program = &self.color_program;

        // Set common render state, while minimizing unnecessary state changes.
        if !std::ptr::eq(program, self.active_program) {
            unsafe { self.gl.use_program(Some(program.program)) };
            self.active_program = program as *const ShaderProgram;

            program.uniform_matrix4fv(&self.gl, ShaderUniform::ViewMatrix, &self.view_matrix);

            self.mult_color = None;
            self.add_color = None;
        };

        self.color_program
            .uniform_matrix4fv(&self.gl, ShaderUniform::WorldMatrix, &world_matrix);
        if Some(mult_color) != self.mult_color {
            self.color_program
                .uniform4fv(&self.gl, ShaderUniform::MultColor, &mult_color);
            self.mult_color = Some(mult_color);
        }
        if Some(add_color) != self.add_color {
            self.color_program
                .uniform4fv(&self.gl, ShaderUniform::AddColor, &add_color);
            self.add_color = Some(add_color);
        }

        let quad = &self.color_quad_draws;
        self.bind_vertex_array(Some(quad[0].vao));
        // Re-bind the quad's own index buffer: registering a shape (e.g. a device
        // font glyph, lazily tessellated mid-frame) binds its index buffer to
        // `ELEMENT_ARRAY_BUFFER` while this persistent VAO happens to be bound,
        // clobbering the VAO's element-buffer binding. Without this, the draw
        // reads indices from a stale buffer and rasterizes nothing.
        unsafe {
            self.gl
                .bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(quad[0].index_buffer.buffer));
        }

        let count = if COUNT < 0 {
            quad[0].num_indices
        } else {
            COUNT
        };
        unsafe {
            self.gl.draw_elements(MODE, count, glow::UNSIGNED_INT, 0);
        }
    }

    /// Draws `texture` 1:1 over the current framebuffer/viewport (premultiplied,
    /// blend disabled), like the MSAA resolve in `end_frame`. Used to seed a
    /// multisampled offscreen buffer with a target texture's existing content
    /// before compositing new commands on top.
    fn fill_with_texture(&self, texture: glow::Texture, replace: bool) {
        // Use the raw-copy program (not the bitmap program) so the seed is an
        // exact passthrough — no un-premultiply/re-premultiply round-trip, which
        // would drift colors and accumulate across repeated BitmapData.draw.
        let program = &self.copy_program;
        unsafe {
            self.gl.use_program(Some(program.program));
            if replace {
                self.gl.disable(glow::BLEND);
            }
        }
        program.uniform_matrix4fv(
            &self.gl,
            ShaderUniform::WorldMatrix,
            &[
                [2.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [-1.0, -1.0, 0.0, 1.0],
            ],
        );
        program.uniform_matrix4fv(
            &self.gl,
            ShaderUniform::ViewMatrix,
            &[
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        );
        program.uniform4fv(&self.gl, ShaderUniform::MultColor, &[1.0, 1.0, 1.0, 1.0]);
        program.uniform4fv(&self.gl, ShaderUniform::AddColor, &[0.0, 0.0, 0.0, 0.0]);
        program.uniform_matrix3fv(
            &self.gl,
            ShaderUniform::TextureMatrix,
            &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        );
        unsafe {
            self.gl.active_texture(glow::TEXTURE0);
            self.gl.bind_texture(glow::TEXTURE_2D, Some(texture));
            program.uniform1i(&self.gl, ShaderUniform::BitmapTexture, 0);
            let quad = &self.bitmap_quad_draws;
            self.bind_vertex_array(Some(quad[0].vao));
            self.gl
                .draw_elements(glow::TRIANGLE_FAN, quad[0].num_indices, glow::UNSIGNED_INT, 0);
            if replace {
                self.gl.enable(glow::BLEND);
            }
        }
    }

    /// Renders a `CommandList` into the given texture via a temporary FBO.
    ///
    /// `clear` of `Some(color)` clears the color buffer first (cacheAsBitmap);
    /// `None` preserves existing texture content and composites on top
    /// (BitmapData.draw). The stencil buffer is always cleared. Uses the
    /// offscreen projection (Flash-top at texel row 0) so the texture stays
    /// consistent with uploaded bitmaps and CPU readback.
    fn render_commands_to_texture(
        &mut self,
        texture: glow::Texture,
        width: i32,
        height: i32,
        clear: Option<Color>,
        view: [[f32; 4]; 4],
        msaa: bool,
        commands: CommandList,
    ) {
        // Flush any pending batch drawn with the outer target/view before
        // switching to this offscreen pass.
        self.flush_batch();
        let gl = self.gl.clone();
        // Antialias the offscreen content when requested (and supported) by
        // rendering into a multisampled renderbuffer and resolving into the
        // target texture afterward.
        // A nested offscreen pass (e.g. a complex blend's source rendered while
        // we're already inside a cacheAsBitmap pass) must not reuse the shared
        // multisampled `blend_msaa` buffer — it would clobber the outer pass's
        // in-progress content. Fall back to the single-sample `offscreen_fbo`
        // there; the parent stays MSAA and is read back via the resolve buffer.
        let use_msaa = msaa
            && self.caps.is_gles3_or_webgl2
            && self.msaa_sample_count > 1
            && !self.in_offscreen;
        unsafe {
            if use_msaa {
                let samples = self.msaa_sample_count as i32;
                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.blend_msaa_fbo));
                if self.blend_msaa_dims != (width, height) {
                    gl.bind_renderbuffer(glow::RENDERBUFFER, Some(self.blend_msaa_color));
                    gl.renderbuffer_storage_multisample(
                        glow::RENDERBUFFER,
                        samples,
                        glow::RGBA8,
                        width,
                        height,
                    );
                    gl.bind_renderbuffer(glow::RENDERBUFFER, Some(self.blend_msaa_stencil));
                    gl.renderbuffer_storage_multisample(
                        glow::RENDERBUFFER,
                        samples,
                        glow::STENCIL_INDEX8,
                        width,
                        height,
                    );
                    // Keep the single-sample resolve target the same size.
                    gl.bind_texture(glow::TEXTURE_2D, Some(self.blend_msaa_resolve_color));
                    gl.tex_image_2d(
                        glow::TEXTURE_2D,
                        0,
                        glow::RGBA8 as i32,
                        width,
                        height,
                        0,
                        glow::RGBA,
                        glow::UNSIGNED_BYTE,
                        glow::PixelUnpackData::Slice(None),
                    );
                    gl.tex_parameter_i32(
                        glow::TEXTURE_2D,
                        glow::TEXTURE_MIN_FILTER,
                        glow::NEAREST as i32,
                    );
                    gl.tex_parameter_i32(
                        glow::TEXTURE_2D,
                        glow::TEXTURE_MAG_FILTER,
                        glow::NEAREST as i32,
                    );
                    gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.blend_msaa_resolve_fbo));
                    gl.framebuffer_texture_2d(
                        glow::FRAMEBUFFER,
                        glow::COLOR_ATTACHMENT0,
                        glow::TEXTURE_2D,
                        Some(self.blend_msaa_resolve_color),
                        0,
                    );
                    gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.blend_msaa_fbo));
                    self.blend_msaa_dims = (width, height);
                }
                gl.framebuffer_renderbuffer(
                    glow::FRAMEBUFFER,
                    glow::COLOR_ATTACHMENT0,
                    glow::RENDERBUFFER,
                    Some(self.blend_msaa_color),
                );
                gl.framebuffer_renderbuffer(
                    glow::FRAMEBUFFER,
                    glow::STENCIL_ATTACHMENT,
                    glow::RENDERBUFFER,
                    Some(self.blend_msaa_stencil),
                );
            } else {
                // Reuse the shared offscreen FBO and its stencil renderbuffer,
                // attaching this call's color texture. WebGL1 requires all
                // attachments to share dimensions, so resize the stencil storage
                // only when the size changes.
                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.offscreen_fbo));
                gl.framebuffer_texture_2d(
                    glow::FRAMEBUFFER,
                    glow::COLOR_ATTACHMENT0,
                    glow::TEXTURE_2D,
                    Some(texture),
                    0,
                );
                gl.bind_renderbuffer(glow::RENDERBUFFER, Some(self.offscreen_stencil));
                if self.offscreen_stencil_dims != (width, height) {
                    gl.renderbuffer_storage(glow::RENDERBUFFER, glow::STENCIL_INDEX8, width, height);
                    self.offscreen_stencil_dims = (width, height);
                }
                gl.framebuffer_renderbuffer(
                    glow::FRAMEBUFFER,
                    glow::STENCIL_ATTACHMENT,
                    glow::RENDERBUFFER,
                    Some(self.offscreen_stencil),
                );
            }
        }

        // Save and reconfigure render state for offscreen rendering.
        let saved_view = self.view_matrix;
        let saved_w = self.renderbuffer_width;
        let saved_h = self.renderbuffer_height;
        let saved_msaa = self.msaa_buffers.take();
        let saved_offscreen = self.in_offscreen;
        let saved_offscreen_msaa = self.offscreen_msaa;
        let saved_offscreen_color = self.offscreen_color;
        let saved_active_blend = self.active_hw_blend;
        let saved_batch_blend = self.batch_blend;
        self.in_offscreen = true;
        self.offscreen_msaa = use_msaa;
        // Single-sample passes render straight into `texture`; a nested complex
        // blend reads its parent from there.
        self.offscreen_color = if use_msaa { None } else { Some(texture) };
        // The offscreen content starts from a clean Normal blend (its own blends
        // push/pop from there); the outer batch was already flushed above.
        self.active_hw_blend = NORMAL_BLEND_KEY;
        self.batch_blend = NORMAL_BLEND_KEY;
        self.apply_hw_blend(NORMAL_BLEND_KEY);

        self.view_matrix = view;
        self.renderbuffer_width = width;
        self.renderbuffer_height = height;

        self.active_program = std::ptr::null();
        self.mask_state = MaskState::NoMask;
        self.num_masks = 0;
        self.mask_state_dirty = true;
        self.mult_color = None;
        self.add_color = None;

        unsafe {
            gl.viewport(0, 0, width, height);
            self.set_stencil_state();
            gl.stencil_mask(0xff);
            // Masks clear the stencil to a 0 fill value. `glClearStencil` is
            // global GL state shared with Context3D (Stage3D), which sets its own
            // clear value — so set ours explicitly rather than assuming the
            // default, or offscreen masks silently break after a Stage3D clear.
            gl.clear_stencil(0);
            if let Some(c) = clear {
                gl.clear_color(
                    c.r as f32 / 255.0,
                    c.g as f32 / 255.0,
                    c.b as f32 / 255.0,
                    c.a as f32 / 255.0,
                );
                gl.clear(glow::COLOR_BUFFER_BIT | glow::STENCIL_BUFFER_BIT);
            } else {
                gl.clear(glow::STENCIL_BUFFER_BIT);
            }
        }

        // For the multisampled compose-on-top path (clear=None), the target's
        // existing content lives only in the single-sample texture, not the MSAA
        // buffer — seed the buffer with it before drawing the new commands.
        if use_msaa && clear.is_none() {
            self.fill_with_texture(texture, true);
            self.active_program = std::ptr::null();
        }

        commands.execute(self);
        // Flush the offscreen content's batch (still under this pass's view)
        // before restoring the outer state.
        self.flush_batch();

        // Restore state and tear down the temporary FBO.
        self.view_matrix = saved_view;
        self.renderbuffer_width = saved_w;
        self.renderbuffer_height = saved_h;
        self.msaa_buffers = saved_msaa;
        self.in_offscreen = saved_offscreen;
        self.offscreen_msaa = saved_offscreen_msaa;
        self.offscreen_color = saved_offscreen_color;
        self.active_hw_blend = saved_active_blend;
        self.batch_blend = saved_batch_blend;
        self.apply_hw_blend(saved_active_blend);
        self.active_program = std::ptr::null();

        unsafe {
            if use_msaa {
                // Resolve the multisampled content into the target texture.
                gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(self.blend_msaa_fbo));
                gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, Some(self.scratch_fbo));
                gl.framebuffer_texture_2d(
                    glow::DRAW_FRAMEBUFFER,
                    glow::COLOR_ATTACHMENT0,
                    glow::TEXTURE_2D,
                    Some(texture),
                    0,
                );
                gl.blit_framebuffer(
                    0,
                    0,
                    width,
                    height,
                    0,
                    0,
                    width,
                    height,
                    glow::COLOR_BUFFER_BIT,
                    glow::NEAREST,
                );
            }
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
        }
    }

    /// Applies a supported filter to `texture` in place (size unchanged),
    /// copying the result back over the source. Returns false for unsupported
    /// filter types. Used by the cacheAsBitmap path in `submit_frame`.
    fn apply_filter_in_place(
        &mut self,
        texture: glow::Texture,
        width: u32,
        height: u32,
        filter: &Filter,
    ) -> bool {
        if width == 0 || height == 0 {
            return true;
        }
        // A ShaderFilter runs a PixelBender program; handle it before borrowing
        // the fixed-function filter set.
        if let Filter::ShaderFilter(sf) = filter {
            return self.apply_shader_filter(texture, width, height, sf);
        }
        if self.filters.is_none() {
            match filters::Filters::new(self.gl.clone(), self.caps.is_embedded) {
                Ok(f) => self.filters = Some(f),
                Err(e) => {
                    log::error!("Couldn't initialize GL filters: {e}");
                    return false;
                }
            }
        }
        let filters = self.filters.as_ref().expect("filters just initialized");
        let result = match filter {
            Filter::ColorMatrixFilter(f) => filters.apply_color_matrix(
                &mut self.pool,
                texture,
                width,
                height,
                (0, 0),
                (width, height),
                &f.matrix,
            ),
            Filter::BlurFilter(f) => filters.apply_blur(
                &mut self.pool,
                texture,
                width,
                height,
                (0, 0),
                (width, height),
                f.blur_x.to_f32(),
                f.blur_y.to_f32(),
                f.num_passes() as u32,
            ),
            Filter::GlowFilter(f) => filters.apply_glow(
                &mut self.pool,
                texture,
                width,
                height,
                (0, 0),
                (width, height),
                color_to_rgba(f.color),
                f.strength.to_f32(),
                f.is_inner(),
                f.is_knockout(),
                f.composite_source(),
                f.blur_x.to_f32(),
                f.blur_y.to_f32(),
                f.num_passes() as u32,
                (0.0, 0.0),
            ),
            Filter::DropShadowFilter(f) => {
                let distance = f.distance.to_f32();
                let angle = f.angle.to_f32();
                let offset = (angle.cos() * distance, angle.sin() * distance);
                filters.apply_glow(
                    &mut self.pool,
                    texture,
                    width,
                    height,
                    (0, 0),
                    (width, height),
                    color_to_rgba(f.color),
                    f.strength.to_f32(),
                    f.is_inner(),
                    f.is_knockout(),
                    !f.hide_object(),
                    f.blur_x.to_f32(),
                    f.blur_y.to_f32(),
                    f.num_passes() as u32,
                    (-offset.0, -offset.1),
                )
            }
            Filter::BevelFilter(f) => {
                let distance = f.distance.to_f32();
                let angle = f.angle.to_f32();
                let offset = (angle.cos() * distance, angle.sin() * distance);
                let bevel_type = if f.is_on_top() {
                    2
                } else if f.is_inner() {
                    1
                } else {
                    0
                };
                filters.apply_bevel(
                    &mut self.pool,
                    texture,
                    width,
                    height,
                    (0, 0),
                    (width, height),
                    premultiplied_rgba(f.highlight_color),
                    premultiplied_rgba(f.shadow_color),
                    f.strength.to_f32(),
                    bevel_type,
                    f.is_knockout(),
                    f.blur_x.to_f32(),
                    f.blur_y.to_f32(),
                    f.num_passes() as u32,
                    offset,
                )
            }
            Filter::ConvolutionFilter(f) => match convolution_params(f) {
                Some((kernel, cols, rows, divisor, bias, default_color, clamp, preserve)) => filters
                    .apply_convolution(
                        &mut self.pool,
                        texture,
                        width,
                        height,
                        (0, 0),
                        (width, height),
                        &kernel,
                        cols,
                        rows,
                        divisor,
                        bias,
                        default_color,
                        clamp,
                        preserve,
                    ),
                None => None,
            },
            Filter::DisplacementMapFilter(f) => match f.map_bitmap.as_ref() {
                Some(map) => {
                    let (map_tex, map_w, map_h) = {
                        let d = as_registry_data(map);
                        (d.texture, d.width, d.height)
                    };
                    filters.apply_displacement(
                        &mut self.pool,
                        texture,
                        width,
                        height,
                        (0, 0),
                        (width, height),
                        map_tex,
                        map_w,
                        map_h,
                        color_to_rgba(f.color),
                        (f.component_x as f32, f.component_y as f32),
                        displacement_mode(f.mode),
                        (f.scale_x, f.scale_y),
                        (f.map_point.0 as f32, f.map_point.1 as f32),
                        (f.viewscale_x, f.viewscale_y),
                    )
                }
                None => None,
            },
            Filter::GradientGlowFilter(f) => {
                let ramp = build_gradient_ramp(&f.colors);
                let distance = f.distance.to_f32();
                let angle = f.angle.to_f32();
                let offset = (angle.cos() * distance, angle.sin() * distance);
                let gtype = if f.is_on_top() {
                    2
                } else if f.is_inner() {
                    1
                } else {
                    0
                };
                filters.apply_gradient_glow(
                    &mut self.pool,
                    texture,
                    width,
                    height,
                    (0, 0),
                    (width, height),
                    &ramp,
                    f.strength.to_f32(),
                    gtype,
                    f.is_knockout(),
                    f.flags.contains(swf::GradientFilterFlags::COMPOSITE_SOURCE),
                    f.blur_x.to_f32(),
                    f.blur_y.to_f32(),
                    f.num_passes() as u32,
                    offset,
                )
            }
            Filter::GradientBevelFilter(f) => {
                let ramp = build_gradient_ramp(&f.colors);
                let distance = f.distance.to_f32();
                let angle = f.angle.to_f32();
                let offset = (angle.cos() * distance, angle.sin() * distance);
                let bevel_type = if f.is_on_top() {
                    2
                } else if f.is_inner() {
                    1
                } else {
                    0
                };
                filters.apply_gradient_bevel(
                    &mut self.pool,
                    texture,
                    width,
                    height,
                    (0, 0),
                    (width, height),
                    &ramp,
                    f.strength.to_f32(),
                    bevel_type,
                    f.is_knockout(),
                    f.blur_x.to_f32(),
                    f.blur_y.to_f32(),
                    f.num_passes() as u32,
                    offset,
                )
            }
            _ => None,
        };
        let Some(result) = result else {
            return false;
        };
        // The filter pass changed program/VAO/blend state outside the normal
        // command path; invalidate the cached active program.
        self.active_program = std::ptr::null();

        let gl = self.gl.clone();
        let copy_w = result.width.min(width) as i32;
        let copy_h = result.height.min(height) as i32;
        unsafe {
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.scratch_fbo));
            gl.framebuffer_texture_2d(
                glow::FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_2D,
                Some(result.texture),
                0,
            );
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));
            gl.copy_tex_sub_image_2d(glow::TEXTURE_2D, 0, 0, 0, 0, 0, copy_w, copy_h);
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
        }
        self.pool.release(result.texture, result.width, result.height);
        true
    }

    /// Clamped region `(rx, ry, rw, rh)` in the *current target's* local pixels
    /// (top-left origin) covering a blend's content, or `None` if it has no
    /// measurable area. Stage-space content bounds are shifted by the target
    /// origin so a `Layer` offscreen sees its own local coordinates.
    fn blend_region(&self, commands: &CommandList) -> Option<(i32, i32, i32, i32)> {
        let bounds = command_bounds(commands)?;
        let fb_w = self.renderbuffer_width;
        let fb_h = self.renderbuffer_height;
        let (ox, oy) = self.target_origin;
        let rx = (bounds.x_min.to_pixels().floor() as i32 - ox).clamp(0, fb_w);
        let ry = (bounds.y_min.to_pixels().floor() as i32 - oy).clamp(0, fb_h);
        let rx_max = (bounds.x_max.to_pixels().ceil() as i32 - ox).clamp(0, fb_w);
        let ry_max = (bounds.y_max.to_pixels().ceil() as i32 - oy).clamp(0, fb_h);
        let rw = rx_max - rx;
        let rh = ry_max - ry;
        if rw <= 0 || rh <= 0 {
            return None;
        }
        Some((rx, ry, rw, rh))
    }

    /// Region-sized complex blend. Renders `commands` into a region-sized texture
    /// (`src`), copies the framebuffer region as the parent (`dst`), then
    /// composites the two with the blend shader straight back onto the
    /// framebuffer over the tight region — so transparent source pixels keep the
    /// existing background and the cost scales with the content, not the stage.
    fn draw_complex_blend(
        &mut self,
        commands: CommandList,
        mode: i32,
        rx: i32,
        ry: i32,
        rw: i32,
        rh: i32,
    ) {
        // Lazily build the filter/blend programs.
        if self.filters.is_none() {
            match filters::Filters::new(self.gl.clone(), self.caps.is_embedded) {
                Ok(f) => self.filters = Some(f),
                Err(e) => {
                    log::error!("Couldn't initialize GL blend programs: {e}");
                    commands.execute(self);
                    return;
                }
            }
        }

        let (rw_u, rh_u) = (rw as u32, rh as u32);
        let Some(src_tex) = self.pool.acquire(rw_u, rh_u) else {
            commands.execute(self);
            return;
        };
        let Some(dst_tex) = self.pool.acquire(rw_u, rh_u) else {
            self.pool.release(src_tex, rw_u, rh_u);
            commands.execute(self);
            return;
        };

        // Render the blended content into `src`. render_commands_to_texture
        // resets the mask state for the nested pass, so save/restore the outer
        // mask that still applies to the composite.
        let saved_mask = self.mask_state;
        let saved_num_masks = self.num_masks;

        // The blended content's commands are in stage space; offset the region
        // projection by the target origin so a `Layer` offscreen places them
        // correctly into its local space.
        let (ox, oy) = self.target_origin;
        // Match the current target's Y orientation: the screen and `Layer`
        // offscreens use a flipped view, but a cacheAsBitmap pass renders
        // non-flipped (offscreen_view_matrix). A mismatch renders the source
        // upside-down relative to the parent (a vertically reversed blend).
        let flipped = self.view_matrix[1][1] < 0.0;
        let view = if flipped {
            region_view_matrix((rx + ox) as f32, (ry + oy) as f32, rw as f32, rh as f32)
        } else {
            region_view_matrix_unflipped((rx + ox) as f32, (ry + oy) as f32, rw as f32, rh as f32)
        };
        let transparent = Color {
            r: 0,
            g: 0,
            b: 0,
            a: 0,
        };
        self.render_commands_to_texture(src_tex, rw, rh, Some(transparent), view, true, commands);

        self.mask_state = saved_mask;
        self.num_masks = saved_num_masks;

        // Framebuffer-space Y of the region. A flipped target has framebuffer
        // row 0 at the bottom (so mirror the region); a non-flipped offscreen
        // target maps stage Y straight to the framebuffer row.
        let ry_fb = if flipped {
            self.renderbuffer_height - (ry + rh)
        } else {
            ry
        };

        let gl = self.gl.clone();
        let target_fbo = self.target_fbo;
        unsafe {
            // Copy/resolve the target region into `dst` (the parent).
            if let Some(layer) = target_fbo {
                // A `Layer` offscreen is single-sample: copy its region directly.
                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(layer));
                gl.bind_texture(glow::TEXTURE_2D, Some(dst_tex));
                gl.copy_tex_sub_image_2d(glow::TEXTURE_2D, 0, 0, 0, rx, ry_fb, rw, rh);
            } else if self.in_offscreen && self.offscreen_msaa {
                // Nested inside an MSAA offscreen pass (e.g. an Overlay inside a
                // cacheAsBitmap). Resolve the multisampled parent region into the
                // single-sample resolve buffer (same bounds), then copy that
                // region down to the region texture's origin.
                gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(self.blend_msaa_fbo));
                gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, Some(self.blend_msaa_resolve_fbo));
                gl.blit_framebuffer(
                    rx,
                    ry_fb,
                    rx + rw,
                    ry_fb + rh,
                    rx,
                    ry_fb,
                    rx + rw,
                    ry_fb + rh,
                    glow::COLOR_BUFFER_BIT,
                    glow::NEAREST,
                );
                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.blend_msaa_resolve_fbo));
                gl.bind_texture(glow::TEXTURE_2D, Some(dst_tex));
                gl.copy_tex_sub_image_2d(glow::TEXTURE_2D, 0, 0, 0, rx, ry_fb, rw, rh);
            } else if let Some(off_tex) = self.offscreen_color {
                // Nested inside a single-sample offscreen pass: the parent is
                // already in the offscreen texture. The src render re-attached the
                // shared FBO to its own texture, so re-attach the parent first.
                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.offscreen_fbo));
                gl.framebuffer_texture_2d(
                    glow::FRAMEBUFFER,
                    glow::COLOR_ATTACHMENT0,
                    glow::TEXTURE_2D,
                    Some(off_tex),
                    0,
                );
                gl.bind_texture(glow::TEXTURE_2D, Some(dst_tex));
                gl.copy_tex_sub_image_2d(glow::TEXTURE_2D, 0, 0, 0, rx, ry_fb, rw, rh);
            } else if let Some(msaa) = &self.msaa_buffers {
                // Multisample-resolve blits require *identical* source and
                // destination rectangles, so we can't blit straight into a
                // region-origin texture. Resolve the region in place into the
                // single-sample resolve framebuffer (same bounds), then copy that
                // region down to the region texture's origin.
                gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(msaa.render_framebuffer));
                gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, Some(msaa.color_framebuffer));
                gl.blit_framebuffer(
                    rx,
                    ry_fb,
                    rx + rw,
                    ry_fb + rh,
                    rx,
                    ry_fb,
                    rx + rw,
                    ry_fb + rh,
                    glow::COLOR_BUFFER_BIT,
                    glow::NEAREST,
                );
                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(msaa.color_framebuffer));
                gl.bind_texture(glow::TEXTURE_2D, Some(dst_tex));
                gl.copy_tex_sub_image_2d(glow::TEXTURE_2D, 0, 0, 0, rx, ry_fb, rw, rh);
            } else {
                gl.bind_framebuffer(glow::FRAMEBUFFER, None);
                gl.bind_texture(glow::TEXTURE_2D, Some(dst_tex));
                gl.copy_tex_sub_image_2d(glow::TEXTURE_2D, 0, 0, 0, rx, ry_fb, rw, rh);
            }

            // Re-bind the draw target and restrict drawing to the region.
            if let Some(layer) = target_fbo {
                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(layer));
            } else if self.in_offscreen && self.offscreen_msaa {
                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.blend_msaa_fbo));
            } else if self.offscreen_color.is_some() {
                // Already re-attached to the parent (offscreen) texture above.
                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.offscreen_fbo));
            } else if let Some(msaa) = &self.msaa_buffers {
                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(msaa.render_framebuffer));
            } else {
                gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            }
            gl.viewport(rx, ry_fb, rw, rh);
            gl.disable(glow::BLEND);
        }

        // Apply the current mask (if any) but don't write the stencil buffer.
        self.mask_state_dirty = true;
        self.set_stencil_state();
        unsafe { gl.stencil_mask(0x00) };

        self.filters
            .as_ref()
            .expect("blend programs initialized")
            .draw_blend(src_tex, dst_tex, mode);

        unsafe {
            gl.stencil_mask(0xff);
            gl.viewport(0, 0, self.renderbuffer_width, self.renderbuffer_height);
            gl.enable(glow::BLEND);
        }

        // The composite changed program/VAO/blend state outside the normal path;
        // restore the active blend func so later non-batched draws are correct.
        self.active_program = std::ptr::null();
        self.mask_state_dirty = true;
        self.apply_hw_blend(self.active_hw_blend);

        self.pool.release(src_tex, rw_u, rh_u);
        self.pool.release(dst_tex, rw_u, rh_u);
    }

    /// Region-sized PixelBender-shader blend: isolates the group into a foreground
    /// texture, copies the framebuffer region as the background, runs the shader
    /// (input 0 = background, input 1 = foreground — matching wgpu) in Filter mode,
    /// and replaces the region with its output. Works in any target — screen, a
    /// `Layer`, or an MSAA/single-sample offscreen pass — by reading the backdrop
    /// from whichever is bound (see `draw_complex_blend`).
    fn draw_shader_blend(
        &mut self,
        commands: CommandList,
        pb: &GlPixelBenderShader,
        rx: i32,
        ry: i32,
        rw: i32,
        rh: i32,
    ) {
        let (rw_u, rh_u) = (rw as u32, rh as u32);
        let fallback = |this: &mut Self, commands: CommandList| {
            this.push_blend_mode(RenderBlendMode::Builtin(BlendMode::Normal));
            commands.execute(this);
            this.pop_blend_mode();
        };
        let Some(src_tex) = self.pool.acquire(rw_u, rh_u) else {
            return fallback(self, commands);
        };
        let Some(dst_tex) = self.pool.acquire(rw_u, rh_u) else {
            self.pool.release(src_tex, rw_u, rh_u);
            return fallback(self, commands);
        };
        let Some(res_tex) = self.pool.acquire(rw_u, rh_u) else {
            self.pool.release(src_tex, rw_u, rh_u);
            self.pool.release(dst_tex, rw_u, rh_u);
            return fallback(self, commands);
        };

        // Foreground: isolate the group into `src_tex`.
        let saved_mask = self.mask_state;
        let saved_num_masks = self.num_masks;
        let (ox, oy) = self.target_origin;
        // Match the target's Y orientation (see `draw_complex_blend`): a top-down
        // on-screen framebuffer maps stage Y straight to the framebuffer row.
        let flipped = self.view_matrix[1][1] < 0.0;
        let view = if flipped {
            region_view_matrix((rx + ox) as f32, (ry + oy) as f32, rw as f32, rh as f32)
        } else {
            region_view_matrix_unflipped((rx + ox) as f32, (ry + oy) as f32, rw as f32, rh as f32)
        };
        let transparent = Color { r: 0, g: 0, b: 0, a: 0 };
        self.render_commands_to_texture(src_tex, rw, rh, Some(transparent), view, true, commands);
        self.mask_state = saved_mask;
        self.num_masks = saved_num_masks;

        // Background: copy the target region into `dst_tex`. The source depends on
        // the current target — screen, an enclosing `Layer`, or an MSAA/
        // single-sample offscreen pass (cacheAsBitmap, BitmapData.draw). Mirror
        // `draw_complex_blend` so a shader blend works in every target, not just
        // on-screen.
        let ry_fb = if flipped {
            self.renderbuffer_height - (ry + rh)
        } else {
            ry
        };
        let gl = self.gl.clone();
        let target_fbo = self.target_fbo;
        unsafe {
            if let Some(layer) = target_fbo {
                // A `Layer` offscreen is single-sample: copy its region directly.
                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(layer));
                gl.bind_texture(glow::TEXTURE_2D, Some(dst_tex));
                gl.copy_tex_sub_image_2d(glow::TEXTURE_2D, 0, 0, 0, rx, ry_fb, rw, rh);
            } else if self.in_offscreen && self.offscreen_msaa {
                // Nested inside an MSAA offscreen pass: resolve the multisampled
                // parent region, then copy it down to the region texture's origin.
                gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(self.blend_msaa_fbo));
                gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, Some(self.blend_msaa_resolve_fbo));
                gl.blit_framebuffer(
                    rx, ry_fb, rx + rw, ry_fb + rh, rx, ry_fb, rx + rw, ry_fb + rh,
                    glow::COLOR_BUFFER_BIT, glow::NEAREST,
                );
                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.blend_msaa_resolve_fbo));
                gl.bind_texture(glow::TEXTURE_2D, Some(dst_tex));
                gl.copy_tex_sub_image_2d(glow::TEXTURE_2D, 0, 0, 0, rx, ry_fb, rw, rh);
            } else if let Some(off_tex) = self.offscreen_color {
                // Nested inside a single-sample offscreen pass: the parent lives in
                // the offscreen texture. The `src` render re-attached the shared FBO
                // to its own texture, so re-attach the parent first.
                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.offscreen_fbo));
                gl.framebuffer_texture_2d(
                    glow::FRAMEBUFFER,
                    glow::COLOR_ATTACHMENT0,
                    glow::TEXTURE_2D,
                    Some(off_tex),
                    0,
                );
                gl.bind_texture(glow::TEXTURE_2D, Some(dst_tex));
                gl.copy_tex_sub_image_2d(glow::TEXTURE_2D, 0, 0, 0, rx, ry_fb, rw, rh);
            } else if let Some(msaa) = &self.msaa_buffers {
                gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(msaa.render_framebuffer));
                gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, Some(msaa.color_framebuffer));
                gl.blit_framebuffer(
                    rx, ry_fb, rx + rw, ry_fb + rh, rx, ry_fb, rx + rw, ry_fb + rh,
                    glow::COLOR_BUFFER_BIT, glow::NEAREST,
                );
                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(msaa.color_framebuffer));
                gl.bind_texture(glow::TEXTURE_2D, Some(dst_tex));
                gl.copy_tex_sub_image_2d(glow::TEXTURE_2D, 0, 0, 0, rx, ry_fb, rw, rh);
            } else {
                gl.bind_framebuffer(glow::FRAMEBUFFER, None);
                gl.bind_texture(glow::TEXTURE_2D, Some(dst_tex));
                gl.copy_tex_sub_image_2d(glow::TEXTURE_2D, 0, 0, 0, rx, ry_fb, rw, rh);
            }
        }

        // Run the shader (Filter mode) into `res_tex`.
        let float_data = vec![0.0f32; pb.float_slots * 4];
        let int_data = vec![0i32; pb.int_slots * 4];
        self.bind_and_draw_pixelbender(
            pb,
            res_tex,
            rw,
            rh,
            true,
            &[
                (0, dst_tex, rw as f32, rh as f32),
                (1, src_tex, rw as f32, rh as f32),
            ],
            &float_data,
            &int_data,
        );

        // Replace the region with the shader's output. Re-bind the draw target the
        // same way the background was read (screen, `Layer`, or offscreen).
        unsafe {
            if let Some(layer) = target_fbo {
                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(layer));
            } else if self.in_offscreen && self.offscreen_msaa {
                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.blend_msaa_fbo));
            } else if self.offscreen_color.is_some() {
                // Already re-attached to the parent (offscreen) texture above; the
                // shader draw used `scratch_fbo`, so `offscreen_fbo` still holds it.
                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.offscreen_fbo));
            } else if let Some(msaa) = &self.msaa_buffers {
                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(msaa.render_framebuffer));
            } else {
                gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            }
            gl.viewport(rx, ry_fb, rw, rh);
        }
        self.mask_state_dirty = true;
        self.set_stencil_state();
        unsafe { gl.stencil_mask(0x00) };
        self.fill_with_texture(res_tex, true);
        unsafe {
            gl.stencil_mask(0xff);
            gl.viewport(0, 0, self.renderbuffer_width, self.renderbuffer_height);
        }
        self.active_program = std::ptr::null();
        self.mask_state_dirty = true;
        self.apply_hw_blend(self.active_hw_blend);
        self.pool.release(src_tex, rw_u, rh_u);
        self.pool.release(dst_tex, rw_u, rh_u);
        self.pool.release(res_tex, rw_u, rh_u);
    }

    /// Region-sized alpha mask: renders the maskee and the mask into region
    /// textures, then composites `maskee * mask.a` over the target with normal
    /// blend. Unlike a complex blend it never reads the framebuffer, so it works
    /// in any target (screen, cache, layer). Cost scales with the content, not
    /// the stage (wgpu uses a full-surface pass).
    fn draw_alpha_mask(
        &mut self,
        maskee_commands: CommandList,
        mask_commands: CommandList,
        rx: i32,
        ry: i32,
        rw: i32,
        rh: i32,
    ) {
        if self.filters.is_none() {
            match filters::Filters::new(self.gl.clone(), self.caps.is_embedded) {
                Ok(f) => self.filters = Some(f),
                Err(e) => {
                    log::error!("Couldn't initialize GL blend programs: {e}");
                    maskee_commands.execute(self);
                    return;
                }
            }
        }

        let (rw_u, rh_u) = (rw as u32, rh as u32);
        let Some(maskee_tex) = self.pool.acquire(rw_u, rh_u) else {
            maskee_commands.execute(self);
            return;
        };
        let Some(mask_tex) = self.pool.acquire(rw_u, rh_u) else {
            self.pool.release(maskee_tex, rw_u, rh_u);
            maskee_commands.execute(self);
            return;
        };

        // render_commands_to_texture resets the mask state for the nested pass;
        // save/restore the outer mask that still applies to the composite.
        let saved_mask = self.mask_state;
        let saved_num_masks = self.num_masks;

        let (ox, oy) = self.target_origin;
        let flipped = self.view_matrix[1][1] < 0.0;
        let view = if flipped {
            region_view_matrix((rx + ox) as f32, (ry + oy) as f32, rw as f32, rh as f32)
        } else {
            region_view_matrix_unflipped((rx + ox) as f32, (ry + oy) as f32, rw as f32, rh as f32)
        };
        let transparent = Color {
            r: 0,
            g: 0,
            b: 0,
            a: 0,
        };

        self.render_commands_to_texture(maskee_tex, rw, rh, Some(transparent), view, true, maskee_commands);
        self.mask_state = saved_mask;
        self.num_masks = saved_num_masks;
        self.render_commands_to_texture(mask_tex, rw, rh, Some(transparent), view, true, mask_commands);
        self.mask_state = saved_mask;
        self.num_masks = saved_num_masks;

        let ry_fb = if flipped {
            self.renderbuffer_height - (ry + rh)
        } else {
            ry
        };

        let gl = self.gl.clone();
        let target_fbo = self.target_fbo;
        unsafe {
            // Bind the draw target and restrict drawing to the region. Keep normal
            // (premultiplied-over) blend so the masked content composites over the
            // existing background.
            if let Some(layer) = target_fbo {
                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(layer));
            } else if self.in_offscreen && self.offscreen_msaa {
                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.blend_msaa_fbo));
            } else if let Some(msaa) = &self.msaa_buffers {
                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(msaa.render_framebuffer));
            } else {
                gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            }
            gl.viewport(rx, ry_fb, rw, rh);
        }
        self.apply_hw_blend(NORMAL_BLEND_KEY);

        // Apply the current mask (if any) but don't write the stencil buffer.
        self.mask_state_dirty = true;
        self.set_stencil_state();
        unsafe { gl.stencil_mask(0x00) };

        // mode 9: maskee (u_current) modulated by mask alpha (u_parent.a).
        self.filters
            .as_ref()
            .expect("blend programs initialized")
            .draw_blend(maskee_tex, mask_tex, 9);

        unsafe {
            gl.stencil_mask(0xff);
            gl.viewport(0, 0, self.renderbuffer_width, self.renderbuffer_height);
        }

        self.active_program = std::ptr::null();
        self.mask_state_dirty = true;
        self.apply_hw_blend(self.active_hw_blend);

        self.pool.release(maskee_tex, rw_u, rh_u);
        self.pool.release(mask_tex, rw_u, rh_u);
    }

    /// Applies a `ShaderFilter` (a PixelBender program) to `texture` in place:
    /// runs the shader with `texture` as the source input, then copies the result
    /// back over it.
    fn apply_shader_filter(
        &mut self,
        texture: glow::Texture,
        width: u32,
        height: u32,
        sf: &ruffle_render::filters::ShaderFilter,
    ) -> bool {
        let arc = sf.shader.0.clone();
        let pb: &GlPixelBenderShader = match <dyn Any>::downcast_ref(&*arc) {
            Some(pb) => pb,
            None => return false,
        };
        let Some(out_tex) = self.pool.acquire(width, height) else {
            return false;
        };
        // Filter mode: samples outside the source become transparent.
        self.execute_pixelbender(
            pb,
            &sf.shader_args,
            out_tex,
            width as i32,
            height as i32,
            true,
            Some((texture, width as f32, height as f32)),
        );
        // Copy the result back into the in-place texture.
        let gl = self.gl.clone();
        unsafe {
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.scratch_fbo));
            gl.framebuffer_texture_2d(
                glow::FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_2D,
                Some(out_tex),
                0,
            );
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));
            gl.copy_tex_sub_image_2d(glow::TEXTURE_2D, 0, 0, 0, 0, 0, width as i32, height as i32);
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
        }
        self.pool.release(out_tex, width, height);
        self.active_program = std::ptr::null();
        self.mask_state_dirty = true;
        true
    }

    /// Renders a compiled PixelBender shader over `out_tex` (`out_w`×`out_h`),
    /// binding value params into the uniform arrays and image inputs into
    /// samplers. `zeroed` selects Filter mode (samples outside inputs become
    /// transparent) vs ShaderJob mode (clamp).
    fn execute_pixelbender(
        &mut self,
        pb: &GlPixelBenderShader,
        arguments: &[PixelBenderShaderArgument],
        out_tex: glow::Texture,
        out_w: i32,
        out_h: i32,
        zeroed: bool,
        // Bound to the first image input (the filter source); overrides its arg.
        source_override: Option<(glow::Texture, f32, f32)>,
    ) {
        let gl = self.gl.clone();
        let mut float_data = vec![0.0f32; pb.float_slots * 4];
        let mut int_data = vec![0i32; pb.int_slots * 4];
        // (input index, texture, width, height)
        let mut tex_bindings: Vec<(u8, glow::Texture, f32, f32)> = Vec::new();
        // Temporary float-input textures (from `Vector.<Number>` ShaderJob inputs);
        // deleted after the draw.
        let mut temp_textures: Vec<glow::Texture> = Vec::new();
        let mut source_used = source_override.is_none();
        for arg in arguments {
            match arg {
                PixelBenderShaderArgument::ImageInput { index, texture, .. } => {
                    if !source_used {
                        source_used = true;
                        let (t, w, h) = source_override.unwrap();
                        tex_bindings.push((*index, t, w, h));
                    } else if let Some(ImageInputTexture::Bitmap(h)) = texture.as_ref() {
                        let r = as_registry_data(h);
                        tex_bindings.push((*index, r.texture, r.width as f32, r.height as f32));
                    } else if let Some(ImageInputTexture::Floats {
                        width,
                        height,
                        data,
                    }) = texture.as_ref()
                    {
                        // A `Vector.<Number>` image input (e.g. the shallow-water
                        // fluid simulation's ping-pong buffers). Upload the floats
                        // as an RGBA32F texture. The shader only reads the channels
                        // it was compiled for, so padding 1/2-channel data to RGBA
                        // is harmless and keeps a single (widely-supported) format.
                        let padded = data.padded_data();
                        let rgba: Vec<f32> = match data.channel_count() {
                            1 => padded.iter().flat_map(|&r| [r, 0.0, 0.0, 0.0]).collect(),
                            2 => padded
                                .chunks_exact(2)
                                .flat_map(|c| [c[0], c[1], 0.0, 0.0])
                                .collect(),
                            _ => padded.into_owned(), // Rgb is padded to Rgba, Rgba as-is
                        };
                        if let Ok(t) = unsafe { gl.create_texture() } {
                            unsafe {
                                gl.bind_texture(glow::TEXTURE_2D, Some(t));
                                gl.tex_image_2d(
                                    glow::TEXTURE_2D,
                                    0,
                                    glow::RGBA32F as i32,
                                    *width as i32,
                                    *height as i32,
                                    0,
                                    glow::RGBA,
                                    glow::FLOAT,
                                    glow::PixelUnpackData::Slice(Some(bytemuck::cast_slice(&rgba))),
                                );
                            }
                            temp_textures.push(t);
                            tex_bindings.push((*index, t, *width as f32, *height as f32));
                        }
                    }
                }
                PixelBenderShaderArgument::ValueInput { index, value } => {
                    if let Some(Some(slot)) = pb.param_slots.get(*index as usize) {
                        write_param_value(value, *slot, &mut float_data, &mut int_data);
                    }
                }
            }
        }

        self.bind_and_draw_pixelbender(
            pb,
            out_tex,
            out_w,
            out_h,
            zeroed,
            &tex_bindings,
            &float_data,
            &int_data,
        );

        for t in temp_textures {
            unsafe { gl.delete_texture(t) };
        }
    }

    /// Binds a compiled PixelBender shader's params and image inputs (in the given
    /// order — `tex_bindings[i]` goes to texture unit `i`) and draws the NDC quad
    /// into `out_tex` via the scratch FBO. The low-level half of
    /// [`Self::execute_pixelbender`], shared with the shader-blend path which binds
    /// raw region textures rather than argument-derived ones.
    #[allow(clippy::too_many_arguments)]
    fn bind_and_draw_pixelbender(
        &mut self,
        pb: &GlPixelBenderShader,
        out_tex: glow::Texture,
        out_w: i32,
        out_h: i32,
        zeroed: bool,
        tex_bindings: &[(u8, glow::Texture, f32, f32)],
        float_data: &[f32],
        int_data: &[i32],
    ) {
        let gl = self.gl.clone();
        unsafe {
            gl.bind_vertex_array(Some(self.pb_vao));
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.scratch_fbo));
            gl.framebuffer_texture_2d(
                glow::FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_2D,
                Some(out_tex),
                0,
            );
            gl.viewport(0, 0, out_w, out_h);
            gl.disable(glow::BLEND);
            gl.disable(glow::STENCIL_TEST);
            gl.color_mask(true, true, true, true);
            gl.use_program(Some(pb.program));

            if let Some(l) = &pb.u_out_size {
                gl.uniform_2_f32(Some(l), out_w as f32, out_h as f32);
            }
            if let Some(l) = &pb.u_zeroed {
                gl.uniform_1_i32(Some(l), zeroed as i32);
            }
            if let Some(l) = &pb.u_float_params {
                gl.uniform_4_f32_slice(Some(l), float_data);
            }
            if let Some(l) = &pb.u_int_params {
                gl.uniform_4_i32_slice(Some(l), int_data);
            }
            for (unit, (index, tex, w, h)) in tex_bindings.iter().enumerate() {
                gl.active_texture(glow::TEXTURE0 + unit as u32);
                gl.bind_texture(glow::TEXTURE_2D, Some(*tex));
                let clamp = glow::CLAMP_TO_EDGE as i32;
                gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, clamp);
                gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, clamp);
                let info = pb.u_inputs.iter().find(|(i, _, _, _)| i == index);
                // ES 1.00 filtering is per-texture: match the shader's sampler.
                let filter = if info.map(|(_, _, _, n)| *n).unwrap_or(false) {
                    glow::NEAREST
                } else {
                    glow::LINEAR
                } as i32;
                gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, filter);
                gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, filter);
                if let Some((_, sloc, szloc, _)) = info {
                    if let Some(sl) = sloc {
                        gl.uniform_1_i32(Some(sl), unit as i32);
                    }
                    if let Some(zl) = szloc {
                        gl.uniform_2_f32(Some(zl), *w, *h);
                    }
                }
            }

            gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.pb_quad_vbo));
            if let Some(loc) = pb.position_loc {
                gl.vertex_attrib_pointer_f32(loc, 2, glow::FLOAT, false, 8, 0);
                gl.enable_vertex_attrib_array(loc);
            }
            gl.draw_arrays(glow::TRIANGLES, 0, 3);

            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            gl.bind_vertex_array(None);
            gl.enable(glow::BLEND);
            gl.active_texture(glow::TEXTURE0);
        }
        self.active_program = std::ptr::null();
        self.mask_state_dirty = true;
    }

    /// Renders a `Layer` blend's children into a single-sample offscreen region
    /// texture (so Alpha/Erase and nested complex blends composite against the
    /// layer's own transparent content, not the opaque stage), then composites
    /// that texture over the parent target with Normal blend. `(rx, ry, rw, rh)`
    /// is parent-local.
    fn draw_layer(&mut self, commands: CommandList, rx: i32, ry: i32, rw: i32, rh: i32) {
        let (rw_u, rh_u) = (rw as u32, rh as u32);
        let Some(layer_tex) = self.pool.acquire(rw_u, rh_u) else {
            self.push_blend_mode(RenderBlendMode::Builtin(BlendMode::Normal));
            commands.execute(self);
            self.pop_blend_mode();
            return;
        };

        // Stage-space origin of the region (parent-local plus parent origin).
        let (pox, poy) = self.target_origin;
        let stage_rx = rx + pox;
        let stage_ry = ry + poy;

        let gl = self.gl.clone();
        unsafe {
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.layer_fbo));
            gl.framebuffer_texture_2d(
                glow::FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_2D,
                Some(layer_tex),
                0,
            );
            gl.bind_renderbuffer(glow::RENDERBUFFER, Some(self.layer_stencil));
            if self.layer_stencil_dims != (rw, rh) {
                gl.renderbuffer_storage(glow::RENDERBUFFER, glow::STENCIL_INDEX8, rw, rh);
                self.layer_stencil_dims = (rw, rh);
            }
            gl.framebuffer_renderbuffer(
                glow::FRAMEBUFFER,
                glow::STENCIL_ATTACHMENT,
                glow::RENDERBUFFER,
                Some(self.layer_stencil),
            );
        }

        // Switch to the layer target, saving the parent's state.
        let saved_target_fbo = self.target_fbo;
        let saved_target_texture = self.target_texture;
        let saved_origin = self.target_origin;
        let saved_view = self.view_matrix;
        let saved_w = self.renderbuffer_width;
        let saved_h = self.renderbuffer_height;
        let saved_mask = self.mask_state;
        let saved_num_masks = self.num_masks;

        self.target_fbo = Some(self.layer_fbo);
        self.target_texture = Some(layer_tex);
        self.target_origin = (stage_rx, stage_ry);
        self.view_matrix =
            region_view_matrix_unflipped(stage_rx as f32, stage_ry as f32, rw as f32, rh as f32);
        self.renderbuffer_width = rw;
        self.renderbuffer_height = rh;
        self.mask_state = MaskState::NoMask;
        self.num_masks = 0;
        self.mask_state_dirty = true;
        self.active_program = std::ptr::null();
        self.mult_color = None;
        self.add_color = None;

        unsafe {
            gl.viewport(0, 0, rw, rh);
            self.set_stencil_state();
            gl.stencil_mask(0xff);
            gl.clear_color(0.0, 0.0, 0.0, 0.0);
            gl.clear(glow::COLOR_BUFFER_BIT | glow::STENCIL_BUFFER_BIT);
        }

        commands.execute(self);
        // Flush the layer content's batch (under the layer view) before
        // compositing the layer over the parent.
        self.flush_batch();

        // Restore the parent state.
        self.target_fbo = saved_target_fbo;
        self.target_texture = saved_target_texture;
        self.target_origin = saved_origin;
        self.view_matrix = saved_view;
        self.renderbuffer_width = saved_w;
        self.renderbuffer_height = saved_h;
        self.mask_state = saved_mask;
        self.num_masks = saved_num_masks;
        self.mask_state_dirty = true;
        self.active_program = std::ptr::null();
        self.mult_color = None;
        self.add_color = None;

        // Composite the layer over the parent target, restricted to the region.
        let ry_fb = ry; // top-down
        unsafe {
            if let Some(p) = saved_target_fbo {
                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(p));
            } else if let Some(msaa) = &self.msaa_buffers {
                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(msaa.render_framebuffer));
            } else {
                gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            }
            gl.viewport(rx, ry_fb, rw, rh);
            gl.enable(glow::BLEND);
        }
        self.apply_blend_mode(RenderBlendMode::Builtin(BlendMode::Normal));
        self.mask_state_dirty = true;
        self.set_stencil_state();
        unsafe { gl.stencil_mask(0x00) };

        self.fill_with_texture(layer_tex, false);

        unsafe {
            gl.stencil_mask(0xff);
            gl.viewport(0, 0, self.renderbuffer_width, self.renderbuffer_height);
        }
        // Restore the blend func to the surrounding mode.
        let current = self
            .blend_modes
            .last()
            .cloned()
            .unwrap_or(RenderBlendMode::Builtin(BlendMode::Normal));
        self.apply_blend_mode(current);

        self.active_program = std::ptr::null();
        self.mask_state_dirty = true;
        self.pool.release(layer_tex, rw_u, rh_u);
    }
}

/// View matrix for offscreen rendering: like the on-screen one but without the
/// vertical flip, so Flash y=0 (top) maps to texel row 0 — consistent with how
/// bitmaps are uploaded and read back.
fn offscreen_view_matrix(width: i32, height: i32) -> [[f32; 4]; 4] {
    [
        [1.0 / (width as f32 / 2.0), 0.0, 0.0, 0.0],
        [0.0, 1.0 / (height as f32 / 2.0), 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [-1.0, -1.0, 0.0, 1.0],
    ]
}

/// Like [`region_view_matrix`] but with a non-flipped Y, matching an offscreen
/// target rendered with [`offscreen_view_matrix`] (e.g. a cacheAsBitmap pass).
/// Keeps a complex blend's source aligned with the offscreen parent it reads.
fn region_view_matrix_unflipped(rx: f32, ry: f32, rw: f32, rh: f32) -> [[f32; 4]; 4] {
    [
        [1.0 / (rw / 2.0), 0.0, 0.0, 0.0],
        [0.0, 1.0 / (rh / 2.0), 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [-1.0 - rx / (rw / 2.0), -1.0 - ry / (rh / 2.0), 0.0, 1.0],
    ]
}

/// On-screen-style view matrix (flipped Y, matching the framebuffer) restricted
/// to a sub-region: stage pixel `(rx, ry)` maps to the region texture's
/// top-left. Used to render a complex blend's content into a region-sized
/// texture aligned with the framebuffer copy of the same region.
fn region_view_matrix(rx: f32, ry: f32, rw: f32, rh: f32) -> [[f32; 4]; 4] {
    [
        [1.0 / (rw / 2.0), 0.0, 0.0, 0.0],
        [0.0, -1.0 / (rh / 2.0), 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [-1.0 - rx / (rw / 2.0), 1.0 + ry / (rh / 2.0), 0.0, 1.0],
    ]
}

fn union_rect(a: Rectangle<Twips>, b: Rectangle<Twips>) -> Rectangle<Twips> {
    Rectangle {
        x_min: a.x_min.min(b.x_min),
        y_min: a.y_min.min(b.y_min),
        x_max: a.x_max.max(b.x_max),
        y_max: a.y_max.max(b.y_max),
    }
}

/// Tight stage-space (Twips) bounds of a command list's drawn content, or `None`
/// if it draws nothing measurable (e.g. only mask bookkeeping). Used to size the
/// region for a complex blend so it touches only the affected bounding box
/// rather than the whole stage.
fn command_bounds(commands: &CommandList) -> Option<Rectangle<Twips>> {
    let mut acc: Option<Rectangle<Twips>> = None;
    for command in &commands.commands {
        let b = match command {
            Command::RenderShape { shape, transform } => {
                transform.matrix * as_mesh(shape).bounds.clone()
            }
            Command::RenderBitmap {
                bitmap, transform, ..
            } => {
                let d = as_registry_data(bitmap);
                transform.matrix
                    * Rectangle {
                        x_min: Twips::ZERO,
                        y_min: Twips::ZERO,
                        x_max: Twips::from_pixels(d.width as f64),
                        y_max: Twips::from_pixels(d.height as f64),
                    }
            }
            Command::DrawRect { matrix, .. }
            | Command::DrawLine { matrix, .. }
            | Command::DrawLineRect { matrix, .. } => {
                *matrix
                    * Rectangle {
                        x_min: Twips::ZERO,
                        y_min: Twips::ZERO,
                        x_max: Twips::from_pixels(1.0),
                        y_max: Twips::from_pixels(1.0),
                    }
            }
            Command::Blend(sub, _) => match command_bounds(sub) {
                Some(b) => b,
                None => continue,
            },
            Command::RenderAlphaMask {
                maskee_commands, ..
            } => match command_bounds(maskee_commands) {
                Some(b) => b,
                None => continue,
            },
            // Mask bookkeeping and Stage3D contribute no measurable bounds here.
            _ => continue,
        };
        acc = Some(match acc {
            Some(a) => union_rect(a, b),
            None => b,
        });
    }
    acc
}

/// Whether a blend group is a single leaf draw (one shape/bitmap/rect/line), for
/// which the fixed-function fast path composites identically to isolating the
/// group. Anything else — multiple commands, a nested blend, or a mask — can have
/// overlapping content whose internal composition must happen before the group is
/// blended against the backdrop, so it needs the isolate-then-composite path.
fn blend_group_is_single_draw(commands: &CommandList) -> bool {
    matches!(
        commands.commands.as_slice(),
        [Command::RenderShape { .. }
            | Command::RenderBitmap { .. }
            | Command::RenderStage3D { .. }
            | Command::DrawRect { .. }
            | Command::DrawLine { .. }
            | Command::DrawLineRect { .. }]
    )
}

/// Whether a `Layer` blend must be rendered offscreen. It must whenever it
/// contains a child that composites against the layer's backdrop — i.e. any
/// blend other than Normal (Alpha/Erase need a transparent layer to composite
/// into; Multiply/Screen/etc. must read the layer's isolated content rather than
/// the stage behind it). This matches wgpu, which always renders a `Layer` into a
/// fresh texture. Normal-only layers composite associatively, so they can draw
/// straight onto the target; a nested `Layer` isolates its own content, so we
/// don't descend into one.
fn layer_needs_offscreen(commands: &CommandList) -> bool {
    commands.commands.iter().any(|c| match c {
        // A nested Layer isolates itself; a Normal group composites associatively
        // (isolate-then-over == draw-in-place), so neither forces isolation here —
        // but a Normal group may still hide a blend that does, so descend into it.
        Command::Blend(_, RenderBlendMode::Builtin(BlendMode::Alpha))
        | Command::Blend(_, RenderBlendMode::Builtin(BlendMode::Erase)) => true,
        Command::Blend(_, RenderBlendMode::Builtin(BlendMode::Layer)) => false,
        Command::Blend(sub, _) => layer_needs_offscreen(sub),
        Command::RenderAlphaMask {
            maskee_commands, ..
        } => layer_needs_offscreen(maskee_commands),
        _ => false,
    })
}

/// Maps a blend mode to its blend-shader index if it needs the complex
/// (read-destination) path, or `None` for modes expressible with hardware blend
/// state (Normal/Layer/Add/Subtract/Screen) or the unsupported shader blend.
fn complex_blend_index(mode: &RenderBlendMode) -> Option<i32> {
    match mode {
        RenderBlendMode::Builtin(m) => match m {
            BlendMode::Multiply => Some(0),
            BlendMode::Lighten => Some(1),
            BlendMode::Darken => Some(2),
            BlendMode::Difference => Some(3),
            BlendMode::Invert => Some(4),
            BlendMode::Alpha => Some(5),
            BlendMode::Erase => Some(6),
            BlendMode::Overlay => Some(7),
            BlendMode::HardLight => Some(8),
            // Add/Subtract/Screen are fixed-function-expressible (and take the
            // fast path for a single leaf draw), but a multi-child group must be
            // isolated first, then composited via these shader formulas.
            BlendMode::Add => Some(10),
            BlendMode::Subtract => Some(11),
            BlendMode::Screen => Some(12),
            _ => None,
        },
        // No PixelBender backend on GL: shader blends fall back to hardware
        // normal (as before).
        RenderBlendMode::Shader(_) => None,
    }
}

fn same_blend_mode(first: Option<&RenderBlendMode>, second: &RenderBlendMode) -> bool {
    match (first, second) {
        (Some(RenderBlendMode::Builtin(old)), RenderBlendMode::Builtin(new)) => old == new,
        _ => false,
    }
}

/// Hardware blend state for a mode: `[rgb_eq, alpha_eq, rgb_src, rgb_dst,
/// alpha_src, alpha_dst]`. Modes not expressible as fixed-function blend state
/// fall back to Normal here (they take the region-composite path instead).
const NORMAL_BLEND_KEY: [u32; 6] = [
    glow::FUNC_ADD,
    glow::FUNC_ADD,
    glow::ONE,
    glow::ONE_MINUS_SRC_ALPHA,
    glow::ONE,
    glow::ONE_MINUS_SRC_ALPHA,
];

fn blend_key(mode: &RenderBlendMode, minmax_ok: bool) -> [u32; 6] {
    let add = glow::FUNC_ADD;
    match mode {
        RenderBlendMode::Builtin(BlendMode::Add) => {
            [add, add, glow::ONE, glow::ONE, glow::ONE, glow::ONE_MINUS_SRC_ALPHA]
        }
        RenderBlendMode::Builtin(BlendMode::Subtract) => [
            glow::FUNC_REVERSE_SUBTRACT,
            add,
            glow::ONE,
            glow::ONE,
            glow::ONE,
            glow::ONE_MINUS_SRC_ALPHA,
        ],
        RenderBlendMode::Builtin(BlendMode::Screen) => [
            add,
            add,
            glow::ONE,
            glow::ONE_MINUS_SRC_COLOR,
            glow::ONE,
            glow::ONE_MINUS_SRC_ALPHA,
        ],
        RenderBlendMode::Builtin(BlendMode::Multiply) => [
            add,
            add,
            glow::DST_COLOR,
            glow::ONE_MINUS_SRC_ALPHA,
            glow::ONE,
            glow::ONE_MINUS_SRC_ALPHA,
        ],
        RenderBlendMode::Builtin(BlendMode::Darken) if minmax_ok => {
            [glow::MIN, add, glow::ONE, glow::ONE, glow::ZERO, glow::ONE]
        }
        RenderBlendMode::Builtin(BlendMode::Lighten) if minmax_ok => {
            [glow::MAX, add, glow::ONE, glow::ONE, glow::ZERO, glow::ONE]
        }
        _ => NORMAL_BLEND_KEY,
    }
}

/// Whether a mode is fully expressible as fixed-function blend state (so it can
/// be batched), as opposed to needing the region-composite/offscreen path.
fn is_hw_blend(mode: &RenderBlendMode, minmax_ok: bool) -> bool {
    matches!(
        mode,
        RenderBlendMode::Builtin(
            BlendMode::Normal
                | BlendMode::Add
                | BlendMode::Subtract
                | BlendMode::Screen
                | BlendMode::Multiply
        )
    ) || (minmax_ok
        && matches!(
            mode,
            RenderBlendMode::Builtin(BlendMode::Darken | BlendMode::Lighten)
        ))
}

impl Drop for GlRenderBackend {
    fn drop(&mut self) {
        unsafe {
            self.gl.delete_program(self.color_program.program);
            self.gl.delete_program(self.bitmap_program.program);
            self.gl.delete_program(self.gradient_program.program);
            self.gl.delete_texture(self.gradient_texture);
            self.gl.delete_program(self.batch_color_program.program);
            self.gl.delete_program(self.batch_bitmap_program.program);
            self.gl.delete_program(self.copy_program.program);
            self.gl.delete_buffer(self.pb_quad_vbo);
            self.gl.delete_vertex_array(self.pb_vao);
            self.gl.delete_vertex_array(self.batch_vao);
            self.gl.delete_buffer(self.batch_vbo);
            self.gl.delete_buffer(self.batch_ibo);
            self.gl.delete_vertex_array(self.batch_bitmap_vao);
            self.gl.delete_buffer(self.batch_bitmap_vbo);
            self.gl.delete_buffer(self.batch_bitmap_ibo);
            self.gl.delete_framebuffer(self.scratch_fbo);
            self.gl.delete_framebuffer(self.offscreen_fbo);
            self.gl.delete_renderbuffer(self.offscreen_stencil);
            self.gl.delete_framebuffer(self.blend_msaa_fbo);
            self.gl.delete_renderbuffer(self.blend_msaa_color);
            self.gl.delete_renderbuffer(self.blend_msaa_stencil);
            self.gl.delete_framebuffer(self.blend_msaa_resolve_fbo);
            self.gl.delete_texture(self.blend_msaa_resolve_color);
            self.gl.delete_framebuffer(self.layer_fbo);
            self.gl.delete_renderbuffer(self.layer_stencil);
        }
    }
}

impl RenderBackend for GlRenderBackend {
    fn render_offscreen(
        &mut self,
        handle: BitmapHandle,
        batches: RenderOffscreenBatches,
        _quality: StageQuality,
        bounds: PixelRegion,
    ) -> Option<Box<dyn SyncHandle>> {
        if batches.is_empty() {
            return None;
        }

        let (texture, width, height) = {
            let data = as_registry_data(&handle);
            (data.texture, data.width as i32, data.height as i32)
        };

        // Each batch is rendered as its own pass against the running target
        // texture; existing content is preserved (load), commands composite on
        // top. Matches wgpu's `FreshWithTexture` semantics.
        for commands in batches {
            let view = offscreen_view_matrix(width, height);
            self.render_commands_to_texture(texture, width, height, None, view, true, commands);
        }

        // Restore the on-screen viewport (renderbuffer dims were restored by the
        // helper).
        unsafe {
            self.gl
                .viewport(0, 0, self.renderbuffer_width, self.renderbuffer_height);
        }

        Some(Box::new(GlSyncHandle {
            gl: self.gl.clone(),
            handle,
            copy_area: bounds,
        }))
    }

    fn is_offscreen_supported(&self) -> bool {
        true
    }

    fn supports_perspective_triangles(&self) -> bool {
        true
    }

    fn is_filter_supported(&self, filter: &Filter) -> bool {
        matches!(
            filter,
            Filter::ColorMatrixFilter(_)
                | Filter::BlurFilter(_)
                | Filter::GlowFilter(_)
                | Filter::DropShadowFilter(_)
                | Filter::BevelFilter(_)
                | Filter::ConvolutionFilter(_)
                | Filter::DisplacementMapFilter(_)
                | Filter::GradientGlowFilter(_)
                | Filter::GradientBevelFilter(_)
                | Filter::ShaderFilter(_)
        )
    }

    fn apply_filter(
        &mut self,
        source: BitmapHandle,
        source_point: (u32, u32),
        source_size: (u32, u32),
        destination: BitmapHandle,
        dest_point: (i32, i32),
        filter: Filter,
    ) -> Option<Box<dyn SyncHandle>> {
        // Lazily build the filter programs on first use.
        if self.filters.is_none() {
            match filters::Filters::new(self.gl.clone(), self.caps.is_embedded) {
                Ok(f) => self.filters = Some(f),
                Err(e) => {
                    log::error!("Couldn't initialize GL filters: {e}");
                    return None;
                }
            }
        }

        let (src_tex, src_w, src_h) = {
            let d = as_registry_data(&source);
            (d.texture, d.width, d.height)
        };
        let (dst_tex, dst_w, dst_h) = {
            let d = as_registry_data(&destination);
            (d.texture, d.width, d.height)
        };

        // ShaderFilter runs a PixelBender program (uses `self` directly, so it
        // must run before borrowing the fixed-function filter set).
        if let Filter::ShaderFilter(sf) = &filter {
            let arc = sf.shader.0.clone();
            let pb: &GlPixelBenderShader = <dyn Any>::downcast_ref(&*arc)?;
            // The shader output is the size of the filter source region (matching
            // wgpu's `ShaderFilter::apply`), *not* the destination bitmap. The
            // result is then blitted into the destination at `dest_point`, leaving
            // the rest of the destination untouched.
            let (sw, sh) = source_size;
            let out_tex = self.pool.acquire(sw, sh)?;
            self.execute_pixelbender(
                pb,
                &sf.shader_args,
                out_tex,
                sw as i32,
                sh as i32,
                true,
                Some((src_tex, sw as f32, sh as f32)),
            );

            // Blit the result into the destination at `dest_point`, clamping a
            // negative offset by skipping the corresponding source rows/cols
            // (same arithmetic as the fixed-function filter path below).
            let (dest_x, dest_y) = dest_point;
            let src_offset_x = dest_x.min(0).unsigned_abs();
            let src_offset_y = dest_y.min(0).unsigned_abs();
            let final_dest_x = dest_x.max(0) as u32;
            let final_dest_y = dest_y.max(0) as u32;
            let copy_w = sw
                .saturating_sub(src_offset_x)
                .min(dst_w.saturating_sub(final_dest_x));
            let copy_h = sh
                .saturating_sub(src_offset_y)
                .min(dst_h.saturating_sub(final_dest_y));
            let gl = self.gl.clone();
            if copy_w > 0 && copy_h > 0 {
                unsafe {
                    gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.scratch_fbo));
                    gl.framebuffer_texture_2d(
                        glow::FRAMEBUFFER,
                        glow::COLOR_ATTACHMENT0,
                        glow::TEXTURE_2D,
                        Some(out_tex),
                        0,
                    );
                    gl.bind_texture(glow::TEXTURE_2D, Some(dst_tex));
                    gl.copy_tex_sub_image_2d(
                        glow::TEXTURE_2D,
                        0,
                        final_dest_x as i32,
                        final_dest_y as i32,
                        src_offset_x as i32,
                        src_offset_y as i32,
                        copy_w as i32,
                        copy_h as i32,
                    );
                    gl.bind_framebuffer(glow::FRAMEBUFFER, None);
                }
            }
            self.pool.release(out_tex, sw, sh);
            self.active_program = std::ptr::null();
            self.mask_state_dirty = true;
            return Some(Box::new(GlSyncHandle {
                gl: self.gl.clone(),
                handle: destination.clone(),
                copy_area: PixelRegion::for_whole_size(dst_w, dst_h),
            }));
        }

        let filters = self.filters.as_ref().expect("filters just initialized");
        let result = match &filter {
            Filter::ColorMatrixFilter(f) => filters.apply_color_matrix(
                &mut self.pool,
                src_tex,
                src_w,
                src_h,
                source_point,
                source_size,
                &f.matrix,
            ),
            Filter::BlurFilter(f) => filters.apply_blur(
                &mut self.pool,
                src_tex,
                src_w,
                src_h,
                source_point,
                source_size,
                f.blur_x.to_f32(),
                f.blur_y.to_f32(),
                f.num_passes() as u32,
            ),
            Filter::GlowFilter(f) => filters.apply_glow(
                &mut self.pool,
                src_tex,
                src_w,
                src_h,
                source_point,
                source_size,
                color_to_rgba(f.color),
                f.strength.to_f32(),
                f.is_inner(),
                f.is_knockout(),
                f.composite_source(),
                f.blur_x.to_f32(),
                f.blur_y.to_f32(),
                f.num_passes() as u32,
                (0.0, 0.0),
            ),
            Filter::DropShadowFilter(f) => {
                let distance = f.distance.to_f32();
                let angle = f.angle.to_f32();
                let offset = (angle.cos() * distance, angle.sin() * distance);
                filters.apply_glow(
                    &mut self.pool,
                    src_tex,
                    src_w,
                    src_h,
                    source_point,
                    source_size,
                    color_to_rgba(f.color),
                    f.strength.to_f32(),
                    f.is_inner(),
                    f.is_knockout(),
                    !f.hide_object(),
                    f.blur_x.to_f32(),
                    f.blur_y.to_f32(),
                    f.num_passes() as u32,
                    (-offset.0, -offset.1),
                )
            }
            Filter::BevelFilter(f) => {
                let distance = f.distance.to_f32();
                let angle = f.angle.to_f32();
                let offset = (angle.cos() * distance, angle.sin() * distance);
                let bevel_type = if f.is_on_top() {
                    2
                } else if f.is_inner() {
                    1
                } else {
                    0
                };
                filters.apply_bevel(
                    &mut self.pool,
                    src_tex,
                    src_w,
                    src_h,
                    source_point,
                    source_size,
                    premultiplied_rgba(f.highlight_color),
                    premultiplied_rgba(f.shadow_color),
                    f.strength.to_f32(),
                    bevel_type,
                    f.is_knockout(),
                    f.blur_x.to_f32(),
                    f.blur_y.to_f32(),
                    f.num_passes() as u32,
                    offset,
                )
            }
            Filter::ConvolutionFilter(f) => match convolution_params(f) {
                Some((kernel, cols, rows, divisor, bias, default_color, clamp, preserve)) => filters
                    .apply_convolution(
                        &mut self.pool,
                        src_tex,
                        src_w,
                        src_h,
                        source_point,
                        source_size,
                        &kernel,
                        cols,
                        rows,
                        divisor,
                        bias,
                        default_color,
                        clamp,
                        preserve,
                    ),
                None => None,
            },
            Filter::DisplacementMapFilter(f) => match f.map_bitmap.as_ref() {
                Some(map) => {
                    let (map_tex, map_w, map_h) = {
                        let d = as_registry_data(map);
                        (d.texture, d.width, d.height)
                    };
                    filters.apply_displacement(
                        &mut self.pool,
                        src_tex,
                        src_w,
                        src_h,
                        source_point,
                        source_size,
                        map_tex,
                        map_w,
                        map_h,
                        color_to_rgba(f.color),
                        (f.component_x as f32, f.component_y as f32),
                        displacement_mode(f.mode),
                        (f.scale_x, f.scale_y),
                        (f.map_point.0 as f32, f.map_point.1 as f32),
                        (f.viewscale_x, f.viewscale_y),
                    )
                }
                None => None,
            },
            Filter::GradientGlowFilter(f) => {
                let ramp = build_gradient_ramp(&f.colors);
                let distance = f.distance.to_f32();
                let angle = f.angle.to_f32();
                let offset = (angle.cos() * distance, angle.sin() * distance);
                let gtype = if f.is_on_top() {
                    2
                } else if f.is_inner() {
                    1
                } else {
                    0
                };
                filters.apply_gradient_glow(
                    &mut self.pool,
                    src_tex,
                    src_w,
                    src_h,
                    source_point,
                    source_size,
                    &ramp,
                    f.strength.to_f32(),
                    gtype,
                    f.is_knockout(),
                    f.flags.contains(swf::GradientFilterFlags::COMPOSITE_SOURCE),
                    f.blur_x.to_f32(),
                    f.blur_y.to_f32(),
                    f.num_passes() as u32,
                    offset,
                )
            }
            Filter::GradientBevelFilter(f) => {
                let ramp = build_gradient_ramp(&f.colors);
                let distance = f.distance.to_f32();
                let angle = f.angle.to_f32();
                let offset = (angle.cos() * distance, angle.sin() * distance);
                let bevel_type = if f.is_on_top() {
                    2
                } else if f.is_inner() {
                    1
                } else {
                    0
                };
                filters.apply_gradient_bevel(
                    &mut self.pool,
                    src_tex,
                    src_w,
                    src_h,
                    source_point,
                    source_size,
                    &ramp,
                    f.strength.to_f32(),
                    bevel_type,
                    f.is_knockout(),
                    f.blur_x.to_f32(),
                    f.blur_y.to_f32(),
                    f.num_passes() as u32,
                    offset,
                )
            }
            // Other filters not yet supported on the GL backend.
            _ => None,
        }?;

        // The filter pass changed program/VAO/blend state outside the normal
        // command path; invalidate the cached active program.
        self.active_program = std::ptr::null();

        // Copy the filter result into the destination at dest_point, clamping
        // negative offsets the same way wgpu does.
        let (dest_x, dest_y) = dest_point;
        let src_offset_x = dest_x.min(0).unsigned_abs();
        let src_offset_y = dest_y.min(0).unsigned_abs();
        let final_dest_x = dest_x.max(0) as u32;
        let final_dest_y = dest_y.max(0) as u32;
        let copy_w = result
            .width
            .saturating_sub(src_offset_x)
            .min(dst_w.saturating_sub(final_dest_x));
        let copy_h = result
            .height
            .saturating_sub(src_offset_y)
            .min(dst_h.saturating_sub(final_dest_y));

        let gl = self.gl.clone();
        if copy_w == 0 || copy_h == 0 {
            self.pool.release(result.texture, result.width, result.height);
            return None;
        }

        // Result texels stay Flash-top at row 0, matching the destination, so
        // Flash coordinates map directly (no Y flip), as in copy_pixels_to_texture.
        unsafe {
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.scratch_fbo));
            gl.framebuffer_texture_2d(
                glow::FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_2D,
                Some(result.texture),
                0,
            );
            gl.bind_texture(glow::TEXTURE_2D, Some(dst_tex));
            gl.copy_tex_sub_image_2d(
                glow::TEXTURE_2D,
                0,
                final_dest_x as i32,
                final_dest_y as i32,
                src_offset_x as i32,
                src_offset_y as i32,
                copy_w as i32,
                copy_h as i32,
            );
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
        }
        self.pool.release(result.texture, result.width, result.height);

        let dest_region = PixelRegion {
            x_min: final_dest_x,
            y_min: final_dest_y,
            x_max: final_dest_x + copy_w,
            y_max: final_dest_y + copy_h,
        };
        Some(Box::new(GlSyncHandle {
            gl: self.gl.clone(),
            handle: destination,
            copy_area: dest_region,
        }))
    }

    fn copy_pixels_to_texture(
        &mut self,
        source: BitmapHandle,
        source_region: PixelRegion,
        destination: BitmapHandle,
        dest_point: (u32, u32),
    ) -> Option<Box<dyn SyncHandle>> {
        let (src_texture, src_w, src_h) = {
            let d = as_registry_data(&source);
            (d.texture, d.width, d.height)
        };
        let (dst_texture, dst_w, dst_h) = {
            let d = as_registry_data(&destination);
            (d.texture, d.width, d.height)
        };

        let copy_width = source_region
            .width()
            .min(dst_w.saturating_sub(dest_point.0))
            .min(src_w.saturating_sub(source_region.x_min));
        let copy_height = source_region
            .height()
            .min(dst_h.saturating_sub(dest_point.1))
            .min(src_h.saturating_sub(source_region.y_min));
        if copy_width == 0 || copy_height == 0 {
            return None;
        }

        // glCopyTexSubImage2D copies from the read framebuffer to a texture.
        // Both our offscreen textures store Flash-top at texel row 0 and FBO row
        // 0 == texel row 0, so Flash coordinates map directly with no Y flip.
        // Available on GLES2/WebGL1, unlike `blit_framebuffer`.
        let gl = self.gl.clone();
        unsafe {
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.scratch_fbo));
            gl.framebuffer_texture_2d(
                glow::FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_2D,
                Some(src_texture),
                0,
            );
            gl.bind_texture(glow::TEXTURE_2D, Some(dst_texture));
            gl.copy_tex_sub_image_2d(
                glow::TEXTURE_2D,
                0,
                dest_point.0 as i32,
                dest_point.1 as i32,
                source_region.x_min as i32,
                source_region.y_min as i32,
                copy_width as i32,
                copy_height as i32,
            );
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
        }

        let dest_region = PixelRegion {
            x_min: dest_point.0,
            y_min: dest_point.1,
            x_max: dest_point.0 + copy_width,
            y_max: dest_point.1 + copy_height,
        };
        Some(Box::new(GlSyncHandle {
            gl: self.gl.clone(),
            handle: destination,
            copy_area: dest_region,
        }))
    }

    fn viewport_dimensions(&self) -> ViewportDimensions {
        ViewportDimensions {
            width: self.renderbuffer_width as u32,
            height: self.renderbuffer_height as u32,
            scale_factor: self.viewport_scale_factor,
        }
    }

    fn set_viewport_dimensions(&mut self, dimensions: ViewportDimensions) {
        // Build view matrix based on canvas size. Top-down (Flash-top -> row 0)
        // so the GPU top-left fill rule matches wgpu; the present flips it back.
        self.view_matrix = [
            [1.0 / (dimensions.width as f32 / 2.0), 0.0, 0.0, 0.0],
            [0.0, 1.0 / (dimensions.height as f32 / 2.0), 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [-1.0, -1.0, 0.0, 1.0],
        ];

        // On the web, clamp to the actual drawing-buffer size (which reads zero
        // when the context is lost, hence not using `.clamp()`). On native the
        // integration owns the surface size, so we trust the requested values.
        #[cfg(target_family = "wasm")]
        {
            let (max_w, max_h) = self.web_context.drawing_buffer_size();
            self.renderbuffer_width = (dimensions.width.max(1) as i32).min(max_w);
            self.renderbuffer_height = (dimensions.height.max(1) as i32).min(max_h);
        }
        #[cfg(not(target_family = "wasm"))]
        {
            self.renderbuffer_width = dimensions.width.max(1) as i32;
            self.renderbuffer_height = dimensions.height.max(1) as i32;
        }

        // Recreate framebuffers with the new size.
        self.build_msaa_buffers();
        unsafe {
            self.gl
                .viewport(0, 0, self.renderbuffer_width, self.renderbuffer_height);
        }
        self.viewport_scale_factor = dimensions.scale_factor;
    }

    fn register_shape(
        &mut self,
        shape: DistilledShape,
        bitmap_source: &dyn BitmapSource,
    ) -> ShapeHandle {
        self.register_shape_with_scale(shape, bitmap_source, 1.0)
    }

    fn register_shape_with_scale(
        &mut self,
        shape: DistilledShape,
        bitmap_source: &dyn BitmapSource,
        scale: f32,
    ) -> ShapeHandle {
        let bounds = shape.shape_bounds.clone();
        let draws = match self.register_shape_internal(shape, bitmap_source, scale) {
            Ok(draws) => draws,
            Err(e) => {
                log::error!("Couldn't register shape: {e:?}");
                vec![]
            }
        };
        ShapeHandle(Arc::new(Mesh {
            gl: self.gl.clone(),
            draws,
            bounds,
        }))
    }

    fn submit_frame(
        &mut self,
        clear: Color,
        commands: CommandList,
        cache_entries: Vec<BitmapCacheEntry>,
    ) {
        // Render cacheAsBitmap entries into their textures first. Each entry is
        // cleared to its clear color, its commands drawn on top, then its filters
        // applied in sequence.
        for entry in cache_entries {
            let (texture, width, height) = {
                let data = as_registry_data(&entry.handle);
                (data.texture, data.width as i32, data.height as i32)
            };
            let view = offscreen_view_matrix(width, height);
            self.render_commands_to_texture(
                texture,
                width,
                height,
                Some(entry.clear),
                view,
                true,
                entry.commands,
            );
            for filter in &entry.filters {
                if !self.apply_filter_in_place(texture, width as u32, height as u32, filter) {
                    log::warn!("GL backend: unsupported cache filter ignored: {filter:?}");
                }
            }
        }

        self.begin_frame(clear);
        commands.execute(self);
        // Flush the frame's final batch before resolving/presenting.
        self.flush_batch();
        self.end_frame();
    }

    fn register_bitmap(&mut self, bitmap: Bitmap<'_>) -> Result<BitmapHandle, BitmapError> {
        let (format, bitmap) = match bitmap.format() {
            BitmapFormat::Rgb | BitmapFormat::Yuv420p => (glow::RGB, bitmap.to_rgb()),
            BitmapFormat::Rgba | BitmapFormat::Yuva420p => (glow::RGBA, bitmap.to_rgba()),
        };

        let texture = unsafe { self.gl.create_texture() }.map_err(bitmap_gl_error)?;
        unsafe {
            self.gl.bind_texture(glow::TEXTURE_2D, Some(texture));
            self.gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                format as i32,
                bitmap.width() as i32,
                bitmap.height() as i32,
                0,
                format,
                glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(Some(bitmap.data())),
            );

            // Non-power-of-2 textures require these parameters to function in WebGL1.
            self.gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_WRAP_S,
                glow::CLAMP_TO_EDGE as i32,
            );
            self.gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_WRAP_T,
                glow::CLAMP_TO_EDGE as i32,
            );
            self.gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MIN_FILTER,
                glow::LINEAR as i32,
            );
            self.gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MAG_FILTER,
                glow::LINEAR as i32,
            );
        }

        Ok(BitmapHandle(Arc::new(RegistryData {
            gl: self.gl.clone(),
            width: bitmap.width(),
            height: bitmap.height(),
            texture,
        })))
    }

    fn update_texture(
        &mut self,
        handle: &BitmapHandle,
        bitmap: Bitmap<'_>,
        _region: PixelRegion,
    ) -> Result<(), BitmapError> {
        let texture = as_registry_data(handle).texture;

        let (format, bitmap) = match bitmap.format() {
            BitmapFormat::Rgb | BitmapFormat::Yuv420p => (glow::RGB, bitmap.to_rgb()),
            BitmapFormat::Rgba | BitmapFormat::Yuva420p => (glow::RGBA, bitmap.to_rgba()),
        };

        unsafe {
            self.gl.bind_texture(glow::TEXTURE_2D, Some(texture));
            self.gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                format as i32,
                bitmap.width() as i32,
                bitmap.height() as i32,
                0,
                format,
                glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(Some(bitmap.data())),
            );
        }

        Ok(())
    }

    fn create_context3d(
        &mut self,
        profile: Context3DProfile,
    ) -> Result<Box<dyn Context3D>, BitmapError> {
        Ok(Box::new(context3d::GlContext3D::new(
            self.gl.clone(),
            profile,
            self.caps.is_embedded,
            self.caps.is_gles3_or_webgl2,
        )))
    }

    fn debug_info(&self) -> Cow<'static, str> {
        let mut result = vec![];

        if self.caps.is_gles3_or_webgl2 {
            result.push("Renderer: OpenGL (GLES3 / WebGL2 / GL 3+)".to_string());
        } else {
            result.push("Renderer: OpenGL (GLES2 / WebGL1)".to_string());
        }

        unsafe {
            result.push(format!(
                "Adapter Vendor: {}",
                self.gl.get_parameter_string(glow::VENDOR)
            ));
            result.push(format!(
                "Adapter Renderer: {}",
                self.gl.get_parameter_string(glow::RENDERER)
            ));
            result.push(format!(
                "Adapter Version: {}",
                self.gl.get_parameter_string(glow::VERSION)
            ));
        }

        result.push(format!("Surface samples: {} x ", self.msaa_sample_count));
        result.push(format!(
            "Surface size: {} x {}",
            self.renderbuffer_width, self.renderbuffer_height
        ));

        Cow::Owned(result.join("\n"))
    }

    fn name(&self) -> &'static str {
        "gl"
    }

    fn set_quality(&mut self, quality: StageQuality) {
        let samples = context::recommended_msaa_samples(&self.caps, quality);
        if samples != self.msaa_sample_count {
            self.msaa_sample_count = samples;
            // Rebuild the scene buffers at the new sample count (no-op on
            // WebGL1/GLES2). Also refreshes the present buffer for `present_flip`.
            self.build_msaa_buffers();
        }
    }

    fn compile_pixelbender_shader(
        &mut self,
        shader: PixelBenderShader,
    ) -> Result<PixelBenderShaderHandle, BitmapError> {
        let translated = pixelbender::translate(&shader)
            .map_err(|e| BitmapError::Unimplemented(format!("PixelBender: {e}").into()))?;

        let is_embedded = self.caps.is_embedded;
        let gl = self.gl.clone();
        let vertex =
            shader::compile_shader(&gl, is_embedded, glow::VERTEX_SHADER, PIXELBENDER_VERTEX_GLSL)
                .map_err(|e| BitmapError::Unimplemented(format!("PixelBender vertex: {e:?}").into()))?;
        let fragment =
            shader::compile_shader(&gl, is_embedded, glow::FRAGMENT_SHADER, &translated.glsl)
                .map_err(|e| {
                    BitmapError::Unimplemented(format!("PixelBender fragment: {e:?}").into())
                })?;

        let program = unsafe {
            let p = gl
                .create_program()
                .map_err(|e| BitmapError::Unimplemented(format!("PixelBender program: {e}").into()))?;
            gl.attach_shader(p, vertex);
            gl.attach_shader(p, fragment);
            gl.link_program(p);
            let ok = gl.get_program_link_status(p);
            gl.delete_shader(vertex);
            gl.delete_shader(fragment);
            if !ok {
                let msg = gl.get_program_info_log(p);
                gl.delete_program(p);
                return Err(BitmapError::Unimplemented(
                    format!("PixelBender link error: {msg}").into(),
                ));
            }
            p
        };

        let position_loc = unsafe { gl.get_attrib_location(program, "position") };
        let loc = |n: &str| unsafe { gl.get_uniform_location(program, n) };
        let u_inputs = translated
            .inputs
            .iter()
            .map(|&i| {
                (
                    i,
                    loc(&format!("u_tex_{i}")),
                    loc(&format!("u_tex_size_{i}")),
                    translated.nearest_inputs.contains(&i),
                )
            })
            .collect();

        let pb = GlPixelBenderShader {
            gl: self.gl.clone(),
            program,
            float_slots: translated.float_slots,
            int_slots: translated.int_slots,
            param_slots: translated.param_slots,
            output_channels: translated.output_channels,
            position_loc,
            u_float_params: loc("u_float_params[0]"),
            u_int_params: loc("u_int_params[0]"),
            u_zeroed: loc("u_zeroed"),
            u_out_size: loc("u_out_size"),
            u_inputs,
            shader,
        };
        self.active_program = std::ptr::null();
        Ok(PixelBenderShaderHandle(Arc::new(pb)))
    }

    fn resolve_sync_handle(
        &mut self,
        handle: Box<dyn SyncHandle>,
        with_rgba: RgbaBufRead,
    ) -> Result<(), ruffle_render::error::Error> {
        let handle = Box::<dyn Any>::downcast::<GlSyncHandle>(handle)
            .expect("Sync handle must be a gl GlSyncHandle");
        handle.capture(self.scratch_fbo, with_rgba);
        Ok(())
    }

    fn run_pixelbender_shader(
        &mut self,
        handle: PixelBenderShaderHandle,
        arguments: &[PixelBenderShaderArgument],
        target: &PixelBenderTarget,
    ) -> Result<PixelBenderOutput, BitmapError> {
        let arc = handle.0.clone();
        let pb: &GlPixelBenderShader = <dyn Any>::downcast_ref(&*arc)
            .expect("PixelBender handle must be a GL shader");

        let gl = self.gl.clone();
        match target {
            // ShaderJob mode: samples outside a texture are clamped (not zeroed).
            PixelBenderTarget::Bitmap(h) => {
                let r = as_registry_data(h);
                let (tex, w, hh) = (r.texture, r.width as i32, r.height as i32);
                self.execute_pixelbender(pb, arguments, tex, w, hh, false, None);
                Ok(PixelBenderOutput::Bitmap(Box::new(GlSyncHandle {
                    gl: self.gl.clone(),
                    handle: h.clone(),
                    copy_area: PixelRegion::for_whole_size(w as u32, hh as u32),
                })))
            }
            PixelBenderTarget::Bytes { width, height } => {
                let (w, hh) = (*width as i32, *height as i32);
                let channels = pb.output_channels.max(1);
                // ShaderJob output is Vector.<Number>: render to a float target so
                // values aren't clamped to [0,1] like an RGBA8 bitmap.
                let tex = unsafe {
                    let t = gl.create_texture().map_err(bitmap_gl_error)?;
                    gl.bind_texture(glow::TEXTURE_2D, Some(t));
                    gl.tex_image_2d(
                        glow::TEXTURE_2D,
                        0,
                        glow::RGBA32F as i32,
                        w,
                        hh,
                        0,
                        glow::RGBA,
                        glow::FLOAT,
                        glow::PixelUnpackData::Slice(None),
                    );
                    gl.tex_parameter_i32(
                        glow::TEXTURE_2D,
                        glow::TEXTURE_MIN_FILTER,
                        glow::NEAREST as i32,
                    );
                    gl.tex_parameter_i32(
                        glow::TEXTURE_2D,
                        glow::TEXTURE_MAG_FILTER,
                        glow::NEAREST as i32,
                    );
                    t
                };

                self.execute_pixelbender(pb, arguments, tex, w, hh, false, None);

                let mut raw = vec![0u8; (w * hh * 16) as usize];
                unsafe {
                    gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.scratch_fbo));
                    gl.framebuffer_texture_2d(
                        glow::FRAMEBUFFER,
                        glow::COLOR_ATTACHMENT0,
                        glow::TEXTURE_2D,
                        Some(tex),
                        0,
                    );
                    gl.read_pixels(
                        0,
                        0,
                        w,
                        hh,
                        glow::RGBA,
                        glow::FLOAT,
                        glow::PixelPackData::Slice(Some(&mut raw)),
                    );
                    gl.bind_framebuffer(glow::FRAMEBUFFER, None);
                    gl.delete_texture(tex);
                }

                // The core reads exactly `output_channels` floats per pixel, so
                // strip the RGBA padding for 1/2/3-channel outputs.
                let floats: &[f32] = bytemuck::cast_slice(&raw);
                let bytes = if channels >= 4 {
                    raw
                } else {
                    let n = (w * hh) as usize;
                    let mut out = Vec::with_capacity(n * channels);
                    for p in 0..n {
                        for c in 0..channels {
                            out.push(floats[p * 4 + c]);
                        }
                    }
                    bytemuck::cast_slice(&out).to_vec()
                };
                Ok(PixelBenderOutput::Bytes(bytes))
            }
        }
    }

    fn create_empty_texture(
        &mut self,
        width: NonZeroU32,
        height: NonZeroU32,
    ) -> Result<BitmapHandle, BitmapError> {
        let texture = unsafe { self.gl.create_texture() }.map_err(bitmap_gl_error)?;
        unsafe {
            self.gl.bind_texture(glow::TEXTURE_2D, Some(texture));

            // Allocate RGBA storage without uploading a CPU buffer; the contents
            // are cleared to transparent on the GPU below. This avoids a
            // `w * h * 4` zeroed heap allocation per BitmapData, a meaningful
            // source of allocator churn.
            self.gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::RGBA as i32,
                width.get() as i32,
                height.get() as i32,
                0,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(None),
            );

            // Non-power-of-2 textures require these parameters to function in WebGL1.
            self.gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_WRAP_S,
                glow::CLAMP_TO_EDGE as i32,
            );
            self.gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_WRAP_T,
                glow::CLAMP_TO_EDGE as i32,
            );
            self.gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MIN_FILTER,
                glow::LINEAR as i32,
            );
            self.gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MAG_FILTER,
                glow::LINEAR as i32,
            );

            // Give it the defined transparent starting state BitmapData expects,
            // via a GPU clear through the shared scratch framebuffer.
            self.gl
                .bind_framebuffer(glow::FRAMEBUFFER, Some(self.scratch_fbo));
            self.gl.framebuffer_texture_2d(
                glow::FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_2D,
                Some(texture),
                0,
            );
            self.gl.color_mask(true, true, true, true);
            self.gl.clear_color(0.0, 0.0, 0.0, 0.0);
            self.gl.clear(glow::COLOR_BUFFER_BIT);
            self.gl.bind_framebuffer(glow::FRAMEBUFFER, None);
        }
        // The clear touched color-mask/clear-color and the active program's
        // expectations; force the next draw to re-apply its state.
        self.mask_state_dirty = true;
        self.active_program = std::ptr::null();

        Ok(BitmapHandle(Arc::new(RegistryData {
            gl: self.gl.clone(),
            width: width.get(),
            height: height.get(),
            texture,
        })))
    }
}

impl CommandHandler for GlRenderBackend {
    fn render_bitmap(
        &mut self,
        bitmap: BitmapHandle,
        transform: Transform,
        smoothing: bool,
        pixel_snapping: PixelSnapping,
    ) {
        let entry = as_registry_data(&bitmap);
        let texture = entry.texture;

        // Scale the [0,1]² quad to the bitmap's dimensions, in stage space.
        let mut matrix = transform.matrix;
        pixel_snapping.apply(&mut matrix);
        matrix *= Matrix::scale(entry.width as f32, entry.height as f32);

        let key = BitmapBatchKey {
            texture,
            smoothing,
            mult: transform.color_transform.mult_rgba_normalized(),
            add: transform.color_transform.add_rgba_normalized(),
        };

        // Apply the current mask state now so it's bound when the batch flushes.
        self.set_stencil_state();
        // Move the handle in so the batch keeps the GL texture alive until flush.
        self.append_bitmap_draw(key, matrix, bitmap);
    }

    fn render_shape(&mut self, shape: ShapeHandle, transform: Transform) {
        let world_matrix = [
            [transform.matrix.a, transform.matrix.b, 0.0, 0.0],
            [transform.matrix.c, transform.matrix.d, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [
                transform.matrix.tx.to_pixels() as f32,
                transform.matrix.ty.to_pixels() as f32,
                0.0,
                1.0,
            ],
        ];

        let mult_color = transform.color_transform.mult_rgba_normalized();
        let add_color = transform.color_transform.add_rgba_normalized();

        self.set_stencil_state();

        let mesh = as_mesh(&shape);
        for draw in &mesh.draws {
            // Ignore strokes when drawing a mask stencil.
            let num_indices = if self.mask_state != MaskState::DrawMaskStencil
                && self.mask_state != MaskState::ClearMaskStencil
            {
                draw.num_indices
            } else {
                draw.num_mask_indices
            };
            if num_indices == 0 {
                continue;
            }

            // Solid-color draws are accumulated into the shared batch (one draw
            // call for a run of consecutive color shapes); anything else breaks
            // the run and flushes it first.
            if let DrawType::Color = &draw.draw_type {
                if let Some(geom) = &draw.color_cpu {
                    self.append_color_draw(
                        geom,
                        num_indices as usize,
                        transform.matrix,
                        mult_color,
                        add_color,
                    );
                    continue;
                }
            }
            self.flush_batch();

            self.bind_vertex_array(Some(draw.vao));

            let program = match &draw.draw_type {
                DrawType::Color => &self.color_program,
                DrawType::Gradient(_) => &self.gradient_program,
                DrawType::Bitmap { .. } => &self.bitmap_program,
                DrawType::PerspectiveBitmap { .. } => &self.perspective_bitmap_program,
            };

            if !std::ptr::eq(program, self.active_program) {
                unsafe { self.gl.use_program(Some(program.program)) };
                self.active_program = program as *const ShaderProgram;

                program.uniform_matrix4fv(&self.gl, ShaderUniform::ViewMatrix, &self.view_matrix);

                self.mult_color = None;
                self.add_color = None;
            }

            program.uniform_matrix4fv(&self.gl, ShaderUniform::WorldMatrix, &world_matrix);
            if Some(mult_color) != self.mult_color {
                program.uniform4fv(&self.gl, ShaderUniform::MultColor, &mult_color);
                self.mult_color = Some(mult_color);
            }
            if Some(add_color) != self.add_color {
                program.uniform4fv(&self.gl, ShaderUniform::AddColor, &add_color);
                self.add_color = Some(add_color);
            }

            // Set shader-specific uniforms.
            match &draw.draw_type {
                DrawType::Color => (),
                DrawType::Gradient(gradient) => {
                    program.uniform_matrix3fv(
                        &self.gl,
                        ShaderUniform::TextureMatrix,
                        &gradient.matrix,
                    );
                    program.uniform1i(
                        &self.gl,
                        ShaderUniform::GradientType,
                        gradient.gradient_type,
                    );
                    program.uniform1i(
                        &self.gl,
                        ShaderUniform::GradientRepeatMode,
                        gradient.repeat_mode,
                    );
                    program.uniform1f(
                        &self.gl,
                        ShaderUniform::GradientFocalPoint,
                        gradient.focal_point,
                    );
                    program.uniform1i(
                        &self.gl,
                        ShaderUniform::GradientInterpolation,
                        (gradient.interpolation == swf::GradientInterpolation::LinearRgb) as i32,
                    );

                    // Upload the baked ramp into the shared 256x1 texture and
                    // bind it (unit 0). Sampled with linear filtering + clamp;
                    // repeat/reflect is folded into `t` in the shader, so the
                    // wrap mode is always clamp (WebGL1-safe, no NPOT concern).
                    unsafe {
                        self.gl.active_texture(glow::TEXTURE0);
                        self.gl
                            .bind_texture(glow::TEXTURE_2D, Some(self.gradient_texture));
                        self.gl.tex_image_2d(
                            glow::TEXTURE_2D,
                            0,
                            glow::RGBA as i32,
                            GRADIENT_SIZE as i32,
                            1,
                            0,
                            glow::RGBA,
                            glow::UNSIGNED_BYTE,
                            glow::PixelUnpackData::Slice(Some(&gradient.ramp[..])),
                        );
                        self.gl.tex_parameter_i32(
                            glow::TEXTURE_2D,
                            glow::TEXTURE_MAG_FILTER,
                            glow::LINEAR as i32,
                        );
                        self.gl.tex_parameter_i32(
                            glow::TEXTURE_2D,
                            glow::TEXTURE_MIN_FILTER,
                            glow::LINEAR as i32,
                        );
                        self.gl.tex_parameter_i32(
                            glow::TEXTURE_2D,
                            glow::TEXTURE_WRAP_S,
                            glow::CLAMP_TO_EDGE as i32,
                        );
                        self.gl.tex_parameter_i32(
                            glow::TEXTURE_2D,
                            glow::TEXTURE_WRAP_T,
                            glow::CLAMP_TO_EDGE as i32,
                        );
                        program.uniform1i(&self.gl, ShaderUniform::BitmapTexture, 0);
                    }
                }
                DrawType::Bitmap(bitmap) => {
                    let texture = match &bitmap.handle {
                        Some(handle) => as_registry_data(handle).texture,
                        None => {
                            log::warn!("Tried to render a handleless bitmap");
                            continue;
                        }
                    };

                    program.uniform_matrix3fv(
                        &self.gl,
                        ShaderUniform::TextureMatrix,
                        &bitmap.matrix,
                    );

                    unsafe {
                        // Bind texture.
                        self.gl.active_texture(glow::TEXTURE0);
                        self.gl.bind_texture(glow::TEXTURE_2D, Some(texture));
                        program.uniform1i(&self.gl, ShaderUniform::BitmapTexture, 0);

                        // Set texture parameters.
                        let filter = if bitmap.is_smoothed {
                            glow::LINEAR as i32
                        } else {
                            glow::NEAREST as i32
                        };
                        self.gl.tex_parameter_i32(
                            glow::TEXTURE_2D,
                            glow::TEXTURE_MAG_FILTER,
                            filter,
                        );
                        self.gl.tex_parameter_i32(
                            glow::TEXTURE_2D,
                            glow::TEXTURE_MIN_FILTER,
                            filter,
                        );
                        // WebGL1 can't change the wrap parameter of non-power-of-2 textures.
                        let wrap = if self.caps.supports_npot_repeat && bitmap.is_repeating {
                            glow::REPEAT as i32
                        } else {
                            glow::CLAMP_TO_EDGE as i32
                        };
                        self.gl
                            .tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, wrap);
                        self.gl
                            .tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, wrap);
                    }
                }
                DrawType::PerspectiveBitmap(bitmap) => {
                    let texture = match &bitmap.handle {
                        Some(handle) => as_registry_data(handle).texture,
                        None => {
                            log::warn!("Tried to render a handleless perspective bitmap");
                            continue;
                        }
                    };
                    unsafe {
                        self.gl.active_texture(glow::TEXTURE0);
                        self.gl.bind_texture(glow::TEXTURE_2D, Some(texture));
                        program.uniform1i(&self.gl, ShaderUniform::BitmapTexture, 0);
                        let filter = if bitmap.is_smoothed {
                            glow::LINEAR as i32
                        } else {
                            glow::NEAREST as i32
                        };
                        self.gl
                            .tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, filter);
                        self.gl
                            .tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, filter);
                        // NPOT + REPEAT is invalid on WebGL1/GLES2 (falls back to clamp).
                        let wrap = if self.caps.supports_npot_repeat && bitmap.is_repeating {
                            glow::REPEAT as i32
                        } else {
                            glow::CLAMP_TO_EDGE as i32
                        };
                        self.gl
                            .tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, wrap);
                        self.gl
                            .tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, wrap);
                    }
                }
            }

            // Draw the triangles. Perspective meshes are non-indexed (`num_indices`
            // holds the vertex count); all other draws are indexed.
            unsafe {
                if let DrawType::PerspectiveBitmap(_) = &draw.draw_type {
                    self.gl.draw_arrays(glow::TRIANGLES, 0, num_indices);
                } else {
                    self.gl
                        .draw_elements(glow::TRIANGLES, num_indices, glow::UNSIGNED_INT, 0);
                }
            }
        }
    }

    fn render_stage3d(&mut self, bitmap: BitmapHandle, transform: Transform) {
        // Present the Stage3D back buffer as an OPAQUE layer, matching Flash and
        // wgpu's `bitmap_opaque` present: replace the stage's RGB with the back
        // buffer's straight colour (raw passthrough — no alpha-over blend and no
        // premultiply round-trip) and leave the stage's alpha untouched. The back
        // buffer can be semi-transparent (e.g. vertex colours with alpha < 1); it
        // is the bottom-most layer and must not blend with anything behind it, so
        // a normal alpha-over composite would wash it out against the stage.
        let (texture, width, height) = {
            let entry = as_registry_data(&bitmap);
            (entry.texture, entry.width as f32, entry.height as f32)
        };
        let matrix = transform.matrix * Matrix::scale(width, height);

        self.flush_batch();
        self.set_stencil_state();

        let world_matrix = [
            [matrix.a, matrix.b, 0.0, 0.0],
            [matrix.c, matrix.d, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [
                matrix.tx.to_pixels() as f32,
                matrix.ty.to_pixels() as f32,
                0.0,
                1.0,
            ],
        ];

        // Raw passthrough (copy) program: no un-premultiply/re-premultiply, so a
        // straight colour with rgb > a survives intact (the bitmap program would
        // clamp it early and darken it).
        let program = &self.copy_program;
        let gl = self.gl.clone();
        unsafe {
            gl.use_program(Some(program.program));
            gl.disable(glow::BLEND);
            // Write RGB only, leaving the stage opaque (the back-buffer alpha must
            // not make the layer transparent).
            gl.color_mask(true, true, true, false);
        }
        program.uniform_matrix4fv(&self.gl, ShaderUniform::WorldMatrix, &world_matrix);
        program.uniform_matrix4fv(&self.gl, ShaderUniform::ViewMatrix, &self.view_matrix);
        program.uniform_matrix3fv(
            &self.gl,
            ShaderUniform::TextureMatrix,
            &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        );
        unsafe {
            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));
            // 1:1 present: sample the back buffer texel-for-texel (no filtering),
            // otherwise stale sampler state can blur the copy.
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::NEAREST as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::NEAREST as i32);
            program.uniform1i(&self.gl, ShaderUniform::BitmapTexture, 0);
            let quad = &self.bitmap_quad_draws;
            self.bind_vertex_array(Some(quad[0].vao));
            gl.bind_buffer(
                glow::ELEMENT_ARRAY_BUFFER,
                Some(quad[0].index_buffer.buffer),
            );
            gl.draw_elements(
                glow::TRIANGLE_FAN,
                quad[0].num_indices,
                glow::UNSIGNED_INT,
                0,
            );
            gl.color_mask(true, true, true, true);
            gl.enable(glow::BLEND);
        }
        self.active_program = std::ptr::null();
    }

    fn draw_rect(&mut self, color: Color, matrix: Matrix) {
        self.draw_quad::<{ glow::TRIANGLE_FAN }, -1>(color, matrix)
    }

    fn draw_line(&mut self, color: Color, mut matrix: Matrix) {
        matrix.tx += Twips::HALF_PX;
        matrix.ty += Twips::HALF_PX;
        self.draw_quad::<{ glow::LINE_STRIP }, 2>(color, matrix)
    }

    fn draw_line_rect(&mut self, color: Color, mut matrix: Matrix) {
        matrix.tx += Twips::HALF_PX;
        matrix.ty += Twips::HALF_PX;
        self.draw_quad::<{ glow::LINE_LOOP }, -1>(color, matrix)
    }

    fn push_mask(&mut self) {
        self.flush_batch();
        debug_assert!(
            self.mask_state == MaskState::NoMask || self.mask_state == MaskState::DrawMaskedContent
        );
        self.num_masks += 1;
        self.mask_state = MaskState::DrawMaskStencil;
        self.mask_state_dirty = true;
    }

    fn activate_mask(&mut self) {
        self.flush_batch();
        debug_assert!(self.num_masks > 0 && self.mask_state == MaskState::DrawMaskStencil);
        self.mask_state = MaskState::DrawMaskedContent;
        self.mask_state_dirty = true;
    }

    fn deactivate_mask(&mut self) {
        self.flush_batch();
        debug_assert!(self.num_masks > 0 && self.mask_state == MaskState::DrawMaskedContent);
        self.mask_state = MaskState::ClearMaskStencil;
        self.mask_state_dirty = true;
    }

    fn pop_mask(&mut self) {
        self.flush_batch();
        debug_assert!(self.num_masks > 0 && self.mask_state == MaskState::ClearMaskStencil);
        self.num_masks -= 1;
        self.mask_state = if self.num_masks == 0 {
            MaskState::NoMask
        } else {
            MaskState::DrawMaskedContent
        };
        self.mask_state_dirty = true;
    }

    fn blend(&mut self, commands: CommandList, blend: RenderBlendMode) {
        // Modes expressible purely as fixed-function blend state (Normal/Add/
        // Subtract/Screen/Multiply, plus Darken/Lighten via GL_MIN/MAX) compose
        // against whatever target is bound and need no offscreen pass. They go
        // through push/pop, which no longer flush — so a run of same-func draws
        // (e.g. hundreds of Multiply puffs) accumulates into a single batched
        // draw. Darken/Lighten keep the destination alpha and require
        // EXT_blend_minmax on GLES2/WebGL1, hence the `minmax_ok` gate; without
        // it they fall back to the region-composite path below.
        let minmax_ok = self.minmax_ok();
        // The fixed-function fast path is only equivalent to an isolated group
        // when the group is a single leaf draw: then "blend this shape against the
        // target" is the same whether done directly or rendered to a fresh texture
        // and composited. A multi-child group must combine its children among
        // themselves (Normal) first and blend the *result* against the backdrop
        // once — drawing each child with the blend func instead composites them
        // against each other, which is wrong for overlapping content. Single-draw
        // groups keep the batchable path (e.g. hundreds of Multiply puffs).
        //
        // Multiply is the exception when compositing onto a possibly-transparent
        // target — any offscreen pass (BitmapData.draw, cacheAsBitmap, a Layer).
        // Its fixed-function form yields nothing where an operand is transparent,
        // but Flash multiplies the un-premultiplied colors, so there it must go
        // through the complex blend shader (matching wgpu). On the opaque stage
        // the fast path is exact, so keep it there.
        let hw_blend = is_hw_blend(&blend, minmax_ok)
            && !(self.in_offscreen
                && matches!(blend, RenderBlendMode::Builtin(BlendMode::Multiply)));
        if hw_blend && blend_group_is_single_draw(&commands) {
            self.push_blend_mode(blend);
            commands.execute(self);
            self.pop_blend_mode();
            return;
        }

        // Everything else reads the destination or needs an offscreen pass, so
        // flush the pending batch before changing target/state.
        self.flush_batch();

        // A PixelBender-shader blend runs the compiled shader with the backdrop and
        // the isolated group as its two image inputs. Only on-screen targets are
        // handled; a nested pass falls back to a plain draw.
        if let RenderBlendMode::Shader(handle) = &blend {
            // Clone the handle's Arc so the borrow of the compiled shader is tied to
            // this local, leaving `self` free to mutate during the draw.
            let arc = handle.0.clone();
            // Runs on the screen, or inside an MSAA/single-sample offscreen pass
            // (cacheAsBitmap, BitmapData.draw) where `draw_shader_blend` reads the
            // parent region back — same gate as the complex-blend path below. A
            // plain single-sample offscreen with no readable parent falls back to
            // Normal.
            if (!self.in_offscreen || self.offscreen_msaa || self.offscreen_color.is_some())
                && let Some(pb) = <dyn Any>::downcast_ref::<GlPixelBenderShader>(&*arc)
                && let Some((rx, ry, rw, rh)) = self.blend_region(&commands)
            {
                self.draw_shader_blend(commands, pb, rx, ry, rw, rh);
                return;
            }
            self.push_blend_mode(RenderBlendMode::Builtin(BlendMode::Normal));
            commands.execute(self);
            self.pop_blend_mode();
            return;
        }

        // A complex blend reads and composites against the bound framebuffer, so
        // when we're already rendering into a single-sample offscreen with no
        // tracked target it would composite onto the wrong place — fall back to a
        // normal draw there.
        if !self.in_offscreen {
            // A `Layer` containing Alpha/Erase must be rendered offscreen so those
            // modes composite against the layer's transparent content.
            if let RenderBlendMode::Builtin(BlendMode::Layer) = &blend {
                if layer_needs_offscreen(&commands) {
                    if let Some((rx, ry, rw, rh)) = self.blend_region(&commands) {
                        self.draw_layer(commands, rx, ry, rw, rh);
                        return;
                    }
                }
                // Otherwise a Layer composites like Normal.
                self.push_blend_mode(RenderBlendMode::Builtin(BlendMode::Normal));
                commands.execute(self);
                self.pop_blend_mode();
                return;
            }
        }

        // Complex blends compose against whatever target is currently bound. Run
        // them on the screen, or inside an MSAA offscreen pass (e.g. cacheAsBitmap),
        // where `draw_complex_blend` resolves the multisampled parent back. A
        // single-sample offscreen pass has no readable parent here, so it falls
        // through to a plain draw instead of compositing onto the wrong target.
        if !self.in_offscreen || self.offscreen_msaa || self.offscreen_color.is_some() {
            if let Some(mode) = complex_blend_index(&blend) {
                // Alpha/Erase composite against the "nearest layer": an enclosing
                // `Layer` (target_fbo), or the current offscreen surface itself
                // (e.g. a `BitmapData.draw` target, matching wgpu's
                // `LayerRef::Current`). Only the bare stage has no layer to
                // composite into — there they'd erase to black, so skip them.
                let needs_layer = matches!(
                    blend,
                    RenderBlendMode::Builtin(BlendMode::Alpha | BlendMode::Erase)
                );
                if needs_layer && !self.in_offscreen && self.target_fbo.is_none() {
                    return;
                }

                if let Some((rx, ry, rw, rh)) = self.blend_region(&commands) {
                    self.draw_complex_blend(commands, mode, rx, ry, rw, rh);
                } else {
                    // No measurable region: draw normally rather than drop content.
                    self.push_blend_mode(RenderBlendMode::Builtin(BlendMode::Normal));
                    commands.execute(self);
                    self.pop_blend_mode();
                }
                return;
            }
        }

        self.push_blend_mode(blend);
        commands.execute(self);
        self.pop_blend_mode();
    }

    fn render_alpha_mask(&mut self, maskee_commands: CommandList, mask_commands: CommandList) {
        self.flush_batch();
        // The visible result lives within the maskee's bounds (it's what gets
        // masked), so size the region to it.
        if let Some((rx, ry, rw, rh)) = self.blend_region(&maskee_commands) {
            self.draw_alpha_mask(maskee_commands, mask_commands, rx, ry, rw, rh);
        }
    }
}

/// Number of texels in a baked gradient ramp. Must match wgpu's `GRADIENT_SIZE`
/// so both backends quantize gradients identically.
const GRADIENT_SIZE: usize = 256;

#[derive(Clone)]
struct Gradient {
    matrix: [[f32; 3]; 3],
    gradient_type: i32,
    /// Baked `GRADIENT_SIZE`-texel RGBA8 ramp, uploaded to the shared ramp
    /// texture and sampled with hardware linear filtering. Built with the exact
    /// per-texel lerp + `as u8` quantization wgpu uses so the two backends are
    /// bit-identical.
    ramp: [u8; GRADIENT_SIZE * 4],
    repeat_mode: i32,
    focal_point: f32,
    interpolation: swf::GradientInterpolation,
}

impl Gradient {
    fn new(gradient: TessGradient, matrix: [[f32; 3]; 3]) -> Self {
        let ramp = if gradient.records.is_empty() {
            [0u8; GRADIENT_SIZE * 4]
        } else {
            let mut ramp = [0u8; GRADIENT_SIZE * 4];
            // sRGB->linear on a 0-255 channel value, matching wgpu's `convert`.
            let convert = |c: f32| -> f32 {
                match gradient.interpolation {
                    swf::GradientInterpolation::Rgb => c,
                    swf::GradientInterpolation::LinearRgb => srgb_to_linear_scalar(c / 255.0) * 255.0,
                }
            };
            let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;

            let mut last = 0;
            for t in 0..GRADIENT_SIZE {
                if last + 1 < gradient.records.len()
                    && t > gradient.records[last + 1].ratio as usize
                {
                    last += 1;
                }
                let next = (last + 1).min(gradient.records.len() - 1);

                let last_record = &gradient.records[last];
                let next_record = &gradient.records[next];

                let a = if t <= last_record.ratio as usize || last_record.ratio == next_record.ratio
                {
                    0.0
                } else if t > next_record.ratio as usize {
                    1.0
                } else {
                    (t as f32 - last_record.ratio as f32)
                        / (next_record.ratio as f32 - last_record.ratio as f32)
                };

                ramp[t * 4] = lerp(
                    convert(last_record.color.r as f32),
                    convert(next_record.color.r as f32),
                    a,
                ) as u8;
                ramp[t * 4 + 1] = lerp(
                    convert(last_record.color.g as f32),
                    convert(next_record.color.g as f32),
                    a,
                ) as u8;
                ramp[t * 4 + 2] = lerp(
                    convert(last_record.color.b as f32),
                    convert(next_record.color.b as f32),
                    a,
                ) as u8;
                ramp[t * 4 + 3] =
                    lerp(last_record.color.a as f32, next_record.color.a as f32, a) as u8;
            }
            ramp
        };

        Self {
            matrix,
            gradient_type: match gradient.gradient_type {
                GradientType::Linear => 0,
                GradientType::Radial => 1,
                GradientType::Focal => 2,
            },
            ramp,
            repeat_mode: match gradient.repeat_mode {
                swf::GradientSpread::Pad => 0,
                swf::GradientSpread::Repeat => 1,
                swf::GradientSpread::Reflect => 2,
            },
            focal_point: gradient.focal_point.to_f32().clamp(-0.98, 0.98),
            interpolation: gradient.interpolation,
        }
    }
}

#[derive(Clone)]
struct BitmapDraw {
    matrix: [[f32; 3]; 3],
    handle: Option<BitmapHandle>,
    is_repeating: bool,
    is_smoothed: bool,
}

/// Draw state for a perspective textured-triangle mesh — the per-vertex `(u, v, t)`
/// lives in the VBO, so only the texture + sampler state is needed here.
#[derive(Clone)]
struct PerspectiveBitmapDraw {
    handle: Option<BitmapHandle>,
    is_repeating: bool,
    is_smoothed: bool,
}

struct Mesh {
    gl: GlContext,
    draws: Vec<Draw>,
    /// Shape-space bounds (Twips), used to compute tight blend regions.
    bounds: Rectangle<Twips>,
}

impl fmt::Debug for Mesh {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Mesh")
            .field("num_draws", &self.draws.len())
            .finish()
    }
}

impl Drop for Mesh {
    fn drop(&mut self) {
        unsafe {
            for draw in &self.draws {
                self.gl.delete_vertex_array(draw.vao);
            }
        }
    }
}

impl ShapeHandleImpl for Mesh {}

fn as_mesh(handle: &ShapeHandle) -> &Mesh {
    <dyn Any>::downcast_ref(&*handle.0).expect("Shape handle must be a gl Mesh")
}

struct Buffer {
    gl: GlContext,
    buffer: glow::Buffer,
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe { self.gl.delete_buffer(self.buffer) };
    }
}

struct Draw {
    draw_type: DrawType,
    #[expect(dead_code)]
    vertex_buffer: Buffer,
    index_buffer: Buffer,
    vao: glow::VertexArray,
    num_indices: i32,
    num_mask_indices: i32,
    /// CPU-side geometry for `Color` draws, kept so they can be transformed and
    /// appended to the shared draw batch instead of issuing a per-shape draw.
    color_cpu: Option<ColorGeometry>,
}

/// CPU vertex/index data for a solid-color draw, used by the batcher.
struct ColorGeometry {
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
}

enum DrawType {
    Color,
    Gradient(Box<Gradient>),
    Bitmap(BitmapDraw),
    /// Perspective-correct textured triangles: `num_indices` is the vertex count
    /// (drawn non-indexed via `draw_arrays`), the VBO holds `PerspectiveVertex`es.
    PerspectiveBitmap(PerspectiveBitmapDraw),
}

struct MsaaBuffers {
    gl: GlContext,
    color_renderbuffer: glow::Renderbuffer,
    stencil_renderbuffer: glow::Renderbuffer,
    render_framebuffer: glow::Framebuffer,
    color_framebuffer: glow::Framebuffer,
    framebuffer_texture: glow::Texture,
}

impl Drop for MsaaBuffers {
    fn drop(&mut self) {
        unsafe {
            self.gl.delete_renderbuffer(self.color_renderbuffer);
            self.gl.delete_renderbuffer(self.stencil_renderbuffer);
            self.gl.delete_framebuffer(self.render_framebuffer);
            self.gl.delete_framebuffer(self.color_framebuffer);
            self.gl.delete_texture(self.framebuffer_texture);
        }
    }
}

/// A deferred GPU->CPU readback handle. Keeps the source `BitmapHandle` alive so
/// its texture survives until `resolve_sync_handle` reads it back.
struct GlSyncHandle {
    gl: GlContext,
    handle: BitmapHandle,
    copy_area: PixelRegion,
}

impl fmt::Debug for GlSyncHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GlSyncHandle")
            .field("copy_area", &self.copy_area)
            .finish()
    }
}

impl SyncHandle for GlSyncHandle {}

impl GlSyncHandle {
    fn capture(self, fbo: glow::Framebuffer, with_rgba: RgbaBufRead) {
        let texture = as_registry_data(&self.handle).texture;
        let region = self.copy_area;
        let width = region.width();
        let height = region.height();
        if width == 0 || height == 0 {
            with_rgba(&[], 0);
            return;
        }

        // Read back the requested region as tightly-packed premultiplied RGBA.
        // Our textures store Flash-top at texel row 0 and FBO row 0 == texel row
        // 0, so `glReadPixels` returns top-down rows directly (no flip), and the
        // premultiplied values are exactly what BitmapData stores raw.
        let mut pixels = vec![0u8; (width as usize) * (height as usize) * 4];
        unsafe {
            self.gl.bind_framebuffer(glow::FRAMEBUFFER, Some(fbo));
            self.gl.framebuffer_texture_2d(
                glow::FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_2D,
                Some(texture),
                0,
            );
            self.gl.pixel_store_i32(glow::PACK_ALIGNMENT, 1);
            self.gl.read_pixels(
                region.x_min as i32,
                region.y_min as i32,
                width as i32,
                height as i32,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelPackData::Slice(Some(&mut pixels)),
            );
            self.gl.bind_framebuffer(glow::FRAMEBUFFER, None);
        }

        // The callback's second argument is the row stride in bytes.
        with_rgba(&pixels, width * 4);
    }
}

/// Converts a single sRGB channel value (0-1) to linear. Matches wgpu's
/// `srgb_to_linear` so baked gradient ramps quantize identically.
fn srgb_to_linear_scalar(color: f32) -> f32 {
    if color <= 0.04045 {
        color / 12.92
    } else {
        f32::powf((color + 0.055) / 1.055, 2.4)
    }
}
