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

mod context;
mod error;
mod filters;
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
const TEXTURE_VERTEX_GLSL: &str = include_str!("../shaders/texture.vert");
const GRADIENT_FRAGMENT_GLSL: &str = include_str!("../shaders/gradient.frag");
const BITMAP_FRAGMENT_GLSL: &str = include_str!("../shaders/bitmap.frag");

const NUM_VERTEX_ATTRIBUTES: u32 = 2;
const MAX_GRADIENT_COLORS: usize = 15;

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

    color_program: ShaderProgram,
    bitmap_program: ShaderProgram,
    gradient_program: ShaderProgram,
    batch_color_program: ShaderProgram,
    /// Raw texture passthrough used to seed offscreen MSAA buffers.
    copy_program: ShaderProgram,

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
    batch_bitmap_vertices: Vec<BitmapVertex>,
    batch_bitmap_indices: Vec<u32>,
    batch_bitmap_key: Option<BitmapBatchKey>,
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

struct RegistryData {
    gl: GlContext,
    width: u32,
    height: u32,
    texture: glow::Texture,
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
        let created = context::create_for_webgl(canvas, is_transparent, quality)?;
        Self::finish_construction(created, is_transparent)
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
        let caps = created.caps;
        let msaa_sample_count = created.msaa_sample_count;
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

        let mut renderer = Self {
            gl: gl.clone(),
            caps,
            #[cfg(target_family = "wasm")]
            web_context: created.web_context,

            msaa_buffers: None,
            msaa_sample_count,

            color_program,
            gradient_program,
            bitmap_program,
            batch_color_program,
            copy_program,
            batch_vao,
            batch_vbo,
            batch_ibo,
            batch_vertices: Vec::new(),
            batch_indices: Vec::new(),
            batch_bitmap_program,
            batch_bitmap_vao,
            batch_bitmap_vbo,
            batch_bitmap_ibo,
            batch_bitmap_vertices: Vec::new(),
            batch_bitmap_indices: Vec::new(),
            batch_bitmap_key: None,
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

    fn build_msaa_buffers(&mut self) -> Result<(), Error> {
        if !self.caps.is_gles3_or_webgl2 || self.msaa_sample_count <= 1 {
            unsafe {
                self.gl.bind_framebuffer(glow::FRAMEBUFFER, None);
                self.gl.bind_renderbuffer(glow::RENDERBUFFER, None);
            }
            return Ok(());
        }

        // Delete previous buffers, if they exist (Drop deletes the GL objects).
        self.msaa_buffers = None;

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
            gl.renderbuffer_storage_multisample(
                glow::RENDERBUFFER,
                self.msaa_sample_count as i32,
                glow::RGBA8,
                self.renderbuffer_width,
                self.renderbuffer_height,
            );
            check_error(&gl, "renderbuffer_storage_multisample (color)")?;

            let stencil_renderbuffer = gl
                .create_renderbuffer()
                .map_err(Error::UnableToCreateRenderBuffer)?;
            gl.bind_renderbuffer(glow::RENDERBUFFER, Some(stencil_renderbuffer));
            gl.renderbuffer_storage_multisample(
                glow::RENDERBUFFER,
                self.msaa_sample_count as i32,
                glow::STENCIL_INDEX8,
                self.renderbuffer_width,
                self.renderbuffer_height,
            );
            check_error(&gl, "renderbuffer_storage_multisample (stencil)")?;

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
            check_error(&gl, "tex_image_2d (msaa resolve texture)")?;
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

        self.msaa_buffers = Some(buffers);
        Ok(())
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
        let minmax_ok = !self.caps.is_embedded || self.caps.is_gles3_or_webgl2;
        let key = blend_key(&mode, minmax_ok);
        self.active_hw_blend = key;
        self.apply_hw_blend(key);
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
            program.uniform_matrix3fv(
                &self.gl,
                ShaderUniform::TextureMatrix,
                &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            );

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
        self.active_program = std::ptr::null();
        self.mult_color = None;
        self.add_color = None;
    }

    /// Appends a bitmap quad to the bitmap batch, flushing first if the colour
    /// batch is pending (draw order) or the batch key changes.
    fn append_bitmap_draw(&mut self, key: BitmapBatchKey, matrix: Matrix) {
        self.flush_color_batch();
        if !self.batch_bitmap_indices.is_empty()
            && (self.batch_bitmap_key != Some(key) || self.batch_blend != self.active_hw_blend)
        {
            self.flush_bitmap_batch();
        }
        self.batch_blend = self.active_hw_blend;
        self.batch_bitmap_key = Some(key);

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
        self.batch_vertices.reserve(geom.vertices.len());

        // The common case is no color transform (mult identity, add zero), where
        // baking reduces to premultiplying the source color — skip the per-channel
        // mult/add/clamp entirely.
        let identity_color = mult == [1.0, 1.0, 1.0, 1.0] && add == [0.0, 0.0, 0.0, 0.0];
        if identity_color {
            for v in &geom.vertices {
                let x = v.position[0];
                let y = v.position[1];
                let [cr, cg, cb, ca] = v.color.to_le_bytes();
                let af = ca as f32 / 255.0;
                let color = u32::from_le_bytes([
                    (cr as f32 * af).round() as u8,
                    (cg as f32 * af).round() as u8,
                    (cb as f32 * af).round() as u8,
                    ca,
                ]);
                self.batch_vertices.push(Vertex {
                    position: [a * x + c * y + tx, b * x + d * y + ty],
                    color,
                });
            }
        } else {
            for v in &geom.vertices {
                let x = v.position[0];
                let y = v.position[1];
                let [cr, cg, cb, ca] = v.color.to_le_bytes();
                // frag_color = clamp(color * mult + add), then premultiply (matches
                // the per-draw color shader, done here on the CPU instead).
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
                self.batch_vertices.push(Vertex {
                    position: [a * x + c * y + tx, b * x + d * y + ty],
                    color,
                });
            }
        }
        for &i in &geom.indices[..num_indices] {
            self.batch_indices.push(base + i);
        }
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
        let saved_active_blend = self.active_hw_blend;
        let saved_batch_blend = self.batch_blend;
        self.in_offscreen = true;
        self.offscreen_msaa = use_msaa;
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
        self.view_matrix = region_view_matrix(stage_rx as f32, stage_ry as f32, rw as f32, rh as f32);
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
        let ry_fb = saved_h - (ry + rh);
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

/// Whether a `Layer` blend must be rendered offscreen: only if it contains
/// Alpha/Erase children, which require a transparent layer to composite into. A
/// nested `Layer` isolates its own, so we don't descend into one. Other blends
/// inside a layer keep their existing on-target behavior.
fn layer_needs_offscreen(commands: &CommandList) -> bool {
    commands.commands.iter().any(|c| match c {
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
            self.gl.delete_program(self.batch_color_program.program);
            self.gl.delete_program(self.batch_bitmap_program.program);
            self.gl.delete_program(self.copy_program.program);
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
        // Build view matrix based on canvas size.
        self.view_matrix = [
            [1.0 / (dimensions.width as f32 / 2.0), 0.0, 0.0, 0.0],
            [0.0, -1.0 / (dimensions.height as f32 / 2.0), 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0, 1.0],
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
        let _ = self.build_msaa_buffers();
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
        _profile: Context3DProfile,
    ) -> Result<Box<dyn Context3D>, BitmapError> {
        Err(BitmapError::Unimplemented("createContext3D".into()))
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
            // Rebuild MSAA buffers at the new sample count (no-op on WebGL1/GLES2).
            let _ = self.build_msaa_buffers();
        }
    }

    fn compile_pixelbender_shader(
        &mut self,
        _shader: ruffle_render::pixel_bender::PixelBenderShader,
    ) -> Result<ruffle_render::pixel_bender::PixelBenderShaderHandle, BitmapError> {
        Err(BitmapError::Unimplemented(
            "compile_pixelbender_shader".into(),
        ))
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
        _handle: ruffle_render::pixel_bender::PixelBenderShaderHandle,
        _arguments: &[ruffle_render::pixel_bender_support::PixelBenderShaderArgument],
        _target: &PixelBenderTarget,
    ) -> Result<PixelBenderOutput, BitmapError> {
        Err(BitmapError::Unimplemented("run_pixelbender_shader".into()))
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
        self.append_bitmap_draw(key, matrix);
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
                    program.uniform1fv(&self.gl, ShaderUniform::GradientRatios, &gradient.ratios);
                    program.uniform4fv(
                        &self.gl,
                        ShaderUniform::GradientColors,
                        bytemuck::cast_slice(&gradient.colors),
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
            }

            // Draw the triangles.
            unsafe {
                self.gl
                    .draw_elements(glow::TRIANGLES, num_indices, glow::UNSIGNED_INT, 0);
            }
        }
    }

    fn render_stage3d(&mut self, _bitmap: BitmapHandle, _transform: Transform) {
        self.flush_batch();
        panic!("Stage3D should not have been created on GL backend")
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
        let minmax_ok = !self.caps.is_embedded || self.caps.is_gles3_or_webgl2;
        if is_hw_blend(&blend, minmax_ok) {
            self.push_blend_mode(blend);
            commands.execute(self);
            self.pop_blend_mode();
            return;
        }

        // Everything else reads the destination or needs an offscreen pass, so
        // flush the pending batch before changing target/state.
        self.flush_batch();

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
        if !self.in_offscreen || self.offscreen_msaa {
            if let Some(mode) = complex_blend_index(&blend) {
                // Alpha/Erase only make sense compositing into a Layer's
                // transparent content; on the bare stage they'd erase to black, so
                // skip them rather than draw the wrong thing.
                let needs_layer = matches!(
                    blend,
                    RenderBlendMode::Builtin(BlendMode::Alpha | BlendMode::Erase)
                );
                if needs_layer && self.target_fbo.is_none() {
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

    fn render_alpha_mask(&mut self, maskee_commands: CommandList, _mask_commands: CommandList) {
        self.flush_batch();
        // TODO Add support for alpha masks
        maskee_commands.execute(self);
    }
}

#[derive(Clone)]
struct Gradient {
    matrix: [[f32; 3]; 3],
    gradient_type: i32,
    ratios: [f32; MAX_GRADIENT_COLORS],
    colors: [[f32; 4]; MAX_GRADIENT_COLORS],
    repeat_mode: i32,
    focal_point: f32,
    interpolation: swf::GradientInterpolation,
}

impl Gradient {
    fn new(gradient: TessGradient, matrix: [[f32; 3]; 3]) -> Self {
        // TODO: Support more than MAX_GRADIENT_COLORS.
        let num_colors = gradient.records.len().min(MAX_GRADIENT_COLORS);
        let mut ratios = [0.0; MAX_GRADIENT_COLORS];
        let mut colors = [[0.0; 4]; MAX_GRADIENT_COLORS];
        for i in 0..num_colors {
            let record = &gradient.records[i];
            let mut color = [
                f32::from(record.color.r) / 255.0,
                f32::from(record.color.g) / 255.0,
                f32::from(record.color.b) / 255.0,
                f32::from(record.color.a) / 255.0,
            ];
            // Convert to linear color space if this is a linear-interpolated gradient.
            match gradient.interpolation {
                swf::GradientInterpolation::Rgb => {}
                swf::GradientInterpolation::LinearRgb => srgb_to_linear(&mut color),
            }

            colors[i] = color;
            ratios[i] = f32::from(record.ratio) / 255.0;
        }

        for i in num_colors..MAX_GRADIENT_COLORS {
            ratios[i] = ratios[i - 1];
            colors[i] = colors[i - 1];
        }

        Self {
            matrix,
            gradient_type: match gradient.gradient_type {
                GradientType::Linear => 0,
                GradientType::Radial => 1,
                GradientType::Focal => 2,
            },
            ratios,
            colors,
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
    #[expect(dead_code)]
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

/// Converts an RGBA color from sRGB space to linear color space.
fn srgb_to_linear(color: &mut [f32; 4]) {
    for n in &mut color[..3] {
        *n = if *n <= 0.04045 {
            *n / 12.92
        } else {
            f32::powf((*n + 0.055) / 1.055, 2.4)
        };
    }
}
