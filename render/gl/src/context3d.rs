//! Native-OpenGL implementation of Flash's Stage3D (`Context3D`).
//!
//! This is the GL counterpart to `ruffle_render_wgpu`'s `context3d` module. It
//! owns a double-buffered back/front render target (composited onto the stage as
//! an ordinary bitmap), GL-backed vertex/index buffers and textures, and the
//! fixed-function render state set through `process_command`.
//!
//! AGAL shader translation and the actual `DrawTriangles` pipeline are built on
//! top of this foundation (see `agal.rs`); until a program is compiled, draws are
//! skipped, but the buffer/texture/clear/present lifecycle is fully functional.

use crate::RegistryData;
use crate::context::GlContext;
use glow::HasContext as _;
use ruffle_render::backend::{
    BufferUsage, Context3D, Context3DBlendFactor, Context3DCommand, Context3DCompareMode,
    Context3DProfile, Context3DStencilAction, Context3DTextureFilter, Context3DTextureFormat,
    Context3DTriangleFace, Context3DVertexBufferFormat, Context3DWrapMode, IndexBuffer,
    ProgramType, ShaderModule, Texture, VertexBuffer,
};
use ruffle_render::bitmap::BitmapHandle;
use ruffle_render::error::Error;
use std::any::Any;
use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::Arc;

const MAX_VERTEX_ATTRIBUTES: usize = 8;
const MAX_SAMPLERS: usize = 8;

/// A GL buffer object wrapper shared by the index/vertex buffer handles.
struct GlBuffer {
    gl: GlContext,
    buffer: glow::Buffer,
}

impl Drop for GlBuffer {
    fn drop(&mut self) {
        unsafe { self.gl.delete_buffer(self.buffer) };
    }
}

pub struct GlIndexBuffer {
    inner: Option<GlBuffer>,
    /// Number of `u32` indices currently allocated (read by the draw pipeline).
    #[allow(dead_code)]
    count: Cell<u32>,
}

impl IndexBuffer for GlIndexBuffer {}

pub struct GlVertexBuffer {
    inner: Option<GlBuffer>,
    #[allow(dead_code)]
    data32_per_vertex: u8,
}

impl VertexBuffer for GlVertexBuffer {}

#[derive(Debug)]
pub struct GlTexture3D {
    gl: GlContext,
    texture: glow::Texture,
    width: u32,
    height: u32,
    #[allow(dead_code)]
    format: Context3DTextureFormat,
    cube: bool,
    /// Lazily-created framebuffer (+ depth) for render-to-texture.
    render_fbo: Cell<Option<glow::Framebuffer>>,
    render_depth: Cell<Option<glow::Renderbuffer>>,
}

impl GlTexture3D {
    /// Returns (creating on first use) the FBO that renders into this texture,
    /// attaching a depth/stencil buffer when requested.
    fn render_fbo(&self, enable_depth: bool) -> glow::Framebuffer {
        let gl = &self.gl;
        let fbo = self.render_fbo.get().unwrap_or_else(|| {
            let fbo = unsafe { gl.create_framebuffer() }.expect("Context3D RTT FBO");
            unsafe {
                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(fbo));
                let target = if self.cube {
                    glow::TEXTURE_CUBE_MAP_POSITIVE_X
                } else {
                    glow::TEXTURE_2D
                };
                gl.framebuffer_texture_2d(
                    glow::FRAMEBUFFER,
                    glow::COLOR_ATTACHMENT0,
                    target,
                    Some(self.texture),
                    0,
                );
            }
            self.render_fbo.set(Some(fbo));
            fbo
        });
        if enable_depth && self.render_depth.get().is_none() {
            unsafe {
                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(fbo));
                let rb = gl.create_renderbuffer().expect("Context3D RTT depth");
                gl.bind_renderbuffer(glow::RENDERBUFFER, Some(rb));
                gl.renderbuffer_storage(
                    glow::RENDERBUFFER,
                    glow::DEPTH24_STENCIL8,
                    self.width as i32,
                    self.height as i32,
                );
                gl.framebuffer_renderbuffer(
                    glow::FRAMEBUFFER,
                    glow::DEPTH_STENCIL_ATTACHMENT,
                    glow::RENDERBUFFER,
                    Some(rb),
                );
                self.render_depth.set(Some(rb));
            }
        }
        fbo
    }
}

impl Drop for GlTexture3D {
    fn drop(&mut self) {
        unsafe {
            if let Some(fbo) = self.render_fbo.get() {
                self.gl.delete_framebuffer(fbo);
            }
            if let Some(rb) = self.render_depth.get() {
                self.gl.delete_renderbuffer(rb);
            }
            self.gl.delete_texture(self.texture);
        }
    }
}

impl Texture for GlTexture3D {
    fn width(&self) -> u32 {
        self.width
    }
    fn height(&self) -> u32 {
        self.height
    }
}

/// A linked GL program translated from an AGAL vertex/fragment pair, with the
/// uniform locations and attribute/sampler metadata the draw pipeline needs.
pub struct CompiledProgram {
    gl: GlContext,
    program: glow::Program,
    vc_loc: Option<glow::UniformLocation>,
    fc_loc: Option<glow::UniformLocation>,
    num_vc: usize,
    num_fc: usize,
    /// Vertex-attribute register numbers (each bound to attribute location = reg).
    attributes: Vec<u16>,
    /// (sampler reg, uniform location, is-cube).
    samplers: Vec<(u16, Option<glow::UniformLocation>, bool)>,
}

impl Drop for CompiledProgram {
    fn drop(&mut self) {
        unsafe { self.gl.delete_program(self.program) };
    }
}

pub struct GlShaderModule {
    program: Option<CompiledProgram>,
}

impl ShaderModule for GlShaderModule {}

/// One side of the double-buffered render target. The single-sample resolve
/// texture is exposed to the stage as a bitmap; when `samples > 1` the content is
/// drawn into a multisampled FBO and resolved into that texture on `present`.
struct RenderBuffer {
    gl: GlContext,
    handle: BitmapHandle,
    /// The resolve texture id (owned by `handle`; kept for render-to-texture).
    #[allow(dead_code)]
    texture: glow::Texture,
    /// FBO with the resolve texture attached (also the draw target when 1x).
    resolve_fbo: glow::Framebuffer,
    /// `(msaa fbo, color renderbuffer, depth renderbuffer)` when multisampled.
    msaa: Option<(glow::Framebuffer, glow::Renderbuffer, Option<glow::Renderbuffer>)>,
    /// Single-sample depth/stencil (only when not multisampled).
    depth_stencil: Option<glow::Renderbuffer>,
    width: u32,
    height: u32,
}

impl RenderBuffer {
    fn new(gl: &GlContext, width: u32, height: u32, depth_and_stencil: bool, samples: u32) -> Self {
        let (w, h) = (width as i32, height as i32);
        let (texture, resolve_fbo, msaa, depth_stencil) = unsafe {
            let texture = gl.create_texture().expect("Context3D back-buffer texture");
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));
            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::RGBA as i32,
                w,
                h,
                0,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(None),
            );
            let clamp = glow::CLAMP_TO_EDGE as i32;
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, clamp);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, clamp);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::LINEAR as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);

            let resolve_fbo = gl.create_framebuffer().expect("Context3D resolve FBO");
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(resolve_fbo));
            gl.framebuffer_texture_2d(
                glow::FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_2D,
                Some(texture),
                0,
            );

            let (msaa, depth_stencil) = if samples > 1 {
                let fbo = gl.create_framebuffer().expect("Context3D MSAA FBO");
                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(fbo));
                let color = gl.create_renderbuffer().expect("Context3D MSAA color");
                gl.bind_renderbuffer(glow::RENDERBUFFER, Some(color));
                gl.renderbuffer_storage_multisample(
                    glow::RENDERBUFFER,
                    samples as i32,
                    glow::RGBA8,
                    w,
                    h,
                );
                gl.framebuffer_renderbuffer(
                    glow::FRAMEBUFFER,
                    glow::COLOR_ATTACHMENT0,
                    glow::RENDERBUFFER,
                    Some(color),
                );
                let depth = if depth_and_stencil {
                    let d = gl.create_renderbuffer().expect("Context3D MSAA depth");
                    gl.bind_renderbuffer(glow::RENDERBUFFER, Some(d));
                    gl.renderbuffer_storage_multisample(
                        glow::RENDERBUFFER,
                        samples as i32,
                        glow::DEPTH24_STENCIL8,
                        w,
                        h,
                    );
                    gl.framebuffer_renderbuffer(
                        glow::FRAMEBUFFER,
                        glow::DEPTH_STENCIL_ATTACHMENT,
                        glow::RENDERBUFFER,
                        Some(d),
                    );
                    Some(d)
                } else {
                    None
                };
                (Some((fbo, color, depth)), None)
            } else if depth_and_stencil {
                let rb = gl.create_renderbuffer().expect("Context3D depth/stencil");
                gl.bind_renderbuffer(glow::RENDERBUFFER, Some(rb));
                gl.renderbuffer_storage(glow::RENDERBUFFER, glow::DEPTH24_STENCIL8, w, h);
                gl.framebuffer_renderbuffer(
                    glow::FRAMEBUFFER,
                    glow::DEPTH_STENCIL_ATTACHMENT,
                    glow::RENDERBUFFER,
                    Some(rb),
                );
                (None, Some(rb))
            } else {
                (None, None)
            };
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            (texture, resolve_fbo, msaa, depth_stencil)
        };

        let handle = BitmapHandle(Arc::new(RegistryData {
            gl: gl.clone(),
            width,
            height,
            texture,
        }));

        Self {
            gl: gl.clone(),
            handle,
            texture,
            resolve_fbo,
            msaa,
            depth_stencil,
            width,
            height,
        }
    }

    /// The framebuffer to draw into.
    fn draw_fbo(&self) -> glow::Framebuffer {
        match &self.msaa {
            Some((fbo, _, _)) => *fbo,
            None => self.resolve_fbo,
        }
    }

    fn has_depth(&self) -> bool {
        self.depth_stencil.is_some() || matches!(&self.msaa, Some((_, _, Some(_))))
    }

    /// Resolves the multisampled buffer into the exposed texture (no-op at 1x).
    fn resolve(&self) {
        if self.msaa.is_none() {
            return;
        }
        let (w, h) = (self.width as i32, self.height as i32);
        unsafe {
            self.gl
                .bind_framebuffer(glow::READ_FRAMEBUFFER, Some(self.draw_fbo()));
            self.gl
                .bind_framebuffer(glow::DRAW_FRAMEBUFFER, Some(self.resolve_fbo));
            self.gl.blit_framebuffer(
                0,
                0,
                w,
                h,
                0,
                0,
                w,
                h,
                glow::COLOR_BUFFER_BIT,
                glow::NEAREST,
            );
            self.gl.bind_framebuffer(glow::FRAMEBUFFER, None);
        }
    }
}

impl Drop for RenderBuffer {
    fn drop(&mut self) {
        unsafe {
            self.gl.delete_framebuffer(self.resolve_fbo);
            if let Some((fbo, color, depth)) = self.msaa {
                self.gl.delete_framebuffer(fbo);
                self.gl.delete_renderbuffer(color);
                if let Some(d) = depth {
                    self.gl.delete_renderbuffer(d);
                }
            }
            if let Some(rb) = self.depth_stencil {
                self.gl.delete_renderbuffer(rb);
            }
        }
        // `texture` is owned by `handle`'s `RegistryData` and freed with it.
    }
}

/// Bound vertex attribute (set via `SetVertexBufferAt`; read by the draw pipeline).
#[derive(Clone)]
#[allow(dead_code)]
struct VertexAttribute {
    buffer: Rc<dyn VertexBuffer>,
    offset: u32,
    format: Context3DVertexBufferFormat,
}

pub struct GlContext3D {
    gl: GlContext,
    profile: Context3DProfile,
    is_embedded: bool,
    /// Multisampled renderbuffers need GLES3/WebGL2 (or desktop GL); oldest-GL
    /// targets fall back to a 1x back buffer.
    supports_msaa: bool,
    /// A dedicated VAO — GL 3.3 core requires one bound for vertex-attribute draws.
    vao: Option<glow::VertexArray>,

    back: Option<RenderBuffer>,
    front: Option<RenderBuffer>,
    /// `Some(texture)` keeps the render-to-texture target alive while active.
    render_to_texture: Option<Rc<dyn Texture>>,
    /// Active render-to-texture target `(fbo, width, height, has_depth)`; `None`
    /// draws to the back buffer.
    render_target: Option<(glow::Framebuffer, u32, u32, bool)>,
    configured: bool,
    seen_clear: bool,
    clear_color: Option<(f32, f32, f32, f32)>,

    // ---- Pipeline state (retained; consumed by the draw pipeline). ----
    module: Option<Rc<dyn ShaderModule>>,
    vertex_attributes: [Option<VertexAttribute>; MAX_VERTEX_ATTRIBUTES],
    textures: [Option<Rc<dyn Texture>>; MAX_SAMPLERS],
    vertex_constants: Vec<[f32; 4]>,
    fragment_constants: Vec<[f32; 4]>,
    cull: Context3DTriangleFace,
    depth_mask: bool,
    depth_compare: Context3DCompareMode,
    blend_src: Context3DBlendFactor,
    blend_dst: Context3DBlendFactor,
    color_mask: [bool; 4],
    // Stencil state from `setStencilActions` / `setStencilReferenceValue`. The
    // default (Always compare, Keep actions) is a no-op pass-through so content
    // without explicit stencil use draws unaffected.
    stencil_compare: Context3DCompareMode,
    stencil_face: Context3DTriangleFace,
    stencil_both_pass: Context3DStencilAction,
    stencil_depth_fail: Context3DStencilAction,
    stencil_fail: Context3DStencilAction,
    stencil_ref: u32,
    stencil_read_mask: u32,
    stencil_write_mask: u32,
    /// Scissor rectangle in back-buffer pixels (Flash top-left origin), if set.
    scissor: Option<(i32, i32, i32, i32)>,
    /// Per-sampler `(wrap, filter)` state from `setSamplerStateAt`.
    sampler_states: [Option<(Context3DWrapMode, Context3DTextureFilter)>; MAX_SAMPLERS],

    disposed_index: Rc<GlIndexBuffer>,
    disposed_vertex: Rc<GlVertexBuffer>,
}

impl GlContext3D {
    pub fn new(
        gl: GlContext,
        profile: Context3DProfile,
        is_embedded: bool,
        supports_msaa: bool,
    ) -> Self {
        let disposed_index = Rc::new(GlIndexBuffer {
            inner: None,
            count: Cell::new(0),
        });
        let disposed_vertex = Rc::new(GlVertexBuffer {
            inner: None,
            data32_per_vertex: 0,
        });
        let vao = unsafe { gl.create_vertex_array() }.ok();
        Self {
            gl,
            profile,
            is_embedded,
            supports_msaa,
            vao,
            back: None,
            front: None,
            render_to_texture: None,
            render_target: None,
            configured: false,
            seen_clear: false,
            clear_color: None,
            module: None,
            vertex_attributes: Default::default(),
            textures: Default::default(),
            vertex_constants: Vec::new(),
            fragment_constants: Vec::new(),
            cull: Context3DTriangleFace::None,
            depth_mask: true,
            depth_compare: Context3DCompareMode::Less,
            blend_src: Context3DBlendFactor::One,
            blend_dst: Context3DBlendFactor::Zero,
            color_mask: [true; 4],
            stencil_compare: Context3DCompareMode::Always,
            stencil_face: Context3DTriangleFace::FrontAndBack,
            stencil_both_pass: Context3DStencilAction::Keep,
            stencil_depth_fail: Context3DStencilAction::Keep,
            stencil_fail: Context3DStencilAction::Keep,
            stencil_ref: 0,
            stencil_read_mask: 0xff,
            stencil_write_mask: 0xff,
            scissor: None,
            sampler_states: Default::default(),
            disposed_index,
            disposed_vertex,
        }
    }

    fn draw_triangles(&self, index_buffer: &dyn IndexBuffer, first_index: usize, num_triangles: isize) {
        let Some((fbo, target_w, target_h, target_depth)) = self.current_target() else {
            return;
        };
        let Some(module) = self.module.as_ref() else { return };
        let Some(module) = (&**module as &dyn Any).downcast_ref::<GlShaderModule>() else {
            return;
        };
        let Some(prog) = module.program.as_ref() else {
            return;
        };
        let Some(ib) = (index_buffer as &dyn Any).downcast_ref::<GlIndexBuffer>() else {
            return;
        };
        let Some(ib_inner) = ib.inner.as_ref() else {
            return;
        };
        let count = if num_triangles < 0 {
            ib.count.get() as i32
        } else {
            num_triangles as i32 * 3
        };

        let gl = self.gl.clone();
        let mut enabled_attrs: Vec<u32> = Vec::new();
        unsafe {
            // GL 3.3 core needs a VAO bound; glow maps this onto WebGL1/GLES2's
            // `OES_vertex_array_object` (and gracefully no-ops when unavailable).
            gl.bind_vertex_array(self.vao);
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(fbo));
            gl.viewport(0, 0, target_w, target_h);
            apply_scissor(&gl, self.scissor, target_h);
            gl.use_program(Some(prog.program));

            upload_constants(&gl, prog.vc_loc.as_ref(), prog.num_vc, &self.vertex_constants);
            upload_constants(&gl, prog.fc_loc.as_ref(), prog.num_fc, &self.fragment_constants);

            // Vertex attributes: each `SetVertexBufferAt` slot feeds attribute
            // location = its index (matching `va{index}` in the translated shader).
            for &reg in &prog.attributes {
                let slot = reg as usize;
                let Some(Some(attr)) = self.vertex_attributes.get(slot) else {
                    continue;
                };
                let Some(vb) = (&*attr.buffer as &dyn Any)
                    .downcast_ref::<GlVertexBuffer>()
                    .and_then(|b| b.inner.as_ref().map(|i| (b, i)))
                else {
                    continue;
                };
                let (vb, inner) = vb;
                let (size, ty, norm) = attribute_format(attr.format);
                gl.bind_buffer(glow::ARRAY_BUFFER, Some(inner.buffer));
                gl.enable_vertex_attrib_array(reg as u32);
                gl.vertex_attrib_pointer_f32(
                    reg as u32,
                    size,
                    ty,
                    norm,
                    vb.data32_per_vertex as i32 * 4,
                    attr.offset as i32 * 4,
                );
                enabled_attrs.push(reg as u32);
            }

            // Samplers.
            for (unit, (reg, loc, cube)) in prog.samplers.iter().enumerate() {
                let Some(Some(tex)) = self.textures.get(*reg as usize) else {
                    continue;
                };
                let Some(glt) = (&**tex as &dyn Any).downcast_ref::<GlTexture3D>() else {
                    continue;
                };
                let target = if *cube {
                    glow::TEXTURE_CUBE_MAP
                } else {
                    glow::TEXTURE_2D
                };
                gl.active_texture(glow::TEXTURE0 + unit as u32);
                gl.bind_texture(target, Some(glt.texture));
                if let Some((wrap, filter)) = self.sampler_states.get(*reg as usize).copied().flatten()
                {
                    let (wrap_s, wrap_t) = wrap_modes(wrap);
                    let f = filter_mode(filter);
                    gl.tex_parameter_i32(target, glow::TEXTURE_WRAP_S, wrap_s);
                    gl.tex_parameter_i32(target, glow::TEXTURE_WRAP_T, wrap_t);
                    gl.tex_parameter_i32(target, glow::TEXTURE_MIN_FILTER, f);
                    gl.tex_parameter_i32(target, glow::TEXTURE_MAG_FILTER, f);
                }
                if let Some(loc) = loc {
                    gl.uniform_1_i32(Some(loc), unit as i32);
                }
            }

            // Render state. We render into our own back-buffer FBO (its own
            // stencil buffer), so setting the func explicitly also overrides any
            // stale `GL_STENCIL_TEST` the 2D renderer left enabled.
            let sface = stencil_face(self.stencil_face);
            gl.enable(glow::STENCIL_TEST);
            gl.stencil_func_separate(
                sface,
                compare_func(self.stencil_compare),
                self.stencil_ref as i32,
                self.stencil_read_mask,
            );
            gl.stencil_op_separate(
                sface,
                stencil_action(self.stencil_fail),
                stencil_action(self.stencil_depth_fail),
                stencil_action(self.stencil_both_pass),
            );
            gl.stencil_mask(self.stencil_write_mask);
            if target_depth {
                gl.enable(glow::DEPTH_TEST);
                gl.depth_mask(self.depth_mask);
                gl.depth_func(compare_func(self.depth_compare));
            } else {
                gl.disable(glow::DEPTH_TEST);
            }
            gl.enable(glow::BLEND);
            let (src_rgb, src_a) = blend_factor(self.blend_src);
            let (dst_rgb, dst_a) = blend_factor(self.blend_dst);
            gl.blend_func_separate(src_rgb, dst_rgb, src_a, dst_a);
            match self.cull {
                Context3DTriangleFace::None => gl.disable(glow::CULL_FACE),
                face => {
                    gl.enable(glow::CULL_FACE);
                    // Flash's front face is clockwise, but the vertex shader negates
                    // Y (to present the render-to-texture top-down), which reverses
                    // the on-screen winding — so front becomes counter-clockwise.
                    gl.front_face(glow::CCW);
                    gl.cull_face(match face {
                        Context3DTriangleFace::Back => glow::BACK,
                        Context3DTriangleFace::Front => glow::FRONT,
                        _ => glow::FRONT_AND_BACK,
                    });
                }
            }
            let [r, g, b, a] = self.color_mask;
            gl.color_mask(r, g, b, a);

            // Stage3D index buffers are 16-bit.
            gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(ib_inner.buffer));
            gl.draw_elements(
                glow::TRIANGLES,
                count,
                glow::UNSIGNED_SHORT,
                first_index as i32 * 2,
            );

            // Restore the shared GL state the 2D renderer expects, so our Stage3D
            // state doesn't leak into 2D rendering.
            for reg in enabled_attrs {
                gl.disable_vertex_attrib_array(reg);
            }
            gl.disable(glow::DEPTH_TEST);
            gl.disable(glow::CULL_FACE);
            gl.disable(glow::SCISSOR_TEST);
            gl.disable(glow::STENCIL_TEST);
            gl.stencil_mask(0xff);
            gl.color_mask(true, true, true, true);
            gl.bind_vertex_array(None);
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
        }
    }

    /// The current draw target: `(fbo, width, height, has_depth)`. A render-to-
    /// texture target takes precedence over the back buffer.
    fn current_target(&self) -> Option<(glow::Framebuffer, i32, i32, bool)> {
        if let Some((fbo, w, h, depth)) = self.render_target {
            return Some((fbo, w as i32, h as i32, depth));
        }
        self.back
            .as_ref()
            .map(|b| (b.draw_fbo(), b.width as i32, b.height as i32, b.has_depth()))
    }

    fn configure_back_buffer(
        &mut self,
        width: u32,
        height: u32,
        anti_alias: u32,
        depth_and_stencil: bool,
    ) {
        let (w, h) = (width.max(1), height.max(1));
        let samples = if self.supports_msaa && anti_alias > 1 {
            anti_alias.min(4)
        } else {
            1
        };
        self.back = Some(RenderBuffer::new(&self.gl, w, h, depth_and_stencil, samples));
        self.front = Some(RenderBuffer::new(&self.gl, w, h, depth_and_stencil, samples));
        self.configured = true;
    }

    fn clear(&mut self, r: f32, g: f32, b: f32, a: f32, depth: f32, stencil: u32, mask: u32) {
        self.seen_clear = true;
        self.clear_color = Some((r, g, b, a));
        let Some((fbo, _, fb_height, _)) = self.current_target() else {
            return;
        };
        let gl = &self.gl;
        unsafe {
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(fbo));
            apply_scissor(gl, self.scissor, fb_height);
            // AS3 `clear(mask)` bitflags: 1 = color, 2 = depth, 4 = stencil.
            let mut bits = 0;
            if mask & 1 != 0 {
                gl.clear_color(r, g, b, a);
                gl.color_mask(true, true, true, true);
                bits |= glow::COLOR_BUFFER_BIT;
            }
            if mask & 2 != 0 {
                gl.depth_mask(true);
                #[cfg(not(target_family = "wasm"))]
                gl.clear_depth_f64(depth as f64);
                bits |= glow::DEPTH_BUFFER_BIT;
            }
            if mask & 4 != 0 {
                gl.stencil_mask(0xff);
                gl.clear_stencil(stencil as i32);
                bits |= glow::STENCIL_BUFFER_BIT;
            }
            if bits != 0 {
                gl.clear(bits);
            }
            if mask & 4 != 0 {
                // `glClearStencil` is global GL state shared with the 2D renderer,
                // whose mask passes clear the stencil buffer expecting a 0 fill
                // value. Restore it so a Stage3D clear (e.g. Starling clearing to a
                // non-zero stencil) doesn't break 2D masking — which is how a
                // masked TextField drawn to a BitmapData ends up empty.
                gl.clear_stencil(0);
            }
            gl.disable(glow::SCISSOR_TEST);
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
        }
    }
}

impl Drop for GlContext3D {
    fn drop(&mut self) {
        if let Some(vao) = self.vao {
            unsafe { self.gl.delete_vertex_array(vao) };
        }
    }
}

impl Context3D for GlContext3D {
    fn profile(&self) -> Context3DProfile {
        self.profile
    }

    fn bitmap_handle(&self) -> BitmapHandle {
        // The stage composites the *front* buffer (the last presented frame). Fall
        // back to the back buffer before the first present.
        self.front
            .as_ref()
            .or(self.back.as_ref())
            .expect("Context3D has no back buffer")
            .handle
            .clone()
    }

    fn should_render(&self) -> bool {
        self.configured
    }

    fn disposed_index_buffer_handle(&self) -> Rc<dyn IndexBuffer> {
        self.disposed_index.clone()
    }

    fn disposed_vertex_buffer_handle(&self) -> Rc<dyn VertexBuffer> {
        self.disposed_vertex.clone()
    }

    fn create_index_buffer(&mut self, _usage: BufferUsage, num_indices: u32) -> Box<dyn IndexBuffer> {
        // Allocate the full store up front (16-bit indices) so partial uploads use
        // `bufferSubData` and never orphan previously-uploaded data.
        let buffer = alloc_buffer(&self.gl, glow::ELEMENT_ARRAY_BUFFER, num_indices as usize * 2);
        Box::new(GlIndexBuffer {
            inner: Some(GlBuffer {
                gl: self.gl.clone(),
                buffer,
            }),
            count: Cell::new(num_indices),
        })
    }

    fn create_vertex_buffer(
        &mut self,
        _usage: BufferUsage,
        num_vertices: u32,
        data_32_per_vertex: u8,
    ) -> Rc<dyn VertexBuffer> {
        let size = num_vertices as usize * data_32_per_vertex as usize * 4;
        let buffer = alloc_buffer(&self.gl, glow::ARRAY_BUFFER, size);
        Rc::new(GlVertexBuffer {
            inner: Some(GlBuffer {
                gl: self.gl.clone(),
                buffer,
            }),
            data32_per_vertex: data_32_per_vertex,
        })
    }

    fn create_texture(
        &mut self,
        width: u32,
        height: u32,
        format: Context3DTextureFormat,
        _optimize_for_render_to_texture: bool,
        _streaming_levels: u32,
    ) -> Result<Rc<dyn Texture>, Error> {
        let texture = unsafe { self.gl.create_texture() }
            .map_err(|e| Error::Unimplemented(format!("Context3D texture: {e}").into()))?;
        // Zero-initialize: `uploadFromBitmapData` may only fill part of the texture
        // (a smaller bitmap than the texture, or a sub-region), and the rest must
        // read back as transparent black. A NULL upload leaves it undefined on GL,
        // so a partially-uploaded texture would sample driver garbage; wgpu zero-
        // fills new textures, so match that.
        let zeros = vec![0u8; (width as usize) * (height as usize) * 4];
        unsafe {
            self.gl.bind_texture(glow::TEXTURE_2D, Some(texture));
            self.gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::RGBA as i32,
                width as i32,
                height as i32,
                0,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(Some(&zeros)),
            );
            let clamp = glow::CLAMP_TO_EDGE as i32;
            self.gl
                .tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, clamp);
            self.gl
                .tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, clamp);
            self.gl
                .tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::LINEAR as i32);
            self.gl
                .tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);
        }
        Ok(Rc::new(GlTexture3D {
            gl: self.gl.clone(),
            texture,
            width,
            height,
            format,
            cube: false,
            render_fbo: Cell::new(None),
            render_depth: Cell::new(None),
        }))
    }

    fn create_cube_texture(
        &mut self,
        size: u32,
        format: Context3DTextureFormat,
        _optimize_for_render_to_texture: bool,
        _streaming_levels: u32,
    ) -> Result<Rc<dyn Texture>, Error> {
        // Cube sampling is handled by the draw pipeline; allocate storage here.
        let texture = unsafe { self.gl.create_texture() }
            .map_err(|e| Error::Unimplemented(format!("Context3D cube texture: {e}").into()))?;
        unsafe {
            self.gl.bind_texture(glow::TEXTURE_CUBE_MAP, Some(texture));
            for face in 0..6 {
                self.gl.tex_image_2d(
                    glow::TEXTURE_CUBE_MAP_POSITIVE_X + face,
                    0,
                    glow::RGBA as i32,
                    size as i32,
                    size as i32,
                    0,
                    glow::RGBA,
                    glow::UNSIGNED_BYTE,
                    glow::PixelUnpackData::Slice(None),
                );
            }
            // Without an explicit filter the min-filter defaults to
            // NEAREST_MIPMAP_LINEAR; with no mip levels uploaded that leaves the
            // cube map *incomplete*, so every sample returns black (e.g. a skybox
            // and any water reflecting it). Default to LINEAR + clamp, matching
            // `create_texture`; a later SetSamplerStateAt can still override.
            let clamp = glow::CLAMP_TO_EDGE as i32;
            self.gl
                .tex_parameter_i32(glow::TEXTURE_CUBE_MAP, glow::TEXTURE_WRAP_S, clamp);
            self.gl
                .tex_parameter_i32(glow::TEXTURE_CUBE_MAP, glow::TEXTURE_WRAP_T, clamp);
            self.gl.tex_parameter_i32(
                glow::TEXTURE_CUBE_MAP,
                glow::TEXTURE_MIN_FILTER,
                glow::LINEAR as i32,
            );
            self.gl.tex_parameter_i32(
                glow::TEXTURE_CUBE_MAP,
                glow::TEXTURE_MAG_FILTER,
                glow::LINEAR as i32,
            );
        }
        Ok(Rc::new(GlTexture3D {
            gl: self.gl.clone(),
            texture,
            width: size,
            height: size,
            format,
            cube: true,
            render_fbo: Cell::new(None),
            render_depth: Cell::new(None),
        }))
    }

    fn upload_shaders(
        &mut self,
        module: &RefCell<Option<Rc<dyn ShaderModule>>>,
        vertex_shader_agal: Vec<u8>,
        fragment_shader_agal: Vec<u8>,
    ) -> Result<(), naga_agal::AgalError> {
        // Parse (surfaces malformed-program errors), translate to GLSL ES, and link.
        let vertex = naga_agal::parse_bytecode(&vertex_shader_agal)?;
        let fragment = naga_agal::parse_bytecode(&fragment_shader_agal)?;
        // Validate sampler usage (a sampler register reused with mismatched
        // properties is AGAL error #3696), matching the wgpu backend.
        naga_agal::extract_sampler_configs(&fragment)?;
        let program = compile_program(&self.gl, self.is_embedded, &vertex, &fragment);
        *module.borrow_mut() = Some(Rc::new(GlShaderModule { program }));
        Ok(())
    }

    fn process_command(&mut self, command: Context3DCommand<'_>) {
        match command {
            Context3DCommand::Clear {
                red,
                green,
                blue,
                alpha,
                depth,
                stencil,
                mask,
            } => {
                self.clear(
                    red as f32,
                    green as f32,
                    blue as f32,
                    alpha as f32,
                    depth as f32,
                    stencil,
                    mask,
                );
            }
            Context3DCommand::ConfigureBackBuffer {
                width,
                height,
                anti_alias,
                depth_and_stencil,
                ..
            } => {
                self.configure_back_buffer(width, height, anti_alias, depth_and_stencil);
            }
            Context3DCommand::SetRenderToTexture {
                texture,
                enable_depth_and_stencil,
                ..
            } => {
                if let Some(glt) = (&*texture as &dyn Any).downcast_ref::<GlTexture3D>() {
                    let fbo = glt.render_fbo(enable_depth_and_stencil);
                    self.render_target =
                        Some((fbo, glt.width, glt.height, enable_depth_and_stencil));
                }
                self.render_to_texture = Some(texture);
            }
            Context3DCommand::SetRenderToBackBuffer => {
                self.render_to_texture = None;
                self.render_target = None;
            }
            Context3DCommand::UploadToIndexBuffer {
                buffer,
                start_offset,
                data,
            } => {
                if let Some(ib) = (buffer as &dyn Any).downcast_ref::<GlIndexBuffer>() {
                    if let Some(inner) = &ib.inner {
                        // 16-bit indices → 2 bytes per index.
                        upload_buffer(&self.gl, glow::ELEMENT_ARRAY_BUFFER, inner.buffer, start_offset * 2, data);
                    }
                }
            }
            Context3DCommand::UploadToVertexBuffer {
                buffer,
                start_vertex,
                data32_per_vertex,
                data,
            } => {
                if let Some(vb) = (&*buffer as &dyn Any).downcast_ref::<GlVertexBuffer>() {
                    if let Some(inner) = &vb.inner {
                        let stride = data32_per_vertex as usize * 4;
                        upload_buffer(&self.gl, glow::ARRAY_BUFFER, inner.buffer, start_vertex * stride, data);
                    }
                }
            }
            Context3DCommand::SetVertexBufferAt {
                index,
                buffer,
                buffer_offset,
            } => {
                let slot = index as usize;
                if slot < MAX_VERTEX_ATTRIBUTES {
                    self.vertex_attributes[slot] = buffer.map(|(buffer, format)| VertexAttribute {
                        buffer,
                        offset: buffer_offset,
                        format,
                    });
                }
            }
            Context3DCommand::SetShaders { module } => {
                self.module = module;
            }
            Context3DCommand::SetProgramConstants {
                program_type,
                first_register,
                matrix_raw_data_column_major,
            } => {
                let dest = match program_type {
                    ProgramType::Vertex => &mut self.vertex_constants,
                    ProgramType::Fragment => &mut self.fragment_constants,
                };
                let first = first_register as usize;
                let needed = first + matrix_raw_data_column_major.len() / 4;
                if dest.len() < needed {
                    dest.resize(needed, [0.0; 4]);
                }
                for (i, chunk) in matrix_raw_data_column_major.chunks_exact(4).enumerate() {
                    dest[first + i] = [
                        f32::from_le_bytes(chunk[0]),
                        f32::from_le_bytes(chunk[1]),
                        f32::from_le_bytes(chunk[2]),
                        f32::from_le_bytes(chunk[3]),
                    ];
                }
            }
            Context3DCommand::SetCulling { face } => self.cull = face,
            Context3DCommand::SetSamplerStateAt {
                sampler,
                wrap,
                filter,
            } => {
                if let Some(slot) = self.sampler_states.get_mut(sampler as usize) {
                    *slot = Some((wrap, filter));
                }
            }
            Context3DCommand::SetScissorRectangle { rect } => {
                self.scissor = rect.map(|r| {
                    (
                        r.x_min.to_pixels().round() as i32,
                        r.y_min.to_pixels().round() as i32,
                        r.width().to_pixels().round() as i32,
                        r.height().to_pixels().round() as i32,
                    )
                });
            }
            Context3DCommand::SetTextureAt {
                sampler, texture, ..
            } => {
                let slot = sampler as usize;
                if slot < MAX_SAMPLERS {
                    self.textures[slot] = texture;
                }
            }
            Context3DCommand::CopyBitmapToTexture {
                source,
                source_width,
                source_height,
                dest,
                layer,
            } => {
                if let Some(glt) = (&*dest as &dyn Any).downcast_ref::<GlTexture3D>() {
                    let (bind_target, face) = if glt.cube {
                        (
                            glow::TEXTURE_CUBE_MAP,
                            glow::TEXTURE_CUBE_MAP_POSITIVE_X + layer,
                        )
                    } else {
                        (glow::TEXTURE_2D, glow::TEXTURE_2D)
                    };

                    // ATF `compressed`/`compressedAlpha` textures arrive here as
                    // raw DXT5 (BC3) blocks when the ATF payload is block-
                    // compressed (e.g. Starling's compressedAlpha textures). The
                    // wgpu backend uploads these to a native `Bc3RgbaUnorm`
                    // texture, but the oldest-GL targets can't rely on the S3TC
                    // extension — so decode DXT5 to RGBA on the CPU and upload it
                    // as an ordinary RGBA texture. When the ATF payload was JPEG-XR
                    // instead, `source` is already RGBA (length w*h*4, i.e. 4x the
                    // DXT5 size), so the two cases are unambiguous by length.
                    let dxt5_len = (source_width as usize).div_ceil(4)
                        * (source_height as usize).div_ceil(4)
                        * 16;
                    let decoded;
                    let pixels: &[u8] = if matches!(
                        glt.format,
                        Context3DTextureFormat::Compressed | Context3DTextureFormat::CompressedAlpha
                    ) && source.len() == dxt5_len
                    {
                        decoded = decode_dxt5(source, source_width, source_height);
                        &decoded
                    } else {
                        source
                    };

                    unsafe {
                        self.gl.bind_texture(bind_target, Some(glt.texture));
                        self.gl.tex_sub_image_2d(
                            face,
                            0,
                            0,
                            0,
                            source_width as i32,
                            source_height as i32,
                            glow::RGBA,
                            glow::UNSIGNED_BYTE,
                            glow::PixelUnpackData::Slice(Some(pixels)),
                        );
                    }
                }
            }
            Context3DCommand::SetColorMask {
                red,
                green,
                blue,
                alpha,
            } => self.color_mask = [red, green, blue, alpha],
            Context3DCommand::SetDepthTest {
                depth_mask,
                pass_compare_mode,
            } => {
                self.depth_mask = depth_mask;
                self.depth_compare = pass_compare_mode;
            }
            Context3DCommand::SetBlendFactors {
                source_factor,
                destination_factor,
            } => {
                self.blend_src = source_factor;
                self.blend_dst = destination_factor;
            }
            Context3DCommand::SetStencilActions {
                triangle_face,
                compare_mode,
                on_both_pass,
                on_depth_fail,
                on_depth_pass_stencil_fail,
            } => {
                self.stencil_face = triangle_face;
                self.stencil_compare = compare_mode;
                self.stencil_both_pass = on_both_pass;
                self.stencil_depth_fail = on_depth_fail;
                self.stencil_fail = on_depth_pass_stencil_fail;
            }
            Context3DCommand::SetStencilReferenceValue {
                reference_value,
                read_mask,
                write_mask,
            } => {
                self.stencil_ref = reference_value;
                self.stencil_read_mask = read_mask;
                self.stencil_write_mask = write_mask;
            }
            Context3DCommand::DrawTriangles {
                index_buffer,
                first_index,
                num_triangles,
            } => {
                self.draw_triangles(index_buffer, first_index, num_triangles);
            }
        }
    }

    fn present(&mut self) {
        // Resolve the freshly-drawn multisampled buffer into its exposed texture
        // before it becomes the front buffer.
        if let Some(back) = self.back.as_ref() {
            back.resolve();
        }
        std::mem::swap(&mut self.back, &mut self.front);
        self.seen_clear = false;
        self.clear_color = None;
        self.render_to_texture = None;
        self.render_target = None;
    }
}

/// Translates and links an AGAL vertex/fragment pair into a GL program.
fn compile_program(
    gl: &GlContext,
    is_embedded: bool,
    vertex: &naga_agal::ParsedBytecode,
    fragment: &naga_agal::ParsedBytecode,
) -> Option<CompiledProgram> {
    let translated = match crate::agal::translate(vertex, fragment) {
        Ok(t) => t,
        Err(e) => {
            log::warn!("GL Stage3D: AGAL translation failed: {e}");
            return None;
        }
    };
    let compile = |stage, src: &str| match crate::shader::compile_shader(gl, is_embedded, stage, src) {
        Ok(s) => Some(s),
        Err(e) => {
            log::warn!("GL Stage3D: shader compile failed: {e:?}\n{src}");
            None
        }
    };
    let vs = compile(glow::VERTEX_SHADER, &translated.vertex_glsl)?;
    let fs = compile(glow::FRAGMENT_SHADER, &translated.fragment_glsl)?;
    unsafe {
        let program = gl.create_program().ok()?;
        gl.attach_shader(program, vs);
        gl.attach_shader(program, fs);
        for &reg in &translated.attributes {
            gl.bind_attrib_location(program, reg as u32, &format!("va{reg}"));
        }
        gl.link_program(program);
        gl.delete_shader(vs);
        gl.delete_shader(fs);
        if !gl.get_program_link_status(program) {
            log::warn!(
                "GL Stage3D: link failed: {}",
                gl.get_program_info_log(program)
            );
            gl.delete_program(program);
            return None;
        }
        let vc_loc = gl.get_uniform_location(program, "vc[0]");
        let fc_loc = gl.get_uniform_location(program, "fc[0]");
        let samplers = translated
            .samplers
            .iter()
            .map(|s| {
                (
                    s.reg,
                    gl.get_uniform_location(program, &format!("fs{}", s.reg)),
                    s.cube,
                )
            })
            .collect();
        Some(CompiledProgram {
            gl: gl.clone(),
            program,
            vc_loc,
            fc_loc,
            num_vc: translated.num_vertex_constants,
            num_fc: translated.num_fragment_constants,
            attributes: translated.attributes,
            samplers,
        })
    }
}

/// Uploads the first `num` constant registers into a `vec4[num]` uniform.
unsafe fn upload_constants(
    gl: &GlContext,
    loc: Option<&glow::UniformLocation>,
    num: usize,
    constants: &[[f32; 4]],
) {
    let Some(loc) = loc else { return };
    if num == 0 {
        return;
    }
    let mut buf = vec![0.0f32; num * 4];
    for (i, c) in constants.iter().take(num).enumerate() {
        buf[i * 4..i * 4 + 4].copy_from_slice(c);
    }
    unsafe { gl.uniform_4_f32_slice(Some(loc), &buf) };
}

/// Expand a 16-bit RGB565 color to 8-bit-per-channel RGB (bit-replication).
fn rgb565(c: u16) -> [u8; 3] {
    let r = ((c >> 11) & 0x1f) as u8;
    let g = ((c >> 5) & 0x3f) as u8;
    let b = (c & 0x1f) as u8;
    [(r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2)]
}

/// Decode DXT5 (BC3) compressed blocks into tightly-packed RGBA8.
///
/// `width`/`height` are the full texture dimensions; the image is stored as
/// 4x4 blocks padded up to a multiple of 4 in each dimension, and texels past
/// the real edge are discarded. Used for ATF `compressedAlpha` textures, which
/// the oldest-GL targets must decode on the CPU (no guaranteed S3TC support).
fn decode_dxt5(data: &[u8], width: u32, height: u32) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let mut out = vec![0u8; w * h * 4];
    let blocks_x = w.div_ceil(4);
    let blocks_y = h.div_ceil(4);
    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let Some(block) = data.get((by * blocks_x + bx) * 16..(by * blocks_x + bx) * 16 + 16)
            else {
                continue;
            };

            // Alpha block (BC4-style): two 8-bit endpoints + 16 3-bit indices.
            let (a0, a1) = (block[0] as u16, block[1] as u16);
            let mut alpha = [0u8; 8];
            alpha[0] = a0 as u8;
            alpha[1] = a1 as u8;
            if a0 > a1 {
                for i in 1..7u16 {
                    alpha[i as usize + 1] = (((7 - i) * a0 + i * a1) / 7) as u8;
                }
            } else {
                for i in 1..5u16 {
                    alpha[i as usize + 1] = (((5 - i) * a0 + i * a1) / 5) as u8;
                }
                alpha[6] = 0;
                alpha[7] = 255;
            }
            let alpha_bits = u64::from_le_bytes([
                block[2], block[3], block[4], block[5], block[6], block[7], 0, 0,
            ]);

            // Color block (BC1-style, always 4-color mode for DXT5).
            let c0 = u16::from_le_bytes([block[8], block[9]]);
            let c1 = u16::from_le_bytes([block[10], block[11]]);
            let mut col = [[0u8; 3]; 4];
            col[0] = rgb565(c0);
            col[1] = rgb565(c1);
            for c in 0..3 {
                col[2][c] = ((2 * col[0][c] as u16 + col[1][c] as u16) / 3) as u8;
                col[3][c] = ((col[0][c] as u16 + 2 * col[1][c] as u16) / 3) as u8;
            }
            let color_bits =
                u32::from_le_bytes([block[12], block[13], block[14], block[15]]);

            for py in 0..4 {
                for px in 0..4 {
                    let (x, y) = (bx * 4 + px, by * 4 + py);
                    if x >= w || y >= h {
                        continue;
                    }
                    let idx = py * 4 + px;
                    let a = alpha[((alpha_bits >> (idx * 3)) & 0x7) as usize];
                    let rgb = col[((color_bits >> (idx * 2)) & 0x3) as usize];
                    let o = (y * w + x) * 4;
                    out[o] = rgb[0];
                    out[o + 1] = rgb[1];
                    out[o + 2] = rgb[2];
                    out[o + 3] = a;
                }
            }
        }
    }
    out
}

fn attribute_format(format: Context3DVertexBufferFormat) -> (i32, u32, bool) {
    match format {
        Context3DVertexBufferFormat::Float1 => (1, glow::FLOAT, false),
        Context3DVertexBufferFormat::Float2 => (2, glow::FLOAT, false),
        Context3DVertexBufferFormat::Float3 => (3, glow::FLOAT, false),
        Context3DVertexBufferFormat::Float4 => (4, glow::FLOAT, false),
        Context3DVertexBufferFormat::Bytes4 => (4, glow::UNSIGNED_BYTE, true),
    }
}

fn wrap_modes(wrap: Context3DWrapMode) -> (i32, i32) {
    let clamp = glow::CLAMP_TO_EDGE as i32;
    let repeat = glow::REPEAT as i32;
    match wrap {
        Context3DWrapMode::Clamp => (clamp, clamp),
        Context3DWrapMode::Repeat => (repeat, repeat),
        Context3DWrapMode::ClampURepeatV => (clamp, repeat),
        Context3DWrapMode::RepeatUClampV => (repeat, clamp),
    }
}

fn filter_mode(filter: Context3DTextureFilter) -> i32 {
    match filter {
        Context3DTextureFilter::Nearest => glow::NEAREST as i32,
        _ => glow::LINEAR as i32,
    }
}

fn stencil_action(action: Context3DStencilAction) -> u32 {
    match action {
        Context3DStencilAction::Keep => glow::KEEP,
        Context3DStencilAction::Zero => glow::ZERO,
        Context3DStencilAction::Set => glow::REPLACE,
        Context3DStencilAction::IncrementSaturate => glow::INCR,
        Context3DStencilAction::DecrementSaturate => glow::DECR,
        Context3DStencilAction::IncrementWrap => glow::INCR_WRAP,
        Context3DStencilAction::DecrementWrap => glow::DECR_WRAP,
        Context3DStencilAction::Invert => glow::INVERT,
    }
}

/// GL face enum for a Stage3D triangle face (`None` applies to both faces).
fn stencil_face(face: Context3DTriangleFace) -> u32 {
    match face {
        Context3DTriangleFace::Front => glow::FRONT,
        Context3DTriangleFace::Back => glow::BACK,
        Context3DTriangleFace::FrontAndBack | Context3DTriangleFace::None => glow::FRONT_AND_BACK,
    }
}

fn compare_func(mode: Context3DCompareMode) -> u32 {
    match mode {
        Context3DCompareMode::Never => glow::NEVER,
        Context3DCompareMode::Less => glow::LESS,
        Context3DCompareMode::Equal => glow::EQUAL,
        Context3DCompareMode::LessEqual => glow::LEQUAL,
        Context3DCompareMode::Greater => glow::GREATER,
        Context3DCompareMode::NotEqual => glow::NOTEQUAL,
        Context3DCompareMode::GreaterEqual => glow::GEQUAL,
        Context3DCompareMode::Always => glow::ALWAYS,
    }
}

/// Maps a Flash blend factor to a `(color, alpha)` GL factor pair. For the
/// `*_COLOR` factors the alpha channel uses the corresponding `*_ALPHA` factor
/// (a colour has no meaningful value for the alpha channel), matching wgpu's
/// separate colour/alpha blend components — otherwise the resulting back-buffer
/// alpha (and thus later `*_DESTINATION_*` blends) drifts.
fn blend_factor(factor: Context3DBlendFactor) -> (u32, u32) {
    match factor {
        Context3DBlendFactor::Zero => (glow::ZERO, glow::ZERO),
        Context3DBlendFactor::One => (glow::ONE, glow::ONE),
        Context3DBlendFactor::SourceColor => (glow::SRC_COLOR, glow::SRC_ALPHA),
        Context3DBlendFactor::OneMinusSourceColor => {
            (glow::ONE_MINUS_SRC_COLOR, glow::ONE_MINUS_SRC_ALPHA)
        }
        Context3DBlendFactor::SourceAlpha => (glow::SRC_ALPHA, glow::SRC_ALPHA),
        Context3DBlendFactor::OneMinusSourceAlpha => {
            (glow::ONE_MINUS_SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA)
        }
        Context3DBlendFactor::DestinationColor => (glow::DST_COLOR, glow::DST_ALPHA),
        Context3DBlendFactor::OneMinusDestinationColor => {
            (glow::ONE_MINUS_DST_COLOR, glow::ONE_MINUS_DST_ALPHA)
        }
        Context3DBlendFactor::DestinationAlpha => (glow::DST_ALPHA, glow::DST_ALPHA),
        Context3DBlendFactor::OneMinusDestinationAlpha => {
            (glow::ONE_MINUS_DST_ALPHA, glow::ONE_MINUS_DST_ALPHA)
        }
    }
}

/// Enables/sets the GL scissor from a Flash (top-left origin) rectangle, flipping
/// Y into the framebuffer's bottom-left origin.
unsafe fn apply_scissor(gl: &GlContext, scissor: Option<(i32, i32, i32, i32)>, fb_height: i32) {
    unsafe {
        // Flash ignores a degenerate (zero/negative) scissor rectangle.
        if let Some((x, y, w, h)) = scissor.filter(|&(_, _, w, h)| w > 0 && h > 0) {
            gl.enable(glow::SCISSOR_TEST);
            gl.scissor(x, fb_height - (y + h), w, h);
        } else {
            gl.disable(glow::SCISSOR_TEST);
        }
    }
}

/// Creates a buffer with `size` bytes of (zeroed) storage.
fn alloc_buffer(gl: &GlContext, target: u32, size: usize) -> glow::Buffer {
    unsafe {
        let buffer = gl.create_buffer().expect("Context3D buffer");
        gl.bind_buffer(target, Some(buffer));
        gl.buffer_data_size(target, size.max(1) as i32, glow::DYNAMIC_DRAW);
        buffer
    }
}

/// Uploads `data` into a pre-allocated `buffer` at `byte_offset` via `bufferSubData`
/// (so earlier uploads to other regions are preserved).
fn upload_buffer(gl: &GlContext, target: u32, buffer: glow::Buffer, byte_offset: usize, data: &[u8]) {
    unsafe {
        gl.bind_buffer(target, Some(buffer));
        gl.buffer_sub_data_u8_slice(target, byte_offset as i32, data);
    }
}
