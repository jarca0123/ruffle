//! GPU filters (Flash `BitmapFilter`), modeled on `render/wgpu/src/filters`.
//!
//! Each filter renders a quad covering its output target, sampling the source
//! texture, into a fresh temporary texture which the backend then copies into
//! the destination. All filter textures keep the offscreen convention
//! (Flash-top at texel row 0). Filters operate on premultiplied RGBA.

use crate::context::GlContext;
use crate::error::Error;
use crate::pool::{set_sampling, TexturePool};
use crate::shader;
use glow::HasContext as _;

const FILTER_VERT: &str = include_str!("../shaders/filter.vert");
const COLOR_MATRIX_FRAG: &str = include_str!("../shaders/color_matrix.frag");
const BLUR_VERT: &str = include_str!("../shaders/blur.vert");
const BLUR_FRAG: &str = include_str!("../shaders/blur.frag");
const GLOW_VERT: &str = include_str!("../shaders/glow.vert");
const GLOW_FRAG: &str = include_str!("../shaders/glow.frag");
const BEVEL_VERT: &str = include_str!("../shaders/bevel.vert");
const BEVEL_FRAG: &str = include_str!("../shaders/bevel.frag");
const BLEND_VERT: &str = include_str!("../shaders/blend.vert");
const BLEND_FRAG: &str = include_str!("../shaders/blend.frag");
const CONVOLUTION_FRAG: &str = include_str!("../shaders/convolution.frag");
const DISPLACEMENT_VERT: &str = include_str!("../shaders/displacement.vert");
const DISPLACEMENT_FRAG: &str = include_str!("../shaders/displacement.frag");
const GRADIENT_GLOW_FRAG: &str = include_str!("../shaders/gradient_glow.frag");
const GRADIENT_BEVEL_FRAG: &str = include_str!("../shaders/gradient_bevel.frag");

/// Width of the gradient ramp lookup texture (256 entries, 1 row).
pub(crate) const GRADIENT_RAMP_SIZE: usize = 256;

/// Maximum convolution kernel taps (7x7). Larger kernels are unsupported.
pub(crate) const MAX_CONVOLUTION_TAPS: usize = 49;

/// All filter programs bind the `position` attribute to this fixed location so
/// a single quad VAO works for every program.
const POSITION_LOC: u32 = 0;

/// A filter render result: a freshly-allocated texture the caller owns and must
/// delete after copying it into the destination.
pub(crate) struct FilterResult {
    pub texture: glow::Texture,
    pub width: u32,
    pub height: u32,
}

pub(crate) struct Filters {
    gl: GlContext,
    quad_vbo: glow::Buffer,
    quad_vao: glow::VertexArray,
    /// Reused for every filter pass: a target texture is attached, drawn into,
    /// then the next pass re-attaches its own target. Created once.
    scratch_fbo: glow::Framebuffer,
    color_matrix: ColorMatrixProgram,
    blur: BlurProgram,
    glow: GlowProgram,
    bevel: BevelProgram,
    blend: BlendProgram,
    convolution: ConvolutionProgram,
    displacement: DisplacementProgram,
    gradient_glow: GradientGlowProgram,
    gradient_bevel: GradientBevelProgram,
    /// Reused 256x1 RGBA8 ramp texture for gradient glow/bevel; re-uploaded each
    /// apply.
    ramp_texture: glow::Texture,
}

struct ColorMatrixProgram {
    program: glow::Program,
    u_texture: Option<glow::UniformLocation>,
    u_uv_rect: Option<glow::UniformLocation>,
    u_cm: Option<glow::UniformLocation>,
}

struct BlurProgram {
    program: glow::Program,
    u_texture: Option<glow::UniformLocation>,
    u_uv_rect: Option<glow::UniformLocation>,
    u_dir: Option<glow::UniformLocation>,
    u_m: Option<glow::UniformLocation>,
    u_full_size: Option<glow::UniformLocation>,
    u_m2: Option<glow::UniformLocation>,
    u_first_weight: Option<glow::UniformLocation>,
    u_last_offset: Option<glow::UniformLocation>,
    u_last_weight: Option<glow::UniformLocation>,
}

struct GlowProgram {
    program: glow::Program,
    u_texture: Option<glow::UniformLocation>,
    u_blurred: Option<glow::UniformLocation>,
    u_color: Option<glow::UniformLocation>,
    u_strength: Option<glow::UniformLocation>,
    u_inner: Option<glow::UniformLocation>,
    u_knockout: Option<glow::UniformLocation>,
    u_composite_source: Option<glow::UniformLocation>,
    u_uv_rect: Option<glow::UniformLocation>,
    u_blur_offset: Option<glow::UniformLocation>,
}

struct BevelProgram {
    program: glow::Program,
    u_texture: Option<glow::UniformLocation>,
    u_blurred: Option<glow::UniformLocation>,
    u_highlight: Option<glow::UniformLocation>,
    u_shadow: Option<glow::UniformLocation>,
    u_strength: Option<glow::UniformLocation>,
    u_bevel_type: Option<glow::UniformLocation>,
    u_knockout: Option<glow::UniformLocation>,
    u_uv_rect: Option<glow::UniformLocation>,
    u_blur_offset: Option<glow::UniformLocation>,
}

struct BlendProgram {
    program: glow::Program,
    u_current: Option<glow::UniformLocation>,
    u_parent: Option<glow::UniformLocation>,
    u_blend_mode: Option<glow::UniformLocation>,
}

struct ConvolutionProgram {
    program: glow::Program,
    u_texture: Option<glow::UniformLocation>,
    u_uv_rect: Option<glow::UniformLocation>,
    u_texel: Option<glow::UniformLocation>,
    u_kernel: Option<glow::UniformLocation>,
    u_cols: Option<glow::UniformLocation>,
    u_rows: Option<glow::UniformLocation>,
    u_divisor: Option<glow::UniformLocation>,
    u_bias: Option<glow::UniformLocation>,
    u_default_color: Option<glow::UniformLocation>,
    u_clamp: Option<glow::UniformLocation>,
    u_preserve_alpha: Option<glow::UniformLocation>,
}

struct DisplacementProgram {
    program: glow::Program,
    u_source: Option<glow::UniformLocation>,
    u_map: Option<glow::UniformLocation>,
    u_uv_rect: Option<glow::UniformLocation>,
    u_color: Option<glow::UniformLocation>,
    u_components: Option<glow::UniformLocation>,
    u_mode: Option<glow::UniformLocation>,
    u_scale: Option<glow::UniformLocation>,
    u_source_size: Option<glow::UniformLocation>,
    u_map_size: Option<glow::UniformLocation>,
    u_offset: Option<glow::UniformLocation>,
    u_viewscale: Option<glow::UniformLocation>,
}

struct GradientGlowProgram {
    program: glow::Program,
    u_texture: Option<glow::UniformLocation>,
    u_blurred: Option<glow::UniformLocation>,
    u_ramp: Option<glow::UniformLocation>,
    u_strength: Option<glow::UniformLocation>,
    u_type: Option<glow::UniformLocation>,
    u_knockout: Option<glow::UniformLocation>,
    u_composite_source: Option<glow::UniformLocation>,
    u_uv_rect: Option<glow::UniformLocation>,
    u_blur_offset: Option<glow::UniformLocation>,
}

struct GradientBevelProgram {
    program: glow::Program,
    u_texture: Option<glow::UniformLocation>,
    u_blurred: Option<glow::UniformLocation>,
    u_ramp: Option<glow::UniformLocation>,
    u_strength: Option<glow::UniformLocation>,
    u_bevel_type: Option<glow::UniformLocation>,
    u_knockout: Option<glow::UniformLocation>,
    u_uv_rect: Option<glow::UniformLocation>,
    u_blur_offset: Option<glow::UniformLocation>,
}

/// Links a vertex+fragment pair, forcing `position` to [`POSITION_LOC`].
fn link_program(
    gl: &glow::Context,
    vert: glow::Shader,
    frag: glow::Shader,
) -> Result<glow::Program, Error> {
    unsafe {
        let program = gl.create_program().map_err(Error::UnableToCreateProgram)?;
        gl.attach_shader(program, vert);
        gl.attach_shader(program, frag);
        gl.bind_attrib_location(program, POSITION_LOC, "position");
        gl.link_program(program);
        if !gl.get_program_link_status(program) {
            return Err(Error::LinkingShaderProgram(
                gl.get_program_info_log(program),
            ));
        }
        Ok(program)
    }
}

impl Filters {
    pub(crate) fn new(gl: GlContext, is_embedded: bool) -> Result<Self, Error> {
        let filter_vert =
            shader::compile_shader(&gl, is_embedded, glow::VERTEX_SHADER, FILTER_VERT)?;
        let blur_vert = shader::compile_shader(&gl, is_embedded, glow::VERTEX_SHADER, BLUR_VERT)?;
        let cm_frag =
            shader::compile_shader(&gl, is_embedded, glow::FRAGMENT_SHADER, COLOR_MATRIX_FRAG)?;
        let blur_frag = shader::compile_shader(&gl, is_embedded, glow::FRAGMENT_SHADER, BLUR_FRAG)?;

        let glow_vert = shader::compile_shader(&gl, is_embedded, glow::VERTEX_SHADER, GLOW_VERT)?;
        let glow_frag = shader::compile_shader(&gl, is_embedded, glow::FRAGMENT_SHADER, GLOW_FRAG)?;

        let bevel_vert = shader::compile_shader(&gl, is_embedded, glow::VERTEX_SHADER, BEVEL_VERT)?;
        let bevel_frag =
            shader::compile_shader(&gl, is_embedded, glow::FRAGMENT_SHADER, BEVEL_FRAG)?;

        let blend_vert = shader::compile_shader(&gl, is_embedded, glow::VERTEX_SHADER, BLEND_VERT)?;
        let blend_frag =
            shader::compile_shader(&gl, is_embedded, glow::FRAGMENT_SHADER, BLEND_FRAG)?;

        let conv_frag =
            shader::compile_shader(&gl, is_embedded, glow::FRAGMENT_SHADER, CONVOLUTION_FRAG)?;
        let disp_vert =
            shader::compile_shader(&gl, is_embedded, glow::VERTEX_SHADER, DISPLACEMENT_VERT)?;
        let disp_frag =
            shader::compile_shader(&gl, is_embedded, glow::FRAGMENT_SHADER, DISPLACEMENT_FRAG)?;

        // The color-matrix vertex shader is reused for convolution (both emit the
        // region UV from `u_uv_rect`); re-compile it for the second link.
        let filter_vert_conv =
            shader::compile_shader(&gl, is_embedded, glow::VERTEX_SHADER, FILTER_VERT)?;

        let color_matrix = ColorMatrixProgram::new(&gl, link_program(&gl, filter_vert, cm_frag)?);
        let blur = BlurProgram::new(&gl, link_program(&gl, blur_vert, blur_frag)?);
        let glow = GlowProgram::new(&gl, link_program(&gl, glow_vert, glow_frag)?);
        let bevel = BevelProgram::new(&gl, link_program(&gl, bevel_vert, bevel_frag)?);
        let blend = BlendProgram::new(&gl, link_program(&gl, blend_vert, blend_frag)?);
        let convolution =
            ConvolutionProgram::new(&gl, link_program(&gl, filter_vert_conv, conv_frag)?);
        let displacement =
            DisplacementProgram::new(&gl, link_program(&gl, disp_vert, disp_frag)?);

        // Gradient glow/bevel reuse the glow/bevel vertex shaders (Copy handles).
        let gg_frag =
            shader::compile_shader(&gl, is_embedded, glow::FRAGMENT_SHADER, GRADIENT_GLOW_FRAG)?;
        let gb_frag =
            shader::compile_shader(&gl, is_embedded, glow::FRAGMENT_SHADER, GRADIENT_BEVEL_FRAG)?;
        let gradient_glow =
            GradientGlowProgram::new(&gl, link_program(&gl, glow_vert, gg_frag)?);
        let gradient_bevel =
            GradientBevelProgram::new(&gl, link_program(&gl, bevel_vert, gb_frag)?);

        // 256x1 RGBA ramp lookup, re-uploaded per gradient-filter apply.
        let ramp_texture = unsafe {
            let tex = gl.create_texture().map_err(Error::UnableToCreateTexture)?;
            gl.bind_texture(glow::TEXTURE_2D, Some(tex));
            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::RGBA as i32,
                GRADIENT_RAMP_SIZE as i32,
                1,
                0,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(None),
            );
            set_sampling(&gl, glow::LINEAR);
            tex
        };

        // Quad with positions in [0, 1] (drawn as a TRIANGLE_FAN).
        let quad: [f32; 8] = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        let (quad_vbo, quad_vao) = unsafe {
            let vao = gl.create_vertex_array().map_err(Error::UnableToCreateVAO)?;
            gl.bind_vertex_array(Some(vao));
            let vbo = gl.create_buffer().map_err(Error::UnableToCreateBuffer)?;
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
            gl.buffer_data_u8_slice(
                glow::ARRAY_BUFFER,
                bytemuck::cast_slice(&quad),
                glow::STATIC_DRAW,
            );
            gl.vertex_attrib_pointer_f32(POSITION_LOC, 2, glow::FLOAT, false, 8, 0);
            gl.enable_vertex_attrib_array(POSITION_LOC);
            gl.bind_vertex_array(None);
            (vbo, vao)
        };

        let scratch_fbo = unsafe { gl.create_framebuffer() }.map_err(Error::UnableToCreateFrameBuffer)?;

        Ok(Self {
            gl,
            quad_vbo,
            quad_vao,
            scratch_fbo,
            color_matrix,
            blur,
            glow,
            bevel,
            blend,
            convolution,
            displacement,
            gradient_glow,
            gradient_bevel,
            ramp_texture,
        })
    }

    /// Composites a complex blend onto the currently-bound framebuffer: samples
    /// `src` (current, texture unit 0) and `dst` (parent, unit 1) with the chosen
    /// `mode` and draws the region quad. The caller owns the framebuffer binding,
    /// viewport, stencil and (disabled) blend state; this only touches the
    /// program, texture units and the quad VAO.
    pub(crate) fn draw_blend(&self, src: glow::Texture, dst: glow::Texture, mode: i32) {
        let gl = &self.gl;
        let bp = &self.blend;
        unsafe {
            gl.use_program(Some(bp.program));
            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, Some(src));
            gl.active_texture(glow::TEXTURE1);
            gl.bind_texture(glow::TEXTURE_2D, Some(dst));
            gl.uniform_1_i32(bp.u_current.as_ref(), 0);
            gl.uniform_1_i32(bp.u_parent.as_ref(), 1);
            gl.uniform_1_i32(bp.u_blend_mode.as_ref(), mode);

            gl.bind_vertex_array(Some(self.quad_vao));
            gl.draw_arrays(glow::TRIANGLE_FAN, 0, 4);
            gl.bind_vertex_array(None);

            // Leave unit 0 active for subsequent code.
            gl.active_texture(glow::TEXTURE0);
        }
    }

    /// Renders the quad through `program` (sampling `source`) into an existing
    /// `target` texture, via the shared scratch FBO. Blend/stencil are disabled;
    /// blend is restored after.
    unsafe fn render_into(
        &self,
        program: glow::Program,
        source: glow::Texture,
        target: glow::Texture,
        out_w: u32,
        out_h: u32,
        set_uniforms: impl FnOnce(&glow::Context),
    ) {
        let gl = &self.gl;
        unsafe {
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.scratch_fbo));
            gl.framebuffer_texture_2d(
                glow::FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_2D,
                Some(target),
                0,
            );
            gl.viewport(0, 0, out_w as i32, out_h as i32);
            gl.disable(glow::BLEND);
            gl.disable(glow::STENCIL_TEST);
            gl.color_mask(true, true, true, true);

            gl.use_program(Some(program));
            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, Some(source));
            set_uniforms(gl);

            gl.bind_vertex_array(Some(self.quad_vao));
            gl.draw_arrays(glow::TRIANGLE_FAN, 0, 4);
            gl.bind_vertex_array(None);

            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            gl.enable(glow::BLEND);
        }
    }

    pub(crate) fn apply_color_matrix(
        &self,
        pool: &mut TexturePool,
        source: glow::Texture,
        src_w: u32,
        src_h: u32,
        src_point: (u32, u32),
        src_size: (u32, u32),
        matrix: &[f32; 20],
    ) -> Option<FilterResult> {
        let (out_w, out_h) = src_size;
        if out_w == 0 || out_h == 0 || src_w == 0 || src_h == 0 {
            return None;
        }
        let uv_rect = uv_rect(src_point, src_size, src_w, src_h);
        let cm = &self.color_matrix;
        let target = pool.acquire(out_w, out_h)?;
        unsafe {
            self.render_into(cm.program, source, target, out_w, out_h, |gl| {
                gl.uniform_1_i32(cm.u_texture.as_ref(), 0);
                gl.uniform_4_f32_slice(cm.u_uv_rect.as_ref(), &uv_rect);
                gl.uniform_1_f32_slice(cm.u_cm.as_ref(), matrix);
            });
        }
        Some(FilterResult {
            texture: target,
            width: out_w,
            height: out_h,
        })
    }

    /// Separable box blur, ping-ponging between two targets. `blur_x`/`blur_y`
    /// are the per-axis blur amounts (px), applied `num_passes` times each.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn apply_blur(
        &self,
        pool: &mut TexturePool,
        source: glow::Texture,
        src_w: u32,
        src_h: u32,
        src_point: (u32, u32),
        src_size: (u32, u32),
        blur_x: f32,
        blur_y: f32,
        num_passes: u32,
    ) -> Option<FilterResult> {
        let (out_w, out_h) = src_size;
        if out_w == 0 || out_h == 0 || src_w == 0 || src_h == 0 || num_passes == 0 {
            return None;
        }

        // The fused sampling relies on bilinear filtering of the source.
        unsafe {
            self.gl.bind_texture(glow::TEXTURE_2D, Some(source));
            set_sampling(&self.gl, glow::LINEAR);
        }

        let targets = match (pool.acquire(out_w, out_h), pool.acquire(out_w, out_h)) {
            (Some(a), Some(b)) => [a, b],
            (a, b) => {
                if let Some(a) = a {
                    pool.release(a, out_w, out_h);
                }
                if let Some(b) = b {
                    pool.release(b, out_w, out_h);
                }
                return None;
            }
        };

        let bp = &self.blur;
        let mut cur_src = source;
        let mut cur_w = src_w as f32;
        let mut cur_h = src_h as f32;
        let mut cur_uv = uv_rect(src_point, src_size, src_w, src_h);
        let mut write_idx = 0usize;
        let mut result: Option<glow::Texture> = None;

        for _ in 0..num_passes {
            for axis in 0..2 {
                let horizontal = axis == 0;
                let strength = if horizontal { blur_x } else { blur_y };
                let full_size = strength.min(255.0);
                if full_size <= 1.0 {
                    // A width <= 1 is a no-op (it would just sample itself).
                    continue;
                }

                let radius = (full_size - 1.0) / 2.0;
                let m = radius.ceil() - 1.0;
                let alpha = ((radius - m) * 255.0).floor() / 255.0;
                let last_offset = 1.0 / ((1.0 / alpha) + 1.0);
                let last_weight = alpha + 1.0;
                let dir = if horizontal {
                    [1.0 / cur_w, 0.0]
                } else {
                    [0.0, 1.0 / cur_h]
                };
                let m2 = m * 2.0;
                let uv = cur_uv;
                let target = targets[write_idx];

                unsafe {
                    self.render_into(bp.program, cur_src, target, out_w, out_h, |gl| {
                        gl.uniform_1_i32(bp.u_texture.as_ref(), 0);
                        gl.uniform_4_f32_slice(bp.u_uv_rect.as_ref(), &uv);
                        gl.uniform_2_f32(bp.u_dir.as_ref(), dir[0], dir[1]);
                        gl.uniform_1_f32(bp.u_m.as_ref(), m);
                        gl.uniform_1_f32(bp.u_full_size.as_ref(), full_size);
                        gl.uniform_1_f32(bp.u_m2.as_ref(), m2);
                        gl.uniform_1_f32(bp.u_first_weight.as_ref(), alpha);
                        gl.uniform_1_f32(bp.u_last_offset.as_ref(), last_offset);
                        gl.uniform_1_f32(bp.u_last_weight.as_ref(), last_weight);
                    });
                }

                cur_src = target;
                cur_w = out_w as f32;
                cur_h = out_h as f32;
                cur_uv = [0.0, 0.0, 1.0, 1.0];
                result = Some(target);
                write_idx ^= 1;
            }
        }

        match result {
            None => {
                // Every axis was a no-op.
                pool.release(targets[0], out_w, out_h);
                pool.release(targets[1], out_w, out_h);
                None
            }
            Some(result_tex) => {
                let unused = if result_tex == targets[0] {
                    targets[1]
                } else {
                    targets[0]
                };
                pool.release(unused, out_w, out_h);
                Some(FilterResult {
                    texture: result_tex,
                    width: out_w,
                    height: out_h,
                })
            }
        }
    }
}

impl Filters {
    /// Glow / DropShadow: blur the source, then composite the colorized blurred
    /// alpha with the source. `blur_offset` is the shadow offset in pixels (0,0
    /// for glow). Returns a premultiplied result texture of `src_size`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn apply_glow(
        &self,
        pool: &mut TexturePool,
        source: glow::Texture,
        src_w: u32,
        src_h: u32,
        src_point: (u32, u32),
        src_size: (u32, u32),
        color: [f32; 4],
        strength: f32,
        inner: bool,
        knockout: bool,
        composite_source: bool,
        blur_x: f32,
        blur_y: f32,
        num_passes: u32,
        blur_offset: (f32, f32),
    ) -> Option<FilterResult> {
        let (out_w, out_h) = src_size;
        if out_w == 0 || out_h == 0 || src_w == 0 || src_h == 0 {
            return None;
        }

        let blurred = self.apply_blur(
            pool, source, src_w, src_h, src_point, src_size, blur_x, blur_y, num_passes,
        );
        // If the blur was a no-op, composite against the source itself.
        let blurred_tex = blurred.as_ref().map(|b| b.texture).unwrap_or(source);

        let uv = uv_rect(src_point, src_size, src_w, src_h);
        let off = [blur_offset.0 / out_w as f32, blur_offset.1 / out_h as f32];
        let gp = &self.glow;

        let target = match pool.acquire(out_w, out_h) {
            Some(t) => t,
            None => {
                log::error!("Couldn't acquire glow target");
                if let Some(b) = blurred {
                    pool.release(b.texture, b.width, b.height);
                }
                return None;
            }
        };

        let gl = &self.gl;
        unsafe {
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.scratch_fbo));
            gl.framebuffer_texture_2d(
                glow::FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_2D,
                Some(target),
                0,
            );
            gl.viewport(0, 0, out_w as i32, out_h as i32);
            gl.disable(glow::BLEND);
            gl.disable(glow::STENCIL_TEST);
            gl.color_mask(true, true, true, true);

            gl.use_program(Some(gp.program));
            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, Some(source));
            gl.active_texture(glow::TEXTURE1);
            gl.bind_texture(glow::TEXTURE_2D, Some(blurred_tex));

            gl.uniform_1_i32(gp.u_texture.as_ref(), 0);
            gl.uniform_1_i32(gp.u_blurred.as_ref(), 1);
            gl.uniform_4_f32_slice(gp.u_color.as_ref(), &color);
            gl.uniform_1_f32(gp.u_strength.as_ref(), strength);
            gl.uniform_1_i32(gp.u_inner.as_ref(), inner as i32);
            gl.uniform_1_i32(gp.u_knockout.as_ref(), knockout as i32);
            gl.uniform_1_i32(gp.u_composite_source.as_ref(), composite_source as i32);
            gl.uniform_4_f32_slice(gp.u_uv_rect.as_ref(), &uv);
            gl.uniform_2_f32(gp.u_blur_offset.as_ref(), off[0], off[1]);

            gl.bind_vertex_array(Some(self.quad_vao));
            gl.draw_arrays(glow::TRIANGLE_FAN, 0, 4);
            gl.bind_vertex_array(None);

            // Reset texture unit 0 as the active one for subsequent code.
            gl.active_texture(glow::TEXTURE0);
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            gl.enable(glow::BLEND);
        }

        if let Some(b) = blurred {
            pool.release(b.texture, b.width, b.height);
        }
        Some(FilterResult {
            texture: target,
            width: out_w,
            height: out_h,
        })
    }

    /// Bevel: blur the source, then composite a highlight/shadow rim from the
    /// difference of the blurred alpha sampled at +/- the bevel offset.
    /// `highlight`/`shadow` are premultiplied RGBA; `bevel_type` is 0 outer,
    /// 1 inner, 2 full. Returns a premultiplied result texture of `src_size`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn apply_bevel(
        &self,
        pool: &mut TexturePool,
        source: glow::Texture,
        src_w: u32,
        src_h: u32,
        src_point: (u32, u32),
        src_size: (u32, u32),
        highlight: [f32; 4],
        shadow: [f32; 4],
        strength: f32,
        bevel_type: i32,
        knockout: bool,
        blur_x: f32,
        blur_y: f32,
        num_passes: u32,
        blur_offset: (f32, f32),
    ) -> Option<FilterResult> {
        let (out_w, out_h) = src_size;
        if out_w == 0 || out_h == 0 || src_w == 0 || src_h == 0 {
            return None;
        }

        let blurred = self.apply_blur(
            pool, source, src_w, src_h, src_point, src_size, blur_x, blur_y, num_passes,
        );
        // If the blur was a no-op, sample the (unblurred) source alpha instead.
        let blurred_tex = blurred.as_ref().map(|b| b.texture).unwrap_or(source);

        let uv = uv_rect(src_point, src_size, src_w, src_h);
        let off = [blur_offset.0 / out_w as f32, blur_offset.1 / out_h as f32];
        let bp = &self.bevel;

        let target = match pool.acquire(out_w, out_h) {
            Some(t) => t,
            None => {
                log::error!("Couldn't acquire bevel target");
                if let Some(b) = blurred {
                    pool.release(b.texture, b.width, b.height);
                }
                return None;
            }
        };

        let gl = &self.gl;
        unsafe {
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.scratch_fbo));
            gl.framebuffer_texture_2d(
                glow::FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_2D,
                Some(target),
                0,
            );
            gl.viewport(0, 0, out_w as i32, out_h as i32);
            gl.disable(glow::BLEND);
            gl.disable(glow::STENCIL_TEST);
            gl.color_mask(true, true, true, true);

            gl.use_program(Some(bp.program));
            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, Some(source));
            gl.active_texture(glow::TEXTURE1);
            gl.bind_texture(glow::TEXTURE_2D, Some(blurred_tex));

            gl.uniform_1_i32(bp.u_texture.as_ref(), 0);
            gl.uniform_1_i32(bp.u_blurred.as_ref(), 1);
            gl.uniform_4_f32_slice(bp.u_highlight.as_ref(), &highlight);
            gl.uniform_4_f32_slice(bp.u_shadow.as_ref(), &shadow);
            gl.uniform_1_f32(bp.u_strength.as_ref(), strength);
            gl.uniform_1_i32(bp.u_bevel_type.as_ref(), bevel_type);
            gl.uniform_1_i32(bp.u_knockout.as_ref(), knockout as i32);
            gl.uniform_4_f32_slice(bp.u_uv_rect.as_ref(), &uv);
            gl.uniform_2_f32(bp.u_blur_offset.as_ref(), off[0], off[1]);

            gl.bind_vertex_array(Some(self.quad_vao));
            gl.draw_arrays(glow::TRIANGLE_FAN, 0, 4);
            gl.bind_vertex_array(None);

            // Reset texture unit 0 as the active one for subsequent code.
            gl.active_texture(glow::TEXTURE0);
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            gl.enable(glow::BLEND);
        }

        if let Some(b) = blurred {
            pool.release(b.texture, b.width, b.height);
        }
        Some(FilterResult {
            texture: target,
            width: out_w,
            height: out_h,
        })
    }
}

impl Filters {
    /// Flash ConvolutionFilter: an N x M kernel over the source region. `kernel`
    /// is row-major, padded to [`MAX_CONVOLUTION_TAPS`]; `default_color` is
    /// premultiplied. Returns a premultiplied result texture of `src_size`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn apply_convolution(
        &self,
        pool: &mut TexturePool,
        source: glow::Texture,
        src_w: u32,
        src_h: u32,
        src_point: (u32, u32),
        src_size: (u32, u32),
        kernel: &[f32; MAX_CONVOLUTION_TAPS],
        cols: f32,
        rows: f32,
        divisor: f32,
        bias: f32,
        default_color: [f32; 4],
        clamp: bool,
        preserve_alpha: bool,
    ) -> Option<FilterResult> {
        let (out_w, out_h) = src_size;
        if out_w == 0 || out_h == 0 || src_w == 0 || src_h == 0 {
            return None;
        }
        let uv = uv_rect(src_point, src_size, src_w, src_h);
        let texel = [1.0 / src_w as f32, 1.0 / src_h as f32];
        let cp = &self.convolution;
        let target = pool.acquire(out_w, out_h)?;
        unsafe {
            self.render_into(cp.program, source, target, out_w, out_h, |gl| {
                gl.uniform_1_i32(cp.u_texture.as_ref(), 0);
                gl.uniform_4_f32_slice(cp.u_uv_rect.as_ref(), &uv);
                gl.uniform_2_f32(cp.u_texel.as_ref(), texel[0], texel[1]);
                gl.uniform_1_f32_slice(cp.u_kernel.as_ref(), kernel);
                gl.uniform_1_f32(cp.u_cols.as_ref(), cols);
                gl.uniform_1_f32(cp.u_rows.as_ref(), rows);
                gl.uniform_1_f32(cp.u_divisor.as_ref(), divisor);
                gl.uniform_1_f32(cp.u_bias.as_ref(), bias);
                gl.uniform_4_f32_slice(cp.u_default_color.as_ref(), &default_color);
                gl.uniform_1_f32(cp.u_clamp.as_ref(), if clamp { 1.0 } else { 0.0 });
                gl.uniform_1_f32(
                    cp.u_preserve_alpha.as_ref(),
                    if preserve_alpha { 1.0 } else { 0.0 },
                );
            });
        }
        Some(FilterResult {
            texture: target,
            width: out_w,
            height: out_h,
        })
    }

    /// Flash DisplacementMapFilter: `map` (unit 1) offsets sampling of `source`
    /// (unit 0). `color` is straight RGBA. Returns a premultiplied result of
    /// `src_size`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn apply_displacement(
        &self,
        pool: &mut TexturePool,
        source: glow::Texture,
        src_w: u32,
        src_h: u32,
        src_point: (u32, u32),
        src_size: (u32, u32),
        map: glow::Texture,
        map_w: u32,
        map_h: u32,
        color: [f32; 4],
        components: (f32, f32),
        mode: f32,
        scale: (f32, f32),
        offset: (f32, f32),
        viewscale: (f32, f32),
    ) -> Option<FilterResult> {
        let (out_w, out_h) = src_size;
        if out_w == 0 || out_h == 0 || src_w == 0 || src_h == 0 || map_w == 0 || map_h == 0 {
            return None;
        }
        let uv = uv_rect(src_point, src_size, src_w, src_h);
        let dp = &self.displacement;
        let target = pool.acquire(out_w, out_h)?;
        unsafe {
            self.render_into(dp.program, source, target, out_w, out_h, |gl| {
                // Bind the map to texture unit 1; leave unit 0 active for the
                // source that `render_into` already bound.
                gl.active_texture(glow::TEXTURE1);
                gl.bind_texture(glow::TEXTURE_2D, Some(map));
                gl.active_texture(glow::TEXTURE0);

                gl.uniform_1_i32(dp.u_source.as_ref(), 0);
                gl.uniform_1_i32(dp.u_map.as_ref(), 1);
                gl.uniform_4_f32_slice(dp.u_uv_rect.as_ref(), &uv);
                gl.uniform_4_f32_slice(dp.u_color.as_ref(), &color);
                gl.uniform_2_f32(dp.u_components.as_ref(), components.0, components.1);
                gl.uniform_1_f32(dp.u_mode.as_ref(), mode);
                gl.uniform_2_f32(dp.u_scale.as_ref(), scale.0, scale.1);
                gl.uniform_2_f32(dp.u_source_size.as_ref(), out_w as f32, out_h as f32);
                gl.uniform_2_f32(dp.u_map_size.as_ref(), map_w as f32, map_h as f32);
                gl.uniform_2_f32(dp.u_offset.as_ref(), offset.0, offset.1);
                gl.uniform_2_f32(dp.u_viewscale.as_ref(), viewscale.0, viewscale.1);
            });
        }
        Some(FilterResult {
            texture: target,
            width: out_w,
            height: out_h,
        })
    }

    /// Uploads a 256-entry premultiplied RGBA ramp into the reused ramp texture.
    unsafe fn upload_ramp(&self, ramp: &[u8]) {
        let gl = &self.gl;
        unsafe {
            gl.bind_texture(glow::TEXTURE_2D, Some(self.ramp_texture));
            gl.tex_sub_image_2d(
                glow::TEXTURE_2D,
                0,
                0,
                0,
                GRADIENT_RAMP_SIZE as i32,
                1,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(Some(ramp)),
            );
        }
    }

    /// GradientGlow: like glow, but the color comes from `ramp` (256x1
    /// premultiplied) indexed by intensity. `gtype` is 0 outer, 1 inner, 2 full.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn apply_gradient_glow(
        &self,
        pool: &mut TexturePool,
        source: glow::Texture,
        src_w: u32,
        src_h: u32,
        src_point: (u32, u32),
        src_size: (u32, u32),
        ramp: &[u8],
        strength: f32,
        gtype: i32,
        knockout: bool,
        composite_source: bool,
        blur_x: f32,
        blur_y: f32,
        num_passes: u32,
        blur_offset: (f32, f32),
    ) -> Option<FilterResult> {
        let (out_w, out_h) = src_size;
        if out_w == 0 || out_h == 0 || src_w == 0 || src_h == 0 {
            return None;
        }
        let blurred = self.apply_blur(
            pool, source, src_w, src_h, src_point, src_size, blur_x, blur_y, num_passes,
        );
        let blurred_tex = blurred.as_ref().map(|b| b.texture).unwrap_or(source);
        let uv = uv_rect(src_point, src_size, src_w, src_h);
        let off = [blur_offset.0 / out_w as f32, blur_offset.1 / out_h as f32];
        let gp = &self.gradient_glow;

        let target = match pool.acquire(out_w, out_h) {
            Some(t) => t,
            None => {
                if let Some(b) = blurred {
                    pool.release(b.texture, b.width, b.height);
                }
                return None;
            }
        };

        let gl = &self.gl;
        unsafe {
            self.upload_ramp(ramp);
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.scratch_fbo));
            gl.framebuffer_texture_2d(
                glow::FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_2D,
                Some(target),
                0,
            );
            gl.viewport(0, 0, out_w as i32, out_h as i32);
            gl.disable(glow::BLEND);
            gl.disable(glow::STENCIL_TEST);
            gl.color_mask(true, true, true, true);

            gl.use_program(Some(gp.program));
            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, Some(source));
            gl.active_texture(glow::TEXTURE1);
            gl.bind_texture(glow::TEXTURE_2D, Some(blurred_tex));
            gl.active_texture(glow::TEXTURE2);
            gl.bind_texture(glow::TEXTURE_2D, Some(self.ramp_texture));

            gl.uniform_1_i32(gp.u_texture.as_ref(), 0);
            gl.uniform_1_i32(gp.u_blurred.as_ref(), 1);
            gl.uniform_1_i32(gp.u_ramp.as_ref(), 2);
            gl.uniform_1_f32(gp.u_strength.as_ref(), strength);
            gl.uniform_1_i32(gp.u_type.as_ref(), gtype);
            gl.uniform_1_i32(gp.u_knockout.as_ref(), knockout as i32);
            gl.uniform_1_i32(gp.u_composite_source.as_ref(), composite_source as i32);
            gl.uniform_4_f32_slice(gp.u_uv_rect.as_ref(), &uv);
            gl.uniform_2_f32(gp.u_blur_offset.as_ref(), off[0], off[1]);

            gl.bind_vertex_array(Some(self.quad_vao));
            gl.draw_arrays(glow::TRIANGLE_FAN, 0, 4);
            gl.bind_vertex_array(None);

            gl.active_texture(glow::TEXTURE0);
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            gl.enable(glow::BLEND);
        }

        if let Some(b) = blurred {
            pool.release(b.texture, b.width, b.height);
        }
        Some(FilterResult {
            texture: target,
            width: out_w,
            height: out_h,
        })
    }

    /// GradientBevel: like bevel, but the rim color comes from `ramp` (256x1
    /// premultiplied). `bevel_type` is 0 outer, 1 inner, 2 full.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn apply_gradient_bevel(
        &self,
        pool: &mut TexturePool,
        source: glow::Texture,
        src_w: u32,
        src_h: u32,
        src_point: (u32, u32),
        src_size: (u32, u32),
        ramp: &[u8],
        strength: f32,
        bevel_type: i32,
        knockout: bool,
        blur_x: f32,
        blur_y: f32,
        num_passes: u32,
        blur_offset: (f32, f32),
    ) -> Option<FilterResult> {
        let (out_w, out_h) = src_size;
        if out_w == 0 || out_h == 0 || src_w == 0 || src_h == 0 {
            return None;
        }
        let blurred = self.apply_blur(
            pool, source, src_w, src_h, src_point, src_size, blur_x, blur_y, num_passes,
        );
        let blurred_tex = blurred.as_ref().map(|b| b.texture).unwrap_or(source);
        let uv = uv_rect(src_point, src_size, src_w, src_h);
        let off = [blur_offset.0 / out_w as f32, blur_offset.1 / out_h as f32];
        let gp = &self.gradient_bevel;

        let target = match pool.acquire(out_w, out_h) {
            Some(t) => t,
            None => {
                if let Some(b) = blurred {
                    pool.release(b.texture, b.width, b.height);
                }
                return None;
            }
        };

        let gl = &self.gl;
        unsafe {
            self.upload_ramp(ramp);
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.scratch_fbo));
            gl.framebuffer_texture_2d(
                glow::FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_2D,
                Some(target),
                0,
            );
            gl.viewport(0, 0, out_w as i32, out_h as i32);
            gl.disable(glow::BLEND);
            gl.disable(glow::STENCIL_TEST);
            gl.color_mask(true, true, true, true);

            gl.use_program(Some(gp.program));
            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, Some(source));
            gl.active_texture(glow::TEXTURE1);
            gl.bind_texture(glow::TEXTURE_2D, Some(blurred_tex));
            gl.active_texture(glow::TEXTURE2);
            gl.bind_texture(glow::TEXTURE_2D, Some(self.ramp_texture));

            gl.uniform_1_i32(gp.u_texture.as_ref(), 0);
            gl.uniform_1_i32(gp.u_blurred.as_ref(), 1);
            gl.uniform_1_i32(gp.u_ramp.as_ref(), 2);
            gl.uniform_1_f32(gp.u_strength.as_ref(), strength);
            gl.uniform_1_i32(gp.u_bevel_type.as_ref(), bevel_type);
            gl.uniform_1_i32(gp.u_knockout.as_ref(), knockout as i32);
            gl.uniform_4_f32_slice(gp.u_uv_rect.as_ref(), &uv);
            gl.uniform_2_f32(gp.u_blur_offset.as_ref(), off[0], off[1]);

            gl.bind_vertex_array(Some(self.quad_vao));
            gl.draw_arrays(glow::TRIANGLE_FAN, 0, 4);
            gl.bind_vertex_array(None);

            gl.active_texture(glow::TEXTURE0);
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            gl.enable(glow::BLEND);
        }

        if let Some(b) = blurred {
            pool.release(b.texture, b.width, b.height);
        }
        Some(FilterResult {
            texture: target,
            width: out_w,
            height: out_h,
        })
    }
}

fn uv_rect(src_point: (u32, u32), src_size: (u32, u32), src_w: u32, src_h: u32) -> [f32; 4] {
    [
        src_point.0 as f32 / src_w as f32,
        src_point.1 as f32 / src_h as f32,
        src_size.0 as f32 / src_w as f32,
        src_size.1 as f32 / src_h as f32,
    ]
}

impl Drop for Filters {
    fn drop(&mut self) {
        unsafe {
            self.gl.delete_program(self.color_matrix.program);
            self.gl.delete_program(self.blur.program);
            self.gl.delete_program(self.glow.program);
            self.gl.delete_program(self.bevel.program);
            self.gl.delete_program(self.blend.program);
            self.gl.delete_program(self.convolution.program);
            self.gl.delete_program(self.displacement.program);
            self.gl.delete_program(self.gradient_glow.program);
            self.gl.delete_program(self.gradient_bevel.program);
            self.gl.delete_texture(self.ramp_texture);
            self.gl.delete_buffer(self.quad_vbo);
            self.gl.delete_vertex_array(self.quad_vao);
            self.gl.delete_framebuffer(self.scratch_fbo);
        }
    }
}

impl ColorMatrixProgram {
    fn new(gl: &glow::Context, program: glow::Program) -> Self {
        unsafe {
            Self {
                u_texture: gl.get_uniform_location(program, "u_texture"),
                u_uv_rect: gl.get_uniform_location(program, "u_uv_rect"),
                u_cm: gl.get_uniform_location(program, "u_cm"),
                program,
            }
        }
    }
}

impl BlurProgram {
    fn new(gl: &glow::Context, program: glow::Program) -> Self {
        unsafe {
            Self {
                u_texture: gl.get_uniform_location(program, "u_texture"),
                u_uv_rect: gl.get_uniform_location(program, "u_uv_rect"),
                u_dir: gl.get_uniform_location(program, "u_dir"),
                u_m: gl.get_uniform_location(program, "u_m"),
                u_full_size: gl.get_uniform_location(program, "u_full_size"),
                u_m2: gl.get_uniform_location(program, "u_m2"),
                u_first_weight: gl.get_uniform_location(program, "u_first_weight"),
                u_last_offset: gl.get_uniform_location(program, "u_last_offset"),
                u_last_weight: gl.get_uniform_location(program, "u_last_weight"),
                program,
            }
        }
    }
}

impl GlowProgram {
    fn new(gl: &glow::Context, program: glow::Program) -> Self {
        unsafe {
            Self {
                u_texture: gl.get_uniform_location(program, "u_texture"),
                u_blurred: gl.get_uniform_location(program, "u_blurred"),
                u_color: gl.get_uniform_location(program, "u_color"),
                u_strength: gl.get_uniform_location(program, "u_strength"),
                u_inner: gl.get_uniform_location(program, "u_inner"),
                u_knockout: gl.get_uniform_location(program, "u_knockout"),
                u_composite_source: gl.get_uniform_location(program, "u_composite_source"),
                u_uv_rect: gl.get_uniform_location(program, "u_uv_rect"),
                u_blur_offset: gl.get_uniform_location(program, "u_blur_offset"),
                program,
            }
        }
    }
}

impl BevelProgram {
    fn new(gl: &glow::Context, program: glow::Program) -> Self {
        unsafe {
            Self {
                u_texture: gl.get_uniform_location(program, "u_texture"),
                u_blurred: gl.get_uniform_location(program, "u_blurred"),
                u_highlight: gl.get_uniform_location(program, "u_highlight"),
                u_shadow: gl.get_uniform_location(program, "u_shadow"),
                u_strength: gl.get_uniform_location(program, "u_strength"),
                u_bevel_type: gl.get_uniform_location(program, "u_bevel_type"),
                u_knockout: gl.get_uniform_location(program, "u_knockout"),
                u_uv_rect: gl.get_uniform_location(program, "u_uv_rect"),
                u_blur_offset: gl.get_uniform_location(program, "u_blur_offset"),
                program,
            }
        }
    }
}

impl BlendProgram {
    fn new(gl: &glow::Context, program: glow::Program) -> Self {
        unsafe {
            Self {
                u_current: gl.get_uniform_location(program, "u_current"),
                u_parent: gl.get_uniform_location(program, "u_parent"),
                u_blend_mode: gl.get_uniform_location(program, "u_blend_mode"),
                program,
            }
        }
    }
}

impl ConvolutionProgram {
    fn new(gl: &glow::Context, program: glow::Program) -> Self {
        unsafe {
            Self {
                u_texture: gl.get_uniform_location(program, "u_texture"),
                u_uv_rect: gl.get_uniform_location(program, "u_uv_rect"),
                u_texel: gl.get_uniform_location(program, "u_texel"),
                u_kernel: gl.get_uniform_location(program, "u_kernel"),
                u_cols: gl.get_uniform_location(program, "u_cols"),
                u_rows: gl.get_uniform_location(program, "u_rows"),
                u_divisor: gl.get_uniform_location(program, "u_divisor"),
                u_bias: gl.get_uniform_location(program, "u_bias"),
                u_default_color: gl.get_uniform_location(program, "u_default_color"),
                u_clamp: gl.get_uniform_location(program, "u_clamp"),
                u_preserve_alpha: gl.get_uniform_location(program, "u_preserve_alpha"),
                program,
            }
        }
    }
}

impl DisplacementProgram {
    fn new(gl: &glow::Context, program: glow::Program) -> Self {
        unsafe {
            Self {
                u_source: gl.get_uniform_location(program, "u_source"),
                u_map: gl.get_uniform_location(program, "u_map"),
                u_uv_rect: gl.get_uniform_location(program, "u_uv_rect"),
                u_color: gl.get_uniform_location(program, "u_color"),
                u_components: gl.get_uniform_location(program, "u_components"),
                u_mode: gl.get_uniform_location(program, "u_mode"),
                u_scale: gl.get_uniform_location(program, "u_scale"),
                u_source_size: gl.get_uniform_location(program, "u_source_size"),
                u_map_size: gl.get_uniform_location(program, "u_map_size"),
                u_offset: gl.get_uniform_location(program, "u_offset"),
                u_viewscale: gl.get_uniform_location(program, "u_viewscale"),
                program,
            }
        }
    }
}

impl GradientGlowProgram {
    fn new(gl: &glow::Context, program: glow::Program) -> Self {
        unsafe {
            Self {
                u_texture: gl.get_uniform_location(program, "u_texture"),
                u_blurred: gl.get_uniform_location(program, "u_blurred"),
                u_ramp: gl.get_uniform_location(program, "u_ramp"),
                u_strength: gl.get_uniform_location(program, "u_strength"),
                u_type: gl.get_uniform_location(program, "u_type"),
                u_knockout: gl.get_uniform_location(program, "u_knockout"),
                u_composite_source: gl.get_uniform_location(program, "u_composite_source"),
                u_uv_rect: gl.get_uniform_location(program, "u_uv_rect"),
                u_blur_offset: gl.get_uniform_location(program, "u_blur_offset"),
                program,
            }
        }
    }
}

impl GradientBevelProgram {
    fn new(gl: &glow::Context, program: glow::Program) -> Self {
        unsafe {
            Self {
                u_texture: gl.get_uniform_location(program, "u_texture"),
                u_blurred: gl.get_uniform_location(program, "u_blurred"),
                u_ramp: gl.get_uniform_location(program, "u_ramp"),
                u_strength: gl.get_uniform_location(program, "u_strength"),
                u_bevel_type: gl.get_uniform_location(program, "u_bevel_type"),
                u_knockout: gl.get_uniform_location(program, "u_knockout"),
                u_uv_rect: gl.get_uniform_location(program, "u_uv_rect"),
                u_blur_offset: gl.get_uniform_location(program, "u_blur_offset"),
                program,
            }
        }
    }
}
