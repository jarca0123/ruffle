//! Shader compilation, including the per-target preamble that lets a single set
//! of GLSL ES 1.00 shader bodies compile on WebGL1, WebGL2 and desktop OpenGL.

use crate::error::Error;
use glow::HasContext as _;

/// Preamble for WebGL / OpenGL ES contexts — exactly what the original `.glsl`
/// files carried, so behavior on the web is unchanged.
const ES_PREAMBLE: &str = "#version 100\n\
#ifdef GL_FRAGMENT_PRECISION_HIGH\n\
precision highp float;\n\
#else\n\
precision mediump float;\n\
#endif\n";

/// Vertex preamble for desktop OpenGL: map the ES-1.00 keywords used by the
/// bodies onto GLSL 1.30.
const DESKTOP_VERTEX_PREAMBLE: &str = "#version 130\n\
#define attribute in\n\
#define varying out\n\
#define texture2D texture\n";

/// Fragment preamble for desktop OpenGL.
const DESKTOP_FRAGMENT_PREAMBLE: &str = "#version 130\n\
#define varying in\n\
#define texture2D texture\n\
out vec4 _ruffle_FragColor;\n\
#define gl_FragColor _ruffle_FragColor\n";

fn preamble(is_embedded: bool, stage: u32) -> &'static str {
    if is_embedded {
        ES_PREAMBLE
    } else if stage == glow::VERTEX_SHADER {
        DESKTOP_VERTEX_PREAMBLE
    } else {
        DESKTOP_FRAGMENT_PREAMBLE
    }
}

/// Compiles a single shader stage, prepending the target-appropriate preamble.
pub(crate) fn compile_shader(
    gl: &glow::Context,
    is_embedded: bool,
    stage: u32,
    body: &str,
) -> Result<glow::Shader, Error> {
    let source = format!("{}\n{}", preamble(is_embedded, stage), body);
    // SAFETY: a GL context is current on this thread.
    unsafe {
        let shader = gl
            .create_shader(stage)
            .map_err(Error::UnableToCreateShader)?;
        gl.shader_source(shader, &source);
        gl.compile_shader(shader);
        if !gl.get_shader_compile_status(shader) {
            let log = gl.get_shader_info_log(shader);
            gl.delete_shader(shader);
            return Err(Error::CompilingShader(log));
        }
        Ok(shader)
    }
}

// These should match the uniform names in the shaders.
const NUM_UNIFORMS: usize = 12;
const UNIFORM_NAMES: [&str; NUM_UNIFORMS] = [
    "world_matrix",
    "view_matrix",
    "mult_color",
    "add_color",
    "u_matrix",
    "u_gradient_type",
    "u_ratios",
    "u_colors",
    "u_repeat_mode",
    "u_focal_point",
    "u_interpolation",
    "u_texture",
];

#[derive(Copy, Clone)]
pub(crate) enum ShaderUniform {
    WorldMatrix = 0,
    ViewMatrix,
    MultColor,
    AddColor,
    TextureMatrix,
    GradientType,
    GradientRatios,
    GradientColors,
    GradientRepeatMode,
    GradientFocalPoint,
    GradientInterpolation,
    BitmapTexture,
}

/// Because the shaders are currently simple and few in number, we use a
/// straightforward shader model: an enum of every possible uniform, and each
/// program grabs the location of each uniform it has.
pub(crate) struct ShaderProgram {
    pub program: glow::Program,
    uniforms: [Option<glow::UniformLocation>; NUM_UNIFORMS],
    pub vertex_position_location: Option<u32>,
    pub vertex_color_location: Option<u32>,
    pub num_vertex_attributes: u32,
}

impl ShaderProgram {
    pub(crate) fn new(
        gl: &glow::Context,
        vertex_shader: glow::Shader,
        fragment_shader: glow::Shader,
    ) -> Result<Self, Error> {
        // SAFETY: a GL context is current on this thread.
        unsafe {
            let program = gl.create_program().map_err(Error::UnableToCreateProgram)?;
            gl.attach_shader(program, vertex_shader);
            gl.attach_shader(program, fragment_shader);
            gl.link_program(program);
            if !gl.get_program_link_status(program) {
                let msg = gl.get_program_info_log(program);
                log::error!("Error linking shader program: {msg}");
                return Err(Error::LinkingShaderProgram(msg));
            }

            // `from_fn` avoids requiring `UniformLocation: Copy` (it isn't on
            // the web backend).
            let uniforms: [Option<glow::UniformLocation>; NUM_UNIFORMS] =
                std::array::from_fn(|i| gl.get_uniform_location(program, UNIFORM_NAMES[i]));

            // Vertex attribute locations are not guaranteed to be in a fixed
            // position between shaders in GLSL ES (no layout qualifiers), so we
            // query them per-program.
            let vertex_position_location = gl.get_attrib_location(program, "position");
            let vertex_color_location = gl.get_attrib_location(program, "color");
            let num_vertex_attributes =
                vertex_position_location.is_some() as u32 + vertex_color_location.is_some() as u32;

            Ok(ShaderProgram {
                program,
                uniforms,
                vertex_position_location,
                vertex_color_location,
                num_vertex_attributes,
            })
        }
    }

    fn loc(&self, uniform: ShaderUniform) -> Option<&glow::UniformLocation> {
        self.uniforms[uniform as usize].as_ref()
    }

    pub(crate) fn uniform1f(&self, gl: &glow::Context, uniform: ShaderUniform, value: f32) {
        unsafe { gl.uniform_1_f32(self.loc(uniform), value) }
    }

    pub(crate) fn uniform1fv(&self, gl: &glow::Context, uniform: ShaderUniform, values: &[f32]) {
        unsafe { gl.uniform_1_f32_slice(self.loc(uniform), values) }
    }

    pub(crate) fn uniform1i(&self, gl: &glow::Context, uniform: ShaderUniform, value: i32) {
        unsafe { gl.uniform_1_i32(self.loc(uniform), value) }
    }

    pub(crate) fn uniform4fv(&self, gl: &glow::Context, uniform: ShaderUniform, values: &[f32]) {
        unsafe { gl.uniform_4_f32_slice(self.loc(uniform), values) }
    }

    pub(crate) fn uniform_matrix3fv(
        &self,
        gl: &glow::Context,
        uniform: ShaderUniform,
        values: &[[f32; 3]; 3],
    ) {
        unsafe {
            gl.uniform_matrix_3_f32_slice(self.loc(uniform), false, bytemuck::cast_slice(values))
        }
    }

    pub(crate) fn uniform_matrix4fv(
        &self,
        gl: &glow::Context,
        uniform: ShaderUniform,
        values: &[[f32; 4]; 4],
    ) {
        unsafe {
            gl.uniform_matrix_4_f32_slice(self.loc(uniform), false, bytemuck::cast_slice(values))
        }
    }
}
