use thiserror::Error;

/// Errors that can occur while constructing or driving the GL backend.
///
/// This is portable across native OpenGL and WebGL — glow surfaces failures as
/// `String`s rather than the web-specific `JsValue`s the old `webgl` backend
/// used.
#[derive(Error, Debug)]
pub enum Error {
    #[error("Couldn't create GL context")]
    CantCreateGLContext,

    #[error("Couldn't create frame buffer: {0}")]
    UnableToCreateFrameBuffer(String),

    #[error("Couldn't create program: {0}")]
    UnableToCreateProgram(String),

    #[error("Couldn't create texture: {0}")]
    UnableToCreateTexture(String),

    #[error("Couldn't create shader: {0}")]
    UnableToCreateShader(String),

    #[error("Couldn't create render buffer: {0}")]
    UnableToCreateRenderBuffer(String),

    #[error("Couldn't create vertex array object: {0}")]
    UnableToCreateVAO(String),

    #[error("Couldn't create buffer: {0}")]
    UnableToCreateBuffer(String),

    #[error("OES_element_index_uint extension not available")]
    OESExtensionNotFound,

    #[error("Couldn't compile shader: {0}")]
    CompilingShader(String),

    #[error("Couldn't link shader program: {0}")]
    LinkingShaderProgram(String),

    #[error("GL Error in {0}: {1}")]
    GLError(&'static str, u32),

    #[cfg(target_family = "wasm")]
    #[error("Javascript error: {0}")]
    JavascriptError(#[from] ruffle_web_common::JsError),
}
