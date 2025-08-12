//! Safe wrappers around OpenGL operations
//!
//! This module provides safe wrappers around unsafe OpenGL operations,
//! with proper error checking and parameter validation.

use crate::error::{GLError, GraphicsApiError, MiniquadError};
use crate::native::gl::*;
use crate::graphics::*;

/// Maximum number of buffers that can be generated at once
const MAX_BUFFERS: i32 = 1024;

/// Maximum number of textures that can be generated at once  
const MAX_TEXTURES: i32 = 1024;

/// Parameters for texture image upload
#[derive(Debug, Clone)]
pub struct TexImageParams {
    pub target: GLenum,
    pub level: i32,
    pub internal_format: i32,
    pub width: i32,
    pub height: i32,
    pub border: i32,
    pub format: GLenum,
    pub type_: GLenum,
}

/// Safe OpenGL wrapper with error checking
pub struct SafeGL;

impl SafeGL {
    /// Check for OpenGL errors and convert to our error type
    pub fn check_error() -> Result<(), GLError> {
        let error = unsafe { glGetError() };
        match error {
            GL_NO_ERROR => Ok(()),
            _ => Err(GLError::from_gl_enum(error)),
        }
    }

    /// Check for OpenGL errors with context message
    pub fn check_error_with_context(context: &str) -> Result<(), MiniquadError> {
        Self::check_error().map_err(|e| {
            MiniquadError::GraphicsApi(GraphicsApiError::OpenGL(e))
        }).map_err(|e| {
            eprintln!("OpenGL error in {}: {}", context, e);
            e
        })
    }

    /// Safely generate buffers with validation
    pub fn gen_buffers(n: i32) -> Result<Vec<GLuint>, MiniquadError> {
        if n <= 0 {
            return Err(MiniquadError::InvalidParameter(
                "Buffer count must be positive".to_string()
            ));
        }
        if n > MAX_BUFFERS {
            return Err(MiniquadError::InvalidParameter(
                format!("Cannot generate more than {} buffers at once", MAX_BUFFERS)
            ));
        }

        let mut buffers = vec![0; n as usize];
        unsafe { glGenBuffers(n, buffers.as_mut_ptr()) };
        Self::check_error_with_context("glGenBuffers")?;
        
        // Verify all buffers were generated successfully
        if buffers.contains(&0) {
            return Err(MiniquadError::GraphicsApi(GraphicsApiError::OpenGL(
                GLError::Unknown(0)
            )));
        }
        
        Ok(buffers)
    }

    /// Safely generate a single buffer
    pub fn gen_buffer() -> Result<GLuint, MiniquadError> {
        let buffers = Self::gen_buffers(1)?;
        Ok(buffers[0])
    }

    /// Safely generate textures with validation
    pub fn gen_textures(n: i32) -> Result<Vec<GLuint>, MiniquadError> {
        if n <= 0 {
            return Err(MiniquadError::InvalidParameter(
                "Texture count must be positive".to_string()
            ));
        }
        if n > MAX_TEXTURES {
            return Err(MiniquadError::InvalidParameter(
                format!("Cannot generate more than {} textures at once", MAX_TEXTURES)
            ));
        }

        let mut textures = vec![0; n as usize];
        unsafe { glGenTextures(n, textures.as_mut_ptr()) };
        Self::check_error_with_context("glGenTextures")?;
        
        // Verify all textures were generated successfully
        if textures.contains(&0) {
            return Err(MiniquadError::GraphicsApi(GraphicsApiError::OpenGL(
                GLError::Unknown(0)
            )));
        }
        
        Ok(textures)
    }

    /// Safely generate a single texture
    pub fn gen_texture() -> Result<GLuint, MiniquadError> {
        let textures = Self::gen_textures(1)?;
        Ok(textures[0])
    }

    /// Safely bind buffer with validation
    pub fn bind_buffer(target: GLenum, buffer: GLuint) -> Result<(), MiniquadError> {
        // Validate target
        match target {
            GL_ARRAY_BUFFER | GL_ELEMENT_ARRAY_BUFFER => {},
            _ => return Err(MiniquadError::InvalidParameter(
                format!("Invalid buffer target: 0x{:X}", target)
            )),
        }

        unsafe { glBindBuffer(target, buffer) };
        Self::check_error_with_context("glBindBuffer")
    }

    /// Safely bind texture with validation
    pub fn bind_texture(target: GLenum, texture: GLuint) -> Result<(), MiniquadError> {
        // Validate target
        match target {
            GL_TEXTURE_2D | GL_TEXTURE_CUBE_MAP => {},
            _ => return Err(MiniquadError::InvalidParameter(
                format!("Invalid texture target: 0x{:X}", target)
            )),
        }

        unsafe { glBindTexture(target, texture) };
        Self::check_error_with_context("glBindTexture")
    }

    /// Safely upload buffer data with validation
    /// 
    /// # Safety
    /// The caller must ensure that `data` points to valid memory of at least `size` bytes,
    /// or is null for uninitialized buffer allocation.
    pub unsafe fn buffer_data(
        target: GLenum,
        size: GLsizeiptr,
        data: *const std::ffi::c_void,
        usage: GLenum,
    ) -> Result<(), MiniquadError> {
        // Validate parameters
        if size < 0 {
            return Err(MiniquadError::InvalidParameter(
                "Buffer size cannot be negative".to_string()
            ));
        }

        match target {
            GL_ARRAY_BUFFER | GL_ELEMENT_ARRAY_BUFFER => {},
            _ => return Err(MiniquadError::InvalidParameter(
                format!("Invalid buffer target: 0x{:X}", target)
            )),
        }

        match usage {
            GL_STATIC_DRAW | GL_DYNAMIC_DRAW | GL_STREAM_DRAW => {},
            _ => return Err(MiniquadError::InvalidParameter(
                format!("Invalid buffer usage: 0x{:X}", usage)
            )),
        }

        glBufferData(target, size, data, usage);
        Self::check_error_with_context("glBufferData")
    }


    /// Safely upload texture data with validation
    /// 
    /// # Safety
    /// The caller must ensure that `pixels` points to valid texture data of the appropriate
    /// size for the given dimensions and format, or is null for uninitialized allocation.
    pub unsafe fn tex_image_2d(
        params: TexImageParams,
        pixels: *const std::ffi::c_void,
    ) -> Result<(), MiniquadError> {
        // Validate parameters
        if params.width <= 0 || params.height <= 0 {
            return Err(MiniquadError::InvalidParameter(
                "Texture dimensions must be positive".to_string()
            ));
        }

        if params.level < 0 {
            return Err(MiniquadError::InvalidParameter(
                "Mipmap level cannot be negative".to_string()
            ));
        }

        if params.border != 0 {
            return Err(MiniquadError::InvalidParameter(
                "Border must be 0 in OpenGL ES".to_string()
            ));
        }

        // Validate texture target
        match params.target {
            GL_TEXTURE_2D | 
            GL_TEXTURE_CUBE_MAP_POSITIVE_X |
            GL_TEXTURE_CUBE_MAP_NEGATIVE_X |
            GL_TEXTURE_CUBE_MAP_POSITIVE_Y |
            GL_TEXTURE_CUBE_MAP_NEGATIVE_Y |
            GL_TEXTURE_CUBE_MAP_POSITIVE_Z |
            GL_TEXTURE_CUBE_MAP_NEGATIVE_Z => {},
            _ => return Err(MiniquadError::InvalidParameter(
                format!("Invalid texture target: 0x{:X}", params.target)
            )),
        }

        glTexImage2D(
            params.target, params.level, params.internal_format, 
            params.width, params.height, params.border, 
            params.format, params.type_, pixels
        );
        
        Self::check_error_with_context("glTexImage2D")
    }

    /// Safely create and compile shader with validation
    pub fn create_shader(shader_type: GLenum, source: &str) -> Result<GLuint, MiniquadError> {
        // Validate shader type
        match shader_type {
            GL_VERTEX_SHADER | GL_FRAGMENT_SHADER => {},
            _ => return Err(MiniquadError::InvalidParameter(
                format!("Invalid shader type: 0x{:X}", shader_type)
            )),
        }

        // Check for null bytes in source
        if source.contains('\0') {
            return Err(MiniquadError::InvalidParameter(
                "Shader source cannot contain null bytes".to_string()
            ));
        }

        let shader = unsafe { glCreateShader(shader_type) };
        if shader == 0 {
            Self::check_error_with_context("glCreateShader")?;
            return Err(MiniquadError::GraphicsApi(GraphicsApiError::OpenGL(
                GLError::Unknown(0)
            )));
        }

        // Convert to C string safely
        let c_source = std::ffi::CString::new(source)
            .map_err(|_| MiniquadError::InvalidParameter(
                "Shader source contains null bytes".to_string()
            ))?;

        unsafe {
            let c_source_ptr = c_source.as_ptr();
            glShaderSource(shader, 1, &c_source_ptr, std::ptr::null());
            glCompileShader(shader);
        }

        // Check compilation status
        let mut success: GLint = 0;
        unsafe { glGetShaderiv(shader, GL_COMPILE_STATUS, &mut success) };

        if success == 0 {
            // Get error message
            let mut len: GLint = 0;
            unsafe { glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &mut len) };

            let mut log = vec![0u8; len as usize];
            unsafe { 
                glGetShaderInfoLog(
                    shader, 
                    len, 
                    std::ptr::null_mut(), 
                    log.as_mut_ptr() as *mut i8
                ) 
            };

            unsafe { glDeleteShader(shader) };

            let error_msg = String::from_utf8_lossy(&log).trim_end_matches('\0').to_string();
            let shader_type_name = match shader_type {
                GL_VERTEX_SHADER => ShaderType::Vertex,
                GL_FRAGMENT_SHADER => ShaderType::Fragment,
                _ => unreachable!(),
            };

            return Err(MiniquadError::Shader(ShaderError::CompilationError {
                shader_type: shader_type_name,
                error_message: error_msg,
            }));
        }

        Self::check_error_with_context("shader compilation")?;
        Ok(shader)
    }

    /// Safely delete buffers
    pub fn delete_buffers(buffers: &[GLuint]) -> Result<(), MiniquadError> {
        if buffers.is_empty() {
            return Ok(());
        }

        unsafe { glDeleteBuffers(buffers.len() as i32, buffers.as_ptr()) };
        Self::check_error_with_context("glDeleteBuffers")
    }

    /// Safely delete textures
    pub fn delete_textures(textures: &[GLuint]) -> Result<(), MiniquadError> {
        if textures.is_empty() {
            return Ok(());
        }

        unsafe { glDeleteTextures(textures.len() as i32, textures.as_ptr()) };
        Self::check_error_with_context("glDeleteTextures")
    }

    /// Safely delete shader
    pub fn delete_shader(shader: GLuint) -> Result<(), MiniquadError> {
        if shader == 0 {
            return Ok(());
        }

        unsafe { glDeleteShader(shader) };
        Self::check_error_with_context("glDeleteShader")
    }

    /// Get current OpenGL context info safely
    pub fn get_context_info() -> Result<String, MiniquadError> {
        let version_ptr = unsafe { glGetString(GL_VERSION) };
        if version_ptr.is_null() {
            return Err(MiniquadError::GraphicsApi(GraphicsApiError::OpenGL(
                GLError::InvalidOperation
            )));
        }

        let version_cstr = unsafe { std::ffi::CStr::from_ptr(version_ptr as *const i8) };
        let version_str = version_cstr.to_string_lossy().into_owned();
        
        Self::check_error_with_context("glGetString")?;
        Ok(version_str)
    }
}

/// Macro for safely calling OpenGL functions with automatic error checking
#[macro_export]
macro_rules! gl_call {
    ($func:ident($($arg:expr),*)) => {{
        let result = unsafe { $func($($arg),*) };
        $crate::graphics::gl_safety::SafeGL::check_error_with_context(stringify!($func))?;
        result
    }};
}

/// Macro for OpenGL calls that don't return values
#[macro_export]
macro_rules! gl_call_void {
    ($func:ident($($arg:expr),*)) => {{
        unsafe { $func($($arg),*) };
        $crate::graphics::gl_safety::SafeGL::check_error_with_context(stringify!($func))?;
    }};
}