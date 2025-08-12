//! Error types for miniquad operations
//!
//! This module provides comprehensive error handling for all miniquad operations,
//! replacing the previous panic-heavy approach with proper Result types.

use std::error::Error;
use std::fmt::{self, Display};

/// Main error type for all miniquad operations
#[derive(Debug, Clone)]
pub enum MiniquadError {
    /// Resource management errors
    Resource(ResourceError),
    /// Shader compilation and linking errors  
    Shader(crate::graphics::ShaderError),
    /// Graphics context errors
    GraphicsContext(GraphicsError),
    /// Platform-specific errors
    Platform(PlatformError),
    /// Invalid parameter errors
    InvalidParameter(String),
    /// OpenGL/graphics API errors
    GraphicsApi(GraphicsApiError),
}

/// Resource management errors
#[derive(Debug, Clone, PartialEq)]
pub enum ResourceError {
    /// Resource with given ID not found
    NotFound(usize),
    /// Resource has already been deleted
    AlreadyDeleted(usize),
    /// Resource is in invalid state for operation
    InvalidState(String),
    /// Resource limit exceeded
    LimitExceeded { limit: usize, requested: usize },
}

/// Graphics context errors
#[derive(Debug, Clone)]
pub enum GraphicsError {
    /// Context lost or not available
    ContextLost,
    /// Context creation failed
    CreationFailed(String),
    /// Unsupported operation
    Unsupported(String),
    /// Invalid texture format
    InvalidTextureFormat(String),
    /// Buffer creation failed
    BufferCreationFailed(String),
}

/// Platform-specific errors
#[derive(Debug, Clone)]
pub enum PlatformError {
    /// Display initialization failed
    DisplayInitFailed(String),
    /// Window creation failed
    WindowCreationFailed(String),
    /// Library loading failed
    LibraryLoadFailed(String),
    /// Feature not supported on platform
    FeatureUnsupported(String),
}

/// Graphics API errors (OpenGL, Metal, etc.)
#[derive(Debug, Clone)]
pub enum GraphicsApiError {
    /// OpenGL error
    OpenGL(GLError),
    /// Metal error
    #[cfg(target_vendor = "apple")]
    Metal(String),
    /// WebGL error
    #[cfg(target_arch = "wasm32")]
    WebGL(String),
}

/// OpenGL specific errors
#[derive(Debug, Clone, PartialEq)]
pub enum GLError {
    /// GL_INVALID_ENUM
    InvalidEnum,
    /// GL_INVALID_VALUE
    InvalidValue,
    /// GL_INVALID_OPERATION
    InvalidOperation,
    /// GL_INVALID_FRAMEBUFFER_OPERATION
    InvalidFramebufferOperation,
    /// GL_OUT_OF_MEMORY
    OutOfMemory,
    /// GL_STACK_UNDERFLOW
    StackUnderflow,
    /// GL_STACK_OVERFLOW
    StackOverflow,
    /// Unknown GL error
    Unknown(u32),
}

impl GLError {
    /// Convert OpenGL error code to GLError
    pub fn from_gl_enum(error: u32) -> Self {
        // OpenGL error constants (standard values across implementations)
        const GL_INVALID_ENUM: u32 = 0x0500;
        const GL_INVALID_VALUE: u32 = 0x0501;
        const GL_INVALID_OPERATION: u32 = 0x0502;
        const GL_INVALID_FRAMEBUFFER_OPERATION: u32 = 0x0506;
        const GL_OUT_OF_MEMORY: u32 = 0x0505;
        const GL_STACK_UNDERFLOW: u32 = 0x0504;
        const GL_STACK_OVERFLOW: u32 = 0x0503;

        match error {
            GL_INVALID_ENUM => GLError::InvalidEnum,
            GL_INVALID_VALUE => GLError::InvalidValue,
            GL_INVALID_OPERATION => GLError::InvalidOperation,
            GL_INVALID_FRAMEBUFFER_OPERATION => GLError::InvalidFramebufferOperation,
            GL_OUT_OF_MEMORY => GLError::OutOfMemory,
            GL_STACK_UNDERFLOW => GLError::StackUnderflow,
            GL_STACK_OVERFLOW => GLError::StackOverflow,
            code => GLError::Unknown(code),
        }
    }
}

// Display implementations
impl Display for MiniquadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MiniquadError::Resource(e) => write!(f, "Resource error: {}", e),
            MiniquadError::Shader(e) => write!(f, "Shader error: {}", e),
            MiniquadError::GraphicsContext(e) => write!(f, "Graphics context error: {}", e),
            MiniquadError::Platform(e) => write!(f, "Platform error: {}", e),
            MiniquadError::InvalidParameter(msg) => write!(f, "Invalid parameter: {}", msg),
            MiniquadError::GraphicsApi(e) => write!(f, "Graphics API error: {}", e),
        }
    }
}

impl Display for ResourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ResourceError::NotFound(id) => write!(f, "Resource with ID {} not found", id),
            ResourceError::AlreadyDeleted(id) => {
                write!(f, "Resource with ID {} already deleted", id)
            }
            ResourceError::InvalidState(msg) => write!(f, "Resource in invalid state: {}", msg),
            ResourceError::LimitExceeded { limit, requested } => {
                write!(
                    f,
                    "Resource limit exceeded: requested {}, limit {}",
                    requested, limit
                )
            }
        }
    }
}

impl Display for GraphicsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GraphicsError::ContextLost => write!(f, "Graphics context lost"),
            GraphicsError::CreationFailed(msg) => write!(f, "Context creation failed: {}", msg),
            GraphicsError::Unsupported(msg) => write!(f, "Unsupported operation: {}", msg),
            GraphicsError::InvalidTextureFormat(msg) => {
                write!(f, "Invalid texture format: {}", msg)
            }
            GraphicsError::BufferCreationFailed(msg) => {
                write!(f, "Buffer creation failed: {}", msg)
            }
        }
    }
}

impl Display for PlatformError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PlatformError::DisplayInitFailed(msg) => {
                write!(f, "Display initialization failed: {}", msg)
            }
            PlatformError::WindowCreationFailed(msg) => {
                write!(f, "Window creation failed: {}", msg)
            }
            PlatformError::LibraryLoadFailed(msg) => write!(f, "Library loading failed: {}", msg),
            PlatformError::FeatureUnsupported(msg) => write!(f, "Feature not supported: {}", msg),
        }
    }
}

impl Display for GraphicsApiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GraphicsApiError::OpenGL(e) => write!(f, "OpenGL error: {}", e),
            #[cfg(target_vendor = "apple")]
            GraphicsApiError::Metal(msg) => write!(f, "Metal error: {}", msg),
            #[cfg(target_arch = "wasm32")]
            GraphicsApiError::WebGL(msg) => write!(f, "WebGL error: {}", msg),
        }
    }
}

impl Display for GLError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GLError::InvalidEnum => write!(f, "Invalid enum value"),
            GLError::InvalidValue => write!(f, "Invalid parameter value"),
            GLError::InvalidOperation => write!(f, "Invalid operation"),
            GLError::InvalidFramebufferOperation => write!(f, "Invalid framebuffer operation"),
            GLError::OutOfMemory => write!(f, "Out of memory"),
            GLError::StackUnderflow => write!(f, "Stack underflow"),
            GLError::StackOverflow => write!(f, "Stack overflow"),
            GLError::Unknown(code) => write!(f, "Unknown OpenGL error: 0x{:X}", code),
        }
    }
}

// Error trait implementations
impl Error for MiniquadError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            MiniquadError::Resource(e) => Some(e),
            MiniquadError::Shader(e) => Some(e),
            MiniquadError::GraphicsContext(e) => Some(e),
            MiniquadError::Platform(e) => Some(e),
            MiniquadError::GraphicsApi(e) => Some(e),
            _ => None,
        }
    }
}

impl Error for ResourceError {}
impl Error for GraphicsError {}
impl Error for PlatformError {}
impl Error for GraphicsApiError {}
impl Error for GLError {}

// From implementations for easy error conversion
impl From<ResourceError> for MiniquadError {
    fn from(e: ResourceError) -> Self {
        MiniquadError::Resource(e)
    }
}

impl From<crate::graphics::ShaderError> for MiniquadError {
    fn from(e: crate::graphics::ShaderError) -> Self {
        MiniquadError::Shader(e)
    }
}

impl From<GraphicsError> for MiniquadError {
    fn from(e: GraphicsError) -> Self {
        MiniquadError::GraphicsContext(e)
    }
}

impl From<PlatformError> for MiniquadError {
    fn from(e: PlatformError) -> Self {
        MiniquadError::Platform(e)
    }
}

impl From<GraphicsApiError> for MiniquadError {
    fn from(e: GraphicsApiError) -> Self {
        MiniquadError::GraphicsApi(e)
    }
}

impl From<GLError> for MiniquadError {
    fn from(e: GLError) -> Self {
        MiniquadError::GraphicsApi(GraphicsApiError::OpenGL(e))
    }
}

/// Result type alias for miniquad operations
pub type Result<T> = std::result::Result<T, MiniquadError>;

/// Result type alias for resource operations
pub type ResourceResult<T> = std::result::Result<T, ResourceError>;

/// Result type alias for graphics operations  
pub type GraphicsResult<T> = std::result::Result<T, GraphicsError>;
