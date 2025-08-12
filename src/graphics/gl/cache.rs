use crate::graphics::profiling;
use crate::graphics::*;

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct VertexAttributeInternal {
    pub attr_loc: GLuint,
    pub size: i32,
    pub type_: GLuint,
    pub offset: i64,
    pub stride: i32,
    pub buffer_index: usize,
    pub divisor: i32,
    pub gl_pass_as_float: bool,
}

#[derive(Default, Copy, Clone)]
pub struct CachedAttribute {
    pub attribute: VertexAttributeInternal,
    pub gl_vbuf: GLuint,
}

#[derive(Clone, Copy)]
pub struct CachedTexture {
    // GL_TEXTURE_2D or GL_TEXTURE_CUBEMAP
    pub target: GLuint,
    pub texture: GLuint,
}

pub struct GlCache {
    pub stored_index_buffer: GLuint,
    pub stored_index_type: Option<u32>,
    pub stored_vertex_buffer: GLuint,
    pub stored_target: GLuint,
    pub stored_texture: GLuint,
    pub index_buffer: GLuint,
    pub index_type: Option<u32>,
    pub vertex_buffer: GLuint,
    pub textures: [CachedTexture; MAX_SHADERSTAGE_IMAGES],
    pub cur_pipeline: Option<Pipeline>,
    pub cur_pass: Option<RenderPass>,
    pub color_blend: Option<BlendState>,
    pub alpha_blend: Option<BlendState>,
    pub stencil: Option<StencilState>,
    pub color_write: ColorMask,
    pub cull_face: CullFace,
    pub attributes: [Option<CachedAttribute>; MAX_VERTEX_ATTRIBUTES],

    // Enhanced caching for performance optimization
    pub current_program: GLuint,
    pub viewport: (i32, i32, i32, i32),
    pub scissor: Option<(i32, i32, i32, i32)>,

    // Dirty flags to avoid redundant state changes
    pub program_dirty: bool,
    pub viewport_dirty: bool,
    pub scissor_dirty: bool,
}

impl GlCache {
    pub fn bind_buffer(&mut self, target: GLenum, buffer: GLuint, index_type: Option<u32>) {
        if target == GL_ARRAY_BUFFER {
            if self.vertex_buffer != buffer {
                let _ = profiling::get_profiler()
                    .lock()
                    .map(|mut p| p.record_buffer_bind(target, buffer));
                self.vertex_buffer = buffer;
                unsafe {
                    glBindBuffer(target, buffer);
                }
            }
        } else {
            if self.index_buffer != buffer {
                let _ = profiling::get_profiler()
                    .lock()
                    .map(|mut p| p.record_buffer_bind(target, buffer));
                self.index_buffer = buffer;
                unsafe {
                    glBindBuffer(target, buffer);
                }
            }
            self.index_type = index_type;
        }
    }

    pub fn store_buffer_binding(&mut self, target: GLenum) {
        if target == GL_ARRAY_BUFFER {
            self.stored_vertex_buffer = self.vertex_buffer;
        } else {
            self.stored_index_buffer = self.index_buffer;
            self.stored_index_type = self.index_type;
        }
    }

    pub fn restore_buffer_binding(&mut self, target: GLenum) {
        if target == GL_ARRAY_BUFFER {
            if self.stored_vertex_buffer != 0 {
                self.bind_buffer(target, self.stored_vertex_buffer, None);
                self.stored_vertex_buffer = 0;
            }
        } else if self.stored_index_buffer != 0 {
            self.bind_buffer(target, self.stored_index_buffer, self.stored_index_type);
            self.stored_index_buffer = 0;
        }
    }

    pub fn bind_texture(&mut self, slot_index: usize, target: GLuint, texture: GLuint) {
        unsafe {
            glActiveTexture(GL_TEXTURE0 + slot_index as GLuint);
            if self.textures[slot_index].target != target
                || self.textures[slot_index].texture != texture
            {
                let _ = profiling::get_profiler()
                    .lock()
                    .map(|mut p| p.record_texture_bind(slot_index as u32, texture));
                let target = if target == 0 { GL_TEXTURE_2D } else { target };
                glBindTexture(target, texture);
                self.textures[slot_index] = CachedTexture { target, texture };
            }
        }
    }

    pub fn store_texture_binding(&mut self, slot_index: usize) {
        self.stored_target = self.textures[slot_index].target;
        self.stored_texture = self.textures[slot_index].texture;
    }

    pub fn restore_texture_binding(&mut self, slot_index: usize) {
        self.bind_texture(slot_index, self.stored_target, self.stored_texture);
    }

    pub fn clear_buffer_bindings(&mut self) {
        self.bind_buffer(GL_ARRAY_BUFFER, 0, None);
        self.vertex_buffer = 0;

        self.bind_buffer(GL_ELEMENT_ARRAY_BUFFER, 0, None);
        self.index_buffer = 0;
    }

    pub fn clear_texture_bindings(&mut self) {
        for ix in 0..MAX_SHADERSTAGE_IMAGES {
            if self.textures[ix].texture != 0 {
                self.bind_texture(ix, self.textures[ix].target, 0);
                self.textures[ix] = CachedTexture {
                    target: 0,
                    texture: 0,
                };
            }
        }
    }

    pub fn clear_vertex_attributes(&mut self) {
        for attr_index in 0..MAX_VERTEX_ATTRIBUTES {
            let cached_attr = &mut self.attributes[attr_index];

            if cached_attr.is_some() {
                unsafe { glDisableVertexAttribArray(attr_index as GLuint) };
            }
            *cached_attr = None;
        }
    }

    /// Enhanced program caching with profiling
    pub fn use_program(&mut self, program: GLuint) {
        if self.current_program != program || self.program_dirty {
            let _ = profiling::get_profiler()
                .lock()
                .map(|mut p| p.record_program_use(program));
            self.current_program = program;
            self.program_dirty = false;
            unsafe {
                glUseProgram(program);
            }
        }
    }

    /// Enhanced viewport caching
    pub fn apply_viewport(&mut self, x: i32, y: i32, w: i32, h: i32) {
        let new_viewport = (x, y, w, h);
        if self.viewport != new_viewport || self.viewport_dirty {
            self.viewport = new_viewport;
            self.viewport_dirty = false;
            unsafe {
                glViewport(x, y, w, h);
            }
        }
    }

    /// Enhanced scissor caching
    pub fn apply_scissor(&mut self, x: i32, y: i32, w: i32, h: i32) {
        let new_scissor = Some((x, y, w, h));
        if self.scissor != new_scissor || self.scissor_dirty {
            self.scissor = new_scissor;
            self.scissor_dirty = false;
            unsafe {
                glEnable(GL_SCISSOR_TEST);
                glScissor(x, y, w, h);
            }
        }
    }
}

impl Default for GlCache {
    fn default() -> Self {
        Self {
            stored_index_buffer: 0,
            stored_index_type: None,
            stored_vertex_buffer: 0,
            stored_target: 0,
            stored_texture: 0,
            index_buffer: 0,
            index_type: None,
            vertex_buffer: 0,
            textures: [CachedTexture {
                target: 0,
                texture: 0,
            }; MAX_SHADERSTAGE_IMAGES],
            cur_pipeline: None,
            cur_pass: None,
            color_blend: None,
            alpha_blend: None,
            stencil: None,
            color_write: (true, true, true, true),
            cull_face: CullFace::Nothing,
            attributes: [None; MAX_VERTEX_ATTRIBUTES],

            // Enhanced caching state
            current_program: 0,
            viewport: (0, 0, 0, 0),
            scissor: None,

            // All dirty on init to force first setup
            program_dirty: true,
            viewport_dirty: true,
            scissor_dirty: true,
        }
    }
}
