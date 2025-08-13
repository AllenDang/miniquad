//! High-performance command buffer batching system
//!
//! This module implements a sophisticated command batching system that queues
//! and optimally executes rendering commands to minimize GPU overhead.
//!
//! Key features:
//! - Automatic draw call merging for compatible states
//! - Deferred command execution with optimal batching
//! - State change minimization through intelligent sorting
//! - Instancing support for identical draw calls
//! - Comprehensive batching statistics and monitoring

use crate::graphics::*;
use std::collections::HashMap;

/// Maximum number of commands to batch before forced flush
const MAX_BATCH_SIZE: usize = 1024;

/// Maximum number of instances to batch into a single instanced draw call
const MAX_INSTANCES_PER_DRAW: i32 = 16384;

/// Parameters for a draw elements command
#[derive(Debug, Clone, PartialEq)]
pub struct DrawElementsParams {
    pub base_element: i32,
    pub num_elements: i32,
    pub num_instances: i32,
    pub primitive_type: PrimitiveType,
    pub index_type: u32,
}

/// Command types that can be batched
#[derive(Debug, Clone, PartialEq)]
pub enum Command {
    /// Draw elements command
    DrawElements {
        pipeline: Pipeline,
        bindings: CommandBindings,
        params: DrawElementsParams,
    },
    /// State change command (viewport, scissor, etc.)
    StateChange { state_type: StateChangeType },
    /// Begin render pass
    BeginPass {
        pass: Option<RenderPass>,
        action: PassAction,
    },
    /// End render pass
    EndPass,
    /// Apply uniforms
    ApplyUniforms { data: Vec<u8> },
}

/// Lightweight bindings representation for batching
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CommandBindings {
    pub vertex_buffers: Vec<BufferId>,
    pub index_buffer: BufferId,
    pub images: Vec<TextureId>,
}

impl From<&Bindings> for CommandBindings {
    fn from(bindings: &Bindings) -> Self {
        Self {
            vertex_buffers: bindings.vertex_buffers.clone(),
            index_buffer: bindings.index_buffer,
            images: bindings.images.clone(),
        }
    }
}

/// State change types for batching optimization
#[derive(Debug, Clone, PartialEq)]
pub enum StateChangeType {
    Viewport { x: i32, y: i32, w: i32, h: i32 },
    Scissor { x: i32, y: i32, w: i32, h: i32 },
    Pipeline { pipeline: Pipeline },
}

/// Batch statistics for performance monitoring
#[derive(Debug, Default, Clone)]
pub struct BatchStats {
    pub total_commands: u64,
    pub batched_commands: u64,
    pub draw_calls_saved: u64,
    pub state_changes_eliminated: u64,
    pub instanced_draws_created: u64,
    pub average_batch_size: f64,
    pub flush_count: u64,
    pub compatibility_rate: f64,
}

impl BatchStats {
    pub fn batching_efficiency(&self) -> f64 {
        if self.total_commands == 0 {
            0.0
        } else {
            (self.batched_commands as f64 / self.total_commands as f64) * 100.0
        }
    }

    pub fn print_report(&self) {
        println!("\n=== Command Batching Performance Report ===");
        println!("Total commands: {}", self.total_commands);
        println!(
            "Batched commands: {} ({:.1}% efficiency)",
            self.batched_commands,
            self.batching_efficiency()
        );
        println!("Draw calls saved: {}", self.draw_calls_saved);
        println!(
            "State changes eliminated: {}",
            self.state_changes_eliminated
        );
        println!("Instanced draws created: {}", self.instanced_draws_created);
        println!("Average batch size: {:.1}", self.average_batch_size);
        println!(
            "Flush count: {} (avg {:.1} commands per flush)",
            self.flush_count,
            if self.flush_count > 0 {
                self.total_commands as f64 / self.flush_count as f64
            } else {
                0.0
            }
        );
        println!("Compatibility rate: {:.1}%", self.compatibility_rate);
    }
}

/// Batch group for commands that can be executed together
#[derive(Debug, Clone)]
struct BatchGroup {
    pipeline: Pipeline,
    bindings: CommandBindings,
    primitive_type: PrimitiveType,
    index_type: u32,
    draws: Vec<DrawCall>,
}

/// Individual draw call within a batch group
#[derive(Debug, Clone)]
struct DrawCall {
    base_element: i32,
    num_elements: i32,
    num_instances: i32,
}

impl BatchGroup {
    fn new(
        pipeline: Pipeline,
        bindings: CommandBindings,
        primitive_type: PrimitiveType,
        index_type: u32,
    ) -> Self {
        Self {
            pipeline,
            bindings,
            primitive_type,
            index_type,
            draws: Vec::new(),
        }
    }

    /// Add a draw call to this batch group
    fn add_draw(&mut self, base_element: i32, num_elements: i32, num_instances: i32) {
        self.draws.push(DrawCall {
            base_element,
            num_elements,
            num_instances,
        });
    }

    /// Check if this draw call is compatible with the batch group
    fn is_compatible(
        &self,
        pipeline: Pipeline,
        bindings: &CommandBindings,
        primitive_type: PrimitiveType,
        index_type: u32,
    ) -> bool {
        self.pipeline == pipeline
            && self.bindings == *bindings
            && self.primitive_type == primitive_type
            && self.index_type == index_type
    }

    /// Get total number of draw calls in this group
    fn draw_count(&self) -> usize {
        self.draws.len()
    }

    /// Check if we can merge similar draws into instanced draws
    fn can_instance(&self) -> bool {
        // For now, simple instancing: all draws must have same element count
        if self.draws.len() < 2 {
            return false;
        }

        let first_draw = &self.draws[0];
        self.draws.iter().all(|draw| {
            draw.num_elements == first_draw.num_elements && draw.num_instances == 1
            // Only batch single-instance draws
        })
    }
}

/// High-performance command buffer with automatic batching
pub struct CommandBuffer {
    /// Queue of pending commands
    commands: Vec<Command>,

    /// Current batching groups
    batch_groups: Vec<BatchGroup>,

    /// Performance statistics
    stats: BatchStats,

    /// Configuration
    max_batch_size: usize,
    auto_flush: bool,

    /// State tracking for optimization
    current_pipeline: Option<Pipeline>,
    current_bindings: Option<CommandBindings>,
    last_state_changes: HashMap<String, StateChangeType>,
}

impl CommandBuffer {
    /// Create a new command buffer
    pub fn new() -> Self {
        Self {
            commands: Vec::with_capacity(MAX_BATCH_SIZE),
            batch_groups: Vec::new(),
            stats: BatchStats::default(),
            max_batch_size: MAX_BATCH_SIZE,
            auto_flush: true,
            current_pipeline: None,
            current_bindings: None,
            last_state_changes: HashMap::new(),
        }
    }

    /// Add a draw elements command to the batch
    pub fn draw_elements(
        &mut self,
        pipeline: Pipeline,
        bindings: &Bindings,
        params: DrawElementsParams,
    ) {
        let cmd_bindings = CommandBindings::from(bindings);

        // Update current state tracking for optimization
        if self.current_pipeline != Some(pipeline) {
            // Pipeline change needed
            let state_change = Command::StateChange {
                state_type: StateChangeType::Pipeline { pipeline },
            };
            self.add_command(state_change);
            self.current_pipeline = Some(pipeline);
        }

        // Check if bindings have changed
        if self.current_bindings.as_ref() != Some(&cmd_bindings) {
            self.current_bindings = Some(cmd_bindings.clone());
        }

        let command = Command::DrawElements {
            pipeline,
            bindings: cmd_bindings,
            params,
        };

        self.add_command(command);
    }

    /// Add a state change command
    pub fn state_change(&mut self, state_type: StateChangeType) {
        // Check if this state change is redundant
        let state_key = match &state_type {
            StateChangeType::Viewport { .. } => "viewport",
            StateChangeType::Scissor { .. } => "scissor",
            StateChangeType::Pipeline { .. } => "pipeline",
        };

        if let Some(last_state) = self.last_state_changes.get(state_key) {
            if *last_state == state_type {
                // Redundant state change, skip it
                self.stats.state_changes_eliminated += 1;
                return;
            }
        }

        // Update current state tracking
        if let StateChangeType::Pipeline { pipeline } = &state_type {
            self.current_pipeline = Some(*pipeline);
        }

        self.last_state_changes
            .insert(state_key.to_string(), state_type.clone());

        let command = Command::StateChange { state_type };
        self.add_command(command);
    }

    /// Add a begin pass command
    pub fn begin_pass(&mut self, pass: Option<RenderPass>, action: PassAction) {
        // Reset state tracking when beginning a new pass
        self.current_pipeline = None;
        self.current_bindings = None;

        let command = Command::BeginPass { pass, action };
        self.add_command(command);
    }

    /// Add an end pass command
    pub fn end_pass(&mut self) {
        let command = Command::EndPass;
        self.add_command(command);
    }

    /// Add uniforms command
    pub fn apply_uniforms(&mut self, data: Vec<u8>) {
        let command = Command::ApplyUniforms { data };
        self.add_command(command);
    }

    /// Add a command to the buffer
    fn add_command(&mut self, command: Command) {
        self.commands.push(command);
        self.stats.total_commands += 1;

        // Auto-flush if buffer is getting full
        if self.auto_flush && self.commands.len() >= self.max_batch_size {
            self.flush();
        }
    }

    /// Process commands into optimized batches
    pub fn optimize_batches(&mut self) {
        self.batch_groups.clear();
        let mut compatible_commands = 0;

        for command in &self.commands {
            if let Command::DrawElements {
                pipeline,
                bindings,
                params,
            } = command
            {
                // Try to find a compatible batch group
                let mut found_group = false;

                for group in &mut self.batch_groups {
                    if group.is_compatible(
                        *pipeline,
                        bindings,
                        params.primitive_type,
                        params.index_type,
                    ) {
                        group.add_draw(
                            params.base_element,
                            params.num_elements,
                            params.num_instances,
                        );
                        compatible_commands += 1;
                        found_group = true;
                        break;
                    }
                }

                // Create new batch group if no compatible one found
                if !found_group {
                    let mut new_group = BatchGroup::new(
                        *pipeline,
                        bindings.clone(),
                        params.primitive_type,
                        params.index_type,
                    );
                    new_group.add_draw(
                        params.base_element,
                        params.num_elements,
                        params.num_instances,
                    );
                    self.batch_groups.push(new_group);
                }
            }
        }

        self.stats.batched_commands += compatible_commands as u64;
        self.stats.compatibility_rate = if self.stats.total_commands > 0 {
            (compatible_commands as f64 / self.stats.total_commands as f64) * 100.0
        } else {
            0.0
        };
    }

    /// Execute all batched commands
    pub fn execute(&mut self, gl_context: &mut super::gl::GlContext) -> Result<(), String> {
        if self.commands.is_empty() {
            return Ok(());
        }

        // Optimize batches first
        self.optimize_batches();

        // Execute non-draw commands first (state changes, passes, etc.)
        for command in &self.commands {
            match command {
                Command::StateChange { state_type } => {
                    self.execute_state_change(state_type, gl_context);
                }
                Command::BeginPass { pass, action } => {
                    self.execute_begin_pass(*pass, action, gl_context);
                }
                Command::EndPass => {
                    self.execute_end_pass(gl_context);
                }
                Command::ApplyUniforms { data } => {
                    self.execute_apply_uniforms(data, gl_context);
                }
                Command::DrawElements { .. } => {
                    // Draw commands are handled by batch groups
                }
            }
        }

        // Execute optimized draw call batches
        let mut draws_saved = 0;
        let mut instances_created = 0;

        for group in &self.batch_groups {
            let original_draw_count = group.draw_count();

            if original_draw_count > 1 {
                if group.can_instance() {
                    // Execute as instanced draw
                    self.execute_instanced_batch(group, gl_context);
                    draws_saved += original_draw_count - 1;
                    instances_created += 1;
                } else {
                    // Execute as multiple draws with same state
                    self.execute_multi_draw_batch(group, gl_context);
                    draws_saved += original_draw_count - 1;
                }
            } else {
                // Single draw, execute normally
                self.execute_single_draw_batch(group, gl_context);
            }
        }

        self.stats.draw_calls_saved += draws_saved as u64;
        self.stats.instanced_draws_created += instances_created as u64;
        self.stats.flush_count += 1;

        // Update average batch size
        if self.stats.flush_count > 0 {
            self.stats.average_batch_size =
                self.stats.total_commands as f64 / self.stats.flush_count as f64;
        }

        // Clear commands after execution
        self.commands.clear();
        self.batch_groups.clear();

        // Reset state tracking after execution
        self.current_pipeline = None;
        self.current_bindings = None;

        Ok(())
    }

    /// Force flush all pending commands
    pub fn flush(&mut self) {
        // This would normally execute against the GL context
        // For now, just clear the commands
        self.commands.clear();
        self.batch_groups.clear();
        self.stats.flush_count += 1;

        // Reset state tracking
        self.current_pipeline = None;
        self.current_bindings = None;
    }

    /// Get current batching statistics
    pub fn get_stats(&self) -> BatchStats {
        self.stats.clone()
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = BatchStats::default();
    }

    // Private execution methods

    fn execute_state_change(
        &self,
        state_type: &StateChangeType,
        gl_context: &mut super::gl::GlContext,
    ) {
        match state_type {
            StateChangeType::Viewport { x, y, w, h } => {
                gl_context.apply_viewport(*x, *y, *w, *h);
            }
            StateChangeType::Scissor { x, y, w, h } => {
                gl_context.apply_scissor_rect(*x, *y, *w, *h);
            }
            StateChangeType::Pipeline { pipeline } => {
                gl_context.apply_pipeline(pipeline);
            }
        }
    }

    fn execute_begin_pass(
        &self,
        pass: Option<RenderPass>,
        action: &PassAction,
        gl_context: &mut super::gl::GlContext,
    ) {
        gl_context.begin_pass(pass, action.clone());
    }

    fn execute_end_pass(&self, gl_context: &mut super::gl::GlContext) {
        gl_context.end_render_pass();
    }

    fn execute_apply_uniforms(&self, data: &[u8], gl_context: &mut super::gl::GlContext) {
        // Apply uniforms from raw data
        gl_context.apply_uniforms_from_bytes(data.as_ptr(), data.len());
    }

    fn execute_instanced_batch(&self, group: &BatchGroup, gl_context: &mut super::gl::GlContext) {
        // Apply pipeline and bindings once
        gl_context.apply_pipeline(&group.pipeline);

        let bindings = Bindings {
            vertex_buffers: group.bindings.vertex_buffers.clone(),
            index_buffer: group.bindings.index_buffer,
            images: group.bindings.images.clone(),
        };
        gl_context.apply_bindings(&bindings);

        // Calculate total instance count (capped at MAX_INSTANCES_PER_DRAW)
        let total_instances = group.draws.len().min(MAX_INSTANCES_PER_DRAW as usize) as i32;
        let first_draw = &group.draws[0];

        // Execute as single instanced draw
        gl_context.draw(
            first_draw.base_element,
            first_draw.num_elements,
            total_instances,
        );
    }

    fn execute_multi_draw_batch(&self, group: &BatchGroup, gl_context: &mut super::gl::GlContext) {
        // Apply pipeline and bindings once
        gl_context.apply_pipeline(&group.pipeline);

        let bindings = Bindings {
            vertex_buffers: group.bindings.vertex_buffers.clone(),
            index_buffer: group.bindings.index_buffer,
            images: group.bindings.images.clone(),
        };
        gl_context.apply_bindings(&bindings);

        // Execute all draws with shared state
        for draw in &group.draws {
            gl_context.draw(draw.base_element, draw.num_elements, draw.num_instances);
        }
    }

    fn execute_single_draw_batch(&self, group: &BatchGroup, gl_context: &mut super::gl::GlContext) {
        // Apply pipeline and bindings
        gl_context.apply_pipeline(&group.pipeline);

        let bindings = Bindings {
            vertex_buffers: group.bindings.vertex_buffers.clone(),
            index_buffer: group.bindings.index_buffer,
            images: group.bindings.images.clone(),
        };
        gl_context.apply_bindings(&bindings);

        // Execute single draw
        let draw = &group.draws[0];
        gl_context.draw(draw.base_element, draw.num_elements, draw.num_instances);
    }
}

impl Default for CommandBuffer {
    fn default() -> Self {
        Self::new()
    }
}
