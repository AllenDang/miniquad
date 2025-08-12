//! GPU state change profiling for performance analysis
//!
//! This module provides instrumentation to measure redundant GL state changes
//! which are the primary target for optimization in the state caching system.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Global profiler instance for tracking GL state changes
static PROFILER: std::sync::OnceLock<Arc<Mutex<GlStateProfiler>>> = std::sync::OnceLock::new();

/// Statistics about GL state changes
#[derive(Debug, Default, Clone)]
pub struct StateChangeStats {
    pub total_calls: u64,
    pub redundant_calls: u64,
    pub buffer_binds: u64,
    pub texture_binds: u64,
    pub program_uses: u64,
    pub redundant_buffer_binds: u64,
    pub redundant_texture_binds: u64,
    pub redundant_program_uses: u64,
}

impl StateChangeStats {
    pub fn redundancy_percentage(&self) -> f64 {
        if self.total_calls == 0 {
            0.0
        } else {
            (self.redundant_calls as f64 / self.total_calls as f64) * 100.0
        }
    }

    pub fn print_report(&self) {
        println!("\n=== GL State Change Profile ===");
        println!("Total GL calls: {}", self.total_calls);
        println!(
            "Redundant calls: {} ({:.1}%)",
            self.redundant_calls,
            self.redundancy_percentage()
        );
        println!();
        println!(
            "Buffer bindings: {} (redundant: {})",
            self.buffer_binds, self.redundant_buffer_binds
        );
        println!(
            "Texture bindings: {} (redundant: {})",
            self.texture_binds, self.redundant_texture_binds
        );
        println!(
            "Program uses: {} (redundant: {})",
            self.program_uses, self.redundant_program_uses
        );
        println!();

        if self.redundant_calls > 0 {
            println!("Potential savings from state caching:");
            println!(
                "- {:.1}% reduction in GL calls",
                self.redundancy_percentage()
            );
            println!("- {} fewer buffer binds", self.redundant_buffer_binds);
            println!("- {} fewer texture binds", self.redundant_texture_binds);
            println!("- {} fewer program switches", self.redundant_program_uses);
        }
    }
}

/// Tracks current GL state to detect redundant changes
#[derive(Debug, Default)]
struct GlStateTracker {
    current_array_buffer: Option<u32>,
    current_element_buffer: Option<u32>,
    current_program: Option<u32>,
    current_textures: HashMap<u32, u32>, // slot -> texture_id
}

/// Profiler for GL state changes
#[derive(Debug, Default)]
pub struct GlStateProfiler {
    stats: StateChangeStats,
    tracker: GlStateTracker,
    enabled: bool,
}

impl GlStateProfiler {
    pub fn new() -> Self {
        Self {
            stats: StateChangeStats::default(),
            tracker: GlStateTracker::default(),
            enabled: true,
        }
    }

    pub fn enable(&mut self) {
        self.enabled = true;
    }

    pub fn disable(&mut self) {
        self.enabled = false;
    }

    pub fn reset(&mut self) {
        self.stats = StateChangeStats::default();
        self.tracker = GlStateTracker::default();
    }

    pub fn get_stats(&self) -> StateChangeStats {
        self.stats.clone()
    }

    /// Record a buffer binding operation
    pub fn record_buffer_bind(&mut self, target: u32, buffer: u32) {
        if !self.enabled {
            return;
        }

        self.stats.total_calls += 1;
        self.stats.buffer_binds += 1;

        let current_buffer = match target {
            crate::native::gl::GL_ARRAY_BUFFER => &mut self.tracker.current_array_buffer,
            crate::native::gl::GL_ELEMENT_ARRAY_BUFFER => &mut self.tracker.current_element_buffer,
            _ => {
                // Unknown buffer type, can't track redundancy
                return;
            }
        };

        if let Some(current) = current_buffer {
            if *current == buffer {
                // Redundant bind
                self.stats.redundant_calls += 1;
                self.stats.redundant_buffer_binds += 1;
            }
        }

        *current_buffer = Some(buffer);
    }

    /// Record a texture binding operation
    pub fn record_texture_bind(&mut self, slot: u32, texture: u32) {
        if !self.enabled {
            return;
        }

        self.stats.total_calls += 1;
        self.stats.texture_binds += 1;

        if let Some(&current_texture) = self.tracker.current_textures.get(&slot) {
            if current_texture == texture {
                // Redundant bind
                self.stats.redundant_calls += 1;
                self.stats.redundant_texture_binds += 1;
            }
        }

        self.tracker.current_textures.insert(slot, texture);
    }

    /// Record a program use operation
    pub fn record_program_use(&mut self, program: u32) {
        if !self.enabled {
            return;
        }

        self.stats.total_calls += 1;
        self.stats.program_uses += 1;

        if let Some(current_program) = self.tracker.current_program {
            if current_program == program {
                // Redundant program use
                self.stats.redundant_calls += 1;
                self.stats.redundant_program_uses += 1;
            }
        }

        self.tracker.current_program = Some(program);
    }
}

/// Initialize the global profiler
pub fn init_profiler() {
    PROFILER
        .set(Arc::new(Mutex::new(GlStateProfiler::new())))
        .unwrap_or(());
}

/// Get the global profiler instance
pub fn get_profiler() -> Arc<Mutex<GlStateProfiler>> {
    PROFILER
        .get()
        .unwrap_or_else(|| {
            init_profiler();
            PROFILER.get().unwrap()
        })
        .clone()
}

/// Enable profiling
pub fn enable_profiling() {
    if let Ok(mut profiler) = get_profiler().lock() {
        profiler.enable();
    }
}

/// Disable profiling
pub fn disable_profiling() {
    if let Ok(mut profiler) = get_profiler().lock() {
        profiler.disable();
    }
}

/// Reset profiling statistics
pub fn reset_profiling() {
    if let Ok(mut profiler) = get_profiler().lock() {
        profiler.reset();
    }
}

/// Get current profiling statistics
pub fn get_stats() -> Option<StateChangeStats> {
    get_profiler()
        .lock()
        .ok()
        .map(|profiler| profiler.get_stats())
}

/// Print a profiling report
pub fn print_report() {
    if let Some(stats) = get_stats() {
        stats.print_report();
    }
}

// Macros for easy profiling instrumentation
#[macro_export]
macro_rules! profile_buffer_bind {
    ($target:expr, $buffer:expr) => {
        #[cfg(feature = "profiling")]
        {
            if let Ok(mut profiler) = $crate::graphics::profiling::get_profiler().lock() {
                profiler.record_buffer_bind($target, $buffer);
            }
        }
    };
}

#[macro_export]
macro_rules! profile_texture_bind {
    ($slot:expr, $texture:expr) => {
        #[cfg(feature = "profiling")]
        {
            if let Ok(mut profiler) = $crate::graphics::profiling::get_profiler().lock() {
                profiler.record_texture_bind($slot, $texture);
            }
        }
    };
}

#[macro_export]
macro_rules! profile_program_use {
    ($program:expr) => {
        #[cfg(feature = "profiling")]
        {
            if let Ok(mut profiler) = $crate::graphics::profiling::get_profiler().lock() {
                profiler.record_program_use($program);
            }
        }
    };
}
