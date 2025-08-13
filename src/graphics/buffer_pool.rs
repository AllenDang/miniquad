//! High-performance buffer pooling system for GPU memory management
//!
//! This module implements a sophisticated buffer pooling system that eliminates
//! the performance overhead of frequent GPU buffer allocation and deallocation.
//!
//! Key features:
//! - Size-based bucket allocation (powers of 2)
//! - Separate pools for vertex and index buffers  
//! - Usage pattern tracking (static, dynamic, stream)
//! - Automatic pool size management with limits
//! - Comprehensive statistics for monitoring

use crate::graphics::*;
use crate::native::gl::{
    glBindBuffer, glBufferData, glDeleteBuffers, glGenBuffers, GLuint, GL_ARRAY_BUFFER,
    GL_DYNAMIC_DRAW, GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW, GL_STREAM_DRAW,
};
use std::collections::HashMap;

/// Buffer pool configuration
const MIN_POOL_SIZE: usize = 8; // Minimum buffers per bucket
const MAX_POOL_SIZE: usize = 64; // Maximum buffers per bucket
const MAX_TOTAL_BUFFERS: usize = 512; // Total buffer limit across all pools

/// Size buckets for efficient allocation (powers of 2)
const SIZE_BUCKETS: &[usize] = &[
    512,     // 512B - Small vertex data
    2048,    // 2KB - Medium vertex data
    8192,    // 8KB - Large vertex data
    32768,   // 32KB - Very large vertex data
    131072,  // 128KB - Huge vertex data
    524288,  // 512KB - Massive vertex data
    2097152, // 2MB - Maximum reasonable size
];

/// Pooled buffer entry
#[derive(Debug, Clone)]
struct PooledBuffer {
    gl_buf: GLuint,
    size: usize,
    buffer_type: BufferType,
    usage: BufferUsage,
    last_used: std::time::Instant,
}

/// Buffer pool statistics for monitoring
#[derive(Debug, Default, Clone)]
pub struct BufferPoolStats {
    pub total_buffers: usize,
    pub buffers_in_use: usize,
    pub buffers_available: usize,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub pool_allocations: u64,
    pub pool_deallocations: u64,
    pub gpu_allocations_saved: u64,
    pub memory_usage_bytes: usize,
    pub pool_efficiency: f64,
}

impl BufferPoolStats {
    pub fn hit_rate(&self) -> f64 {
        let total_requests = self.cache_hits + self.cache_misses;
        if total_requests == 0 {
            0.0
        } else {
            (self.cache_hits as f64 / total_requests as f64) * 100.0
        }
    }

    pub fn print_report(&self) {
        println!("\n=== Buffer Pool Performance Report ===");
        println!(
            "Total buffers: {} (in use: {}, available: {})",
            self.total_buffers, self.buffers_in_use, self.buffers_available
        );
        println!(
            "Cache performance: {} hits, {} misses ({:.1}% hit rate)",
            self.cache_hits,
            self.cache_misses,
            self.hit_rate()
        );
        println!("GPU allocations saved: {}", self.gpu_allocations_saved);
        println!(
            "Memory usage: {:.1} MB",
            self.memory_usage_bytes as f64 / 1024.0 / 1024.0
        );
        println!("Pool efficiency: {:.1}%", self.pool_efficiency);
    }
}

/// Key for identifying buffer pools
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PoolKey {
    buffer_type: BufferType,
    usage: BufferUsage,
    size_bucket: usize,
}

/// High-performance buffer pool manager
#[derive(Debug)]
pub struct BufferPool {
    // Pool storage organized by type, usage, and size
    pools: HashMap<PoolKey, Vec<PooledBuffer>>,

    // Track buffers currently in use
    active_buffers: HashMap<GLuint, PooledBuffer>,

    // Performance statistics
    stats: BufferPoolStats,

    // Configuration
    max_age: std::time::Duration,
}

impl BufferPool {
    /// Create a new buffer pool
    pub fn new() -> Self {
        Self {
            pools: HashMap::new(),
            active_buffers: HashMap::new(),
            stats: BufferPoolStats::default(),
            max_age: std::time::Duration::from_secs(30), // Clean up unused buffers after 30s
        }
    }

    /// Get the appropriate size bucket for a given size
    fn get_size_bucket(size: usize) -> usize {
        // Find the smallest bucket that can fit this size
        for &bucket_size in SIZE_BUCKETS {
            if size <= bucket_size {
                return bucket_size;
            }
        }

        // For sizes larger than our biggest bucket, round up to nearest power of 2
        let mut bucket = SIZE_BUCKETS[SIZE_BUCKETS.len() - 1];
        while bucket < size {
            bucket *= 2;
            // Cap at reasonable maximum to prevent excessive memory usage
            if bucket > 16 * 1024 * 1024 {
                // 16MB cap
                return bucket;
            }
        }
        bucket
    }

    /// Acquire a buffer from the pool or create new one
    pub fn acquire_buffer(
        &mut self,
        buffer_type: BufferType,
        usage: BufferUsage,
        size: usize,
    ) -> Result<GLuint, String> {
        let size_bucket = Self::get_size_bucket(size);
        let pool_key = PoolKey {
            buffer_type,
            usage,
            size_bucket,
        };

        // Try to get buffer from pool first
        if let Some(pool) = self.pools.get_mut(&pool_key) {
            if let Some(mut buffer) = pool.pop() {
                buffer.last_used = std::time::Instant::now();
                let gl_buf = buffer.gl_buf;
                self.active_buffers.insert(gl_buf, buffer);

                self.stats.cache_hits += 1;
                self.stats.buffers_in_use += 1;
                self.stats.buffers_available = self.stats.buffers_available.saturating_sub(1);

                return Ok(gl_buf);
            }
        }

        // Pool miss - need to create new buffer
        self.stats.cache_misses += 1;

        // Check if we're at the total buffer limit
        if self.stats.total_buffers >= MAX_TOTAL_BUFFERS {
            return Err(format!("Buffer pool limit reached: {}", MAX_TOTAL_BUFFERS));
        }

        // Create new GPU buffer
        let mut gl_buf: GLuint = 0;
        let gl_target = match buffer_type {
            BufferType::VertexBuffer => GL_ARRAY_BUFFER,
            BufferType::IndexBuffer => GL_ELEMENT_ARRAY_BUFFER,
        };
        let gl_usage = match usage {
            BufferUsage::Immutable => GL_STATIC_DRAW,
            BufferUsage::Dynamic => GL_DYNAMIC_DRAW,
            BufferUsage::Stream => GL_STREAM_DRAW,
        };

        unsafe {
            glGenBuffers(1, &mut gl_buf as *mut _);
            if gl_buf == 0 {
                return Err("Failed to generate GL buffer".to_string());
            }

            // Pre-allocate buffer with the bucket size for optimal reuse
            glBindBuffer(gl_target, gl_buf);
            glBufferData(gl_target, size_bucket as _, std::ptr::null(), gl_usage);
            glBindBuffer(gl_target, 0);
        }

        let buffer = PooledBuffer {
            gl_buf,
            size: size_bucket,
            buffer_type,
            usage,
            last_used: std::time::Instant::now(),
        };

        self.active_buffers.insert(gl_buf, buffer.clone());
        self.stats.total_buffers += 1;
        self.stats.buffers_in_use += 1;
        self.stats.pool_allocations += 1;
        self.stats.memory_usage_bytes += size_bucket;

        Ok(gl_buf)
    }

    /// Release a buffer back to the pool
    pub fn release_buffer(&mut self, gl_buf: GLuint) -> Result<(), String> {
        let buffer = self
            .active_buffers
            .remove(&gl_buf)
            .ok_or_else(|| format!("Buffer {} not found in active buffers", gl_buf))?;

        let pool_key = PoolKey {
            buffer_type: buffer.buffer_type,
            usage: buffer.usage,
            size_bucket: buffer.size,
        };

        // Add to appropriate pool if not at capacity
        let pool = self.pools.entry(pool_key).or_default();

        if pool.len() < MAX_POOL_SIZE {
            pool.push(buffer);
            self.stats.buffers_in_use = self.stats.buffers_in_use.saturating_sub(1);
            self.stats.buffers_available += 1;
            self.stats.pool_deallocations += 1;
        } else {
            // Pool is full, actually delete the buffer
            unsafe {
                glDeleteBuffers(1, &gl_buf as *const _);
            }
            self.stats.total_buffers = self.stats.total_buffers.saturating_sub(1);
            self.stats.buffers_in_use = self.stats.buffers_in_use.saturating_sub(1);
            self.stats.memory_usage_bytes =
                self.stats.memory_usage_bytes.saturating_sub(buffer.size);
        }

        self.update_efficiency();
        Ok(())
    }

    /// Clean up old unused buffers to free memory
    pub fn cleanup_old_buffers(&mut self) {
        let now = std::time::Instant::now();
        let mut total_cleaned = 0;
        let mut memory_freed = 0;

        // Collect buffers to delete first to avoid borrow conflicts
        let mut buffers_to_delete = Vec::new();

        for (_pool_key, pool) in self.pools.iter_mut() {
            let before_len = pool.len();

            // Separate buffers into keep vs delete
            let mut i = 0;
            while i < pool.len() {
                if now.duration_since(pool[i].last_used) >= self.max_age {
                    let old_buffer = pool.swap_remove(i);
                    memory_freed += old_buffer.size;
                    buffers_to_delete.push(old_buffer.gl_buf);
                } else {
                    i += 1;
                }
            }

            let cleaned = before_len - pool.len();
            total_cleaned += cleaned;
        }

        // Delete the GL buffers
        for gl_buf in buffers_to_delete {
            unsafe {
                glDeleteBuffers(1, &gl_buf as *const _);
            }
        }

        // Update stats
        self.stats.total_buffers = self.stats.total_buffers.saturating_sub(total_cleaned);
        self.stats.buffers_available = self.stats.buffers_available.saturating_sub(total_cleaned);
        self.stats.memory_usage_bytes = self.stats.memory_usage_bytes.saturating_sub(memory_freed);

        // Remove empty pools
        self.pools.retain(|_, pool| !pool.is_empty());

        if total_cleaned > 0 {
            self.update_efficiency();
        }
    }

    /// Force cleanup of all pooled buffers (useful for context loss)
    pub fn clear_all(&mut self) {
        for (_, pool) in self.pools.iter() {
            for buffer in pool {
                unsafe {
                    glDeleteBuffers(1, &buffer.gl_buf as *const _);
                }
            }
        }

        // Also clean up active buffers if needed
        for (_, buffer) in self.active_buffers.iter() {
            unsafe {
                glDeleteBuffers(1, &buffer.gl_buf as *const _);
            }
        }

        self.pools.clear();
        self.active_buffers.clear();

        // Reset stats except hit/miss counters which are useful to keep
        let old_hits = self.stats.cache_hits;
        let old_misses = self.stats.cache_misses;
        let old_allocations = self.stats.pool_allocations;
        let old_deallocations = self.stats.pool_deallocations;
        let old_saved = self.stats.gpu_allocations_saved;

        self.stats = BufferPoolStats::default();
        self.stats.cache_hits = old_hits;
        self.stats.cache_misses = old_misses;
        self.stats.pool_allocations = old_allocations;
        self.stats.pool_deallocations = old_deallocations;
        self.stats.gpu_allocations_saved = old_saved;
    }

    /// Get current pool statistics
    pub fn get_stats(&self) -> BufferPoolStats {
        self.stats.clone()
    }

    /// Update pool efficiency calculation
    fn update_efficiency(&mut self) {
        if self.stats.total_buffers == 0 {
            self.stats.pool_efficiency = 100.0;
        } else {
            self.stats.pool_efficiency =
                (self.stats.buffers_in_use as f64 / self.stats.total_buffers as f64) * 100.0;
        }

        // Update saved allocations estimate
        self.stats.gpu_allocations_saved = self.stats.cache_hits;
    }

    /// Warm up the pool with commonly used buffer sizes
    pub fn warm_up(&mut self) -> Result<(), String> {
        let common_configs = [
            (BufferType::VertexBuffer, BufferUsage::Dynamic, 2048), // Common dynamic vertex buffer
            (BufferType::VertexBuffer, BufferUsage::Immutable, 8192), // Common static vertex buffer
            (BufferType::IndexBuffer, BufferUsage::Immutable, 2048), // Common index buffer
        ];

        for &(buffer_type, usage, size) in &common_configs {
            // Pre-allocate minimum number of buffers for each common config
            for _ in 0..MIN_POOL_SIZE {
                let gl_buf = self.acquire_buffer(buffer_type, usage, size)?;
                self.release_buffer(gl_buf)?;
            }
        }

        Ok(())
    }
}

impl Default for BufferPool {
    fn default() -> Self {
        Self::new()
    }
}
