//! Performance benchmark to identify bottlenecks in miniquad
//!
//! This benchmark measures key performance metrics:
//! - Buffer creation/deletion overhead
//! - Texture binding performance  
//! - GPU state change frequency
//! - Draw call batching efficiency

use miniquad::*;
use std::time::Instant;

struct PerformanceBenchmark {
    ctx: Box<dyn RenderingBackend>,

    // Test resources
    test_textures: Vec<TextureId>,
    test_shader: Option<ShaderId>,
    test_pipeline: Option<Pipeline>,

    // Metrics tracking
    frame_count: u32,

    // Performance counters
    buffer_creates: u32,
    draw_calls: u32,
    state_changes: u32,
}

impl PerformanceBenchmark {
    fn run_buffer_benchmark(&mut self) {
        println!("\n=== Buffer Creation Benchmark ===");

        // Small buffer benchmark (1KB)
        let small_data = vec![0u8; 1024];
        let start = Instant::now();

        for _ in 0..1000 {
            let buffer = self.ctx.new_buffer(
                BufferType::VertexBuffer,
                BufferUsage::Immutable,
                BufferSource::slice(&small_data),
            );
            self.ctx.delete_buffer(buffer);
            self.buffer_creates += 1;
        }

        let small_time = start.elapsed();
        println!(
            "Small buffers (1KB): {:?} for 1000 creates/deletes ({:.2} µs/op)",
            small_time,
            small_time.as_micros() as f64 / 1000.0
        );

        // Medium buffer benchmark (64KB)
        let medium_data = vec![0u8; 64 * 1024];
        let start = Instant::now();

        for _ in 0..100 {
            let buffer = self.ctx.new_buffer(
                BufferType::VertexBuffer,
                BufferUsage::Immutable,
                BufferSource::slice(&medium_data),
            );
            self.ctx.delete_buffer(buffer);
            self.buffer_creates += 1;
        }

        let medium_time = start.elapsed();
        println!(
            "Medium buffers (64KB): {:?} for 100 creates/deletes ({:.2} µs/op)",
            medium_time,
            medium_time.as_micros() as f64 / 100.0
        );

        // Large buffer benchmark (1MB)
        let large_data = vec![0u8; 1024 * 1024];
        let start = Instant::now();

        for _ in 0..10 {
            let buffer = self.ctx.new_buffer(
                BufferType::VertexBuffer,
                BufferUsage::Immutable,
                BufferSource::slice(&large_data),
            );
            self.ctx.delete_buffer(buffer);
            self.buffer_creates += 1;
        }

        let large_time = start.elapsed();
        println!(
            "Large buffers (1MB): {:?} for 10 creates/deletes ({:.2} µs/op)",
            large_time,
            large_time.as_micros() as f64 / 10.0
        );
    }

    fn run_texture_benchmark(&mut self) {
        println!("\n=== Texture Creation & Binding Benchmark ===");

        // Texture creation benchmark
        let texture_data_64 = vec![255u8; 64 * 64 * 4];
        let start = Instant::now();

        for _ in 0..100 {
            let texture = self.ctx.new_texture_from_data_and_format(
                &texture_data_64,
                TextureParams {
                    width: 64,
                    height: 64,
                    format: TextureFormat::RGBA8,
                    ..Default::default()
                },
            );
            self.ctx.delete_texture(texture);
        }

        let tex_create_time = start.elapsed();
        println!(
            "64x64 texture creation: {:?} for 100 creates/deletes ({:.2} µs/op)",
            tex_create_time,
            tex_create_time.as_micros() as f64 / 100.0
        );

        // Create persistent textures for binding test
        self.test_textures.clear();
        for i in 0..8 {
            let data = vec![(i * 32) as u8; 64 * 64 * 4];
            let texture = self.ctx.new_texture_from_data_and_format(
                &data,
                TextureParams {
                    width: 64,
                    height: 64,
                    format: TextureFormat::RGBA8,
                    ..Default::default()
                },
            );
            self.test_textures.push(texture);
        }
    }

    fn run_state_change_benchmark(&mut self) {
        println!("\n=== GPU State Change Benchmark ===");

        // Create test vertex buffer
        let vertex_data: Vec<f32> = vec![
            -0.5, -0.5, 1.0, 0.0, 0.0, 1.0, 0.5, -0.5, 0.0, 1.0, 0.0, 1.0, 0.0, 0.5, 0.0, 0.0, 1.0,
            1.0,
        ];

        let vertex_buffer = self.ctx.new_buffer(
            BufferType::VertexBuffer,
            BufferUsage::Immutable,
            BufferSource::slice(&vertex_data),
        );

        // Create proper index buffer
        let index_data: Vec<u16> = vec![0, 1, 2];
        let index_buffer = self.ctx.new_buffer(
            BufferType::IndexBuffer,
            BufferUsage::Immutable,
            BufferSource::slice(&index_data),
        );

        // Single draw call benchmark
        let start = Instant::now();
        let iterations = 1000;

        for _ in 0..iterations {
            self.ctx
                .begin_default_pass(PassAction::clear_color(0.0, 0.0, 0.0, 1.0));
            self.ctx
                .apply_pipeline(self.test_pipeline.as_ref().unwrap());
            self.ctx.apply_bindings(&Bindings {
                vertex_buffers: vec![vertex_buffer],
                index_buffer,
                images: vec![],
            });
            self.ctx.draw(0, 3, 1);
            self.ctx.end_render_pass();
            self.draw_calls += 1;
        }

        let single_draw_time = start.elapsed();
        println!(
            "Single draw calls: {:?} for {} calls ({:.2} µs/call)",
            single_draw_time,
            iterations,
            single_draw_time.as_micros() as f64 / iterations as f64
        );

        // Multiple draw calls with same state
        let start = Instant::now();
        let batches = 100;
        let draws_per_batch = 10;

        for _ in 0..batches {
            self.ctx
                .begin_default_pass(PassAction::clear_color(0.0, 0.0, 0.0, 1.0));
            self.ctx
                .apply_pipeline(self.test_pipeline.as_ref().unwrap());
            self.ctx.apply_bindings(&Bindings {
                vertex_buffers: vec![vertex_buffer],
                index_buffer,
                images: vec![],
            });

            for _ in 0..draws_per_batch {
                self.ctx.draw(0, 3, 1);
                self.draw_calls += 1;
            }
            self.ctx.end_render_pass();
        }

        let batch_draw_time = start.elapsed();
        println!(
            "Batched draw calls: {:?} for {} batches of {} draws ({:.2} µs/draw)",
            batch_draw_time,
            batches,
            draws_per_batch,
            batch_draw_time.as_micros() as f64 / (batches * draws_per_batch) as f64
        );

        // State changing benchmark
        let bindings1 = Bindings {
            vertex_buffers: vec![vertex_buffer],
            index_buffer,
            images: if !self.test_textures.is_empty() {
                vec![self.test_textures[0]]
            } else {
                vec![]
            },
        };
        let bindings2 = Bindings {
            vertex_buffers: vec![vertex_buffer],
            index_buffer,
            images: if self.test_textures.len() > 1 {
                vec![self.test_textures[1]]
            } else {
                vec![]
            },
        };

        let start = Instant::now();
        let state_change_iterations = 100;

        for _ in 0..state_change_iterations {
            self.ctx
                .begin_default_pass(PassAction::clear_color(0.0, 0.0, 0.0, 1.0));
            self.ctx
                .apply_pipeline(self.test_pipeline.as_ref().unwrap());

            for i in 0..20 {
                if i % 2 == 0 {
                    self.ctx.apply_bindings(&bindings1);
                } else {
                    self.ctx.apply_bindings(&bindings2);
                }
                self.ctx.draw(0, 3, 1);
                self.state_changes += 1;
            }
            self.ctx.end_render_pass();
        }

        let state_change_time = start.elapsed();
        println!(
            "State changing draws: {:?} for {} batches of 20 alternating draws ({:.2} µs/draw)",
            state_change_time,
            state_change_iterations,
            state_change_time.as_micros() as f64 / (state_change_iterations * 20) as f64
        );

        self.ctx.delete_buffer(vertex_buffer);
        self.ctx.delete_buffer(index_buffer);
    }

    fn run_all_benchmarks(&mut self) {
        println!("=== MINIQUAD PERFORMANCE BENCHMARK ===");
        println!("This will identify performance bottlenecks for optimization");

        // Initialize and enable profiling to measure state change redundancy
        miniquad::graphics::profiling::init_profiler();
        miniquad::graphics::profiling::enable_profiling();
        miniquad::graphics::profiling::reset_profiling();

        let total_start = Instant::now();

        self.run_buffer_benchmark();
        self.run_texture_benchmark();
        self.run_state_change_benchmark();

        let total_time = total_start.elapsed();

        println!("\n=== BENCHMARK SUMMARY ===");
        println!("Total benchmark time: {:?}", total_time);
        println!("Buffer creates: {}", self.buffer_creates);
        println!("Draw calls: {}", self.draw_calls);
        println!("State changes: {}", self.state_changes);

        // Cleanup
        for texture in &self.test_textures {
            self.ctx.delete_texture(*texture);
        }
        self.test_textures.clear();

        println!("\nBenchmark complete! Use these results to prioritize optimizations.");
        println!("Key areas to focus on:");
        println!("- Buffer allocation patterns (if buffer creates are slow)");
        println!("- GPU state caching (if state changes are expensive)");
        println!("- Draw call batching (compare single vs batched performance)");

        // Print profiling report showing state change redundancy
        println!();
        miniquad::graphics::profiling::print_report();
    }
}

impl EventHandler for PerformanceBenchmark {
    fn update(&mut self) {
        // Run benchmarks once on first update
        if self.frame_count == 0 {
            // Setup test resources first
            let shader = self
                .ctx
                .new_shader(
                    ShaderSource::Glsl {
                        vertex: r#"#version 100
                        attribute vec2 pos;
                        attribute vec4 color;
                        varying lowp vec4 color0;
                        void main() {
                            gl_Position = vec4(pos, 0, 1);
                            color0 = color;
                        }"#,
                        fragment: r#"#version 100
                        varying lowp vec4 color0;
                        void main() {
                            gl_FragColor = color0;
                        }"#,
                    },
                    ShaderMeta {
                        images: vec![],
                        uniforms: UniformBlockLayout { uniforms: vec![] },
                    },
                )
                .expect("Failed to create test shader");
            self.test_shader = Some(shader);

            let pipeline = self.ctx.new_pipeline(
                &[BufferLayout::default()],
                &[
                    VertexAttribute::new("pos", VertexFormat::Float2),
                    VertexAttribute::new("color", VertexFormat::Float4),
                ],
                shader,
                PipelineParams::default(),
            );
            self.test_pipeline = Some(pipeline);

            // Run all benchmarks
            self.run_all_benchmarks();
        }

        self.frame_count += 1;

        // Exit after benchmarks complete
        if self.frame_count > 10 {
            std::process::exit(0);
        }
    }

    fn draw(&mut self) {
        // Simple render to keep the window responsive
        self.ctx
            .begin_default_pass(PassAction::clear_color(0.1, 0.2, 0.3, 1.0));
        self.ctx.end_render_pass();
        self.ctx.commit_frame();
    }
}

fn main() {
    let conf = conf::Conf {
        window_title: "Miniquad Performance Benchmark".to_string(),
        window_width: 800,
        window_height: 600,
        ..Default::default()
    };

    miniquad::start(conf, || {
        Box::new(PerformanceBenchmark {
            ctx: window::new_rendering_backend(),
            test_textures: Vec::new(),
            test_shader: None,   // Will be set properly in update
            test_pipeline: None, // Will be set properly in update
            frame_count: 0,
            buffer_creates: 0,
            draw_calls: 0,
            state_changes: 0,
        })
    });
}
