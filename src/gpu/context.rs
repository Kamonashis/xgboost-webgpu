use futures_executor::block_on;
use wgpu::*;

/// Encapsulates GPU device, queue, and precompiled compute pipelines.
pub struct GpuContext {
    pub instance: Instance,
    pub adapter_info: AdapterInfo,
    pub device: Device,
    pub queue: Queue,
    pub histogram_pipeline: ComputePipeline,
    pub histogram_bgl: BindGroupLayout,
}

impl GpuContext {
    /// Initialize headless GPU context, picking the highest performance available adapter.
    pub fn new() -> Result<Self, String> {
        let instance = Instance::default();

        let adapter = block_on(instance.request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .map_err(|e| format!("Failed to request WebGPU adapter: {:?}", e))?;

        let adapter_info = adapter.get_info();

        let (device, queue) = block_on(adapter.request_device(&DeviceDescriptor {
            label: Some("xgboost-webgpu-device"),
            ..Default::default()
        }))
        .map_err(|e| format!("Failed to create WebGPU device: {:?}", e))?;

        let shader_src = include_str!("shaders/histogram.wgsl");
        let shader_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("xgboost_histogram_shader"),
            source: ShaderSource::Wgsl(shader_src.into()),
        });

        let histogram_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("histogram_bgl"),
            entries: &[
                // binding 0: uniform params
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 1: storage binned_data
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 2: storage sample_indices
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 3: storage sample_grads
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 4: storage histograms (atomic<i32>)
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("histogram_pipeline_layout"),
            bind_group_layouts: &[Some(&histogram_bgl)],
            immediate_size: 0,
        });

        let histogram_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("histogram_compute_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        Ok(Self {
            instance,
            adapter_info,
            device,
            queue,
            histogram_pipeline,
            histogram_bgl,
        })
    }

    /// Check if a WebGPU adapter is available on this system.
    pub fn is_available() -> bool {
        let instance = Instance::default();
        block_on(instance.request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .is_ok()
    }
}
