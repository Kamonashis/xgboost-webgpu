use crate::data::DMatrix;
use crate::gpu::context::GpuContext;
use rayon::prelude::*;
use std::sync::mpsc;
use wgpu::util::DeviceExt;
use wgpu::*;

/// Histogram for a single feature containing 256 bins of (G, H).
#[derive(Debug, Clone)]
pub struct FeatureHistogram {
    pub g: [f32; 256],
    pub h: [f32; 256],
    pub total_g: f32,
    pub total_h: f32,
}

impl FeatureHistogram {
    pub fn empty() -> Self {
        Self {
            g: [0.0; 256],
            h: [0.0; 256],
            total_g: 0.0,
            total_h: 0.0,
        }
    }

    /// Compute sibling histogram by subtraction: Sibling = Parent - Child.
    pub fn subtract(parent: &Self, child: &Self) -> Self {
        let mut sib = Self::empty();
        let mut total_g = 0.0;
        let mut total_h = 0.0;

        for k in 0..256 {
            let g = (parent.g[k] - child.g[k]).max(-1e12);
            let h = (parent.h[k] - child.h[k]).max(0.0);
            sib.g[k] = g;
            sib.h[k] = h;
            total_g += g;
            total_h += h;
        }

        sib.total_g = total_g;
        sib.total_h = total_h;
        sib
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct HistogramUniforms {
    num_node_samples: u32,
    num_features: u32,
    scale_g: f32,
    scale_h: f32,
}

/// GPU session buffers preallocated for training a DMatrix on WebGPU.
pub struct GpuHistSession<'a> {
    pub context: &'a GpuContext,
    pub nrows: usize,
    pub ncols: usize,
    pub binned_data_buffer: Buffer,
    pub sample_grads_buffer: Buffer,
    pub hist_buffer: Buffer,
    pub readback_buffer: Buffer,
    pub hist_buffer_size: u64,
}

impl<'a> GpuHistSession<'a> {
    /// Allocate GPU buffers and upload static binned matrix data.
    pub fn new(context: &'a GpuContext, dmatrix: &DMatrix) -> Self {
        let nrows = dmatrix.nrows;
        let ncols = dmatrix.ncols;

        // Pack row-major u8 binned data into u32 words (4 bytes per u32)
        let total_bytes = nrows * ncols;
        let num_u32 = (total_bytes + 3) / 4;
        let mut packed_u32 = vec![0u32; num_u32];

        // Zero-copy or safe copy into u32 slice
        let byte_slice: &mut [u8] = bytemuck::cast_slice_mut(&mut packed_u32);
        byte_slice[..total_bytes].copy_from_slice(&dmatrix.binned_data);

        let binned_data_buffer = context.device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some("gpu_binned_data"),
            contents: bytemuck::cast_slice(&packed_u32),
            usage: BufferUsages::STORAGE,
        });

        // Gradients buffer: vec2<f32> per sample
        let sample_grads_buffer = context.device.create_buffer(&BufferDescriptor {
            label: Some("gpu_sample_grads"),
            size: (nrows * std::mem::size_of::<[f32; 2]>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Histogram buffer: ncols * 256 * 2 * 4 bytes (atomic<i32>)
        let hist_buffer_size = (ncols * 256 * 2 * std::mem::size_of::<i32>()) as u64;
        let hist_buffer = context.device.create_buffer(&BufferDescriptor {
            label: Some("gpu_histograms"),
            size: hist_buffer_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let readback_buffer = context.device.create_buffer(&BufferDescriptor {
            label: Some("gpu_hist_readback"),
            size: hist_buffer_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            context,
            nrows,
            ncols,
            binned_data_buffer,
            sample_grads_buffer,
            hist_buffer,
            readback_buffer,
            hist_buffer_size,
        }
    }

    /// Upload current boosting round gradients and Hessians to GPU.
    pub fn update_gradients(&self, grads: &[f32], hess: &[f32]) {
        assert_eq!(grads.len(), self.nrows);
        assert_eq!(hess.len(), self.nrows);

        let mut gh_pairs = Vec::with_capacity(self.nrows * 2);
        for i in 0..self.nrows {
            gh_pairs.push(grads[i]);
            gh_pairs.push(hess[i]);
        }

        self.context.queue.write_buffer(
            &self.sample_grads_buffer,
            0,
            bytemuck::cast_slice(&gh_pairs),
        );
    }

    /// Compute histograms on GPU for a given list of sample indices.
    pub fn build_histograms(
        &self,
        sample_indices: &[u32],
        grads: &[f32],
        hess: &[f32],
    ) -> Vec<FeatureHistogram> {
        let num_node_samples = sample_indices.len();
        if num_node_samples == 0 {
            return vec![FeatureHistogram::empty(); self.ncols];
        }

        // Determine dynamic fixed-point scaling factor to prevent atomic<i32> overflow
        let mut max_g: f32 = 1.0;
        let mut max_h: f32 = 1.0;
        for &idx in sample_indices {
            let i = idx as usize;
            max_g = max_g.max(grads[i].abs());
            max_h = max_h.max(hess[i].abs());
        }

        let max_val = max_g.max(max_h);
        let scale = ((1.8e9 / ((num_node_samples as f64) * (max_val as f64) + 1.0)) as f32)
            .clamp(1.0, 65536.0);

        let uniforms = HistogramUniforms {
            num_node_samples: num_node_samples as u32,
            num_features: self.ncols as u32,
            scale_g: scale,
            scale_h: scale,
        };

        // Create uniform buffer
        let uniform_buffer = self.context.device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some("hist_uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: BufferUsages::UNIFORM,
        });

        // Create node sample indices buffer
        let sample_indices_buffer = self.context.device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some("node_sample_indices"),
            contents: bytemuck::cast_slice(sample_indices),
            usage: BufferUsages::STORAGE,
        });

        // Zero out the histogram buffer on GPU
        let zero_data = vec![0u8; self.hist_buffer_size as usize];
        self.context.queue.write_buffer(&self.hist_buffer, 0, &zero_data);

        // Bind group
        let bind_group = self.context.device.create_bind_group(&BindGroupDescriptor {
            label: Some("hist_bind_group"),
            layout: &self.context.histogram_bgl,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: self.binned_data_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: sample_indices_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: self.sample_grads_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: self.hist_buffer.as_entire_binding(),
                },
            ],
        });

        let workgroups_x = ((num_node_samples as u32) + 63) / 64;
        let workgroups_y = ((self.ncols as u32) + 3) / 4;

        let mut encoder = self.context.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("hist_encoder"),
        });

        {
            let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("hist_cpass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.context.histogram_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        encoder.copy_buffer_to_buffer(&self.hist_buffer, 0, &self.readback_buffer, 0, self.hist_buffer_size);
        self.context.queue.submit(Some(encoder.finish()));

        // Map readback buffer
        let slice = self.readback_buffer.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(MapMode::Read, move |res| {
            let _ = tx.send(res);
        });

        self.context.instance.poll_all(true);
        rx.recv()
            .map_err(|e| format!("Channel recv error: {:?}", e))
            .and_then(|r| r.map_err(|e| format!("Map buffer error: {:?}", e)))
            .expect("Failed to map histogram readback buffer");

        let mapped_view = slice.get_mapped_range();
        let int_hist: &[i32] = bytemuck::cast_slice(&mapped_view);

        // Convert scaled i32 results back to f32 FeatureHistograms
        let inv_scale = 1.0 / scale;
        let mut feature_hists = Vec::with_capacity(self.ncols);

        for j in 0..self.ncols {
            let mut hist = FeatureHistogram::empty();
            let mut total_g = 0.0f32;
            let mut total_h = 0.0f32;

            let feat_offset = j * 256 * 2;
            for k in 0..256 {
                let g = (int_hist[feat_offset + k * 2] as f32) * inv_scale;
                let h = ((int_hist[feat_offset + k * 2 + 1] as f32) * inv_scale).max(0.0);
                hist.g[k] = g;
                hist.h[k] = h;
                total_g += g;
                total_h += h;
            }

            hist.total_g = total_g;
            hist.total_h = total_h;
            feature_hists.push(hist);
        }

        drop(mapped_view);
        self.readback_buffer.unmap();

        feature_hists
    }
}

/// High-performance CPU multi-threaded histogram construction using Rayon.
pub fn build_histograms_cpu(
    dmatrix: &DMatrix,
    sample_indices: &[u32],
    grads: &[f32],
    hess: &[f32],
) -> Vec<FeatureHistogram> {
    let ncols = dmatrix.ncols;
    if sample_indices.is_empty() {
        return vec![FeatureHistogram::empty(); ncols];
    }

    (0..ncols)
        .into_par_iter()
        .map(|j| {
            let mut hist = FeatureHistogram::empty();
            let mut total_g = 0.0f32;
            let mut total_h = 0.0f32;

            for &idx in sample_indices {
                let row = idx as usize;
                let bin = dmatrix.get_bin(row, j) as usize;
                let g = grads[row];
                let h = hess[row];
                hist.g[bin] += g;
                hist.h[bin] += h;
                total_g += g;
                total_h += h;
            }

            hist.total_g = total_g;
            hist.total_h = total_h;
            hist
        })
        .collect()
}
