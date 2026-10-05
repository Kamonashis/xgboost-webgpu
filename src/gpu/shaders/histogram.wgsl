struct HistogramParams {
    num_node_samples: u32,
    num_features: u32,
    scale_g: f32,
    scale_h: f32,
};

@group(0) @binding(0) var<uniform> params: HistogramParams;
@group(0) @binding(1) var<storage, read> binned_data: array<u32>;
@group(0) @binding(2) var<storage, read> sample_indices: array<u32>;
@group(0) @binding(3) var<storage, read> sample_grads: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> histograms: array<atomic<i32>>;

fn get_bin(row: u32, col: u32, num_cols: u32) -> u32 {
    let byte_idx = row * num_cols + col;
    let word_idx = byte_idx / 4u;
    let byte_offset = (byte_idx % 4u) * 8u;
    let word = binned_data[word_idx];
    return (word >> byte_offset) & 0xFFu;
}

@compute @workgroup_size(64, 4, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let sample_k = global_id.x;
    let feat_j = global_id.y;

    if (sample_k >= params.num_node_samples || feat_j >= params.num_features) {
        return;
    }

    let sample_idx = sample_indices[sample_k];
    let bin = get_bin(sample_idx, feat_j, params.num_features);

    let gh = sample_grads[sample_idx];
    let g_int = i32(gh.x * params.scale_g);
    let h_int = i32(gh.y * params.scale_h);

    let base_idx = (feat_j * 256u + bin) * 2u;
    atomicAdd(&histograms[base_idx], g_int);
    atomicAdd(&histograms[base_idx + 1u], h_int);
}
