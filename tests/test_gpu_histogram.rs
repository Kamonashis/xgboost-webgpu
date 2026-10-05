use rand::Rng;
use xgboost_webgpu::data::DMatrix;
use xgboost_webgpu::gpu::histogram::{build_histograms_cpu, GpuHistSession};
use xgboost_webgpu::gpu::GpuContext;

#[test]
fn test_gpu_histogram_matches_cpu() {
    if !GpuContext::is_available() {
        println!("WebGPU adapter not available, skipping test");
        return;
    }

    let mut rng = rand::rng();
    let nrows = 1000;
    let ncols = 8;

    let mut raw_data = Vec::with_capacity(nrows * ncols);
    for _ in 0..nrows * ncols {
        raw_data.push(rng.random_range(-10.0f32..10.0f32));
    }

    let mut labels = Vec::with_capacity(nrows);
    let mut grads = Vec::with_capacity(nrows);
    let mut hess = Vec::with_capacity(nrows);
    for _ in 0..nrows {
        labels.push(rng.random_range(0.0f32..1.0f32));
        grads.push(rng.random_range(-2.0f32..2.0f32));
        hess.push(rng.random_range(0.1f32..1.5f32));
    }

    let dmatrix = DMatrix::from_dense(&raw_data, nrows, ncols, Some(&labels), 64).unwrap();

    let context = GpuContext::new().expect("Failed to initialize GpuContext");
    println!("Testing on WebGPU adapter: {}", context.adapter_info.name);

    let session = GpuHistSession::new(&context, &dmatrix);
    session.update_gradients(&grads, &hess);

    let sample_indices: Vec<u32> = (0..nrows as u32).collect();

    let gpu_hists = session.build_histograms(&sample_indices, &grads, &hess);
    let cpu_hists = build_histograms_cpu(&dmatrix, &sample_indices, &grads, &hess);

    assert_eq!(gpu_hists.len(), ncols);
    assert_eq!(cpu_hists.len(), ncols);

    for j in 0..ncols {
        let gh = &gpu_hists[j];
        let ch = &cpu_hists[j];

        // Total G and H match closely
        let g_diff = (gh.total_g - ch.total_g).abs();
        let h_diff = (gh.total_h - ch.total_h).abs();

        assert!(
            g_diff < 0.5,
            "Feature {}: GPU total_g ({}) differs from CPU ({}) by {}",
            j, gh.total_g, ch.total_g, g_diff
        );
        assert!(
            h_diff < 0.5,
            "Feature {}: GPU total_h ({}) differs from CPU ({}) by {}",
            j, gh.total_h, ch.total_h, h_diff
        );

        // Individual bin values match within small tolerance
        for b in 0..256 {
            let bin_g_diff = (gh.g[b] - ch.g[b]).abs();
            let bin_h_diff = (gh.h[b] - ch.h[b]).abs();
            assert!(
                bin_g_diff < 0.1,
                "Feature {} Bin {}: GPU g ({}) vs CPU g ({})",
                j, b, gh.g[b], ch.g[b]
            );
            assert!(
                bin_h_diff < 0.1,
                "Feature {} Bin {}: GPU h ({}) vs CPU h ({})",
                j, b, gh.h[b], ch.h[b]
            );
        }
    }

    println!("GPU and CPU histograms matched successfully across all 8 features and 256 bins!");
}
