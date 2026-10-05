# xgboost-webgpu

Universal, blazing-fast, and memory-efficient XGBoost (Gradient Boosted Decision Trees) library in Rust, powered by **WebGPU** (`wgpu`) compute acceleration.

Standard XGBoost only supports GPU acceleration via NVIDIA CUDA (`device='cuda'` or `tree_method='gpu_hist'`). `xgboost-webgpu` breaks this hardware lock-in, enabling hardware-accelerated model training and inference across **any modern GPU**:
- **AMD Radeon GPUs** (via Vulkan on Linux / Windows)
- **Apple Silicon M1/M2/M3/M4** (via Metal on macOS / iOS)
- **Intel Arc & Integrated GPUs** (via Vulkan / DirectX 12)
- **NVIDIA GPUs** (via Vulkan / DirectX 12)
- **Web Browsers & WebAssembly** (via native WebGPU WGSL)
- **Multi-threaded CPU Fallback** (via Rayon)

---

## Why WebGPU Instead of WebGL or CUDA?

| Feature | CUDA (Standard XGBoost) | WebGL 2.0 | WebGPU (`xgboost-webgpu`) |
| :--- | :--- | :--- | :--- |
| **Vendor Support** | NVIDIA Only | Universal | **Universal (AMD, Apple, Intel, NVIDIA)** |
| **Compute Shaders** | Yes | No (Graphics raster only) | **Yes (`@compute`, WGSL)** |
| **Memory Buffers** | Direct VRAM buffers | Textures only (2D ping-pong) | **Storage Buffers (`read_write`)** |
| **Atomic Operations** | Yes | No atomic arbitrary writes | **Yes (`atomicAdd`, workgroup atomics)** |
| **Headless / Server** | Yes | Cumbersome / requires X11/EGL | **Native headless execution** |
| **Browser Deployment**| No | Slow emulation | **Native WebGPU Wasm support** |

---

## Key Architectural Highlights

1. **Histogram Quantization (`DMatrix`)**:
   - Quantizes continuous features into 256 discrete bins (`u8`), compressing feature storage by **75%** and maximizing GPU memory bandwidth.
   - Packs 4 bins per 32-bit word for aligned, high-throughput GPU storage buffer reads.

2. **Parallel WGSL Compute Shaders**:
   - Dispatches 2D compute workgroups across samples and features.
   - Fixed-point scaled atomic accumulation guarantees 100% universal hardware compatibility without requiring optional float-atomic extensions.

3. **Histogram Subtraction Trick**:
   - For sibling nodes after a split, computes histograms only for the child with fewer samples; derives the sibling histogram in $O(1)$ child operations:
     $$H_{\text{sibling}} = H_{\text{parent}} - H_{\text{child}}$$
   - Halves GPU compute operations across every level of the tree.

4. **Official XGBoost JSON Interoperability**:
   - Exports and loads models using the standard **XGBoost JSON model format**.
   - Train on WebGPU in Rust, deploy anywhere (Python XGBoost, Treelite, C++, etc.).

5. **Multi-threaded CPU Fallback**:
   - Automatic or manual fallback (`DeviceType::Cpu`) parallelized via `rayon`.

---

## Quickstart

Add `xgboost-webgpu` to your `Cargo.toml`:

```toml
[dependencies]
xgboost-webgpu = "0.1.0"
```

### Regression Example

```rust
use xgboost_webgpu::{BoosterParams, DeviceType, DMatrix, train};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x_train = vec![
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
        10.0, 11.0, 12.0,
    ];
    let y_train = vec![1.5, 4.2, 7.1, 10.8];

    // 1. Build DMatrix
    let dtrain = DMatrix::from_dense(&x_train, 4, 3, Some(&y_train), 256)?;

    // 2. Configure training parameters
    let params = BoosterParams::new()
        .with_objective("reg:squarederror")
        .with_max_depth(5)
        .with_learning_rate(0.2)
        .with_device(DeviceType::WebGPU); // Accelerate on AMD, Apple, Intel, Nvidia

    // 3. Train model
    let booster = train(params, &dtrain, 20, &[])?;

    // 4. Predict
    let predictions = booster.predict(&dtrain)?;
    println!("Predictions: {:?}", predictions);

    // 5. Export to standard XGBoost JSON format
    booster.save_model("model.json")?;
    println!("Model saved to model.json (compatible with Python xgboost.Booster)");

    Ok(())
}
```

### Binary Classification Example

```rust
use xgboost_webgpu::{BoosterParams, DeviceType, DMatrix, train, metrics};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x_data = vec![/* features */];
    let y_data = vec![0.0, 1.0, 0.0, 1.0];

    let dtrain = DMatrix::from_dense(&x_data, 4, 2, Some(&y_data), 256)?;

    let params = BoosterParams::new()
        .with_objective("binary:logistic")
        .with_max_depth(4)
        .with_learning_rate(0.3)
        .with_device(DeviceType::WebGPU);

    let booster = train(params, &dtrain, 15, &[])?;
    let probs = booster.predict(&dtrain)?;

    let logloss = metrics::logloss(&y_data, &probs);
    println!("Train LogLoss: {:.4}", logloss);

    Ok(())
}
```

---

## Running Benchmarks and Examples

```bash
# Run California Housing regression benchmark on your GPU
cargo run --release --example california_housing

# Run non-linear binary classification benchmark on your GPU
cargo run --release --example binary_classification

# Run automated test suite (including GPU-vs-CPU histogram validation)
cargo test
```

---

## License

Licensed under the MIT License. See [LICENSE](LICENSE) for details.
