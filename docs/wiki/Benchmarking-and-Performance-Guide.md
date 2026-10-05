# Benchmarking & Performance Guide

This guide details the performance characteristics of `xgboost-webgpu`, memory footprint optimizations, hardware-specific considerations, and tuning strategies.

---

## 1. Real Hardware Benchmarks

All benchmarks below were executed on real hardware using the built-in examples with WebGPU hardware acceleration (`DeviceType::WebGPU`).

### Test Setup
* **Hardware**: AMD Radeon Graphics (`RADV RENOIR`, Vulkan 1.3)
* **OS**: Linux (x86_64)
* **Rust Profile**: `--release` (opt-level = 3)

| Benchmark | Dataset Specs | Parameters | WebGPU Execution Time | Model Metric |
| :--- | :--- | :--- | :--- | :--- |
| **California Housing** | 20,640 samples, 8 features | 30 trees, `max_depth=6`, `lr=0.1` | **662 ms** | RMSE: 0.528 |
| **Binary Classification** | 5,000 samples, 10 features | 25 trees, `max_depth=5`, `lr=0.15` | **245 ms** | Accuracy: 91.6% |
| **Advanced Features** | 3,000 samples, 6 features | 30 trees, `LossGuide`, `subsample=0.8` | **268 ms** | Multi-Class Accuracy: 88.5% |

---

## 2. Memory Footprint Optimizations

### Quantized Storage Buffers
Standard floating-point GBDTs store raw features as 32-bit floats (`f32`), requiring $4 \times N \times M$ bytes of VRAM.

`xgboost-webgpu` maps continuous features to $B = 256$ bins, storing them as unsigned 8-bit integers (`u8`):
* **Memory Reduction**: **75%** smaller VRAM footprint.
* **Packing**: Four consecutive bins are packed into a single 32-bit unsigned integer (`u32`) for GPU storage buffers. A dataset of 100,000 samples with 50 features requires only **5 MB** of GPU buffer storage instead of 20 MB.

### Sibling Subtraction Efficiency
When splitting a node with $N$ samples, computing histograms on the GPU takes time proportional to $N$. By dispatching only the smaller child node ($\le N/2$ samples) to the GPU and computing the larger child via $O(B)$ subtraction on the CPU, total VRAM transfer overhead is reduced by **up to 50%**.

---

## 3. Hardware-Specific Considerations

### AMD Radeon GPUs (Linux / Windows)
* **Backend**: Vulkan via Mesa (`RADV`) or AMDVLK on Linux; Vulkan/DX12 on Windows.
* **Performance Note**: AMD GPUs excel at parallel integer atomic operations. Ensure Mesa drivers are up to date (`vulkan-radeon`).

### Apple Silicon (M1, M2, M3, M4)
* **Backend**: Native Metal 2.4+ via `wgpu`.
* **Unified Memory Benefit**: Apple Silicon shares physical memory between CPU and GPU, eliminating PCI-e bus transfer overhead between host memory and VRAM buffers.

### Intel Arc & Iris Xe GPUs
* **Backend**: Vulkan / DirectX 12.
* **Performance Note**: Intel's Gen12+ execution units handle the 2D workgroup tile dispatch ($16 \times 16$) effectively for histogram bin aggregation.

### NVIDIA GPUs
* **Backend**: Vulkan / DirectX 12.
* **Performance Note**: While NVIDIA users traditionally rely on CUDA, `xgboost-webgpu` runs without requiring the CUDA Toolkit or proprietary NVIDIA SDKs installed.

### WebAssembly & Web Browsers
* **Backend**: Native browser WebGPU.
* **In-Browser ML**: Enables training models on user data completely client-side in the browser without server round-trips, ensuring data privacy and zero cloud compute costs.

---

## 4. Hyperparameter Tuning for GPU Performance

1. **`max_bins` (Default: 256)**:
   * 256 bins perfectly fits in a `u8` byte and maximizes GPU cache efficiency. Reducing to 128 or 64 speeds up histogram reduction if feature precision allows.
2. **`grow_policy = "lossguide"` with `max_leaves`**:
   * Leaf-wise growth focuses computation on splits with the largest loss reduction, requiring fewer total trees to achieve comparable accuracy.
3. **`subsample` (0.7 - 0.9)**:
   * Subsampling reduces the number of samples processed per boosting round on the GPU while acting as an effective regularizer.
