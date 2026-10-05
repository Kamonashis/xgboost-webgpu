# Welcome to the xgboost-webgpu Wiki

[![PyPI version](https://img.shields.io/pypi/v/xgboost-webgpu.svg)](https://pypi.org/project/xgboost-webgpu/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**`xgboost-webgpu`** is an industrial-grade, pure-Rust implementation of the XGBoost Gradient Boosted Decision Tree (GBDT) algorithm with **universal hardware acceleration via WebGPU (`wgpu`)**.

Standard XGBoost restricts GPU acceleration exclusively to NVIDIA hardware via CUDA (`device='cuda'` or `tree_method='gpu_hist'`). For practitioners on **AMD Radeon GPUs**, **Apple Silicon Macs (M1/M2/M3/M4)**, **Intel Arc GPUs**, and **web browsers**, standard XGBoost falls back to CPU execution. 

`xgboost-webgpu` breaks this hardware lock-in, delivering high-performance GPU-accelerated gradient boosting across all modern GPU vendors with exact algorithmic parity to native XGBoost.

---

## Hardware Compatibility Matrix

`xgboost-webgpu` targets the modern WebGPU compute standard through the `wgpu` ecosystem, dynamically selecting the optimal native graphics API on each platform without any vendor proprietary SDKs:

| GPU Architecture | Operating System | Underlying Backend | WebGPU Support |
| :--- | :--- | :--- | :--- |
| **AMD Radeon (RX, Pro, APU)** | Linux, Windows | Vulkan 1.2+ | **Full Acceleration** |
| **Apple Silicon (M1/M2/M3/M4)** | macOS, iOS | Metal 2.4+ | **Full Acceleration** |
| **Intel Arc & Iris Xe** | Linux, Windows | Vulkan / DirectX 12 | **Full Acceleration** |
| **NVIDIA GeForce / RTX / Tesla** | Linux, Windows | Vulkan / DirectX 12 | **Full Acceleration** |
| **Web Browsers / WebAssembly** | Chrome, Edge, Firefox | WebGPU (Wasm) | **Full In-Browser** |
| **CPU Fallback** | All | Multi-threaded Rayon | **Full Multi-Core** |

---

## Core Wiki Navigation

Explore detailed guides, mathematical foundations, and implementation details:

* [Architecture & WebGPU Compute Pipeline](Architecture-and-WebGPU.md)
  * WGSL compute shader architecture, 2D workgroup dispatch, fixed-point integer atomics (`atomic<i32>`), compact 256-bin quantization, and $O(1)$ sibling histogram subtraction.
* [Objectives & Loss Functions](Objectives-and-Loss-Functions.md)
  * Full mathematical formulations, gradients, and Hessians for Regression, Logistic Binary, Softmax/Softprob Multi-Class, Poisson Count, Gamma, Tweedie, Quantile, and Pairwise Ranking.
* [Tree Growth & Policies](Tree-Building-and-Growth-Policies.md)
  * `DepthWise` level-by-level vs. `LossGuide` leaf-wise expansion, $L_1$/$L_2$ regularization, step clamping (`max_delta_step`), row/column subsampling, and monotonic/interaction constraints.
* [DART Booster & Early Stopping](Advanced-Boosting-and-DART.md)
  * Dropouts meet Multiple Additive Regression Trees (DART), early stopping evaluation, and model continuation.
* [Model Interpretability & TreeSHAP](Model-Interpretability-and-TreeSHAP.md)
  * Gain/Weight/Cover feature importances, exact TreeSHAP local attribution polynomial algorithm, and leaf index prediction.
* [Rust API Reference](Rust-API-Reference.md)
  * Rust API guide: `DMatrix`, `BoosterParams`, `train()`, `Booster`, and official XGBoost JSON model serialization.
* [Python Bindings & Scikit-Learn API](Python-Bindings-and-Scikit-Learn-API.md)
  * PyO3 Python bindings with Python Stable ABI (`abi3-py310`), drop-in `XGBRegressor`, `XGBClassifier`, and NumPy integration.
* [Benchmarking & Performance Guide](Benchmarking-and-Performance-Guide.md)
  * Performance profiling, GPU memory footprints, and multi-vendor hardware optimization tips.

---

## Quick Example: 10-Second Setup

### Rust
```rust
use xgboost_webgpu::{BoosterParams, DeviceType, DMatrix, train};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3 rows, 2 cols
    let y = vec![10.0, 20.0, 30.0];

    let dtrain = DMatrix::from_dense(&x, 3, 2, Some(&y), 256)?;
    let params = BoosterParams::new()
        .with_objective("reg:squarederror")
        .with_learning_rate(0.1)
        .with_device(DeviceType::WebGPU); // GPU accelerated!

    let bst = train(params, &dtrain, 20, &[])?;
    let preds = bst.predict(&dtrain)?;
    println!("Predictions: {:?}", preds);
    Ok(())
}
```

### Python
```python
# Install via: pip install xgboost-webgpu
import numpy as np
import xgboost_webgpu as xgb

X = np.random.randn(1000, 10).astype(np.float32)
y = (X[:, 0] + X[:, 1] > 0).astype(np.float32)

# Scikit-learn estimator with WebGPU acceleration
clf = xgb.XGBClassifier(n_estimators=30, learning_rate=0.1, device="webgpu")
clf.fit(X, y)
print("Accuracy:", clf.score(X, y))
```