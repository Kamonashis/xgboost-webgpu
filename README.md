# xgboost-webgpu

[![PyPI version](https://img.shields.io/pypi/v/xgboost-webgpu.svg)](https://pypi.org/project/xgboost-webgpu/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/xgboost-webgpu.svg)](https://pypi.org/project/xgboost-webgpu/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Universal, blazing-fast, and memory-efficient XGBoost (Gradient Boosted Decision Trees) library in Rust, powered by **WebGPU** (`wgpu`) compute acceleration.

Standard XGBoost only supports GPU acceleration via NVIDIA CUDA (`device='cuda'` or `tree_method='gpu_hist'`). `xgboost-webgpu` breaks this hardware lock-in, enabling hardware-accelerated model training and inference across **any modern GPU**:
- **AMD Radeon GPUs** (via Vulkan on Linux / Windows)
- **Apple Silicon M1/M2/M3/M4** (via Metal on macOS / iOS)
- **Intel Arc & Integrated GPUs** (via Vulkan / DirectX 12)
- **NVIDIA GPUs** (via Vulkan / DirectX 12)
- **Web Browsers & WebAssembly** (via native WebGPU WGSL)
- **Multi-threaded CPU Fallback** (via Rayon)

📖 **Documentation & Wiki**: Full architectural guides, mathematical formulations, and API manuals are available on the [Project Wiki](https://tea.ergotresearch.com/Ergot-Research/xgboost-webgpu/wiki) or offline in [`docs/wiki/`](docs/wiki/Home.md).

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

## Complete Feature Matrix

### 1. Objectives & Loss Functions
- **Regression**: `reg:squarederror` (squared error loss).
- **Binary Classification**: `binary:logistic` with `scale_pos_weight` support.
- **Multi-Class Classification**: `multi:softprob` (probabilities) and `multi:softmax` (hard labels).
- **Count & Deviance**: `count:poisson` (Poisson regression for count data).
- **Dispersion Models**: `reg:gamma` (Gamma deviance) and `reg:tweedie` (compound Poisson-Gamma).
- **Quantile Loss**: `reg:quantileerror` (pinball loss for arbitrary conditional quantiles).
- **Ranking**: `rank:pairwise` (LambdaMART-style pairwise ranking with query groups).

### 2. Data Formats & Preprocessing
- **Dense Data**: Row-major dense buffers (`DMatrix::from_dense`).
- **Sparse CSR Format**: Compressed Sparse Row matrices (`DMatrix::from_csr`).
- **Quantization**: 256 discrete bins (`u8`) for 75% memory reduction and packed 4-bins-per-u32 GPU storage buffer reads.
- **Categorical Features**: `FeatureType::Categorical` with dedicated category indexing.
- **Missing Value Handling**: Automatic routing according to optimal split gain (`default_left`).

### 3. Tree Growth & Regularization
- **Growth Policies**:
  - `GrowPolicy::DepthWise`: Level-by-level recursive growth.
  - `GrowPolicy::LossGuide`: Leaf-wise best-first growth (LightGBM style) up to `max_leaves`.
- **Regularization**: $L_1$ (`reg_alpha`), $L_2$ (`reg_lambda`), `gamma` (min split loss), `min_child_weight`, and `max_delta_step`.
- **Subsampling**:
  - `subsample`: Row subsampling (bagging) per round.
  - `colsample_bytree`: Feature subsampling per tree.
  - `colsample_bylevel` & `colsample_bynode`: Feature subsampling per level and per split node.
- **Constraints**:
  - `monotone_constraints`: Enforces strict $+1$ (non-decreasing) or $-1$ (non-increasing) monotonic outputs.
  - `interaction_constraints`: Constrains allowed feature interaction groups.

### 4. Advanced Boosting & Training Controls
- **DART Booster (`booster_type = BoosterType::DART`)**: Dropouts meet Multiple Additive Regression Trees with drop rate, skip drop, and weight normalization.
- **Early Stopping**: `early_stopping_rounds` monitors validation metrics and automatically stops training when progress plateaus.
- **Training Continuation**: `booster.train_continue(...)` appends additional boosting rounds to an existing model.

### 5. Interpretability & Model Analysis
- **Feature Importance**: Scores features by `Weight` (split frequency), `Gain` (average or total gain), and `Cover` (Hessian coverage).
- **Leaf Index Prediction**: `booster.predict_leaf(...)` returns the exact leaf node IDs where samples terminate across all trees.
- **Tree SHAP (`predict_contributions`)**: Exact implementation of TreeSHAP (Lundberg & Lee) returning feature attributions satisfying additive efficiency:
  $$\sum_{j=0}^{M-1} \phi_{i, j} + \phi_{\text{bias}} = \hat{y}_i$$

### 6. Interoperability
- **XGBoost JSON Format**: Direct export and import matching the official XGBoost JSON schema (`to_xgboost_json`, `from_xgboost_json`, `save_model`, `load_model`).

---

## Quickstart

Add `xgboost-webgpu` to your `Cargo.toml`:

```toml
[dependencies]
xgboost-webgpu = "0.1.0"
```

### Multi-Class Classification with Advanced Features

```rust
use xgboost_webgpu::{BoosterParams, DeviceType, DMatrix, GrowPolicy, ImportanceType, train};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = vec![/* features: N x M */];
    let y = vec![/* labels: 0.0, 1.0, 2.0 */];

    let dtrain = DMatrix::from_dense(&x, 1000, 6, Some(&y), 256)?;

    let params = BoosterParams::new()
        .with_objective("multi:softprob")
        .with_num_class(3)
        .with_grow_policy(GrowPolicy::LossGuide)
        .with_max_leaves(16)
        .with_subsample(0.8)
        .with_colsample_bytree(0.8)
        .with_early_stopping(5)
        .with_device(DeviceType::WebGPU); // Accelerate on AMD, Apple, Intel, Nvidia

    let booster = train(params, &dtrain, 30, &[])?;

    // Feature Importance
    let importance = booster.feature_importance(ImportanceType::Gain);
    println!("Feature Importance: {:?}", importance);

    // TreeSHAP Feature Contributions
    let shap_values = booster.predict_contributions(&dtrain)?;
    println!("Sample 0 SHAP values: {:?}", shap_values[0]);

    // Save to official XGBoost JSON
    booster.save_model("model.json")?;

    Ok(())
}
```

---

## Python Bindings & Scikit-Learn API

`xgboost-webgpu` includes high-performance Python bindings built with PyO3 and the Python Stable ABI (`abi3`). It provides drop-in Scikit-Learn estimators (`XGBRegressor`, `XGBClassifier`) and the core `xgb.train()` functional API with hardware acceleration.

### Installation

Install directly from [PyPI](https://pypi.org/project/xgboost-webgpu/):

```bash
pip install xgboost-webgpu
```

*(Alternatively, build from source using `pip install .` or `maturin develop --release`)*

### Scikit-Learn API Example

```python
import numpy as np
import xgboost_webgpu as xgb

# Generate sample data
X = np.random.randn(1000, 10).astype(np.float32)
y = (X[:, 0] + X[:, 1] > 0).astype(np.float32)

# Train with WebGPU acceleration
clf = xgb.XGBClassifier(
    n_estimators=50,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    device="webgpu",  # Accelerate on AMD, Apple Silicon, Intel, or NVIDIA
)
clf.fit(X, y)

# Predictions & Probabilities
preds = clf.predict(X)
probs = clf.predict_proba(X)
print("Accuracy:", clf.score(X, y))

# Feature Importances & Exact TreeSHAP
print("Feature Importances:", clf.feature_importances_)
shap_values = clf.predict_contributions(X)
print("TreeSHAP shape:", np.array(shap_values).shape)

# Save and load model in XGBoost JSON format
clf.save_model("model.json")
```

### Low-Level Functional API

```python
import xgboost_webgpu as xgb

# Ingest data into quantized DMatrix
dtrain = xgb.DMatrix(X_train, labels=y_train)
deval = xgb.DMatrix(X_val, labels=y_val)

# Hyperparameters
params = {
    "objective": "multi:softprob",
    "num_class": 3,
    "max_depth": 5,
    "grow_policy": "lossguide",
    "max_leaves": 31,
    "early_stopping_rounds": 10,
    "eval_metric": "error",
    "device": "webgpu",
}

# Train Booster
bst = xgb.train(params, dtrain, num_boost_round=100, evals=[(deval, "val")])
preds = bst.predict(deval)
```

---

## Running Benchmarks & Examples

```bash
# California Housing regression benchmark
cargo run --release --example california_housing

# Non-linear binary classification benchmark
cargo run --release --example binary_classification

# Multi-class, LossGuide, Subsampling, and TreeSHAP demonstration
cargo run --release --example advanced_features

# Automated tests (17 comprehensive tests)
cargo test

# Python test suite
python3 -m unittest python/tests/test_python_bindings.py
```

---

## License

Licensed under the MIT License. See [LICENSE](LICENSE) for details.
