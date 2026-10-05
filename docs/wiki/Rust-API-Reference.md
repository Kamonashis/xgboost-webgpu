# Rust API Reference

This document provides a comprehensive API reference for the Rust crate `xgboost-webgpu`.

---

## 1. Crate Setup

Add `xgboost-webgpu` to your `Cargo.toml`:

```toml
[dependencies]
xgboost-webgpu = "0.1.0"
```

To enable multi-threaded CPU-only execution without GPU dependencies:
```toml
[dependencies]
xgboost-webgpu = { version = "0.1.0", default-features = false }
```

---

## 2. `DMatrix`

The `DMatrix` struct represents quantized training and evaluation datasets optimized for GPU histogram building.

### Constructors

#### `from_dense`
```rust
pub fn from_dense(
    data: &[f32],
    nrows: usize,
    ncols: usize,
    labels: Option<&[f32]>,
    max_bins: usize,
) -> Result<Self, String>
```
Constructs a `DMatrix` from a flat row-major slice of 32-bit floats. Quantizes each feature column into $B \le \text{max\_bins}$ quantile buckets.

#### `from_csr`
```rust
pub fn from_csr(
    indptr: &[usize],
    indices: &[usize],
    values: &[f32],
    nrows: usize,
    ncols: usize,
    labels: Option<&[f32]>,
    max_bins: usize,
) -> Result<Self, String>
```
Constructs a `DMatrix` directly from Compressed Sparse Row (CSR) arrays. Missing or zero entries are handled with optimal split routing.

### Modifier Methods
* `pub fn set_weights(&mut self, weights: Vec<f32>) -> Result<(), String>`: Sets per-instance sample weights.
* `pub fn set_feature_names(&mut self, names: Vec<String>) -> Result<(), String>`: Sets descriptive names for features.
* `pub fn set_group(&mut self, group: Vec<usize>) -> Result<(), String>`: Sets query group counts for ranking objectives (`rank:pairwise`).

---

## 3. `BoosterParams`

Configures model hyperparameters using an ergonomic builder pattern.

```rust
use xgboost_webgpu::{BoosterParams, BoosterType, DeviceType, GrowPolicy};

let params = BoosterParams::new()
    .with_objective("binary:logistic")
    .with_max_depth(6)
    .with_learning_rate(0.1)
    .with_reg_lambda(1.0)
    .with_reg_alpha(0.0)
    .with_gamma(0.0)
    .with_min_child_weight(1.0)
    .with_subsample(0.8)
    .with_colsample_bytree(0.8)
    .with_scale_pos_weight(1.0)
    .with_max_delta_step(0.0)
    .with_grow_policy(GrowPolicy::DepthWise)
    .with_max_leaves(31)
    .with_booster_type(BoosterType::GBTree)
    .with_device(DeviceType::WebGPU) // Or DeviceType::Cpu
    .with_early_stopping(10)
    .with_eval_metric("logloss");
```

### Parameter Defaults

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `objective` | `String` | `"reg:squarederror"` | Loss function identifier |
| `max_depth` | `usize` | `6` | Maximum tree depth |
| `learning_rate` | `f32` | `0.3` | Shrinkage step size |
| `reg_lambda` | `f32` | `1.0` | $L_2$ regularization on weights |
| `reg_alpha` | `f32` | `0.0` | $L_1$ regularization on weights |
| `gamma` | `f32` | `0.0` | Minimum split gain reduction |
| `min_child_weight` | `f32` | `1.0` | Minimum Hessian sum per child |
| `subsample` | `f32` | `1.0` | Row bagging ratio per round |
| `colsample_bytree` | `f32` | `1.0` | Column subsampling ratio per tree |
| `colsample_bynode` | `f32` | `1.0` | Column subsampling ratio per node |
| `grow_policy` | `GrowPolicy` | `DepthWise` | `DepthWise` or `LossGuide` |
| `max_leaves` | `usize` | `0` | Max leaves for `LossGuide` (0 = unlimited) |
| `device` | `DeviceType` | `WebGPU` | `WebGPU` or `Cpu` |

---

## 4. `train` Function

```rust
pub fn train(
    params: BoosterParams,
    dtrain: &DMatrix,
    num_boost_round: usize,
    evals: &[(&DMatrix, &str)],
) -> Result<Booster, String>
```

Trains a new `Booster` model on `dtrain` for `num_boost_round` iterations. Evaluation datasets can be passed via `evals` slice to monitor metrics and trigger early stopping.

---

## 5. `Booster`

The trained ensemble model providing prediction, persistence, and explanation methods:

* `pub fn predict(&self, dmatrix: &DMatrix) -> Result<Vec<f32>, String>`: Computes transformed predictions (e.g. probabilities for classification).
* `pub fn predict_raw(&self, dmatrix: &DMatrix) -> Result<Vec<f32>, String>`: Computes raw un-transformed margin scores.
* `pub fn predict_leaf(&self, dmatrix: &DMatrix) -> Result<Vec<Vec<usize>>, String>`: Returns leaf indices traversed per tree.
* `pub fn predict_contributions(&self, dmatrix: &DMatrix) -> Result<Vec<Vec<f32>>, String>`: Computes exact TreeSHAP feature attributions ($N \times (M+1)$).
* `pub fn feature_importance(&self, importance_type: ImportanceType) -> Vec<(String, f32)>`: Returns feature importance rankings.
* `pub fn save_model(&self, path: &str) -> Result<(), String>`: Saves model to official XGBoost JSON format.
* `pub fn load_model(path: &str) -> Result<Self, String>`: Loads model from JSON file.
* `pub fn to_xgboost_json(&self) -> Result<String, String>`: Serializes model to JSON string.
* `pub fn from_xgboost_json(json_str: &str) -> Result<Self, String>`: Deserializes model from JSON string.
* `pub fn num_trees(&self) -> usize`: Returns the total number of trees in the ensemble.
