//! # xgboost-webgpu
//!
//! A high-performance, memory-efficient Rust library providing universal GPU-accelerated
//! XGBoost (GBDT) training and inference via WebGPU (`wgpu`).
//!
//! ## Key Features
//! - **Universal GPU Acceleration**: Runs on AMD, Apple Silicon, Intel, and NVIDIA GPUs via WebGPU / Vulkan / Metal / DirectX 12.
//! - **Compact Memory Layout**: Quantizes continuous features to 256 discrete bins (`u8`), cutting memory consumption by 75%.
//! - **WebGPU Compute Shaders**: High-throughput parallel histogram accumulation on GPU with workgroup tiling and scaled integer atomics.
//! - **Histogram Subtraction**: Derives sibling node histograms in $O(1)$ child operations, halving GPU work across all tree levels.
//! - **XGBoost Compatibility**: Direct export and import using standard XGBoost JSON model format.
//! - **CPU Fallback**: Multi-threaded CPU fallback via Rayon if no GPU is available or if requested.

pub mod booster;
pub mod data;
pub mod gpu;
pub mod metrics;
pub mod model;
pub mod objective;
pub mod tree;

pub use booster::{Booster, BoosterParams, DeviceType};
pub use data::{DMatrix, FeatureBinMapper};
pub use objective::{BinaryLogistic, Objective, RegSquaredError};
pub use tree::{Tree, TreeNode};

#[cfg(feature = "webgpu")]
pub use gpu::GpuContext;

/// Convenient top-level training entry point matching XGBoost conventions.
pub fn train(
    params: BoosterParams,
    dtrain: &DMatrix,
    num_boost_round: usize,
    evals: &[(&DMatrix, &str)],
) -> Result<Booster, String> {
    Booster::train(params, dtrain, num_boost_round, evals)
}
