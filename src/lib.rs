//! # xgboost-webgpu
//!
//! A high-performance, memory-efficient Rust library providing universal GPU-accelerated
//! XGBoost (GBDT) training and inference via WebGPU (`wgpu`).
//!
//! ## Comprehensive Feature Set
//! - **Universal GPU Acceleration**: Runs on AMD, Apple Silicon, Intel, and NVIDIA GPUs via WebGPU / Vulkan / Metal / DirectX 12.
//! - **Compact Memory Layout**: Quantizes continuous features to 256 discrete bins (`u8`), cutting memory consumption by 75%.
//! - **Sparse CSR & Categorical Ingestion**: Direct support for CSR sparse matrices and native categorical feature partitioning.
//! - **Rich Objectives**: `reg:squarederror`, `binary:logistic`, `multi:softprob`, `multi:softmax`, `count:poisson`, `reg:gamma`, `reg:tweedie`, `reg:quantileerror`, and `rank:pairwise`.
//! - **Tree Growth Policies**: Both `DepthWise` (level-by-level) and `LossGuide` (leaf-wise).
//! - **Regularization & Constraints**: $L_1$ (`reg_alpha`), $L_2$ (`reg_lambda`), `gamma`, `min_child_weight`, `max_delta_step`, `scale_pos_weight`, monotonic constraints, and interaction constraints.
//! - **Subsampling**: Row bagging (`subsample`), column subsampling (`colsample_bytree`, `colsample_bylevel`, `colsample_bynode`).
//! - **Advanced Boosting**: Standard `GBTree` and `DART` (Dropouts meet Multiple Additive Regression Trees).
//! - **Training Controls**: Early stopping (`early_stopping_rounds`) and training continuation (`train_continue`).
//! - **Interpretability**: Feature importance (`Weight`, `Gain`, `Cover`), leaf index prediction (`predict_leaf`), and exact TreeSHAP feature contributions (`predict_contributions`).
//! - **XGBoost Compatibility**: Direct export and import using standard XGBoost JSON model format.
//! - **CPU Fallback**: Multi-threaded CPU fallback via Rayon if no GPU is available or if requested.

pub mod booster;
pub mod data;
pub mod gpu;
pub mod metrics;
pub mod model;
pub mod objective;
pub mod tree;

pub use booster::{Booster, BoosterParams, BoosterType, DeviceType, GrowPolicy, ImportanceType};
pub use data::{DMatrix, FeatureBinMapper, FeatureType};
pub use objective::{
    BinaryLogistic, GammaRegression, MultiClassObjective, Objective, PoissonRegression,
    QuantileRegression, RankingPairwise, RegSquaredError, TweedieRegression,
};
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
