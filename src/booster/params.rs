use serde::{Deserialize, Serialize};

/// Target computation device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceType {
    /// WebGPU cross-vendor GPU acceleration (AMD, Apple Silicon, Intel, Nvidia, etc.)
    WebGPU,
    /// Multi-threaded CPU fallback using Rayon
    Cpu,
}

impl Default for DeviceType {
    fn default() -> Self {
        #[cfg(feature = "webgpu")]
        {
            DeviceType::WebGPU
        }
        #[cfg(not(feature = "webgpu"))]
        {
            DeviceType::Cpu
        }
    }
}

/// Tree growth policy: DepthWise (level-by-level) or LossGuide (leaf-wise).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GrowPolicy {
    DepthWise,
    LossGuide,
}

impl Default for GrowPolicy {
    fn default() -> Self {
        GrowPolicy::DepthWise
    }
}

/// Booster algorithm type: standard GBTree or DART (Dropouts meet Multiple Additive Regression Trees).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BoosterType {
    GBTree,
    DART,
}

impl Default for BoosterType {
    fn default() -> Self {
        BoosterType::GBTree
    }
}

/// Hyperparameters for XGBoost tree boosting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoosterParams {
    /// Objective function: "reg:squarederror", "binary:logistic", "multi:softprob", "multi:softmax", "count:poisson", etc.
    pub objective: String,
    /// Maximum tree depth (default: 6)
    pub max_depth: usize,
    /// Boosting learning rate / shrinkage (default: 0.3)
    pub learning_rate: f32,
    /// L2 regularization term on weights (default: 1.0)
    pub reg_lambda: f32,
    /// L1 regularization term on weights (default: 0.0)
    pub reg_alpha: f32,
    /// Minimum loss reduction required to make a further partition (default: 0.0)
    pub gamma: f32,
    /// Minimum sum of instance weight (Hessian) needed in a child (default: 1.0)
    pub min_child_weight: f32,
    /// Maximum number of discrete bins for histogram quantization (default: 256)
    pub max_bins: usize,
    /// Initial prediction score (base margin). If None, calculated from training data.
    pub base_score: Option<f32>,
    /// Computation device (WebGPU or Cpu)
    pub device: DeviceType,
    /// Evaluation metric name: "rmse", "mae", "logloss", "error"
    pub eval_metric: Option<String>,

    // --- Advanced Features ---
    /// Subsample ratio of the training instances (default: 1.0, bagging)
    pub subsample: f32,
    /// Subsample ratio of columns when constructing each tree (default: 1.0)
    pub colsample_bytree: f32,
    /// Subsample ratio of columns for each level/depth (default: 1.0)
    pub colsample_bylevel: f32,
    /// Subsample ratio of columns for each split node (default: 1.0)
    pub colsample_bynode: f32,
    /// Controls the balance of positive and negative weights, useful for unbalanced classes (default: 1.0)
    pub scale_pos_weight: f32,
    /// Maximum delta step we allow each leaf's output to be (default: 0.0, disabled)
    pub max_delta_step: f32,
    /// Tree growth policy: DepthWise or LossGuide (leaf-wise)
    pub grow_policy: GrowPolicy,
    /// Maximum number of terminal nodes (leaves) to be added for LossGuide policy (default: 0, unlimited)
    pub max_leaves: usize,
    /// Monotonic constraints: +1 for increasing, -1 for decreasing, 0 for unconstrained per feature
    pub monotone_constraints: Option<Vec<i8>>,
    /// Interaction constraints: list of feature groups allowed to interact
    pub interaction_constraints: Option<Vec<Vec<usize>>>,
    /// Booster type: GBTree or DART
    pub booster_type: BoosterType,
    /// Dropout rate for DART booster (default: 0.0)
    pub rate_drop: f32,
    /// Probability of skipping the dropout procedure during a DART boosting round (default: 0.0)
    pub skip_drop: f32,
    /// Number of classes for multi-class classification
    pub num_class: usize,
    /// Early stopping rounds: training stops if metric does not improve for N consecutive rounds
    pub early_stopping_rounds: Option<usize>,
    /// Best iteration recorded during early stopping
    pub best_iteration: Option<usize>,
    /// Best validation score recorded during early stopping
    pub best_score: Option<f32>,
}

impl Default for BoosterParams {
    fn default() -> Self {
        Self {
            objective: "reg:squarederror".to_string(),
            max_depth: 6,
            learning_rate: 0.3,
            reg_lambda: 1.0,
            reg_alpha: 0.0,
            gamma: 0.0,
            min_child_weight: 1.0,
            max_bins: 256,
            base_score: None,
            device: DeviceType::default(),
            eval_metric: None,
            subsample: 1.0,
            colsample_bytree: 1.0,
            colsample_bylevel: 1.0,
            colsample_bynode: 1.0,
            scale_pos_weight: 1.0,
            max_delta_step: 0.0,
            grow_policy: GrowPolicy::DepthWise,
            max_leaves: 0,
            monotone_constraints: None,
            interaction_constraints: None,
            booster_type: BoosterType::GBTree,
            rate_drop: 0.0,
            skip_drop: 0.0,
            num_class: 0,
            early_stopping_rounds: None,
            best_iteration: None,
            best_score: None,
        }
    }
}

impl BoosterParams {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_objective(mut self, obj: &str) -> Self {
        self.objective = obj.to_string();
        self
    }

    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }

    pub fn with_learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }

    pub fn with_reg_lambda(mut self, lambda: f32) -> Self {
        self.reg_lambda = lambda;
        self
    }

    pub fn with_reg_alpha(mut self, alpha: f32) -> Self {
        self.reg_alpha = alpha;
        self
    }

    pub fn with_gamma(mut self, gamma: f32) -> Self {
        self.gamma = gamma;
        self
    }

    pub fn with_min_child_weight(mut self, mcw: f32) -> Self {
        self.min_child_weight = mcw;
        self
    }

    pub fn with_max_bins(mut self, bins: usize) -> Self {
        self.max_bins = bins;
        self
    }

    pub fn with_device(mut self, dev: DeviceType) -> Self {
        self.device = dev;
        self
    }

    pub fn with_eval_metric(mut self, metric: &str) -> Self {
        self.eval_metric = Some(metric.to_string());
        self
    }

    pub fn with_subsample(mut self, subsample: f32) -> Self {
        self.subsample = subsample.clamp(0.01, 1.0);
        self
    }

    pub fn with_colsample_bytree(mut self, colsample: f32) -> Self {
        self.colsample_bytree = colsample.clamp(0.01, 1.0);
        self
    }

    pub fn with_scale_pos_weight(mut self, weight: f32) -> Self {
        self.scale_pos_weight = weight.max(0.0);
        self
    }

    pub fn with_max_delta_step(mut self, step: f32) -> Self {
        self.max_delta_step = step.max(0.0);
        self
    }

    pub fn with_grow_policy(mut self, policy: GrowPolicy) -> Self {
        self.grow_policy = policy;
        self
    }

    pub fn with_max_leaves(mut self, leaves: usize) -> Self {
        self.max_leaves = leaves;
        self
    }

    pub fn with_early_stopping(mut self, rounds: usize) -> Self {
        self.early_stopping_rounds = Some(rounds);
        self
    }

    pub fn with_booster_type(mut self, booster_type: BoosterType) -> Self {
        self.booster_type = booster_type;
        self
    }

    pub fn with_num_class(mut self, num_class: usize) -> Self {
        self.num_class = num_class;
        self
    }

    pub fn with_monotone_constraints(mut self, constraints: Vec<i8>) -> Self {
        self.monotone_constraints = Some(constraints);
        self
    }
}
