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

/// Hyperparameters for XGBoost tree boosting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoosterParams {
    /// Objective function: "reg:squarederror", "binary:logistic"
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
}
