pub mod classification;
pub mod extended;
pub mod multiclass;
pub mod regression;

pub use classification::BinaryLogistic;
pub use extended::{
    GammaRegression, PoissonRegression, QuantileRegression, RankingPairwise, TweedieRegression,
};
pub use multiclass::MultiClassObjective;
pub use regression::RegSquaredError;

/// Trait defining a loss function / optimization objective for gradient boosting.
pub trait Objective: Send + Sync {
    /// Objective name matching XGBoost conventions.
    fn name(&self) -> &'static str;

    /// Computes first-order gradients `g` and second-order Hessians `h`.
    fn compute_gradients(
        &self,
        y_true: &[f32],
        y_pred: &[f32],
        weights: Option<&[f32]>,
        grads: &mut [f32],
        hess: &mut [f32],
    );

    /// Default base score (initial raw margin prediction) for this objective.
    fn default_base_score(&self, y_true: &[f32]) -> f32;

    /// Transforms raw boosting margin to user-facing prediction.
    fn transform_prediction(&self, raw_margin: f32) -> f32;

    /// Transforms an array of raw margins.
    fn transform_predictions(&self, raw_margins: &[f32]) -> Vec<f32> {
        raw_margins.iter().map(|&m| self.transform_prediction(m)).collect()
    }
}

/// Helper function to parse objective name into boxed Objective.
pub fn get_objective(name: &str) -> Result<Box<dyn Objective>, String> {
    match name {
        "reg:squarederror" | "rmse" => Ok(Box::new(RegSquaredError)),
        "binary:logistic" | "logloss" => Ok(Box::new(BinaryLogistic::default())),
        "count:poisson" => Ok(Box::new(PoissonRegression)),
        "reg:gamma" => Ok(Box::new(GammaRegression)),
        "reg:tweedie" => Ok(Box::new(TweedieRegression::default())),
        "reg:quantileerror" => Ok(Box::new(QuantileRegression::default())),
        "rank:pairwise" => Ok(Box::new(RankingPairwise)),
        _ => Err(format!(
            "Unsupported objective: '{}'. Available: 'reg:squarederror', 'binary:logistic', 'count:poisson', 'reg:gamma', 'reg:tweedie', 'reg:quantileerror', 'rank:pairwise', 'multi:softprob', 'multi:softmax'",
            name
        )),
    }
}
