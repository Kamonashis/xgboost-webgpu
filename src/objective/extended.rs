use super::Objective;

/// Count data regression with Poisson deviance: lambda = exp(margin)
#[derive(Debug, Clone, Default)]
pub struct PoissonRegression;

impl Objective for PoissonRegression {
    fn name(&self) -> &'static str {
        "count:poisson"
    }

    fn compute_gradients(
        &self,
        y_true: &[f32],
        y_pred: &[f32],
        weights: Option<&[f32]>,
        grads: &mut [f32],
        hess: &mut [f32],
    ) {
        let n = y_true.len();
        for i in 0..n {
            let y = y_true[i].max(0.0);
            let margin = y_pred[i].clamp(-20.0, 20.0);
            let lambda = margin.exp();
            let weight = weights.map(|w| w[i]).unwrap_or(1.0);

            grads[i] = (lambda - y) * weight;
            hess[i] = lambda.max(1e-16) * weight;
        }
    }

    fn default_base_score(&self, y_true: &[f32]) -> f32 {
        if y_true.is_empty() {
            0.0
        } else {
            let mean: f32 = y_true.iter().sum::<f32>() / y_true.len() as f32;
            mean.max(1e-5).ln()
        }
    }

    fn transform_prediction(&self, raw_margin: f32) -> f32 {
        raw_margin.clamp(-20.0, 20.0).exp()
    }
}

/// Gamma deviance regression for positive continuous values: mu = exp(margin)
#[derive(Debug, Clone, Default)]
pub struct GammaRegression;

impl Objective for GammaRegression {
    fn name(&self) -> &'static str {
        "reg:gamma"
    }

    fn compute_gradients(
        &self,
        y_true: &[f32],
        y_pred: &[f32],
        weights: Option<&[f32]>,
        grads: &mut [f32],
        hess: &mut [f32],
    ) {
        let n = y_true.len();
        for i in 0..n {
            let y = y_true[i].max(1e-7);
            let margin = y_pred[i].clamp(-20.0, 20.0);
            let exp_neg_margin = (-margin).exp();
            let weight = weights.map(|w| w[i]).unwrap_or(1.0);

            grads[i] = (1.0 - y * exp_neg_margin) * weight;
            hess[i] = (y * exp_neg_margin).max(1e-16) * weight;
        }
    }

    fn default_base_score(&self, y_true: &[f32]) -> f32 {
        if y_true.is_empty() {
            0.0
        } else {
            let mean: f32 = y_true.iter().sum::<f32>() / y_true.len() as f32;
            mean.max(1e-5).ln()
        }
    }

    fn transform_prediction(&self, raw_margin: f32) -> f32 {
        raw_margin.clamp(-20.0, 20.0).exp()
    }
}

/// Tweedie regression with compound Poisson-Gamma distribution (variance power 1 < rho < 2)
#[derive(Debug, Clone)]
pub struct TweedieRegression {
    pub rho: f32,
}

impl Default for TweedieRegression {
    fn default() -> Self {
        Self { rho: 1.5 }
    }
}

impl Objective for TweedieRegression {
    fn name(&self) -> &'static str {
        "reg:tweedie"
    }

    fn compute_gradients(
        &self,
        y_true: &[f32],
        y_pred: &[f32],
        weights: Option<&[f32]>,
        grads: &mut [f32],
        hess: &mut [f32],
    ) {
        let n = y_true.len();
        let rho = self.rho;
        for i in 0..n {
            let y = y_true[i].max(0.0);
            let margin = y_pred[i].clamp(-20.0, 20.0);
            let weight = weights.map(|w| w[i]).unwrap_or(1.0);

            // Tweedie gradients
            let g = -y * ((1.0 - rho) * margin).exp() + ((2.0 - rho) * margin).exp();
            let h = -y * (1.0 - rho) * ((1.0 - rho) * margin).exp()
                + (2.0 - rho) * ((2.0 - rho) * margin).exp();

            grads[i] = g * weight;
            hess[i] = h.max(1e-16) * weight;
        }
    }

    fn default_base_score(&self, y_true: &[f32]) -> f32 {
        if y_true.is_empty() {
            0.0
        } else {
            let mean: f32 = y_true.iter().sum::<f32>() / y_true.len() as f32;
            mean.max(1e-5).ln()
        }
    }

    fn transform_prediction(&self, raw_margin: f32) -> f32 {
        raw_margin.clamp(-20.0, 20.0).exp()
    }
}

/// Quantile (Pinball) regression for estimating conditional quantiles (default alpha = 0.5 for median)
#[derive(Debug, Clone)]
pub struct QuantileRegression {
    pub alpha: f32,
}

impl Default for QuantileRegression {
    fn default() -> Self {
        Self { alpha: 0.5 }
    }
}

impl Objective for QuantileRegression {
    fn name(&self) -> &'static str {
        "reg:quantileerror"
    }

    fn compute_gradients(
        &self,
        y_true: &[f32],
        y_pred: &[f32],
        weights: Option<&[f32]>,
        grads: &mut [f32],
        hess: &mut [f32],
    ) {
        let n = y_true.len();
        let alpha = self.alpha;
        for i in 0..n {
            let diff = y_true[i] - y_pred[i];
            let g = if diff > 0.0 { -alpha } else { 1.0 - alpha };
            let weight = weights.map(|w| w[i]).unwrap_or(1.0);

            grads[i] = g * weight;
            hess[i] = 1.0 * weight; // Constant pseudo-Hessian for robust tree splitting
        }
    }

    fn default_base_score(&self, y_true: &[f32]) -> f32 {
        if y_true.is_empty() {
            0.0
        } else {
            let mut sorted = y_true.to_vec();
            sorted.sort_by(|a, b| a.total_cmp(b));
            let idx = ((self.alpha * (sorted.len() as f32)) as usize).min(sorted.len() - 1);
            sorted[idx]
        }
    }

    fn transform_prediction(&self, raw_margin: f32) -> f32 {
        raw_margin
    }
}

/// Pairwise ranking objective (LambdaMART style)
#[derive(Debug, Clone, Default)]
pub struct RankingPairwise;

impl Objective for RankingPairwise {
    fn name(&self) -> &'static str {
        "rank:pairwise"
    }

    fn compute_gradients(
        &self,
        y_true: &[f32],
        y_pred: &[f32],
        weights: Option<&[f32]>,
        grads: &mut [f32],
        hess: &mut [f32],
    ) {
        let n = y_true.len();
        grads.fill(0.0);
        hess.fill(0.0);

        // Compute pairwise logistic gradients across pairs where y_i > y_j
        for i in 0..n {
            for j in 0..n {
                if y_true[i] > y_true[j] {
                    let diff = y_pred[i] - y_pred[j];
                    let sigma = 1.0 / (1.0 + (-diff).exp());
                    let lambda = -(1.0 - sigma);
                    let h = (sigma * (1.0 - sigma)).max(1e-16);

                    let w = weights.map(|w| (w[i] * w[j]).sqrt()).unwrap_or(1.0);
                    grads[i] += lambda * w;
                    grads[j] -= lambda * w;
                    hess[i] += h * w;
                    hess[j] += h * w;
                }
            }
        }
    }

    fn default_base_score(&self, _y_true: &[f32]) -> f32 {
        0.0
    }

    fn transform_prediction(&self, raw_margin: f32) -> f32 {
        raw_margin
    }
}
