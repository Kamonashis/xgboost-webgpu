use super::Objective;

/// Binary classification with logistic loss (negative log-likelihood)
#[derive(Debug, Clone)]
pub struct BinaryLogistic {
    pub scale_pos_weight: f32,
}

impl Default for BinaryLogistic {
    fn default() -> Self {
        Self {
            scale_pos_weight: 1.0,
        }
    }
}

impl BinaryLogistic {
    pub fn new(scale_pos_weight: f32) -> Self {
        Self { scale_pos_weight }
    }
}

#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

impl Objective for BinaryLogistic {
    fn name(&self) -> &'static str {
        "binary:logistic"
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
        let scale_pos = self.scale_pos_weight;

        for i in 0..n {
            let base_w = weights.map(|w| w[i]).unwrap_or(1.0);
            let is_pos = y_true[i] > 0.5;
            let eff_w = if is_pos { base_w * scale_pos } else { base_w };

            let p = sigmoid(y_pred[i]);
            grads[i] = (p - y_true[i]) * eff_w;
            let h = (p * (1.0 - p)).max(1e-16);
            hess[i] = h * eff_w;
        }
    }

    fn default_base_score(&self, _y_true: &[f32]) -> f32 {
        0.5
    }

    fn transform_prediction(&self, raw_margin: f32) -> f32 {
        sigmoid(raw_margin)
    }
}
