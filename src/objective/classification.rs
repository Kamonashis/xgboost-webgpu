use super::Objective;

/// Binary classification with logistic loss (negative log-likelihood)
#[derive(Debug, Clone, Default)]
pub struct BinaryLogistic;

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
        match weights {
            Some(w) => {
                for i in 0..n {
                    let weight = w[i];
                    let p = sigmoid(y_pred[i]);
                    grads[i] = (p - y_true[i]) * weight;
                    let h = (p * (1.0 - p)).max(1e-16);
                    hess[i] = h * weight;
                }
            }
            None => {
                for i in 0..n {
                    let p = sigmoid(y_pred[i]);
                    grads[i] = p - y_true[i];
                    let h = (p * (1.0 - p)).max(1e-16);
                    hess[i] = h;
                }
            }
        }
    }

    fn default_base_score(&self, _y_true: &[f32]) -> f32 {
        0.5
    }

    fn transform_prediction(&self, raw_margin: f32) -> f32 {
        sigmoid(raw_margin)
    }
}
