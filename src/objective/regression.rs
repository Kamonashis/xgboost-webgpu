use super::Objective;

/// Regression with squared error loss: 0.5 * (y - y_pred)^2
#[derive(Debug, Clone, Default)]
pub struct RegSquaredError;

impl Objective for RegSquaredError {
    fn name(&self) -> &'static str {
        "reg:squarederror"
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
                    grads[i] = (y_pred[i] - y_true[i]) * weight;
                    hess[i] = 1.0 * weight;
                }
            }
            None => {
                for i in 0..n {
                    grads[i] = y_pred[i] - y_true[i];
                    hess[i] = 1.0;
                }
            }
        }
    }

    fn default_base_score(&self, y_true: &[f32]) -> f32 {
        if y_true.is_empty() {
            0.5
        } else {
            let sum: f32 = y_true.iter().sum();
            sum / y_true.len() as f32
        }
    }

    fn transform_prediction(&self, raw_margin: f32) -> f32 {
        raw_margin
    }
}
