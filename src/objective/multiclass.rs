/// Multi-class classification objectives: multi:softprob and multi:softmax
#[derive(Debug, Clone)]
pub struct MultiClassObjective {
    pub num_class: usize,
    pub is_softmax: bool, // true for multi:softmax, false for multi:softprob
}

impl MultiClassObjective {
    pub fn new(num_class: usize, is_softmax: bool) -> Self {
        assert!(num_class >= 2, "num_class must be >= 2 for multi-class");
        Self {
            num_class,
            is_softmax,
        }
    }

    /// Computes gradients and Hessians for class `class_idx` given flattened margins (length = N * K).
    pub fn compute_class_gradients(
        &self,
        y_true: &[f32],
        margins: &[f32], // length N * K
        class_idx: usize,
        weights: Option<&[f32]>,
        grads: &mut [f32], // length N
        hess: &mut [f32],  // length N
    ) {
        let n = y_true.len();
        let k = self.num_class;
        assert_eq!(margins.len(), n * k);
        assert_eq!(grads.len(), n);
        assert_eq!(hess.len(), n);

        for i in 0..n {
            let row_start = i * k;
            let row_margins = &margins[row_start..row_start + k];

            // Softmax with numerical stability
            let max_m = row_margins.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0f32;
            for &m in row_margins {
                sum_exp += (m - max_m).exp();
            }

            let p = (row_margins[class_idx] - max_m).exp() / sum_exp.max(1e-16);
            let target = if (y_true[i] as usize) == class_idx { 1.0 } else { 0.0 };
            let weight = weights.map(|w| w[i]).unwrap_or(1.0);

            grads[i] = (p - target) * weight;
            hess[i] = (2.0 * p * (1.0 - p)).max(1e-16) * weight;
        }
    }

    /// Transform flattened margins (N * K) to predictions:
    /// - multi:softprob: returns flattened probabilities (N * K)
    /// - multi:softmax: returns argmax class label for each sample (length N)
    pub fn transform_margins(&self, margins: &[f32]) -> Vec<f32> {
        let k = self.num_class;
        let n = margins.len() / k;

        if self.is_softmax {
            let mut labels = Vec::with_capacity(n);
            for i in 0..n {
                let row_start = i * k;
                let row_margins = &margins[row_start..row_start + k];
                let mut best_class = 0;
                let mut max_val = f32::NEG_INFINITY;
                for (c, &m) in row_margins.iter().enumerate() {
                    if m > max_val {
                        max_val = m;
                        best_class = c;
                    }
                }
                labels.push(best_class as f32);
            }
            labels
        } else {
            let mut probs = Vec::with_capacity(n * k);
            for i in 0..n {
                let row_start = i * k;
                let row_margins = &margins[row_start..row_start + k];
                let max_m = row_margins.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum_exp = 0.0f32;
                for &m in row_margins {
                    sum_exp += (m - max_m).exp();
                }
                for &m in row_margins {
                    probs.push((m - max_m).exp() / sum_exp.max(1e-16));
                }
            }
            probs
        }
    }
}
