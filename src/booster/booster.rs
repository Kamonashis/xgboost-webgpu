use crate::booster::params::{BoosterParams, DeviceType};
use crate::data::DMatrix;
#[cfg(feature = "webgpu")]
use crate::gpu::context::GpuContext;
#[cfg(feature = "webgpu")]
use crate::gpu::histogram::GpuHistSession;
use crate::metrics;
use crate::objective::{get_objective, Objective};
use crate::tree::builder::{TreeBuilder, TreeBuilderConfig};
use crate::tree::node::Tree;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Trained gradient boosted tree ensemble.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Booster {
    pub params: BoosterParams,
    pub trees: Vec<Tree>,
    pub base_score: f32,
    pub num_features: usize,
    pub feature_names: Vec<String>,
}

impl Booster {
    /// Trains a new Booster model using the provided parameters and training dataset.
    pub fn train(
        params: BoosterParams,
        dtrain: &DMatrix,
        num_boost_round: usize,
        evals: &[(&DMatrix, &str)],
    ) -> Result<Self, String> {
        let labels = dtrain.labels.as_ref().ok_or_else(|| {
            "Training DMatrix must have labels for supervised learning".to_string()
        })?;

        let n = dtrain.nrows;
        if n == 0 {
            return Err("Training DMatrix has 0 samples".to_string());
        }

        let obj: Box<dyn Objective> = get_objective(&params.objective)?;
        let base_score = params
            .base_score
            .unwrap_or_else(|| obj.default_base_score(labels));

        let mut current_margin = vec![base_score; n];
        let mut grads = vec![0.0f32; n];
        let mut hess = vec![0.0f32; n];

        // Track validation margins for evaluation
        let mut eval_margins: Vec<Vec<f32>> = evals
            .iter()
            .map(|(dmat, _)| vec![base_score; dmat.nrows])
            .collect();

        let metric_name = params.eval_metric.clone().unwrap_or_else(|| {
            if params.objective.starts_with("binary:") {
                "logloss".to_string()
            } else {
                "rmse".to_string()
            }
        });

        // Initialize WebGPU session if requested and available
        #[cfg(feature = "webgpu")]
        let (_gpu_ctx, gpu_session) = if params.device == DeviceType::WebGPU {
            match GpuContext::new() {
                Ok(ctx) => {
                    let sess = GpuHistSession::new(Box::leak(Box::new(ctx)), dtrain);
                    (true, Some(sess))
                }
                Err(e) => {
                    eprintln!("Warning: WebGPU requested but initialization failed: {}. Falling back to multi-threaded CPU.", e);
                    (false, None)
                }
            }
        } else {
            (false, None)
        };

        #[cfg(not(feature = "webgpu"))]
        let _ = params.device;

        let tree_config = TreeBuilderConfig {
            max_depth: params.max_depth,
            reg_lambda: params.reg_lambda,
            reg_alpha: params.reg_alpha,
            gamma: params.gamma,
            min_child_weight: params.min_child_weight,
            learning_rate: params.learning_rate,
        };

        let mut trees = Vec::with_capacity(num_boost_round);

        for round in 0..num_boost_round {
            // 1. Compute loss gradients and Hessians
            obj.compute_gradients(
                labels,
                &current_margin,
                dtrain.weights.as_deref(),
                &mut grads,
                &mut hess,
            );

            // 2. Build tree using GPU session or CPU fallback
            let builder = TreeBuilder::new(dtrain, tree_config, &grads, &hess);

            #[cfg(feature = "webgpu")]
            let tree = if let Some(ref sess) = gpu_session {
                sess.update_gradients(&grads, &hess);
                builder.with_gpu_session(sess).build()
            } else {
                builder.build()
            };

            #[cfg(not(feature = "webgpu"))]
            let tree = builder.build();

            // 3. Update current training margins in parallel
            let ncols = dtrain.ncols;
            current_margin
                .par_chunks_exact_mut(1)
                .enumerate()
                .for_each(|(i, margin)| {
                    let row_start = i * ncols;
                    let row_features = &dtrain.data[row_start..row_start + ncols];
                    let leaf_val = tree.predict_sample(row_features);
                    margin[0] += leaf_val;
                });

            // 4. Update and report evaluation sets
            if !evals.is_empty() {
                let mut log_line = format!("[{}]", round);

                // Training metric
                let train_preds = obj.transform_predictions(&current_margin);
                if let Ok(train_score) = metrics::evaluate(&metric_name, labels, &train_preds) {
                    log_line.push_str(&format!("  train-{}:{:.5}", metric_name, train_score));
                }

                // Validation metrics
                for (idx, (eval_dmat, name)) in evals.iter().enumerate() {
                    let ev_ncols = eval_dmat.ncols;
                    eval_margins[idx]
                        .par_chunks_exact_mut(1)
                        .enumerate()
                        .for_each(|(i, margin)| {
                            let row_start = i * ev_ncols;
                            let row_features = &eval_dmat.data[row_start..row_start + ev_ncols];
                            margin[0] += tree.predict_sample(row_features);
                        });

                    if let Some(eval_labels) = &eval_dmat.labels {
                        let eval_preds = obj.transform_predictions(&eval_margins[idx]);
                        if let Ok(score) = metrics::evaluate(&metric_name, eval_labels, &eval_preds) {
                            log_line.push_str(&format!("  {}-{}:{:.5}", name, metric_name, score));
                        }
                    }
                }

                println!("{}", log_line);
            }

            trees.push(tree);
        }

        Ok(Self {
            params,
            trees,
            base_score,
            num_features: dtrain.ncols,
            feature_names: dtrain.feature_names.clone(),
        })
    }

    /// Evaluates the booster ensemble and returns final predictions (after objective transform).
    pub fn predict(&self, dmatrix: &DMatrix) -> Result<Vec<f32>, String> {
        let raw = self.predict_raw(dmatrix)?;
        let obj = get_objective(&self.params.objective)?;
        Ok(obj.transform_predictions(&raw))
    }

    /// Evaluates raw margins (before objective sigmoid or identity transform).
    pub fn predict_raw(&self, dmatrix: &DMatrix) -> Result<Vec<f32>, String> {
        if dmatrix.ncols != self.num_features {
            return Err(format!(
                "DMatrix ncols ({}) does not match model num_features ({})",
                dmatrix.ncols, self.num_features
            ));
        }

        let ncols = dmatrix.ncols;
        let base = self.base_score;

        let predictions: Vec<f32> = (0..dmatrix.nrows)
            .into_par_iter()
            .map(|i| {
                let row_start = i * ncols;
                let row_features = &dmatrix.data[row_start..row_start + ncols];
                let mut margin = base;
                for tree in &self.trees {
                    margin += tree.predict_sample(row_features);
                }
                margin
            })
            .collect();

        Ok(predictions)
    }

    /// Number of trees in the ensemble.
    pub fn num_trees(&self) -> usize {
        self.trees.len()
    }
}
