use crate::booster::params::{BoosterParams, BoosterType, DeviceType};
use crate::data::DMatrix;
#[cfg(feature = "webgpu")]
use crate::gpu::context::GpuContext;
#[cfg(feature = "webgpu")]
use crate::gpu::histogram::GpuHistSession;
use crate::metrics;
use crate::objective::{get_objective, MultiClassObjective, Objective};
use crate::tree::builder::{TreeBuilder, TreeBuilderConfig};
use crate::tree::node::Tree;
use crate::tree::shap::tree_shap_sample;
use rand::Rng;
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
    /// Constructs an empty booster ensemble.
    pub fn empty() -> Self {
        Self {
            params: BoosterParams::new(),
            trees: Vec::new(),
            base_score: 0.5,
            num_features: 0,
            feature_names: Vec::new(),
        }
    }

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

        let is_multiclass = params.objective.starts_with("multi:") || params.num_class >= 2;
        let num_class = if is_multiclass {
            if params.num_class >= 2 {
                params.num_class
            } else {
                let max_lbl = labels.iter().copied().fold(0.0f32, f32::max) as usize;
                max_lbl + 1
            }
        } else {
            1
        };

        let mut booster_params = params.clone();
        booster_params.num_class = if is_multiclass { num_class } else { 0 };

        let tree_config = TreeBuilderConfig {
            max_depth: params.max_depth,
            reg_lambda: params.reg_lambda,
            reg_alpha: params.reg_alpha,
            gamma: params.gamma,
            min_child_weight: params.min_child_weight,
            learning_rate: params.learning_rate,
            max_delta_step: params.max_delta_step,
            grow_policy: params.grow_policy,
            max_leaves: params.max_leaves,
            colsample_bytree: params.colsample_bytree,
            colsample_bylevel: params.colsample_bylevel,
            colsample_bynode: params.colsample_bynode,
        };

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

        let mut trees = Vec::with_capacity(num_boost_round * num_class);
        let mut rng = rand::rng();

        // ----------------- Multi-Class Training Branch -----------------
        if is_multiclass {
            let is_softmax = params.objective == "multi:softmax";
            let mc_obj = MultiClassObjective::new(num_class, is_softmax);
            let mut margins = vec![0.0f32; n * num_class];

            let mut eval_margins: Vec<Vec<f32>> = evals
                .iter()
                .map(|(dmat, _)| vec![0.0f32; dmat.nrows * num_class])
                .collect();

            let mut best_score = f32::INFINITY;
            let mut best_iter = 0;
            let mut rounds_without_improvement = 0;

            for round in 0..num_boost_round {
                for c in 0..num_class {
                    let mut grads = vec![0.0f32; n];
                    let mut hess = vec![0.0f32; n];

                    mc_obj.compute_class_gradients(
                        labels,
                        &margins,
                        c,
                        dtrain.weights.as_deref(),
                        &mut grads,
                        &mut hess,
                    );

                    // Row subsampling (bagging)
                    if params.subsample < 1.0 {
                        for i in 0..n {
                            if rng.random::<f32>() > params.subsample {
                                grads[i] = 0.0;
                                hess[i] = 0.0;
                            }
                        }
                    }

                    let builder = TreeBuilder::new(dtrain, tree_config, &grads, &hess)
                        .with_constraints(
                            params.monotone_constraints.as_deref(),
                            params.interaction_constraints.as_deref(),
                        );

                    #[cfg(feature = "webgpu")]
                    let tree = if let Some(ref sess) = gpu_session {
                        sess.update_gradients(&grads, &hess);
                        builder.with_gpu_session(sess).build()
                    } else {
                        builder.build()
                    };

                    #[cfg(not(feature = "webgpu"))]
                    let tree = builder.build();

                    // Update margins for class c
                    let ncols = dtrain.ncols;
                    for i in 0..n {
                        let row_start = i * ncols;
                        let row_features = &dtrain.data[row_start..row_start + ncols];
                        margins[i * num_class + c] += tree.predict_sample(row_features);
                    }

                    trees.push(tree);
                }

                // Check evaluation and early stopping
                if !evals.is_empty() {
                    let mut log_line = format!("[{}]", round);
                    for (idx, (eval_dmat, name)) in evals.iter().enumerate() {
                        let ev_ncols = eval_dmat.ncols;
                        for i in 0..eval_dmat.nrows {
                            let row_start = i * ev_ncols;
                            let row_features = &eval_dmat.data[row_start..row_start + ev_ncols];
                            for c in 0..num_class {
                                let tree_idx = round * num_class + c;
                                eval_margins[idx][i * num_class + c] +=
                                    trees[tree_idx].predict_sample(row_features);
                            }
                        }

                        if let Some(eval_labels) = &eval_dmat.labels {
                            let preds = mc_obj.transform_margins(&eval_margins[idx]);
                            // Calculate multi-class accuracy / error
                            let mut err_count = 0;
                            for i in 0..eval_dmat.nrows {
                                let pred_class = if is_softmax {
                                    preds[i] as usize
                                } else {
                                    let mut best_c = 0;
                                    let mut max_p = f32::NEG_INFINITY;
                                    for c in 0..num_class {
                                        let p = preds[i * num_class + c];
                                        if p > max_p {
                                            max_p = p;
                                            best_c = c;
                                        }
                                    }
                                    best_c
                                };
                                if pred_class != (eval_labels[i] as usize) {
                                    err_count += 1;
                                }
                            }
                            let error_rate = err_count as f32 / eval_dmat.nrows as f32;
                            log_line.push_str(&format!("  {}-error:{:.5}", name, error_rate));

                            if idx == 0 {
                                if error_rate < best_score {
                                    best_score = error_rate;
                                    best_iter = round;
                                    rounds_without_improvement = 0;
                                } else {
                                    rounds_without_improvement += 1;
                                }
                            }
                        }
                    }
                    println!("{}", log_line);

                    if let Some(patience) = params.early_stopping_rounds {
                        if rounds_without_improvement >= patience {
                            println!("Early stopping at round {} (best score: {:.5})", best_iter, best_score);
                            booster_params.best_iteration = Some(best_iter);
                            booster_params.best_score = Some(best_score);
                            trees.truncate((best_iter + 1) * num_class);
                            break;
                        }
                    }
                }
            }

            return Ok(Self {
                params: booster_params,
                trees,
                base_score: 0.0,
                num_features: dtrain.ncols,
                feature_names: dtrain.feature_names.clone(),
            });
        }

        // ----------------- Single-Output Training Branch -----------------
        let obj: Box<dyn Objective> = get_objective(&params.objective)?;
        let base_score = params
            .base_score
            .unwrap_or_else(|| obj.default_base_score(labels));

        let mut current_margin = vec![base_score; n];
        let mut grads = vec![0.0f32; n];
        let mut hess = vec![0.0f32; n];

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

        let mut best_score = f32::INFINITY;
        let mut best_iter = 0;
        let mut rounds_without_improvement = 0;

        for round in 0..num_boost_round {
            // DART booster tree dropping
            let mut dropped_indices = Vec::new();
            if params.booster_type == BoosterType::DART && !trees.is_empty() {
                if params.rate_drop > 0.0 && rng.random::<f32>() > params.skip_drop {
                    for (t_idx, _) in trees.iter().enumerate() {
                        if rng.random::<f32>() <= params.rate_drop {
                            dropped_indices.push(t_idx);
                        }
                    }
                }
            }

            // Recompute margins if trees were dropped in DART
            let mut effective_margin = current_margin.clone();
            if !dropped_indices.is_empty() {
                let ncols = dtrain.ncols;
                for i in 0..n {
                    let row_start = i * ncols;
                    let row_features = &dtrain.data[row_start..row_start + ncols];
                    for &d_idx in &dropped_indices {
                        effective_margin[i] -= trees[d_idx].predict_sample(row_features);
                    }
                }
            }

            // 1. Compute loss gradients and Hessians
            obj.compute_gradients(
                labels,
                &effective_margin,
                dtrain.weights.as_deref(),
                &mut grads,
                &mut hess,
            );

            // Row subsampling (bagging)
            if params.subsample < 1.0 {
                for i in 0..n {
                    if rng.random::<f32>() > params.subsample {
                        grads[i] = 0.0;
                        hess[i] = 0.0;
                    }
                }
            }

            // 2. Build tree using GPU session or CPU fallback
            let builder = TreeBuilder::new(dtrain, tree_config, &grads, &hess)
                .with_constraints(
                    params.monotone_constraints.as_deref(),
                    params.interaction_constraints.as_deref(),
                );

            #[cfg(feature = "webgpu")]
            let mut tree = if let Some(ref sess) = gpu_session {
                sess.update_gradients(&grads, &hess);
                builder.with_gpu_session(sess).build()
            } else {
                builder.build()
            };

            #[cfg(not(feature = "webgpu"))]
            let mut tree = builder.build();

            // DART normalization
            if params.booster_type == BoosterType::DART && !dropped_indices.is_empty() {
                let k = dropped_indices.len() as f32;
                let factor = 1.0 / (k + 1.0);
                for node in &mut tree.nodes {
                    if node.is_leaf {
                        node.leaf_value *= factor;
                    }
                }
            }

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

                let train_preds = obj.transform_predictions(&current_margin);
                if let Ok(train_score) = metrics::evaluate(&metric_name, labels, &train_preds) {
                    log_line.push_str(&format!("  train-{}:{:.5}", metric_name, train_score));
                }

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

                            if idx == 0 {
                                if score < best_score {
                                    best_score = score;
                                    best_iter = round;
                                    rounds_without_improvement = 0;
                                } else {
                                    rounds_without_improvement += 1;
                                }
                            }
                        }
                    }
                }

                println!("{}", log_line);

                if let Some(patience) = params.early_stopping_rounds {
                    if rounds_without_improvement >= patience {
                        println!("Early stopping at round {} (best score: {:.5})", best_iter, best_score);
                        booster_params.best_iteration = Some(best_iter);
                        booster_params.best_score = Some(best_score);
                        trees.truncate(best_iter + 1);
                        break;
                    }
                }
            }

            trees.push(tree);
        }

        Ok(Self {
            params: booster_params,
            trees,
            base_score,
            num_features: dtrain.ncols,
            feature_names: dtrain.feature_names.clone(),
        })
    }

    /// Continues training an existing booster for additional rounds.
    pub fn train_continue(
        &mut self,
        dtrain: &DMatrix,
        additional_rounds: usize,
        evals: &[(&DMatrix, &str)],
    ) -> Result<(), String> {
        let extension = Self::train(self.params.clone(), dtrain, additional_rounds, evals)?;
        self.trees.extend(extension.trees);
        Ok(())
    }

    /// Predict leaf indices for each sample across all trees.
    /// Returns an N x T matrix where element [i][t] is the index of the leaf node sample i landed on in tree t.
    pub fn predict_leaf(&self, dmatrix: &DMatrix) -> Result<Vec<Vec<usize>>, String> {
        if dmatrix.ncols != self.num_features {
            return Err(format!(
                "DMatrix ncols ({}) does not match model num_features ({})",
                dmatrix.ncols, self.num_features
            ));
        }

        let ncols = self.num_features;
        let num_trees = self.trees.len();

        let leaf_matrix: Vec<Vec<usize>> = (0..dmatrix.nrows)
            .into_par_iter()
            .map(|i| {
                let row_start = i * ncols;
                let row_features = &dmatrix.data[row_start..row_start + ncols];
                let mut leaves = Vec::with_capacity(num_trees);

                for tree in &self.trees {
                    let mut curr_idx = 0;
                    loop {
                        let node = &tree.nodes[curr_idx];
                        if node.is_leaf {
                            leaves.push(curr_idx);
                            break;
                        }
                        let val = row_features[node.split_feature];
                        let go_left = if !val.is_finite() {
                            node.default_left
                        } else {
                            val <= node.split_value
                        };
                        curr_idx = if go_left {
                            node.left_child.unwrap()
                        } else {
                            node.right_child.unwrap()
                        };
                    }
                }
                leaves
            })
            .collect();

        Ok(leaf_matrix)
    }

    /// Computes exact TreeSHAP feature contributions and baseline bias for all samples in the dataset.
    /// Returns an N x (M + 1) matrix satisfying: sum_j phi_ij + phi_bias = raw_margin_i.
    pub fn predict_contributions(&self, dmatrix: &DMatrix) -> Result<Vec<Vec<f32>>, String> {
        if dmatrix.ncols != self.num_features {
            return Err(format!(
                "DMatrix ncols ({}) does not match model num_features ({})",
                dmatrix.ncols, self.num_features
            ));
        }

        let ncols = self.num_features;
        let base = self.base_score;

        let contributions: Vec<Vec<f32>> = (0..dmatrix.nrows)
            .into_par_iter()
            .map(|i| {
                let row_start = i * ncols;
                let row_features = &dmatrix.data[row_start..row_start + ncols];
                let mut total_phi = vec![0.0f32; ncols + 1];
                total_phi[ncols] = base; // Baseline bias starts at base_score

                for tree in &self.trees {
                    let tree_phi = tree_shap_sample(tree, row_features, ncols);
                    for j in 0..=ncols {
                        total_phi[j] += tree_phi[j];
                    }
                }

                total_phi
            })
            .collect();

        Ok(contributions)
    }

    /// Evaluates the booster ensemble and returns final predictions.
    pub fn predict(&self, dmatrix: &DMatrix) -> Result<Vec<f32>, String> {
        if self.params.num_class >= 2 {
            let k = self.params.num_class;
            let is_softmax = self.params.objective == "multi:softmax";
            let mc_obj = MultiClassObjective::new(k, is_softmax);

            let ncols = dmatrix.ncols;
            let mut margins = vec![0.0f32; dmatrix.nrows * k];

            for (tree_idx, tree) in self.trees.iter().enumerate() {
                let c = tree_idx % k;
                for i in 0..dmatrix.nrows {
                    let row_start = i * ncols;
                    let row_features = &dmatrix.data[row_start..row_start + ncols];
                    margins[i * k + c] += tree.predict_sample(row_features);
                }
            }

            return Ok(mc_obj.transform_margins(&margins));
        }

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
