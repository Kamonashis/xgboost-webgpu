use crate::booster::params::GrowPolicy;
use crate::data::DMatrix;
use crate::gpu::histogram::{build_histograms_cpu, FeatureHistogram};
#[cfg(feature = "webgpu")]
use crate::gpu::histogram::GpuHistSession;
use crate::tree::node::{Tree, TreeNode};
use crate::tree::split::{calc_leaf_weight, find_best_split, SplitCandidate};
use rand::Rng;

#[derive(Debug, Clone, Copy)]
pub struct TreeBuilderConfig {
    pub max_depth: usize,
    pub reg_lambda: f32,
    pub reg_alpha: f32,
    pub gamma: f32,
    pub min_child_weight: f32,
    pub learning_rate: f32,
    pub max_delta_step: f32,
    pub grow_policy: GrowPolicy,
    pub max_leaves: usize,
    pub colsample_bytree: f32,
    pub colsample_bylevel: f32,
    pub colsample_bynode: f32,
}

pub struct TreeBuilder<'a> {
    pub dmatrix: &'a DMatrix,
    pub config: TreeBuilderConfig,
    pub grads: &'a [f32],
    pub hess: &'a [f32],
    pub monotone_constraints: Option<&'a [i8]>,
    pub interaction_constraints: Option<&'a [Vec<usize>]>,
    #[cfg(feature = "webgpu")]
    pub gpu_session: Option<&'a GpuHistSession<'a>>,
}

impl<'a> TreeBuilder<'a> {
    pub fn new(
        dmatrix: &'a DMatrix,
        config: TreeBuilderConfig,
        grads: &'a [f32],
        hess: &'a [f32],
    ) -> Self {
        Self {
            dmatrix,
            config,
            grads,
            hess,
            monotone_constraints: None,
            interaction_constraints: None,
            #[cfg(feature = "webgpu")]
            gpu_session: None,
        }
    }

    #[cfg(feature = "webgpu")]
    pub fn with_gpu_session(mut self, session: &'a GpuHistSession<'a>) -> Self {
        self.gpu_session = Some(session);
        self
    }

    pub fn with_constraints(
        mut self,
        monotone: Option<&'a [i8]>,
        interaction: Option<&'a [Vec<usize>]>,
    ) -> Self {
        self.monotone_constraints = monotone;
        self.interaction_constraints = interaction;
        self
    }

    /// Builds a single tree according to the configured grow policy (DepthWise or LossGuide).
    pub fn build(&self) -> Tree {
        let mut tree = Tree::new();
        let n = self.dmatrix.nrows;
        if n == 0 {
            return tree;
        }

        let root_samples: Vec<u32> = (0..n as u32).collect();
        let total_g: f32 = self.grads.iter().sum();
        let total_h: f32 = self.hess.iter().sum();

        // Sample features for colsample_bytree
        let mut rng = rand::rng();
        let ncols = self.dmatrix.ncols;
        let mut bytree_features: Vec<usize> = (0..ncols).collect();
        if self.config.colsample_bytree < 1.0 {
            let keep_count = ((ncols as f32) * self.config.colsample_bytree).ceil() as usize;
            bytree_features.retain(|_| rng.random::<f32>() <= self.config.colsample_bytree);
            if bytree_features.is_empty() {
                bytree_features.push(rng.random_range(0..ncols));
            }
            if bytree_features.len() > keep_count {
                bytree_features.truncate(keep_count);
            }
        }

        // Build root histogram
        let root_hist = self.compute_histogram(&root_samples);

        match self.config.grow_policy {
            GrowPolicy::DepthWise => {
                self.build_depthwise(
                    &mut tree,
                    root_samples,
                    root_hist,
                    total_g,
                    total_h,
                    0,
                    &bytree_features,
                );
            }
            GrowPolicy::LossGuide => {
                self.build_lossguide(
                    &mut tree,
                    root_samples,
                    root_hist,
                    total_g,
                    total_h,
                    &bytree_features,
                );
            }
        }

        tree
    }

    fn compute_histogram(&self, sample_indices: &[u32]) -> Vec<FeatureHistogram> {
        #[cfg(feature = "webgpu")]
        if let Some(gpu) = self.gpu_session {
            return gpu.build_histograms(sample_indices, self.grads, self.hess);
        }

        build_histograms_cpu(self.dmatrix, sample_indices, self.grads, self.hess)
    }

    fn sample_node_features(&self, available: &[usize]) -> Vec<usize> {
        if self.config.colsample_bynode >= 1.0 {
            return available.to_vec();
        }
        let mut rng = rand::rng();
        let mut sampled: Vec<usize> = available
            .iter()
            .copied()
            .filter(|_| rng.random::<f32>() <= self.config.colsample_bynode)
            .collect();
        if sampled.is_empty() && !available.is_empty() {
            sampled.push(available[rng.random_range(0..available.len())]);
        }
        sampled
    }

    fn build_depthwise(
        &self,
        tree: &mut Tree,
        samples: Vec<u32>,
        histograms: Vec<FeatureHistogram>,
        sum_g: f32,
        sum_h: f32,
        depth: usize,
        available_features: &[usize],
    ) -> usize {
        let node_idx = tree.nodes.len();
        let leaf_val = calc_leaf_weight(
            sum_g,
            sum_h,
            self.config.reg_lambda,
            self.config.reg_alpha,
            self.config.learning_rate,
            self.config.max_delta_step,
        );
        tree.nodes.push(TreeNode::new_leaf(leaf_val, sum_h));

        if depth >= self.config.max_depth || samples.len() < 2 {
            return node_idx;
        }

        let node_features = self.sample_node_features(available_features);

        let best_split = find_best_split(
            &histograms,
            &self.dmatrix.bin_mappers,
            sum_g,
            sum_h,
            self.config.reg_lambda,
            self.config.reg_alpha,
            self.config.gamma,
            self.config.min_child_weight,
            Some(&node_features),
            self.monotone_constraints,
        );

        let split = match best_split {
            Some(s) if s.gain > 0.0 => s,
            _ => return node_idx,
        };

        let mut left_samples = Vec::with_capacity(samples.len() / 2);
        let mut right_samples = Vec::with_capacity(samples.len() / 2);
        let split_feature = split.feature;
        let split_bin = split.bin;
        let default_left = split.default_left;
        let missing_bin = self.dmatrix.bin_mappers[split_feature].missing_bin;

        for &idx in &samples {
            let row = idx as usize;
            let bin = self.dmatrix.get_bin(row, split_feature);
            let go_left = if bin == missing_bin {
                default_left
            } else {
                bin <= split_bin
            };

            if go_left {
                left_samples.push(idx);
            } else {
                right_samples.push(idx);
            }
        }

        if left_samples.is_empty() || right_samples.is_empty() {
            return node_idx;
        }

        tree.nodes[node_idx] = TreeNode::new_split(
            split.feature,
            split.bin,
            split.split_value,
            split.default_left,
            split.gain,
            sum_h,
        );

        let (left_hist, right_hist) = if left_samples.len() <= right_samples.len() {
            let lh = self.compute_histogram(&left_samples);
            let rh: Vec<FeatureHistogram> = (0..self.dmatrix.ncols)
                .map(|j| FeatureHistogram::subtract(&histograms[j], &lh[j]))
                .collect();
            (lh, rh)
        } else {
            let rh = self.compute_histogram(&right_samples);
            let lh: Vec<FeatureHistogram> = (0..self.dmatrix.ncols)
                .map(|j| FeatureHistogram::subtract(&histograms[j], &rh[j]))
                .collect();
            (lh, rh)
        };

        let left_child_idx = self.build_depthwise(
            tree,
            left_samples,
            left_hist,
            split.left_g,
            split.left_h,
            depth + 1,
            available_features,
        );
        let right_child_idx = self.build_depthwise(
            tree,
            right_samples,
            right_hist,
            split.right_g,
            split.right_h,
            depth + 1,
            available_features,
        );

        tree.nodes[node_idx].left_child = Some(left_child_idx);
        tree.nodes[node_idx].right_child = Some(right_child_idx);

        node_idx
    }

    fn build_lossguide(
        &self,
        tree: &mut Tree,
        root_samples: Vec<u32>,
        root_hist: Vec<FeatureHistogram>,
        root_g: f32,
        root_h: f32,
        available_features: &[usize],
    ) {
        struct LeafCandidate {
            node_idx: usize,
            depth: usize,
            samples: Vec<u32>,
            histograms: Vec<FeatureHistogram>,
            _sum_g: f32,
            sum_h: f32,
            best_split: Option<SplitCandidate>,
        }

        let root_leaf_val = calc_leaf_weight(
            root_g,
            root_h,
            self.config.reg_lambda,
            self.config.reg_alpha,
            self.config.learning_rate,
            self.config.max_delta_step,
        );
        tree.nodes.push(TreeNode::new_leaf(root_leaf_val, root_h));

        let node_features = self.sample_node_features(available_features);
        let initial_split = find_best_split(
            &root_hist,
            &self.dmatrix.bin_mappers,
            root_g,
            root_h,
            self.config.reg_lambda,
            self.config.reg_alpha,
            self.config.gamma,
            self.config.min_child_weight,
            Some(&node_features),
            self.monotone_constraints,
        );

        let mut queue = Vec::new();
        queue.push(LeafCandidate {
            node_idx: 0,
            depth: 0,
            samples: root_samples,
            histograms: root_hist,
            _sum_g: root_g,
            sum_h: root_h,
            best_split: initial_split,
        });

        let max_leaves = if self.config.max_leaves > 0 {
            self.config.max_leaves
        } else {
            1 << self.config.max_depth.min(16)
        };

        let mut num_leaves = 1;

        while num_leaves < max_leaves {
            // Find leaf candidate with maximum positive gain
            let mut best_idx = None;
            let mut max_gain = 0.0f32;

            for (idx, cand) in queue.iter().enumerate() {
                if cand.depth >= self.config.max_depth || cand.samples.len() < 2 {
                    continue;
                }
                if let Some(ref s) = cand.best_split {
                    if s.gain > max_gain {
                        max_gain = s.gain;
                        best_idx = Some(idx);
                    }
                }
            }

            let cand_idx = match best_idx {
                Some(idx) => idx,
                None => break, // No more profitable splits
            };

            let cand = queue.swap_remove(cand_idx);
            let split = cand.best_split.unwrap();

            // Partition samples
            let mut left_samples = Vec::with_capacity(cand.samples.len() / 2);
            let mut right_samples = Vec::with_capacity(cand.samples.len() / 2);
            let split_feature = split.feature;
            let split_bin = split.bin;
            let default_left = split.default_left;
            let missing_bin = self.dmatrix.bin_mappers[split_feature].missing_bin;

            for &idx in &cand.samples {
                let row = idx as usize;
                let bin = self.dmatrix.get_bin(row, split_feature);
                let go_left = if bin == missing_bin {
                    default_left
                } else {
                    bin <= split_bin
                };
                if go_left {
                    left_samples.push(idx);
                } else {
                    right_samples.push(idx);
                }
            }

            if left_samples.is_empty() || right_samples.is_empty() {
                continue;
            }

            // Convert cand.node_idx from leaf to split node
            let left_node_idx = tree.nodes.len();
            let right_node_idx = left_node_idx + 1;

            tree.nodes[cand.node_idx] = TreeNode::new_split(
                split.feature,
                split.bin,
                split.split_value,
                split.default_left,
                split.gain,
                cand.sum_h,
            );
            tree.nodes[cand.node_idx].left_child = Some(left_node_idx);
            tree.nodes[cand.node_idx].right_child = Some(right_node_idx);

            // Compute histograms with subtraction
            let (left_hist, right_hist) = if left_samples.len() <= right_samples.len() {
                let lh = self.compute_histogram(&left_samples);
                let rh: Vec<FeatureHistogram> = (0..self.dmatrix.ncols)
                    .map(|j| FeatureHistogram::subtract(&cand.histograms[j], &lh[j]))
                    .collect();
                (lh, rh)
            } else {
                let rh = self.compute_histogram(&right_samples);
                let lh: Vec<FeatureHistogram> = (0..self.dmatrix.ncols)
                    .map(|j| FeatureHistogram::subtract(&cand.histograms[j], &rh[j]))
                    .collect();
                (lh, rh)
            };

            let left_leaf_val = calc_leaf_weight(
                split.left_g,
                split.left_h,
                self.config.reg_lambda,
                self.config.reg_alpha,
                self.config.learning_rate,
                self.config.max_delta_step,
            );
            let right_leaf_val = calc_leaf_weight(
                split.right_g,
                split.right_h,
                self.config.reg_lambda,
                self.config.reg_alpha,
                self.config.learning_rate,
                self.config.max_delta_step,
            );

            tree.nodes.push(TreeNode::new_leaf(left_leaf_val, split.left_h));
            tree.nodes.push(TreeNode::new_leaf(right_leaf_val, split.right_h));

            num_leaves += 1; // 1 leaf became 2 leaves, net +1

            let next_depth = cand.depth + 1;

            // Evaluate best splits for newly created leaves
            let left_split = find_best_split(
                &left_hist,
                &self.dmatrix.bin_mappers,
                split.left_g,
                split.left_h,
                self.config.reg_lambda,
                self.config.reg_alpha,
                self.config.gamma,
                self.config.min_child_weight,
                Some(&self.sample_node_features(available_features)),
                self.monotone_constraints,
            );

            let right_split = find_best_split(
                &right_hist,
                &self.dmatrix.bin_mappers,
                split.right_g,
                split.right_h,
                self.config.reg_lambda,
                self.config.reg_alpha,
                self.config.gamma,
                self.config.min_child_weight,
                Some(&self.sample_node_features(available_features)),
                self.monotone_constraints,
            );

            queue.push(LeafCandidate {
                node_idx: left_node_idx,
                depth: next_depth,
                samples: left_samples,
                histograms: left_hist,
                _sum_g: split.left_g,
                sum_h: split.left_h,
                best_split: left_split,
            });

            queue.push(LeafCandidate {
                node_idx: right_node_idx,
                depth: next_depth,
                samples: right_samples,
                histograms: right_hist,
                _sum_g: split.right_g,
                sum_h: split.right_h,
                best_split: right_split,
            });
        }
    }
}
