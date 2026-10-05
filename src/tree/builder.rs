use crate::data::DMatrix;
use crate::gpu::histogram::{build_histograms_cpu, FeatureHistogram};
#[cfg(feature = "webgpu")]
use crate::gpu::histogram::GpuHistSession;
use crate::tree::node::{Tree, TreeNode};
use crate::tree::split::{calc_leaf_weight, find_best_split};

#[derive(Debug, Clone, Copy)]
pub struct TreeBuilderConfig {
    pub max_depth: usize,
    pub reg_lambda: f32,
    pub reg_alpha: f32,
    pub gamma: f32,
    pub min_child_weight: f32,
    pub learning_rate: f32,
}

pub struct TreeBuilder<'a> {
    pub dmatrix: &'a DMatrix,
    pub config: TreeBuilderConfig,
    pub grads: &'a [f32],
    pub hess: &'a [f32],
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
            #[cfg(feature = "webgpu")]
            gpu_session: None,
        }
    }

    #[cfg(feature = "webgpu")]
    pub fn with_gpu_session(mut self, session: &'a GpuHistSession<'a>) -> Self {
        self.gpu_session = Some(session);
        self
    }

    /// Builds a single regression tree using GPU-accelerated histograms and sibling subtraction.
    pub fn build(&self) -> Tree {
        let mut tree = Tree::new();
        let n = self.dmatrix.nrows;
        if n == 0 {
            return tree;
        }

        let root_samples: Vec<u32> = (0..n as u32).collect();
        let total_g: f32 = self.grads.iter().sum();
        let total_h: f32 = self.hess.iter().sum();

        // Build root histogram
        let root_hist = self.compute_histogram(&root_samples);

        self.build_recursive(
            &mut tree,
            root_samples,
            root_hist,
            total_g,
            total_h,
            0,
        );

        tree
    }

    fn compute_histogram(&self, sample_indices: &[u32]) -> Vec<FeatureHistogram> {
        #[cfg(feature = "webgpu")]
        if let Some(gpu) = self.gpu_session {
            return gpu.build_histograms(sample_indices, self.grads, self.hess);
        }

        build_histograms_cpu(self.dmatrix, sample_indices, self.grads, self.hess)
    }

    fn build_recursive(
        &self,
        tree: &mut Tree,
        samples: Vec<u32>,
        histograms: Vec<FeatureHistogram>,
        sum_g: f32,
        sum_h: f32,
        depth: usize,
    ) -> usize {
        let node_idx = tree.nodes.len();
        // Placeholder leaf node
        let leaf_val = calc_leaf_weight(
            sum_g,
            sum_h,
            self.config.reg_lambda,
            self.config.reg_alpha,
            self.config.learning_rate,
        );
        tree.nodes.push(TreeNode::new_leaf(leaf_val, sum_h));

        if depth >= self.config.max_depth || samples.len() < 2 {
            return node_idx;
        }

        let best_split = find_best_split(
            &histograms,
            &self.dmatrix.bin_mappers,
            sum_g,
            sum_h,
            self.config.reg_lambda,
            self.config.reg_alpha,
            self.config.gamma,
            self.config.min_child_weight,
        );

        let split = match best_split {
            Some(s) if s.gain > 0.0 => s,
            _ => return node_idx,
        };

        // Partition samples into left and right
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

        // Convert current node from leaf to split node
        tree.nodes[node_idx] = TreeNode::new_split(
            split.feature,
            split.bin,
            split.split_value,
            split.default_left,
            split.gain,
            sum_h,
        );

        // Histogram subtraction: compute histogram for smaller child, subtract for larger child
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

        let left_child_idx = self.build_recursive(
            tree,
            left_samples,
            left_hist,
            split.left_g,
            split.left_h,
            depth + 1,
        );
        let right_child_idx = self.build_recursive(
            tree,
            right_samples,
            right_hist,
            split.right_g,
            split.right_h,
            depth + 1,
        );

        tree.nodes[node_idx].left_child = Some(left_child_idx);
        tree.nodes[node_idx].right_child = Some(right_child_idx);

        node_idx
    }
}
