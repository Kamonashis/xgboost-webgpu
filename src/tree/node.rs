use serde::{Deserialize, Serialize};

/// Represents a single node in a decision tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeNode {
    pub is_leaf: bool,
    /// Split feature index (if internal node).
    pub split_feature: usize,
    /// Split bin threshold (if internal node). Values <= split_bin go left.
    pub split_bin: u8,
    /// Continuous threshold value for user-facing interpretation and inference.
    pub split_value: f32,
    /// If true, missing values go left; otherwise right.
    pub default_left: bool,
    /// Left child index in the tree's node array.
    pub left_child: Option<usize>,
    /// Right child index in the tree's node array.
    pub right_child: Option<usize>,
    /// Leaf prediction value (if leaf).
    pub leaf_value: f32,
    /// Split gain achieved (if internal node).
    pub gain: f32,
    /// Sum of Hessians in this node (cover).
    pub cover: f32,
}

impl TreeNode {
    pub fn new_leaf(leaf_value: f32, cover: f32) -> Self {
        Self {
            is_leaf: true,
            split_feature: 0,
            split_bin: 0,
            split_value: 0.0,
            default_left: true,
            left_child: None,
            right_child: None,
            leaf_value,
            gain: 0.0,
            cover,
        }
    }

    pub fn new_split(
        split_feature: usize,
        split_bin: u8,
        split_value: f32,
        default_left: bool,
        gain: f32,
        cover: f32,
    ) -> Self {
        Self {
            is_leaf: false,
            split_feature,
            split_bin,
            split_value,
            default_left,
            left_child: None,
            right_child: None,
            leaf_value: 0.0,
            gain,
            cover,
        }
    }
}

/// A decision tree composed of an array of TreeNodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tree {
    pub nodes: Vec<TreeNode>,
}

impl Tree {
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    /// Evaluates a single sample through the decision tree, returning the leaf value.
    #[inline]
    pub fn predict_sample(&self, sample_features: &[f32]) -> f32 {
        if self.nodes.is_empty() {
            return 0.0;
        }

        let mut curr_idx = 0;
        loop {
            let node = &self.nodes[curr_idx];
            if node.is_leaf {
                return node.leaf_value;
            }

            let val = sample_features[node.split_feature];
            let go_left = if !val.is_finite() {
                node.default_left
            } else {
                val <= node.split_value
            };

            curr_idx = if go_left {
                node.left_child.expect("Internal node missing left child")
            } else {
                node.right_child.expect("Internal node missing right child")
            };
        }
    }

    /// Number of leaves in this tree.
    pub fn num_leaves(&self) -> usize {
        self.nodes.iter().filter(|n| n.is_leaf).count()
    }
}
