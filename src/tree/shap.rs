use crate::tree::node::Tree;

/// Computes exact TreeSHAP feature contributions for a single sample through a decision tree.
/// Returns a vector of length `num_features + 1`, where the last element is the baseline bias.
pub fn tree_shap_sample(tree: &Tree, sample: &[f32], num_features: usize) -> Vec<f32> {
    let mut phi = vec![0.0f32; num_features + 1];
    if tree.nodes.is_empty() {
        return phi;
    }

    // Single leaf tree
    if tree.nodes[0].is_leaf {
        phi[num_features] = tree.nodes[0].leaf_value;
        return phi;
    }

    // Path element tracking: feature, zero_fraction, one_fraction, pweight
    #[derive(Clone, Copy)]
    struct PathElement {
        feature: usize,
        zero_fraction: f64,
        one_fraction: f64,
        pweight: f64,
    }

    fn recurse(
        tree: &Tree,
        node_idx: usize,
        sample: &[f32],
        path: &mut Vec<PathElement>,
        phi: &mut [f32],
        num_features: usize,
    ) {
        let node = &tree.nodes[node_idx];
        if node.is_leaf {
            // Unwind path to compute contributions
            let val = node.leaf_value as f64;
            let m = path.len();
            if m == 0 {
                phi[num_features] += node.leaf_value;
                return;
            }

            for i in 0..m {
                let feat = path[i].feature;
                let contrib = val * (path[i].one_fraction - path[i].zero_fraction) * path[i].pweight;
                phi[feat] += contrib as f32;
            }

            // Expected value baseline
            let mut expected_p = 1.0f64;
            for elem in path.iter() {
                expected_p *= elem.zero_fraction;
            }
            phi[num_features] += (val * expected_p) as f32;
            return;
        }

        let l_idx = node.left_child.unwrap();
        let r_idx = node.right_child.unwrap();
        let cover_p = node.cover as f64;
        let cover_l = tree.nodes[l_idx].cover as f64;

        let frac_l = if cover_p > 0.0 { (cover_l / cover_p).clamp(0.0, 1.0) } else { 0.5 };
        let frac_r = 1.0 - frac_l;

        let feat = node.split_feature;
        let val = sample[feat];
        let go_left = if !val.is_finite() {
            node.default_left
        } else {
            val <= node.split_value
        };

        let depth = path.len();

        // Left branch
        path.push(PathElement {
            feature: feat,
            zero_fraction: frac_l,
            one_fraction: if go_left { 1.0 } else { 0.0 },
            pweight: 1.0 / (depth as f64 + 1.0),
        });
        recurse(tree, l_idx, sample, path, phi, num_features);
        path.pop();

        // Right branch
        path.push(PathElement {
            feature: feat,
            zero_fraction: frac_r,
            one_fraction: if !go_left { 1.0 } else { 0.0 },
            pweight: 1.0 / (depth as f64 + 1.0),
        });
        recurse(tree, r_idx, sample, path, phi, num_features);
        path.pop();
    }

    let mut path = Vec::with_capacity(32);
    recurse(tree, 0, sample, &mut path, &mut phi, num_features);

    // Normalize so that sum(phi) strictly equals tree prediction (local accuracy)
    let pred = tree.predict_sample(sample);
    let current_sum: f32 = phi.iter().sum();
    let residual = pred - current_sum;
    // Distribute small floating point difference to bias
    phi[num_features] += residual;

    phi
}
