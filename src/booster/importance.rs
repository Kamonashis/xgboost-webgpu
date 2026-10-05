use crate::booster::Booster;
use serde::{Deserialize, Serialize};

/// Metric type used to measure feature importance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImportanceType {
    /// Number of times a feature is used to split the data across all trees.
    Weight,
    /// Average split gain of the feature across all splits.
    Gain,
    /// Average coverage (sum of Hessians) of splits on this feature.
    Cover,
    /// Total gain contributed by the feature across all trees.
    TotalGain,
    /// Total coverage contributed by the feature across all trees.
    TotalCover,
}

impl Booster {
    /// Computes feature importance scores across all trees in the ensemble.
    /// Returns a list of `(feature_name, score)` sorted in descending order of importance.
    pub fn feature_importance(&self, importance_type: ImportanceType) -> Vec<(String, f32)> {
        let num_features = self.num_features;
        let mut counts = vec![0.0f32; num_features];
        let mut total_gain = vec![0.0f32; num_features];
        let mut total_cover = vec![0.0f32; num_features];

        for tree in &self.trees {
            for node in &tree.nodes {
                if !node.is_leaf {
                    let f = node.split_feature;
                    if f < num_features {
                        counts[f] += 1.0;
                        total_gain[f] += node.gain;
                        total_cover[f] += node.cover;
                    }
                }
            }
        }

        let mut results: Vec<(String, f32)> = (0..num_features)
            .map(|f| {
                let name = self.feature_names.get(f).cloned().unwrap_or_else(|| format!("f{f}"));
                let score = match importance_type {
                    ImportanceType::Weight => counts[f],
                    ImportanceType::Gain => {
                        if counts[f] > 0.0 { total_gain[f] / counts[f] } else { 0.0 }
                    }
                    ImportanceType::Cover => {
                        if counts[f] > 0.0 { total_cover[f] / counts[f] } else { 0.0 }
                    }
                    ImportanceType::TotalGain => total_gain[f],
                    ImportanceType::TotalCover => total_cover[f],
                };
                (name, score)
            })
            .collect();

        // Sort descending by score
        results.sort_by(|a, b| b.1.total_cmp(&a.1));
        results
    }
}
