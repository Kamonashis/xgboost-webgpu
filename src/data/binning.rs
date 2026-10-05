use serde::{Deserialize, Serialize};

/// Type of feature: Numerical (continuous/discrete) or Categorical
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeatureType {
    Numerical,
    Categorical,
}

impl Default for FeatureType {
    fn default() -> Self {
        FeatureType::Numerical
    }
}

/// Quantization and histogram binning logic for continuous and categorical features.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureBinMapper {
    /// Ascending threshold cut points for bins (or unique categories for categorical).
    pub cut_points: Vec<f32>,
    /// Minimum value observed in training data for this feature.
    pub min_val: f32,
    /// Maximum value observed in training data for this feature.
    pub max_val: f32,
    /// Bin assigned to missing (NaN) values.
    pub missing_bin: u8,
    /// Feature type: Numerical or Categorical.
    pub feature_type: FeatureType,
}

impl FeatureBinMapper {
    /// Creates a new bin mapper from continuous values using quantile / equal-frequency binning.
    pub fn fit(values: &[f32], max_bins: usize) -> Self {
        Self::fit_with_type(values, max_bins, FeatureType::Numerical)
    }

    /// Creates a bin mapper with explicit feature type (numerical or categorical).
    pub fn fit_with_type(values: &[f32], max_bins: usize, feature_type: FeatureType) -> Self {
        assert!(max_bins >= 2 && max_bins <= 256, "max_bins must be between 2 and 256");

        let mut valid_vals: Vec<f32> = values
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .collect();

        if valid_vals.is_empty() {
            return Self {
                cut_points: vec![0.0],
                min_val: 0.0,
                max_val: 0.0,
                missing_bin: 0,
                feature_type,
            };
        }

        valid_vals.sort_by(|a, b| a.total_cmp(b));

        let min_val = *valid_vals.first().unwrap();
        let max_val = *valid_vals.last().unwrap();

        let mut distinct_vals = Vec::with_capacity(valid_vals.len());
        for &v in &valid_vals {
            if distinct_vals.is_empty() || distinct_vals.last() != Some(&v) {
                distinct_vals.push(v);
            }
        }

        let mut cut_points = Vec::new();
        let num_distinct = distinct_vals.len();

        if feature_type == FeatureType::Categorical || num_distinct <= max_bins {
            for i in 0..num_distinct.saturating_sub(1) {
                cut_points.push(distinct_vals[i]);
            }
        } else {
            let n = valid_vals.len();
            for i in 1..max_bins {
                let idx = (i * n) / max_bins;
                let val = valid_vals[idx.min(n - 1)];
                if cut_points.is_empty() || *cut_points.last().unwrap() < val {
                    cut_points.push(val);
                }
            }
        }

        if cut_points.is_empty() {
            cut_points.push(max_val);
        }

        let num_bins = cut_points.len() + 1;
        let missing_bin = (num_bins.min(max_bins - 1)) as u8;

        Self {
            cut_points,
            min_val,
            max_val,
            missing_bin,
            feature_type,
        }
    }

    /// Maps a single continuous value to a bin index in `0..=cut_points.len()`.
    #[inline(always)]
    pub fn map_value(&self, value: f32) -> u8 {
        if !value.is_finite() {
            return self.missing_bin;
        }

        let bin = self.cut_points.partition_point(|&cut| cut < value);
        bin.min(255) as u8
    }

    /// Number of bins for this feature.
    pub fn num_bins(&self) -> usize {
        self.cut_points.len() + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bin_mapper_basic() {
        let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mapper = FeatureBinMapper::fit(&vals, 4);
        assert!(!mapper.cut_points.is_empty());
        assert!(mapper.cut_points.len() < 4);

        let bin1 = mapper.map_value(1.0);
        let bin8 = mapper.map_value(8.0);
        assert!(bin1 < bin8);
    }

    #[test]
    fn test_bin_mapper_with_nan() {
        let vals = vec![1.0, f32::NAN, 2.0, 3.0, f32::NAN];
        let mapper = FeatureBinMapper::fit(&vals, 4);
        let nan_bin = mapper.map_value(f32::NAN);
        assert_eq!(nan_bin, mapper.missing_bin);
    }

    #[test]
    fn test_bin_mapper_categorical() {
        let vals = vec![0.0, 1.0, 2.0, 0.0, 1.0];
        let mapper = FeatureBinMapper::fit_with_type(&vals, 10, FeatureType::Categorical);
        assert_eq!(mapper.feature_type, FeatureType::Categorical);
        assert_eq!(mapper.map_value(0.0), 0);
    }
}
