use crate::data::binning::{FeatureBinMapper, FeatureType};
use rayon::prelude::*;

/// Primary dataset matrix structure for XGBoost training and inference.
#[derive(Debug, Clone)]
pub struct DMatrix {
    pub nrows: usize,
    pub ncols: usize,
    /// Raw feature values in row-major layout (length = nrows * ncols).
    pub data: Vec<f32>,
    /// Target labels (length = nrows).
    pub labels: Option<Vec<f32>>,
    /// Sample weights (length = nrows).
    pub weights: Option<Vec<f32>>,
    /// Feature names (length = ncols).
    pub feature_names: Vec<String>,
    /// Feature types (Numerical or Categorical).
    pub feature_types: Vec<FeatureType>,
    /// Per-feature bin mappers.
    pub bin_mappers: Vec<FeatureBinMapper>,
    /// Quantized bin values in row-major format (length = nrows * ncols, values in 0..255).
    pub binned_data: Vec<u8>,
    /// Query group counts for ranking (e.g. sum(group) == nrows).
    pub group: Option<Vec<usize>>,
}

impl DMatrix {
    /// Constructs a `DMatrix` from dense row-major slice of `f32` features, fitting new bin mappers.
    pub fn from_dense(
        data: &[f32],
        nrows: usize,
        ncols: usize,
        labels: Option<&[f32]>,
        max_bins: usize,
    ) -> Result<Self, String> {
        Self::from_dense_with_types(data, nrows, ncols, labels, max_bins, None)
    }

    /// Constructs a `DMatrix` from dense data with optional explicit feature types.
    pub fn from_dense_with_types(
        data: &[f32],
        nrows: usize,
        ncols: usize,
        labels: Option<&[f32]>,
        max_bins: usize,
        feature_types: Option<&[FeatureType]>,
    ) -> Result<Self, String> {
        if data.len() != nrows * ncols {
            return Err(format!(
                "Data slice length ({}) does not match nrows * ncols ({} * {} = {})",
                data.len(),
                nrows,
                ncols,
                nrows * ncols
            ));
        }

        if let Some(lbls) = labels {
            if lbls.len() != nrows {
                return Err(format!(
                    "Labels length ({}) does not match nrows ({})",
                    lbls.len(),
                    nrows
                ));
            }
        }

        let ftypes = match feature_types {
            Some(t) => {
                if t.len() != ncols {
                    return Err(format!("feature_types length must match ncols ({})", ncols));
                }
                t.to_vec()
            }
            None => vec![FeatureType::Numerical; ncols],
        };

        let feature_names: Vec<String> = (0..ncols).map(|j| format!("f{j}")).collect();

        // Fit bin mapper for each column in parallel
        let bin_mappers: Vec<FeatureBinMapper> = (0..ncols)
            .into_par_iter()
            .map(|j| {
                let col_vals: Vec<f32> = (0..nrows).map(|i| data[i * ncols + j]).collect();
                FeatureBinMapper::fit_with_type(&col_vals, max_bins, ftypes[j])
            })
            .collect();

        // Quantize all elements into row-major u8 buffer in parallel
        let mut binned_data = vec![0u8; nrows * ncols];
        binned_data
            .par_chunks_exact_mut(ncols)
            .enumerate()
            .for_each(|(i, row_bins)| {
                let row_start = i * ncols;
                for j in 0..ncols {
                    let val = data[row_start + j];
                    row_bins[j] = bin_mappers[j].map_value(val);
                }
            });

        Ok(Self {
            nrows,
            ncols,
            data: data.to_vec(),
            labels: labels.map(|l| l.to_vec()),
            weights: None,
            feature_names,
            feature_types: ftypes,
            bin_mappers,
            binned_data,
            group: None,
        })
    }

    /// Constructs a `DMatrix` from Compressed Sparse Row (CSR) format.
    pub fn from_csr(
        indptr: &[usize],
        indices: &[usize],
        values: &[f32],
        nrows: usize,
        ncols: usize,
        labels: Option<&[f32]>,
        max_bins: usize,
    ) -> Result<Self, String> {
        if indptr.len() != nrows + 1 {
            return Err(format!(
                "CSR indptr length ({}) must be nrows + 1 ({})",
                indptr.len(),
                nrows + 1
            ));
        }

        let mut dense_data = vec![0.0f32; nrows * ncols];

        for i in 0..nrows {
            let start = indptr[i];
            let end = indptr[i + 1];
            for k in start..end {
                let col = indices[k];
                if col >= ncols {
                    return Err(format!("CSR column index {} out of bounds ({})", col, ncols));
                }
                dense_data[i * ncols + col] = values[k];
            }
        }

        Self::from_dense(&dense_data, nrows, ncols, labels, max_bins)
    }

    /// Constructs a `DMatrix` using existing bin mappers (e.g. for validation/test data).
    pub fn from_dense_with_mappers(
        data: &[f32],
        nrows: usize,
        ncols: usize,
        labels: Option<&[f32]>,
        bin_mappers: &[FeatureBinMapper],
    ) -> Result<Self, String> {
        if data.len() != nrows * ncols {
            return Err(format!(
                "Data length ({}) does not match nrows * ncols ({})",
                data.len(),
                nrows * ncols
            ));
        }
        if bin_mappers.len() != ncols {
            return Err(format!(
                "Bin mappers count ({}) does not match ncols ({})",
                bin_mappers.len(),
                ncols
            ));
        }

        let feature_names: Vec<String> = (0..ncols).map(|j| format!("f{j}")).collect();
        let feature_types: Vec<FeatureType> = bin_mappers.iter().map(|m| m.feature_type).collect();

        let mut binned_data = vec![0u8; nrows * ncols];
        binned_data
            .par_chunks_exact_mut(ncols)
            .enumerate()
            .for_each(|(i, row_bins)| {
                let row_start = i * ncols;
                for j in 0..ncols {
                    let val = data[row_start + j];
                    row_bins[j] = bin_mappers[j].map_value(val);
                }
            });

        Ok(Self {
            nrows,
            ncols,
            data: data.to_vec(),
            labels: labels.map(|l| l.to_vec()),
            weights: None,
            feature_names,
            feature_types,
            bin_mappers: bin_mappers.to_vec(),
            binned_data,
            group: None,
        })
    }

    /// Set query groups for ranking.
    pub fn set_group(&mut self, group: Vec<usize>) -> Result<(), String> {
        let total: usize = group.iter().sum();
        if total != self.nrows {
            return Err(format!(
                "Sum of group elements ({}) must equal nrows ({})",
                total, self.nrows
            ));
        }
        self.group = Some(group);
        Ok(())
    }

    /// Set custom feature names.
    pub fn set_feature_names(&mut self, names: Vec<String>) -> Result<(), String> {
        if names.len() != self.ncols {
            return Err(format!(
                "Feature names length ({}) does not match ncols ({})",
                names.len(),
                self.ncols
            ));
        }
        self.feature_names = names;
        Ok(())
    }

    /// Set sample weights.
    pub fn set_weights(&mut self, weights: Vec<f32>) -> Result<(), String> {
        if weights.len() != self.nrows {
            return Err(format!(
                "Weights length ({}) does not match nrows ({})",
                weights.len(),
                self.nrows
            ));
        }
        self.weights = Some(weights);
        Ok(())
    }

    /// Access bin for sample `i` and feature `j`.
    #[inline(always)]
    pub fn get_bin(&self, row: usize, col: usize) -> u8 {
        self.binned_data[row * self.ncols + col]
    }

    /// Access continuous value for sample `i` and feature `j`.
    #[inline(always)]
    pub fn get_value(&self, row: usize, col: usize) -> f32 {
        self.data[row * self.ncols + col]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dmatrix_csr() {
        // 2x3 matrix:
        // [1.0, 0.0, 2.0]
        // [0.0, 3.0, 0.0]
        let indptr = vec![0, 2, 3];
        let indices = vec![0, 2, 1];
        let values = vec![1.0, 2.0, 3.0];
        let labels = vec![0.5, 1.5];

        let dmat = DMatrix::from_csr(&indptr, &indices, &values, 2, 3, Some(&labels), 10).unwrap();
        assert_eq!(dmat.nrows, 2);
        assert_eq!(dmat.ncols, 3);
        assert_eq!(dmat.get_value(0, 0), 1.0);
        assert_eq!(dmat.get_value(0, 1), 0.0);
        assert_eq!(dmat.get_value(0, 2), 2.0);
        assert_eq!(dmat.get_value(1, 1), 3.0);
    }
}
