use crate::data::binning::FeatureBinMapper;
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
    /// Per-feature bin mappers.
    pub bin_mappers: Vec<FeatureBinMapper>,
    /// Quantized bin values in row-major format (length = nrows * ncols, values in 0..255).
    pub binned_data: Vec<u8>,
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

        let feature_names: Vec<String> = (0..ncols).map(|j| format!("f{j}")).collect();

        // Fit bin mapper for each column in parallel
        let bin_mappers: Vec<FeatureBinMapper> = (0..ncols)
            .into_par_iter()
            .map(|j| {
                let col_vals: Vec<f32> = (0..nrows).map(|i| data[i * ncols + j]).collect();
                FeatureBinMapper::fit(&col_vals, max_bins)
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
            bin_mappers,
            binned_data,
        })
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
            bin_mappers: bin_mappers.to_vec(),
            binned_data,
        })
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
    fn test_dmatrix_construction() {
        let raw_data = vec![
            1.0, 10.0,
            2.0, 20.0,
            3.0, 30.0,
            4.0, 40.0,
        ];
        let labels = vec![0.0, 1.0, 0.0, 1.0];
        let dmat = DMatrix::from_dense(&raw_data, 4, 2, Some(&labels), 4).unwrap();

        assert_eq!(dmat.nrows, 4);
        assert_eq!(dmat.ncols, 2);
        assert_eq!(dmat.binned_data.len(), 8);
        assert_eq!(dmat.labels.as_ref().unwrap().len(), 4);
    }
}
