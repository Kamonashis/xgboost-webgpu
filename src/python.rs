#[cfg(feature = "python")]
use pyo3::exceptions::{PyIOError, PyValueError};
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::{PyAny, PyDict, PyList};

#[cfg(feature = "python")]
use crate::booster::params::{BoosterParams, BoosterType, DeviceType, GrowPolicy};
#[cfg(feature = "python")]
use crate::booster::{Booster, ImportanceType};
#[cfg(feature = "python")]
use crate::data::DMatrix;

/// Python DMatrix wrapper.
#[cfg(feature = "python")]
#[pyclass(name = "DMatrix")]
pub struct PyDMatrix {
    pub inner: DMatrix,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyDMatrix {
    #[new]
    #[pyo3(signature = (data, labels=None, weights=None, feature_names=None, max_bins=None))]
    pub fn new(
        data: &Bound<'_, PyAny>,
        labels: Option<Vec<f32>>,
        weights: Option<Vec<f32>>,
        feature_names: Option<Vec<String>>,
        max_bins: Option<usize>,
    ) -> PyResult<Self> {
        let bins = max_bins.unwrap_or(256);

        let mut flat_data = Vec::new();
        let nrows: usize;
        let ncols: usize;

        if let Ok(py_list) = data.downcast::<PyList>() {
            nrows = py_list.len();
            if nrows > 0 {
                let first_elem = py_list.get_item(0)?;
                if let Ok(first_row) = first_elem.downcast::<PyList>() {
                    ncols = first_row.len();
                    flat_data.reserve(nrows * ncols);
                    for item in py_list.iter() {
                        let row = item.downcast::<PyList>()?;
                        if row.len() != ncols {
                            return Err(PyValueError::new_err("Inconsistent row lengths in 2D list"));
                        }
                        for val in row.iter() {
                            let f: f32 = val.extract()?;
                            flat_data.push(f);
                        }
                    }
                } else {
                    return Err(PyValueError::new_err("Expected 2D list for DMatrix data"));
                }
            } else {
                ncols = 0;
            }
        } else {
            // Attempt numpy extraction: shape attribute
            let shape: (usize, usize) = data
                .getattr("shape")
                .map_err(|_| PyValueError::new_err("Data must be a 2D numpy array or list of lists"))?
                .extract()?;
            nrows = shape.0;
            ncols = shape.1;

            // Flatten numpy array
            let flat = data.call_method0("flatten")?;
            let py_list = flat.call_method0("tolist")?;
            let list = py_list.downcast::<PyList>()?;
            flat_data.reserve(nrows * ncols);
            for val in list.iter() {
                let f: f32 = val.extract()?;
                flat_data.push(f);
            }
        }

        let mut dmatrix = DMatrix::from_dense(
            &flat_data,
            nrows,
            ncols,
            labels.as_deref(),
            bins,
        )
        .map_err(|e| PyValueError::new_err(e))?;

        if let Some(w) = weights {
            dmatrix.set_weights(w).map_err(|e| PyValueError::new_err(e))?;
        }
        if let Some(names) = feature_names {
            dmatrix.set_feature_names(names).map_err(|e| PyValueError::new_err(e))?;
        }

        Ok(PyDMatrix { inner: dmatrix })
    }

    pub fn num_rows(&self) -> usize {
        self.inner.nrows
    }

    pub fn num_cols(&self) -> usize {
        self.inner.ncols
    }

    pub fn feature_names(&self) -> Vec<String> {
        self.inner.feature_names.clone()
    }
}

/// Python Booster wrapper.
#[cfg(feature = "python")]
#[pyclass(name = "Booster")]
pub struct PyBooster {
    pub inner: Booster,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyBooster {
    #[new]
    #[pyo3(signature = (model_file=None))]
    pub fn new(model_file: Option<&str>) -> PyResult<Self> {
        let booster = match model_file {
            Some(path) => Booster::load_model(path).map_err(|e| PyIOError::new_err(e))?,
            None => Booster::empty(),
        };
        Ok(PyBooster { inner: booster })
    }

    pub fn predict(&self, dmatrix: &PyDMatrix) -> PyResult<Vec<f32>> {
        self.inner
            .predict(&dmatrix.inner)
            .map_err(|e| PyValueError::new_err(e))
    }

    pub fn predict_raw(&self, dmatrix: &PyDMatrix) -> PyResult<Vec<f32>> {
        self.inner
            .predict_raw(&dmatrix.inner)
            .map_err(|e| PyValueError::new_err(e))
    }

    pub fn predict_leaf(&self, dmatrix: &PyDMatrix) -> PyResult<Vec<Vec<usize>>> {
        self.inner
            .predict_leaf(&dmatrix.inner)
            .map_err(|e| PyValueError::new_err(e))
    }

    pub fn predict_contributions(&self, dmatrix: &PyDMatrix) -> PyResult<Vec<Vec<f32>>> {
        self.inner
            .predict_contributions(&dmatrix.inner)
            .map_err(|e| PyValueError::new_err(e))
    }

    #[pyo3(signature = (importance_type="gain"))]
    pub fn feature_importance(&self, importance_type: &str) -> PyResult<Vec<(String, f32)>> {
        let imp_type = match importance_type.to_lowercase().as_str() {
            "weight" | "split" => ImportanceType::Weight,
            "gain" => ImportanceType::Gain,
            "cover" => ImportanceType::Cover,
            "total_gain" => ImportanceType::TotalGain,
            "total_cover" => ImportanceType::TotalCover,
            _ => return Err(PyValueError::new_err("Unknown importance_type. Expected: gain, weight, cover, total_gain, total_cover")),
        };
        Ok(self.inner.feature_importance(imp_type))
    }

    pub fn save_model(&self, path: &str) -> PyResult<()> {
        self.inner
            .save_model(path)
            .map_err(|e| PyIOError::new_err(e))
    }

    pub fn load_model(&mut self, path: &str) -> PyResult<()> {
        self.inner = Booster::load_model(path).map_err(|e| PyIOError::new_err(e))?;
        Ok(())
    }

    pub fn to_json(&self) -> PyResult<String> {
        self.inner.to_xgboost_json().map_err(|e| PyValueError::new_err(e))
    }

    pub fn num_trees(&self) -> usize {
        self.inner.num_trees()
    }
}

/// Trains an XGBoost model using WebGPU acceleration from Python.
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (params, dtrain, num_boost_round=10, evals=None))]
pub fn train(
    params: &Bound<'_, PyDict>,
    dtrain: &PyDMatrix,
    num_boost_round: Option<usize>,
    evals: Option<&Bound<'_, PyList>>,
) -> PyResult<PyBooster> {
    let mut booster_params = BoosterParams::new();

    if let Some(val) = params.get_item("max_depth")? {
        if !val.is_none() {
            booster_params.max_depth = val.extract()?;
        }
    }
    if let Some(val) = params.get_item("learning_rate")? {
        if !val.is_none() {
            booster_params.learning_rate = val.extract()?;
        }
    } else if let Some(val) = params.get_item("eta")? {
        if !val.is_none() {
            booster_params.learning_rate = val.extract()?;
        }
    }
    if let Some(val) = params.get_item("objective")? {
        if !val.is_none() {
            let obj_str: String = val.extract()?;
            booster_params.objective = obj_str;
        }
    }
    if let Some(val) = params.get_item("reg_lambda")? {
        if !val.is_none() {
            booster_params.reg_lambda = val.extract()?;
        }
    } else if let Some(val) = params.get_item("lambda")? {
        if !val.is_none() {
            booster_params.reg_lambda = val.extract()?;
        }
    }
    if let Some(val) = params.get_item("reg_alpha")? {
        if !val.is_none() {
            booster_params.reg_alpha = val.extract()?;
        }
    } else if let Some(val) = params.get_item("alpha")? {
        if !val.is_none() {
            booster_params.reg_alpha = val.extract()?;
        }
    }
    if let Some(val) = params.get_item("gamma")? {
        if !val.is_none() {
            booster_params.gamma = val.extract()?;
        }
    }
    if let Some(val) = params.get_item("min_child_weight")? {
        if !val.is_none() {
            booster_params.min_child_weight = val.extract()?;
        }
    }
    if let Some(val) = params.get_item("subsample")? {
        if !val.is_none() {
            booster_params.subsample = val.extract()?;
        }
    }
    if let Some(val) = params.get_item("colsample_bytree")? {
        if !val.is_none() {
            booster_params.colsample_bytree = val.extract()?;
        }
    }
    if let Some(val) = params.get_item("colsample_bynode")? {
        if !val.is_none() {
            booster_params.colsample_bynode = val.extract()?;
        }
    }
    if let Some(val) = params.get_item("scale_pos_weight")? {
        if !val.is_none() {
            booster_params.scale_pos_weight = val.extract()?;
        }
    }
    if let Some(val) = params.get_item("max_delta_step")? {
        if !val.is_none() {
            booster_params.max_delta_step = val.extract()?;
        }
    }
    if let Some(val) = params.get_item("num_class")? {
        if !val.is_none() {
            booster_params.num_class = val.extract()?;
        }
    }
    if let Some(val) = params.get_item("max_leaves")? {
        if !val.is_none() {
            booster_params.max_leaves = val.extract()?;
        }
    }
    if let Some(val) = params.get_item("early_stopping_rounds")? {
        if !val.is_none() {
            booster_params.early_stopping_rounds = Some(val.extract()?);
        }
    }
    if let Some(val) = params.get_item("eval_metric")? {
        if !val.is_none() {
            booster_params.eval_metric = Some(val.extract()?);
        }
    }
    if let Some(val) = params.get_item("grow_policy")? {
        if !val.is_none() {
            let p_str: String = val.extract()?;
            if p_str == "lossguide" {
                booster_params.grow_policy = GrowPolicy::LossGuide;
            }
        }
    }
    if let Some(val) = params.get_item("booster")? {
        if !val.is_none() {
            let b_str: String = val.extract()?;
            if b_str == "dart" {
                booster_params.booster_type = BoosterType::DART;
            }
        }
    }
    if let Some(val) = params.get_item("rate_drop")? {
        if !val.is_none() {
            booster_params.rate_drop = val.extract()?;
        }
    }
    if let Some(val) = params.get_item("skip_drop")? {
        if !val.is_none() {
            booster_params.skip_drop = val.extract()?;
        }
    }
    if let Some(val) = params.get_item("monotone_constraints")? {
        if !val.is_none() {
            booster_params.monotone_constraints = Some(val.extract()?);
        }
    }
    if let Some(val) = params.get_item("device")? {
        if !val.is_none() {
            let d_str: String = val.extract()?;
            if d_str == "cpu" {
                booster_params.device = DeviceType::Cpu;
            } else {
                booster_params.device = DeviceType::WebGPU;
            }
        }
    } else if let Some(val) = params.get_item("tree_method")? {
        if !val.is_none() {
            let t_str: String = val.extract()?;
            if t_str == "cpu" {
                booster_params.device = DeviceType::Cpu;
            } else {
                booster_params.device = DeviceType::WebGPU;
            }
        }
    }

    let rounds = num_boost_round.unwrap_or(10);

    let mut eval_data = Vec::new();
    if let Some(ev_list) = evals {
        for item in ev_list.iter() {
            let tuple = item.extract::<(PyRef<PyDMatrix>, String)>()?;
            eval_data.push(tuple);
        }
    }

    let eval_refs: Vec<(&DMatrix, &str)> = eval_data
        .iter()
        .map(|(dmat, name)| (&dmat.inner, name.as_str()))
        .collect();

    let booster = crate::booster::Booster::train(
        booster_params,
        &dtrain.inner,
        rounds,
        &eval_refs,
    )
    .map_err(|e| PyValueError::new_err(e))?;

    Ok(PyBooster { inner: booster })
}

/// PyO3 Python C-extension module entry point.
#[cfg(feature = "python")]
#[pymodule]
pub fn _xgboost_webgpu(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDMatrix>()?;
    m.add_class::<PyBooster>()?;
    m.add_function(wrap_pyfunction!(train, m)?)?;
    Ok(())
}
