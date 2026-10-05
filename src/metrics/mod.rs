/// Evaluation metrics for model monitoring.

pub fn rmse(y_true: &[f32], y_pred: &[f32]) -> f32 {
    assert_eq!(y_true.len(), y_pred.len());
    if y_true.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = y_true
        .iter()
        .zip(y_pred)
        .map(|(&t, &p)| {
            let diff = t - p;
            diff * diff
        })
        .sum();
    (sum_sq / y_true.len() as f32).sqrt()
}

pub fn mae(y_true: &[f32], y_pred: &[f32]) -> f32 {
    assert_eq!(y_true.len(), y_pred.len());
    if y_true.is_empty() {
        return 0.0;
    }
    let sum_abs: f32 = y_true
        .iter()
        .zip(y_pred)
        .map(|(&t, &p)| (t - p).abs())
        .sum();
    sum_abs / y_true.len() as f32
}

pub fn logloss(y_true: &[f32], y_pred: &[f32]) -> f32 {
    assert_eq!(y_true.len(), y_pred.len());
    if y_true.is_empty() {
        return 0.0;
    }
    let eps = 1e-15f32;
    let sum_ll: f32 = y_true
        .iter()
        .zip(y_pred)
        .map(|(&t, &p)| {
            let p_clamped = p.clamp(eps, 1.0 - eps);
            -(t * p_clamped.ln() + (1.0 - t) * (1.0 - p_clamped).ln())
        })
        .sum();
    sum_ll / y_true.len() as f32
}

pub fn binary_error(y_true: &[f32], y_pred: &[f32], threshold: f32) -> f32 {
    assert_eq!(y_true.len(), y_pred.len());
    if y_true.is_empty() {
        return 0.0;
    }
    let incorrect: usize = y_true
        .iter()
        .zip(y_pred)
        .filter(|(&t, &p)| {
            let pred_label = if p >= threshold { 1.0 } else { 0.0 };
            (pred_label - t).abs() > 0.5
        })
        .count();
    incorrect as f32 / y_true.len() as f32
}

pub fn evaluate(metric_name: &str, y_true: &[f32], y_pred: &[f32]) -> Result<f32, String> {
    match metric_name {
        "rmse" => Ok(rmse(y_true, y_pred)),
        "mae" => Ok(mae(y_true, y_pred)),
        "logloss" => Ok(logloss(y_true, y_pred)),
        "error" => Ok(binary_error(y_true, y_pred, 0.5)),
        _ => Err(format!("Unknown evaluation metric: {}", metric_name)),
    }
}
