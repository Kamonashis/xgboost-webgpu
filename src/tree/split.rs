use crate::data::binning::FeatureBinMapper;
use crate::gpu::histogram::FeatureHistogram;

/// Candidate split found during histogram scanning.
#[derive(Debug, Clone)]
pub struct SplitCandidate {
    pub feature: usize,
    pub bin: u8,
    pub split_value: f32,
    pub default_left: bool,
    pub gain: f32,
    pub left_g: f32,
    pub left_h: f32,
    pub right_g: f32,
    pub right_h: f32,
}

/// Evaluates split candidates across all feature histograms and returns the best split (if any).
pub fn find_best_split(
    histograms: &[FeatureHistogram],
    bin_mappers: &[FeatureBinMapper],
    total_g: f32,
    total_h: f32,
    reg_lambda: f32,
    reg_alpha: f32,
    gamma: f32,
    min_child_weight: f32,
    allowed_features: Option<&[usize]>,
    monotone_constraints: Option<&[i8]>,
) -> Option<SplitCandidate> {
    if total_h < 2.0 * min_child_weight {
        return None;
    }

    let parent_score = calc_split_score(total_g, total_h, reg_lambda, reg_alpha);
    let mut best_split: Option<SplitCandidate> = None;
    let mut max_gain = 0.0f32;

    let num_features = histograms.len();

    for j in 0..num_features {
        // Feature subsampling / interaction filtering
        if let Some(allowed) = allowed_features {
            if !allowed.contains(&j) {
                continue;
            }
        }

        let hist = &histograms[j];
        let mapper = &bin_mappers[j];
        let num_bins = mapper.num_bins().min(256);
        let missing_bin = mapper.missing_bin as usize;

        let mut gl = 0.0f32;
        let mut hl = 0.0f32;

        let missing_g = if missing_bin < 256 { hist.g[missing_bin] } else { 0.0 };
        let missing_h = if missing_bin < 256 { hist.h[missing_bin] } else { 0.0 };

        let mono = monotone_constraints.and_then(|m| m.get(j)).copied().unwrap_or(0);

        for b in 0..num_bins.saturating_sub(1) {
            if b == missing_bin {
                continue;
            }

            gl += hist.g[b];
            hl += hist.h[b];

            let gr = total_g - gl;
            let hr = total_h - hl;

            // Scenario 1: missing values go left
            let (gl_with_miss, hl_with_miss, gr_without, hr_without) =
                (gl + missing_g, hl + missing_h, gr - missing_g, hr - missing_h);
            if hl_with_miss >= min_child_weight && hr_without >= min_child_weight {
                // Check monotonic constraint
                let valid_mono = check_monotonicity(
                    gl_with_miss,
                    hl_with_miss,
                    gr_without,
                    hr_without,
                    reg_lambda,
                    mono,
                );

                if valid_mono {
                    let left_score = calc_split_score(gl_with_miss, hl_with_miss, reg_lambda, reg_alpha);
                    let right_score = calc_split_score(gr_without, hr_without, reg_lambda, reg_alpha);
                    let gain = 0.5 * (left_score + right_score - parent_score) - gamma;

                    if gain > max_gain {
                        let cut = if b < mapper.cut_points.len() {
                            mapper.cut_points[b]
                        } else {
                            mapper.max_val
                        };

                        max_gain = gain;
                        best_split = Some(SplitCandidate {
                            feature: j,
                            bin: b as u8,
                            split_value: cut,
                            default_left: true,
                            gain,
                            left_g: gl_with_miss,
                            left_h: hl_with_miss,
                            right_g: gr_without,
                            right_h: hr_without,
                        });
                    }
                }
            }

            // Scenario 2: missing values go right
            if hl >= min_child_weight && hr >= min_child_weight {
                let valid_mono = check_monotonicity(gl, hl, gr, hr, reg_lambda, mono);

                if valid_mono {
                    let left_score = calc_split_score(gl, hl, reg_lambda, reg_alpha);
                    let right_score = calc_split_score(gr, hr, reg_lambda, reg_alpha);
                    let gain = 0.5 * (left_score + right_score - parent_score) - gamma;

                    if gain > max_gain {
                        let cut = if b < mapper.cut_points.len() {
                            mapper.cut_points[b]
                        } else {
                            mapper.max_val
                        };

                        max_gain = gain;
                        best_split = Some(SplitCandidate {
                            feature: j,
                            bin: b as u8,
                            split_value: cut,
                            default_left: false,
                            gain,
                            left_g: gl,
                            left_h: hl,
                            right_g: gr,
                            right_h: hr,
                        });
                    }
                }
            }
        }
    }

    best_split
}

#[inline(always)]
fn check_monotonicity(gl: f32, hl: f32, gr: f32, hr: f32, reg_lambda: f32, mono: i8) -> bool {
    if mono == 0 {
        return true;
    }
    let wl = -gl / (hl + reg_lambda);
    let wr = -gr / (hr + reg_lambda);
    if mono == 1 {
        wl <= wr
    } else if mono == -1 {
        wl >= wr
    } else {
        true
    }
}

/// Calculate optimal score: G^2 / (H + lambda), with optional L1 regularization.
#[inline(always)]
pub fn calc_split_score(g: f32, h: f32, reg_lambda: f32, reg_alpha: f32) -> f32 {
    let denom = h + reg_lambda;
    if denom <= 0.0 {
        return 0.0;
    }
    if reg_alpha == 0.0 {
        (g * g) / denom
    } else {
        let abs_g = g.abs();
        if abs_g <= reg_alpha {
            0.0
        } else {
            let shrink = abs_g - reg_alpha;
            (shrink * shrink) / denom
        }
    }
}

/// Calculate optimal leaf weight: -G / (H + lambda), with optional L1 thresholding and max_delta_step.
#[inline(always)]
pub fn calc_leaf_weight(
    g: f32,
    h: f32,
    reg_lambda: f32,
    reg_alpha: f32,
    learning_rate: f32,
    max_delta_step: f32,
) -> f32 {
    let denom = h + reg_lambda;
    if denom <= 0.0 {
        return 0.0;
    }
    let mut weight = if reg_alpha == 0.0 {
        -g / denom
    } else if g > reg_alpha {
        -(g - reg_alpha) / denom
    } else if g < -reg_alpha {
        -(g + reg_alpha) / denom
    } else {
        0.0
    };

    if max_delta_step > 0.0 {
        weight = weight.clamp(-max_delta_step, max_delta_step);
    }

    weight * learning_rate
}
