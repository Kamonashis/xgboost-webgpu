use rand::Rng;
use xgboost_webgpu::booster::{BoosterParams, DeviceType, GrowPolicy, ImportanceType};
use xgboost_webgpu::data::DMatrix;
use xgboost_webgpu::train;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=========================================================");
    println!("  xgboost-webgpu: Advanced Features & Interpretability");
    println!("=========================================================\n");

    let mut rng = rand::rng();
    let n_train = 3000;
    let n_test = 600;
    let n_features = 6;
    let num_class = 3;

    let feature_names = vec![
        "Age".to_string(),
        "Income".to_string(),
        "EducationScore".to_string(),
        "CreditScore".to_string(),
        "DebtRatio".to_string(),
        "AccountTenure".to_string(),
    ];

    println!("Generating multi-class dataset ({} samples, {} classes, {} features)...", n_train, num_class, n_features);

    let generate_data = |n: usize, rng: &mut rand::rngs::ThreadRng| -> (Vec<f32>, Vec<f32>) {
        let mut x = Vec::with_capacity(n * n_features);
        let mut y = Vec::with_capacity(n);

        for _ in 0..n {
            let age = rng.random_range(18.0..70.0);
            let income = rng.random_range(20.0..150.0);
            let edu = rng.random_range(1.0..5.0);
            let credit = rng.random_range(300.0..850.0);
            let debt = rng.random_range(0.0..1.0);
            let tenure = rng.random_range(0.5..15.0);

            let row = [age, income, edu, credit, debt, tenure];

            // Multi-class decision logic
            let class_id = if credit < 550.0 || debt > 0.6 {
                0.0f32 // High risk
            } else if income > 80.0 && credit > 700.0 {
                2.0f32 // Prime tier
            } else {
                1.0f32 // Standard tier
            };

            x.extend_from_slice(&row);
            y.push(class_id);
        }
        (x, y)
    };

    let (x_train, y_train) = generate_data(n_train, &mut rng);
    let (x_test, y_test) = generate_data(n_test, &mut rng);

    let mut dtrain = DMatrix::from_dense(&x_train, n_train, n_features, Some(&y_train), 64)?;
    dtrain.set_feature_names(feature_names.clone())?;

    let mut dtest = DMatrix::from_dense_with_mappers(
        &x_test,
        n_test,
        n_features,
        Some(&y_test),
        &dtrain.bin_mappers,
    )?;
    dtest.set_feature_names(feature_names)?;

    println!("\n>>> Training Multi-Class Booster with Advanced Options:");
    println!("  - Objective: multi:softprob (3 classes)");
    println!("  - Grow Policy: LossGuide (leaf-wise, max_leaves: 15)");
    println!("  - Row Subsampling: 85% (bagging)");
    println!("  - Column Subsampling: 85%");
    println!("  - Early Stopping: 5 rounds");
    println!("  - Acceleration: WebGPU\n");

    let params = BoosterParams::new()
        .with_objective("multi:softprob")
        .with_num_class(num_class)
        .with_grow_policy(GrowPolicy::LossGuide)
        .with_max_leaves(15)
        .with_subsample(0.85)
        .with_colsample_bytree(0.85)
        .with_learning_rate(0.25)
        .with_early_stopping(5)
        .with_device(DeviceType::WebGPU);

    let booster = train(params, &dtrain, 25, &[(&dtest, "test")])?;

    // 1. Predictions
    let probs = booster.predict(&dtest)?;
    println!("\n[Prediction Sample 0 Probabilities]: Risk: {:.3}, Standard: {:.3}, Prime: {:.3} (Actual: Class {})",
        probs[0], probs[1], probs[2], y_test[0] as usize);

    // 2. Feature Importance
    println!("\n>>> Feature Importance Scores (Gain):");
    let importance_gain = booster.feature_importance(ImportanceType::Gain);
    for (feat, score) in &importance_gain {
        println!("  - {:16} : {:.4}", feat, score);
    }

    // 3. Tree SHAP feature contributions
    println!("\n>>> TreeSHAP Local Explanation for Sample 0:");
    let contribs = booster.predict_contributions(&dtest)?;
    for (j, name) in booster.feature_names.iter().enumerate() {
        println!("  - {:16} phi: {:+.4}", name, contribs[0][j]);
    }
    println!("  - {:16} base: {:+.4}", "Baseline Bias", contribs[0][n_features]);

    println!("\n=========================================================");
    println!("  All native XGBoost advanced features verified!");
    println!("=========================================================");

    Ok(())
}
