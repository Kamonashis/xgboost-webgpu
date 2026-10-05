use rand::Rng;
use xgboost_webgpu::booster::{BoosterParams, BoosterType, GrowPolicy, ImportanceType};
use xgboost_webgpu::data::DMatrix;
use xgboost_webgpu::train;

#[test]
fn test_subsampling_and_dart() {
    let mut rng = rand::rng();
    let n = 300;
    let m = 5;
    let x: Vec<f32> = (0..n * m).map(|_| rng.random_range(-2.0..2.0)).collect();
    let y: Vec<f32> = (0..n).map(|i| x[i * m] * 2.0 + rng.random_range(-0.1..0.1)).collect();

    let dtrain = DMatrix::from_dense(&x, n, m, Some(&y), 32).unwrap();

    // DART booster with row bagging and column subsampling
    let params = BoosterParams::new()
        .with_booster_type(BoosterType::DART)
        .with_subsample(0.8)
        .with_colsample_bytree(0.8)
        .with_learning_rate(0.2)
        .with_max_depth(4);

    let booster = train(params, &dtrain, 10, &[]).unwrap();
    let preds = booster.predict(&dtrain).unwrap();
    assert_eq!(preds.len(), n);
}

#[test]
fn test_lossguide_leafwise_growth() {
    let mut rng = rand::rng();
    let n = 400;
    let m = 6;
    let x: Vec<f32> = (0..n * m).map(|_| rng.random_range(-2.0..2.0)).collect();
    let y: Vec<f32> = (0..n).map(|i| x[i * m].sin() * 3.0 + x[i * m + 1]).collect();

    let dtrain = DMatrix::from_dense(&x, n, m, Some(&y), 32).unwrap();

    // Leaf-wise growth with max_leaves = 10
    let params = BoosterParams::new()
        .with_grow_policy(GrowPolicy::LossGuide)
        .with_max_leaves(10)
        .with_max_depth(8);

    let booster = train(params, &dtrain, 8, &[]).unwrap();
    for tree in &booster.trees {
        assert!(tree.num_leaves() <= 10);
    }
}

#[test]
fn test_feature_importance_and_shap() {
    let mut rng = rand::rng();
    let n = 200;
    let m = 4;
    let mut x = Vec::with_capacity(n * m);
    let mut y = Vec::with_capacity(n);

    // Feature 0 is dominant
    for _ in 0..n {
        let f0 = rng.random_range(0.0..10.0);
        let f1 = rng.random_range(0.0..1.0);
        let f2 = rng.random_range(0.0..1.0);
        let f3 = rng.random_range(0.0..1.0);
        x.extend_from_slice(&[f0, f1, f2, f3]);
        y.push(f0 * 5.0 + rng.random_range(-0.05..0.05));
    }

    let mut dtrain = DMatrix::from_dense(&x, n, m, Some(&y), 32).unwrap();
    dtrain.set_feature_names(vec!["dominant".into(), "noise1".into(), "noise2".into(), "noise3".into()]).unwrap();

    let params = BoosterParams::new()
        .with_max_depth(3)
        .with_learning_rate(0.3);

    let booster = train(params, &dtrain, 6, &[]).unwrap();

    // 1. Feature Importance
    let imp_gain = booster.feature_importance(ImportanceType::Gain);
    assert_eq!(imp_gain[0].0, "dominant");
    println!("Feature Importance (Gain): {:?}", imp_gain);

    let imp_weight = booster.feature_importance(ImportanceType::Weight);
    println!("Feature Importance (Weight): {:?}", imp_weight);

    // 2. Leaf index prediction
    let leaves = booster.predict_leaf(&dtrain).unwrap();
    assert_eq!(leaves.len(), n);
    assert_eq!(leaves[0].len(), booster.num_trees());

    // 3. Tree SHAP exact feature contributions
    let contribs = booster.predict_contributions(&dtrain).unwrap();
    assert_eq!(contribs.len(), n);
    assert_eq!(contribs[0].len(), m + 1); // 4 features + 1 bias

    let raw_preds = booster.predict_raw(&dtrain).unwrap();
    for i in 0..n {
        let sum_contrib: f32 = contribs[i].iter().sum();
        let diff = (sum_contrib - raw_preds[i]).abs();
        assert!(
            diff < 1e-4,
            "TreeSHAP sum ({}) does not match raw margin prediction ({}) for sample {}",
            sum_contrib, raw_preds[i], i
        );
    }
    println!("TreeSHAP local efficiency verified! sum(phi) == prediction for all samples.");
}

#[test]
fn test_early_stopping() {
    let mut rng = rand::rng();
    let n = 200;
    let m = 3;
    let x: Vec<f32> = (0..n * m).map(|_| rng.random_range(0.0..1.0)).collect();
    let y: Vec<f32> = (0..n).map(|_| rng.random_range(0.0..1.0)).collect();

    let dtrain = DMatrix::from_dense(&x, n, m, Some(&y), 16).unwrap();
    let y_val: Vec<f32> = (0..n).map(|_| rng.random_range(10.0..20.0)).collect();
    let dval = DMatrix::from_dense(&x, n, m, Some(&y_val), 16).unwrap();

    let params = BoosterParams::new()
        .with_early_stopping(3)
        .with_max_depth(2);

    let booster = train(params, &dtrain, 50, &[(&dval, "val")]).unwrap();
    assert!(booster.num_trees() < 50);
    println!("Early stopping successfully stopped after {} trees!", booster.num_trees());
}
