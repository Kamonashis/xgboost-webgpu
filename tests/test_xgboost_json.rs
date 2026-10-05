use xgboost_webgpu::booster::BoosterParams;
use xgboost_webgpu::data::DMatrix;
use xgboost_webgpu::Booster;
use xgboost_webgpu::train;

#[test]
fn test_xgboost_json_roundtrip() {
    let data = vec![
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
        10.0, 11.0, 12.0,
    ];
    let labels = vec![1.5, 4.2, 7.1, 10.8];
    let dmat = DMatrix::from_dense(&data, 4, 3, Some(&labels), 16).unwrap();

    let params = BoosterParams::new()
        .with_objective("reg:squarederror")
        .with_max_depth(3)
        .with_learning_rate(0.3);

    let booster = train(params, &dmat, 5, &[]).expect("Training failed");
    let original_preds = booster.predict(&dmat).expect("Predict failed");

    // 1. Serialize to standard XGBoost JSON string
    let json_str = booster.to_xgboost_json().expect("Serialization failed");
    println!("Serialized XGBoost JSON:\n{}", &json_str[..json_str.len().min(400)]);

    // Check JSON keys conform to official XGBoost schema
    assert!(json_str.contains("\"learner\""));
    assert!(json_str.contains("\"gradient_booster\""));
    assert!(json_str.contains("\"gbtree\""));
    assert!(json_str.contains("\"trees\""));
    assert!(json_str.contains("\"loss_changes\""));
    assert!(json_str.contains("\"split_indices\""));

    // 2. Deserialize from XGBoost JSON string
    let reloaded_booster = Booster::from_xgboost_json(&json_str).expect("Deserialization failed");
    let reloaded_preds = reloaded_booster.predict(&dmat).expect("Reloaded predict failed");

    assert_eq!(booster.num_trees(), reloaded_booster.num_trees());
    assert_eq!(original_preds.len(), reloaded_preds.len());

    for i in 0..original_preds.len() {
        let diff = (original_preds[i] - reloaded_preds[i]).abs();
        assert!(
            diff < 1e-5,
            "Sample {}: original pred {} vs reloaded pred {}",
            i, original_preds[i], reloaded_preds[i]
        );
    }

    println!("XGBoost JSON roundtrip validated! Original and reloaded predictions match identically.");
}
