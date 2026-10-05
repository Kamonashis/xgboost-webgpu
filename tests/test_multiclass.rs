use rand::Rng;
use xgboost_webgpu::booster::BoosterParams;
use xgboost_webgpu::data::DMatrix;
use xgboost_webgpu::train;

#[test]
fn test_multiclass_softprob_and_softmax() {
    let mut rng = rand::rng();
    let n_train = 450;
    let n_test = 150;
    let n_features = 4;
    let num_class = 3;

    // 3 distinct cluster centers in feature space
    let centers = [
        [-2.0f32, -2.0, 0.0, 0.0],
        [2.0f32, 2.0, 0.0, 0.0],
        [0.0f32, 0.0, 3.0, 3.0],
    ];

    let generate_data = |n: usize, rng: &mut rand::rngs::ThreadRng| -> (Vec<f32>, Vec<f32>) {
        let mut x = Vec::with_capacity(n * n_features);
        let mut y = Vec::with_capacity(n);
        for _ in 0..n {
            let class_id = rng.random_range(0..num_class);
            let center = &centers[class_id];
            for j in 0..n_features {
                x.push(center[j] + rng.random_range(-0.6..0.6));
            }
            y.push(class_id as f32);
        }
        (x, y)
    };

    let (x_train, y_train) = generate_data(n_train, &mut rng);
    let (x_test, y_test) = generate_data(n_test, &mut rng);

    let dtrain = DMatrix::from_dense(&x_train, n_train, n_features, Some(&y_train), 32).unwrap();
    let dtest = DMatrix::from_dense_with_mappers(
        &x_test,
        n_test,
        n_features,
        Some(&y_test),
        &dtrain.bin_mappers,
    )
    .unwrap();

    // 1. Train with multi:softprob
    let params_prob = BoosterParams::new()
        .with_objective("multi:softprob")
        .with_num_class(3)
        .with_max_depth(3)
        .with_learning_rate(0.3);

    let booster_prob = train(params_prob, &dtrain, 15, &[(&dtest, "test")]).unwrap();
    let probs = booster_prob.predict(&dtest).unwrap();
    assert_eq!(probs.len(), n_test * 3);

    // Verify probabilities sum to ~1.0 per sample
    for i in 0..n_test {
        let p_sum: f32 = (0..3).map(|c| probs[i * 3 + c]).sum();
        assert!((p_sum - 1.0).abs() < 1e-4);
    }

    // 2. Train with multi:softmax
    let params_softmax = BoosterParams::new()
        .with_objective("multi:softmax")
        .with_num_class(3)
        .with_max_depth(3)
        .with_learning_rate(0.3);

    let booster_softmax = train(params_softmax, &dtrain, 15, &[]).unwrap();
    let class_preds = booster_softmax.predict(&dtest).unwrap();
    assert_eq!(class_preds.len(), n_test);

    let mut correct = 0;
    for i in 0..n_test {
        if (class_preds[i] as usize) == (y_test[i] as usize) {
            correct += 1;
        }
    }
    let accuracy = correct as f32 / n_test as f32;
    println!("Multi-class test accuracy: {:.2}%", accuracy * 100.0);
    assert!(accuracy > 0.90, "Expected accuracy > 90%, got {:.2}%", accuracy * 100.0);
}
