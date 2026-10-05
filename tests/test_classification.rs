use rand::Rng;
use xgboost_webgpu::booster::{BoosterParams, DeviceType};
use xgboost_webgpu::data::DMatrix;
use xgboost_webgpu::metrics;
use xgboost_webgpu::train;

#[test]
fn test_binary_classification_training() {
    let mut rng = rand::rng();
    let n_train = 1000;
    let n_test = 250;
    let n_features = 4;

    // Binary classification: class 1 if x0^2 + x1^2 > 1.2, else 0
    let generate_data = |n: usize, rng: &mut rand::rngs::ThreadRng| -> (Vec<f32>, Vec<f32>) {
        let mut x = Vec::with_capacity(n * n_features);
        let mut y = Vec::with_capacity(n);
        for _ in 0..n {
            let row: Vec<f32> = (0..n_features)
                .map(|_| rng.random_range(-1.5f32..1.5f32))
                .collect();
            let label = if row[0] * row[0] + row[1] * row[1] > 1.2 {
                1.0f32
            } else {
                0.0f32
            };
            x.extend_from_slice(&row);
            y.push(label);
        }
        (x, y)
    };

    let (x_train, y_train) = generate_data(n_train, &mut rng);
    let (x_test, y_test) = generate_data(n_test, &mut rng);

    let dtrain = DMatrix::from_dense(&x_train, n_train, n_features, Some(&y_train), 64).unwrap();
    let dtest = DMatrix::from_dense_with_mappers(
        &x_test,
        n_test,
        n_features,
        Some(&y_test),
        &dtrain.bin_mappers,
    )
    .unwrap();

    let params = BoosterParams::new()
        .with_objective("binary:logistic")
        .with_max_depth(4)
        .with_learning_rate(0.3)
        .with_device(DeviceType::WebGPU);

    let booster = train(params, &dtrain, 20, &[(&dtest, "val")]).expect("Training failed");

    let probs = booster.predict(&dtest).expect("Predict failed");
    let error = metrics::binary_error(&y_test, &probs, 0.5);
    let logloss = metrics::logloss(&y_test, &probs);

    println!("Classification validation LogLoss: {:.4}, Error rate: {:.2}%", logloss, error * 100.0);
    assert!(
        error < 0.20,
        "Expected classification error under 20%, got {:.2}%",
        error * 100.0
    );
}
