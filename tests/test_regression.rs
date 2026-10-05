use rand::Rng;
use xgboost_webgpu::booster::{BoosterParams, DeviceType};
use xgboost_webgpu::data::DMatrix;
use xgboost_webgpu::metrics;
use xgboost_webgpu::train;

#[test]
fn test_regression_training() {
    let mut rng = rand::rng();
    let n_train = 800;
    let n_test = 200;
    let n_features = 6;

    // Generate non-linear function: y = 2.0*x0^2 - 3.0*x1 + 1.5*sin(x2) + noise
    let generate_data = |n: usize, rng: &mut rand::rngs::ThreadRng| -> (Vec<f32>, Vec<f32>) {
        let mut x = Vec::with_capacity(n * n_features);
        let mut y = Vec::with_capacity(n);
        for _ in 0..n {
            let row: Vec<f32> = (0..n_features)
                .map(|_| rng.random_range(-2.0f32..2.0f32))
                .collect();
            let target = 2.0 * row[0] * row[0] - 3.0 * row[1] + 1.5 * row[2].sin()
                + rng.random_range(-0.1f32..0.1f32);
            x.extend_from_slice(&row);
            y.push(target);
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

    let initial_rmse = metrics::rmse(&y_test, &vec![0.0; n_test]);

    // Test GPU training
    let params_gpu = BoosterParams::new()
        .with_objective("reg:squarederror")
        .with_max_depth(5)
        .with_learning_rate(0.2)
        .with_device(DeviceType::WebGPU);

    let booster = train(params_gpu, &dtrain, 25, &[(&dtest, "test")]).expect("Training failed");

    let preds = booster.predict(&dtest).expect("Predict failed");
    let final_rmse = metrics::rmse(&y_test, &preds);

    println!("Initial Test RMSE: {:.4}, Final Test RMSE: {:.4}", initial_rmse, final_rmse);
    assert!(
        final_rmse < initial_rmse * 0.4,
        "Expected RMSE to drop significantly: initial {} vs final {}",
        initial_rmse,
        final_rmse
    );
}
