use rand::Rng;
use std::time::Instant;
use xgboost_webgpu::booster::{BoosterParams, DeviceType};
use xgboost_webgpu::data::DMatrix;
use xgboost_webgpu::metrics;
use xgboost_webgpu::train;

fn main() {
    println!("=========================================================");
    println!("  xgboost-webgpu: Binary Classification Benchmark");
    println!("=========================================================\n");

    let mut rng = rand::rng();
    let n_train = 6000;
    let n_test = 1500;
    let n_features = 10;

    println!("Generating {} training samples with {} features...", n_train, n_features);

    let generate_data = |n: usize, rng: &mut rand::rngs::ThreadRng| -> (Vec<f32>, Vec<f32>) {
        let mut x = Vec::with_capacity(n * n_features);
        let mut y = Vec::with_capacity(n);

        for _ in 0..n {
            let row: Vec<f32> = (0..n_features)
                .map(|_| rng.random_range(-2.0f32..2.0f32))
                .collect();

            // Non-linear decision boundary: circle + interactive terms
            let score = row[0] * row[0] + row[1] * row[1] - row[2] * row[3] + 0.5 * row[4]
                + rng.random_range(-0.2f32..0.2f32);

            let label = if score > 1.0 { 1.0f32 } else { 0.0f32 };
            x.extend_from_slice(&row);
            y.push(label);
        }
        (x, y)
    };

    let (x_train, y_train) = generate_data(n_train, &mut rng);
    let (x_test, y_test) = generate_data(n_test, &mut rng);

    let dtrain = DMatrix::from_dense(&x_train, n_train, n_features, Some(&y_train), 128)
        .expect("Failed to build training DMatrix");
    let dtest = DMatrix::from_dense_with_mappers(
        &x_test,
        n_test,
        n_features,
        Some(&y_test),
        &dtrain.bin_mappers,
    )
    .expect("Failed to build test DMatrix");

    let num_rounds = 25;

    println!("\n>>> Training with WebGPU acceleration ({} trees)...", num_rounds);
    let params = BoosterParams::new()
        .with_objective("binary:logistic")
        .with_max_depth(5)
        .with_learning_rate(0.3)
        .with_eval_metric("logloss")
        .with_device(DeviceType::WebGPU);

    let start_time = Instant::now();
    let booster = train(params, &dtrain, num_rounds, &[(&dtest, "test")])
        .expect("Training failed");
    let elapsed = start_time.elapsed();

    let probs = booster.predict(&dtest).expect("Predict failed");
    let test_logloss = metrics::logloss(&y_test, &probs);
    let test_err = metrics::binary_error(&y_test, &probs, 0.5);

    println!("\n[WebGPU] Training completed in {:.2?}", elapsed);
    println!("[WebGPU] Test LogLoss: {:.4}", test_logloss);
    println!("[WebGPU] Test Accuracy: {:.2}%", (1.0 - test_err) * 100.0);

    let model_path = "classification_model.json";
    booster.save_model(model_path).expect("Failed to save model");
    println!("Exported classification model to '{}'", model_path);
}
