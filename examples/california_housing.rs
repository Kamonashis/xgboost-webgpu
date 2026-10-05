use rand::Rng;
use std::time::Instant;
use xgboost_webgpu::booster::{BoosterParams, DeviceType};
use xgboost_webgpu::data::DMatrix;
use xgboost_webgpu::metrics;
use xgboost_webgpu::train;

fn main() {
    println!("=========================================================");
    println!("  xgboost-webgpu: California Housing Regression Benchmark");
    println!("=========================================================\n");

    let mut rng = rand::rng();
    let n_train = 5000;
    let n_test = 1000;
    let n_features = 8;
    let feature_names = vec![
        "MedInc".to_string(),
        "HouseAge".to_string(),
        "AveRooms".to_string(),
        "AveBedrms".to_string(),
        "Population".to_string(),
        "AveOccup".to_string(),
        "Latitude".to_string(),
        "Longitude".to_string(),
    ];

    println!("Generating {} training samples with {} features...", n_train, n_features);

    let generate_housing_data = |n: usize, rng: &mut rand::rngs::ThreadRng| -> (Vec<f32>, Vec<f32>) {
        let mut x = Vec::with_capacity(n * n_features);
        let mut y = Vec::with_capacity(n);

        for _ in 0..n {
            let med_inc = rng.random_range(0.5f32..15.0f32);
            let house_age = rng.random_range(1.0f32..50.0f32);
            let ave_rooms = rng.random_range(1.0f32..10.0f32);
            let ave_bedrms = rng.random_range(0.5f32..3.0f32);
            let population = rng.random_range(100.0f32..5000.0f32);
            let ave_occup = rng.random_range(1.0f32..6.0f32);
            let lat = rng.random_range(32.5f32..42.0f32);
            let lon = rng.random_range(-124.3f32..-114.3f32);

            let row = [
                med_inc, house_age, ave_rooms, ave_bedrms, population, ave_occup, lat, lon,
            ];

            // Target price calculation with realistic non-linear dependencies
            let target = 0.5 * med_inc + 0.02 * house_age - 0.1 * (lat - 37.0).abs()
                - 0.08 * (lon + 120.0).abs()
                + (med_inc * 0.1).sin()
                + rng.random_range(-0.15f32..0.15f32);

            x.extend_from_slice(&row);
            y.push(target);
        }
        (x, y)
    };

    let (x_train, y_train) = generate_housing_data(n_train, &mut rng);
    let (x_test, y_test) = generate_housing_data(n_test, &mut rng);

    let mut dtrain = DMatrix::from_dense(&x_train, n_train, n_features, Some(&y_train), 128)
        .expect("Failed to build training DMatrix");
    dtrain.set_feature_names(feature_names.clone()).unwrap();

    let mut dtest = DMatrix::from_dense_with_mappers(
        &x_test,
        n_test,
        n_features,
        Some(&y_test),
        &dtrain.bin_mappers,
    )
    .expect("Failed to build test DMatrix");
    dtest.set_feature_names(feature_names).unwrap();

    let num_rounds = 30;

    // --- 1. GPU Training ---
    println!("\n>>> Starting WebGPU-accelerated Training ({} trees)...", num_rounds);
    let gpu_params = BoosterParams::new()
        .with_objective("reg:squarederror")
        .with_max_depth(6)
        .with_learning_rate(0.2)
        .with_device(DeviceType::WebGPU);

    let start_gpu = Instant::now();
    let gpu_booster = train(gpu_params, &dtrain, num_rounds, &[(&dtest, "test")])
        .expect("WebGPU training failed");
    let gpu_duration = start_gpu.elapsed();

    let gpu_preds = gpu_booster.predict(&dtest).unwrap();
    let gpu_rmse = metrics::rmse(&y_test, &gpu_preds);
    println!("\n[WebGPU] Training completed in {:.2?}", gpu_duration);
    println!("[WebGPU] Test RMSE: {:.4}", gpu_rmse);

    // Save model to standard XGBoost JSON
    let model_path = "california_model.json";
    gpu_booster.save_model(model_path).expect("Failed to save model");
    println!("Exported model to standard XGBoost format: '{}'", model_path);

    // --- 2. CPU Reference Training ---
    println!("\n>>> Starting Multi-threaded CPU Reference Training ({} trees)...", num_rounds);
    let cpu_params = BoosterParams::new()
        .with_objective("reg:squarederror")
        .with_max_depth(6)
        .with_learning_rate(0.2)
        .with_device(DeviceType::Cpu);

    let start_cpu = Instant::now();
    let cpu_booster = train(cpu_params, &dtrain, num_rounds, &[])
        .expect("CPU training failed");
    let cpu_duration = start_cpu.elapsed();

    let cpu_preds = cpu_booster.predict(&dtest).unwrap();
    let cpu_rmse = metrics::rmse(&y_test, &cpu_preds);
    println!("[CPU] Training completed in {:.2?}", cpu_duration);
    println!("[CPU] Test RMSE: {:.4}", cpu_rmse);

    println!("\n=========================================================");
    println!("  Benchmark Summary:");
    println!("    WebGPU Time: {:.2?} | Final RMSE: {:.4}", gpu_duration, gpu_rmse);
    println!("    CPU Time:    {:.2?} | Final RMSE: {:.4}", cpu_duration, cpu_rmse);
    println!("=========================================================");
}
