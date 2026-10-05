use rand::Rng;
use xgboost_webgpu::booster::BoosterParams;
use xgboost_webgpu::data::DMatrix;
use xgboost_webgpu::train;

#[test]
fn test_poisson_and_gamma_and_tweedie() {
    let mut rng = rand::rng();
    let n = 200;
    let m = 3;
    let x: Vec<f32> = (0..n * m).map(|_| rng.random_range(0.1..2.0)).collect();
    let y_counts: Vec<f32> = (0..n).map(|i| (x[i * m] * 3.0).floor()).collect();

    let dtrain_count = DMatrix::from_dense(&x, n, m, Some(&y_counts), 16).unwrap();

    // 1. Poisson
    let params_poisson = BoosterParams::new().with_objective("count:poisson").with_max_depth(3);
    let booster_poisson = train(params_poisson, &dtrain_count, 10, &[]).unwrap();
    let preds_poisson = booster_poisson.predict(&dtrain_count).unwrap();
    assert!(preds_poisson.iter().all(|&p| p >= 0.0));

    // 2. Gamma
    let y_pos: Vec<f32> = (0..n).map(|i| x[i * m] * 2.0 + 0.5).collect();
    let dtrain_pos = DMatrix::from_dense(&x, n, m, Some(&y_pos), 16).unwrap();
    let params_gamma = BoosterParams::new().with_objective("reg:gamma").with_max_depth(3);
    let booster_gamma = train(params_gamma, &dtrain_pos, 10, &[]).unwrap();
    let preds_gamma = booster_gamma.predict(&dtrain_pos).unwrap();
    assert!(preds_gamma.iter().all(|&p| p > 0.0));

    // 3. Tweedie
    let params_tweedie = BoosterParams::new().with_objective("reg:tweedie").with_max_depth(3);
    let booster_tweedie = train(params_tweedie, &dtrain_pos, 10, &[]).unwrap();
    let preds_tweedie = booster_tweedie.predict(&dtrain_pos).unwrap();
    assert!(preds_tweedie.iter().all(|&p| p >= 0.0));

    // 4. Quantile regression
    let params_quantile = BoosterParams::new().with_objective("reg:quantileerror").with_max_depth(3);
    let booster_quantile = train(params_quantile, &dtrain_pos, 10, &[]).unwrap();
    let preds_quantile = booster_quantile.predict(&dtrain_pos).unwrap();
    assert_eq!(preds_quantile.len(), n);

    // 5. Ranking pairwise
    let params_ranking = BoosterParams::new().with_objective("rank:pairwise").with_max_depth(3);
    let booster_ranking = train(params_ranking, &dtrain_pos, 10, &[]).unwrap();
    let preds_ranking = booster_ranking.predict(&dtrain_pos).unwrap();
    assert_eq!(preds_ranking.len(), n);
}
