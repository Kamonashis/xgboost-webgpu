use xgboost_webgpu::data::{DMatrix, FeatureBinMapper};

#[test]
fn test_dmatrix_basic() {
    let data = vec![
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
        10.0, 11.0, 12.0,
    ];
    let labels = vec![0.0, 1.0, 0.0, 1.0];
    let dmat = DMatrix::from_dense(&data, 4, 3, Some(&labels), 4).unwrap();

    assert_eq!(dmat.nrows, 4);
    assert_eq!(dmat.ncols, 3);
    assert_eq!(dmat.binned_data.len(), 12);
    assert_eq!(dmat.labels.unwrap(), labels);
}

#[test]
fn test_bin_mapper_cut_points() {
    let vals = vec![0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0];
    let mapper = FeatureBinMapper::fit(&vals, 4);

    assert!(mapper.cut_points.len() <= 4);
    assert!(mapper.num_bins() <= 5);

    let bin_min = mapper.map_value(0.0);
    let bin_max = mapper.map_value(70.0);
    assert!(bin_min < bin_max);

    let bin_nan = mapper.map_value(f32::NAN);
    assert_eq!(bin_nan, mapper.missing_bin);
}

#[test]
fn test_dmatrix_with_existing_mappers() {
    let train_data = vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0];
    let train_mat = DMatrix::from_dense(&train_data, 3, 2, None, 4).unwrap();

    let test_data = vec![1.5, 15.0, 2.5, 25.0];
    let test_mat = DMatrix::from_dense_with_mappers(
        &test_data,
        2,
        2,
        None,
        &train_mat.bin_mappers,
    )
    .unwrap();

    assert_eq!(test_mat.nrows, 2);
    assert_eq!(test_mat.ncols, 2);
}
