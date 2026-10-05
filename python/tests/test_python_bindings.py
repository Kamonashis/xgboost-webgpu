import unittest
import numpy as np
import tempfile
import os
import sys

# Ensure package is imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import xgboost_webgpu as xgb


class TestXGBoostWebGPUPython(unittest.TestCase):

    def test_dmatrix_creation(self):
        # From list of lists
        dmat = xgb.DMatrix([[1.0, 2.0], [3.0, 4.0]], labels=[0.0, 1.0], feature_names=["col1", "col2"])
        self.assertEqual(dmat.num_rows(), 2)
        self.assertEqual(dmat.num_cols(), 2)
        self.assertEqual(dmat.feature_names(), ["col1", "col2"])

        # From numpy
        X = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32)
        dmat_np = xgb.DMatrix(X)
        self.assertEqual(dmat_np.num_rows(), 2)
        self.assertEqual(dmat_np.num_cols(), 3)

    def test_xgb_regressor_webgpu(self):
        X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]], dtype=np.float32)
        y = np.array([2.5, 4.5, 6.5, 8.5, 10.5], dtype=np.float32)

        model = xgb.XGBRegressor(n_estimators=10, max_depth=3, learning_rate=0.2, device="webgpu")
        model.fit(X, y)
        preds = model.predict(X)

        self.assertEqual(len(preds), 5)
        self.assertGreater(model.score(X, y), 0.8)

        # Feature importance
        fi = model.feature_importances_
        self.assertIn("f0", fi)

        # TreeSHAP contributions
        shap = model.predict_contributions(X)
        self.assertEqual(len(shap), 5)
        self.assertEqual(len(shap[0]), 3)  # 2 features + 1 bias

        # Leaf indices
        leaves = model.predict_leaf(X)
        self.assertEqual(len(leaves), 5)
        self.assertEqual(len(leaves[0]), 10)

    def test_xgb_classifier_binary(self):
        X = np.array([[0.1, 0.2], [0.2, 0.1], [0.8, 0.9], [0.9, 0.7]], dtype=np.float32)
        y = np.array([0, 0, 1, 1], dtype=np.float32)

        clf = xgb.XGBClassifier(n_estimators=10, max_depth=2, learning_rate=0.3, device="webgpu")
        clf.fit(X, y)
        preds = clf.predict(X)
        probs = clf.predict_proba(X)

        self.assertEqual(len(preds), 4)
        self.assertEqual(len(probs), 4)
        self.assertEqual(len(probs[0]), 2)

    def test_xgb_classifier_multiclass(self):
        X = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9], [-1.0, -1.0], [-0.9, -0.8]], dtype=np.float32)
        y = np.array([0, 0, 1, 1, 2, 2], dtype=np.float32)

        clf = xgb.XGBClassifier(n_estimators=10, max_depth=2, learning_rate=0.3, device="webgpu", num_class=3)
        clf.fit(X, y)
        preds = clf.predict(X)
        probs = clf.predict_proba(X)

        self.assertEqual(len(preds), 6)
        self.assertEqual(len(probs), 6)
        self.assertEqual(len(probs[0]), 3)
        self.assertGreaterEqual(clf.score(X, y), 0.8)

    def test_model_serialization(self):
        X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], dtype=np.float32)
        y = np.array([10.0, 20.0, 30.0], dtype=np.float32)

        model = xgb.XGBRegressor(n_estimators=5, device="cpu")
        model.fit(X, y)
        orig_preds = model.predict(X)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            model.save_model(tmp_path)
            loaded = xgb.XGBRegressor()
            loaded.load_model(tmp_path)
            loaded_preds = loaded.predict(X)
            np.testing.assert_allclose(orig_preds, loaded_preds, rtol=1e-5)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()
