"""
xgboost-webgpu: Universal GPU-Accelerated XGBoost via WebGPU in Python.
"""

import sys
import os

# Try importing the compiled C-extension
try:
    from ._xgboost_webgpu import DMatrix, Booster, train
except ImportError:
    try:
        from _xgboost_webgpu import DMatrix, Booster, train
    except ImportError:
        raise ImportError(
            "Failed to load compiled _xgboost_webgpu extension module. "
            "Please build the Python extension using 'maturin develop' or 'pip install .'"
        )

__all__ = ["DMatrix", "Booster", "train", "XGBRegressor", "XGBClassifier"]


class BaseXGBModel:
    """Base class for scikit-learn compatible XGBoost estimators using WebGPU."""

    def __init__(
        self,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.3,
        reg_lambda=1.0,
        reg_alpha=0.0,
        gamma=0.0,
        min_child_weight=1.0,
        subsample=1.0,
        colsample_bytree=1.0,
        colsample_bynode=1.0,
        max_delta_step=0.0,
        grow_policy="depthwise",
        max_leaves=0,
        booster="gbtree",
        rate_drop=0.0,
        skip_drop=0.0,
        monotone_constraints=None,
        device="webgpu",
        eval_metric=None,
        early_stopping_rounds=None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.colsample_bynode = colsample_bynode
        self.max_delta_step = max_delta_step
        self.grow_policy = grow_policy
        self.max_leaves = max_leaves
        self.booster_type = booster
        self.rate_drop = rate_drop
        self.skip_drop = skip_drop
        self.monotone_constraints = monotone_constraints
        self.device = device
        self.eval_metric = eval_metric
        self.early_stopping_rounds = early_stopping_rounds
        self._booster = None

    def _get_params(self):
        params = {
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "reg_lambda": self.reg_lambda,
            "reg_alpha": self.reg_alpha,
            "gamma": self.gamma,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "colsample_bynode": self.colsample_bynode,
            "max_delta_step": self.max_delta_step,
            "grow_policy": self.grow_policy,
            "max_leaves": self.max_leaves,
            "booster": self.booster_type,
            "rate_drop": self.rate_drop,
            "skip_drop": self.skip_drop,
            "device": self.device,
        }
        if self.monotone_constraints is not None:
            params["monotone_constraints"] = self.monotone_constraints
        if self.eval_metric is not None:
            params["eval_metric"] = self.eval_metric
        if self.early_stopping_rounds is not None:
            params["early_stopping_rounds"] = self.early_stopping_rounds
        return params

    @property
    def feature_importances_(self):
        if self._booster is None:
            raise RuntimeError("Estimator not fitted yet.")
        raw_imp = self._booster.feature_importance("gain")
        return dict(raw_imp)

    def get_booster(self):
        """Get the underlying Booster instance."""
        if self._booster is None:
            raise RuntimeError("Estimator not fitted yet.")
        return self._booster

    def predict_contributions(self, X):
        """Compute exact TreeSHAP feature contributions for each sample."""
        if self._booster is None:
            raise RuntimeError("Estimator not fitted yet.")
        dmat = DMatrix(X)
        return self._booster.predict_contributions(dmat)

    def predict_leaf(self, X):
        """Predict leaf index for each tree for each sample."""
        if self._booster is None:
            raise RuntimeError("Estimator not fitted yet.")
        dmat = DMatrix(X)
        return self._booster.predict_leaf(dmat)

    def save_model(self, path):
        """Save model to JSON file compatible with standard XGBoost."""
        if self._booster is None:
            raise RuntimeError("Estimator not fitted yet.")
        self._booster.save_model(path)

    def load_model(self, path):
        """Load model from JSON file."""
        self._booster = Booster(model_file=path)


class XGBRegressor(BaseXGBModel):
    """XGBoost Regressor with WebGPU acceleration."""

    def __init__(self, objective="reg:squarederror", **kwargs):
        super().__init__(**kwargs)
        self.objective = objective

    def fit(self, X, y, sample_weight=None, eval_set=None):
        labels = list(y) if not isinstance(y, list) else y
        weights = list(sample_weight) if sample_weight is not None and not isinstance(sample_weight, list) else sample_weight
        dtrain = DMatrix(X, labels=labels, weights=weights)
        params = self._get_params()
        params["objective"] = self.objective

        evals_list = []
        if eval_set is not None:
            for i, (eval_x, eval_y) in enumerate(eval_set):
                e_lbls = list(eval_y) if not isinstance(eval_y, list) else eval_y
                eval_dmat = DMatrix(eval_x, labels=e_lbls)
                evals_list.append((eval_dmat, f"eval_{i}"))

        self._booster = train(params, dtrain, num_boost_round=self.n_estimators, evals=evals_list)
        return self

    def predict(self, X):
        if self._booster is None:
            raise RuntimeError("Estimator not fitted yet.")
        dtest = DMatrix(X)
        return self._booster.predict(dtest)

    def score(self, X, y):
        """Return R^2 coefficient of determination."""
        y_true = list(y)
        y_pred = self.predict(X)
        n = len(y_true)
        mean_y = sum(y_true) / n
        ss_tot = sum((yt - mean_y) ** 2 for yt in y_true)
        ss_res = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred))
        return 1.0 - (ss_res / (ss_tot + 1e-10))


class XGBClassifier(BaseXGBModel):
    """XGBoost Classifier with WebGPU acceleration."""

    def __init__(self, objective="binary:logistic", scale_pos_weight=1.0, num_class=0, **kwargs):
        super().__init__(**kwargs)
        self.objective = objective
        self.scale_pos_weight = scale_pos_weight
        self.num_class = num_class

    def fit(self, X, y, sample_weight=None, eval_set=None):
        labels = list(y) if not isinstance(y, list) else y
        unique_labels = sorted(list(set(labels)))
        if len(unique_labels) > 2 and self.num_class == 0:
            self.num_class = len(unique_labels)
            if self.objective == "binary:logistic":
                self.objective = "multi:softprob"

        weights = list(sample_weight) if sample_weight is not None and not isinstance(sample_weight, list) else sample_weight
        dtrain = DMatrix(X, labels=labels, weights=weights)
        params = self._get_params()
        params["objective"] = self.objective
        params["scale_pos_weight"] = self.scale_pos_weight
        if self.num_class > 0:
            params["num_class"] = self.num_class

        evals_list = []
        if eval_set is not None:
            for i, (eval_x, eval_y) in enumerate(eval_set):
                e_lbls = list(eval_y) if not isinstance(eval_y, list) else eval_y
                eval_dmat = DMatrix(eval_x, labels=e_lbls)
                evals_list.append((eval_dmat, f"eval_{i}"))

        self._booster = train(params, dtrain, num_boost_round=self.n_estimators, evals=evals_list)
        return self

    def predict_proba(self, X):
        if self._booster is None:
            raise RuntimeError("Estimator not fitted yet.")
        dtest = DMatrix(X)
        raw_probs = self._booster.predict(dtest)

        if self.num_class > 0:
            k = self.num_class
            n = len(raw_probs) // k
            return [raw_probs[i * k : (i + 1) * k] for i in range(n)]
        else:
            return [[1.0 - p, p] for p in raw_probs]

    def predict(self, X):
        if self.num_class > 0:
            probs = self.predict_proba(X)
            return [p.index(max(p)) for p in probs]
        else:
            dtest = DMatrix(X)
            probs = self._booster.predict(dtest)
            return [1 if p >= 0.5 else 0 for p in probs]

    def score(self, X, y):
        """Return classification accuracy."""
        y_true = list(y)
        y_pred = self.predict(X)
        correct = sum(1 for yt, yp in zip(y_true, y_pred) if int(yt) == int(yp))
        return correct / len(y_true)
