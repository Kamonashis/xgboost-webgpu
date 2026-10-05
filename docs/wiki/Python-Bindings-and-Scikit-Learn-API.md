# Python Bindings & Scikit-Learn API

`xgboost-webgpu` includes high-performance Python bindings built with **PyO3** and compiled against the **Python Stable ABI (`abi3-py310`)**. It exposes both drop-in Scikit-Learn estimators (`XGBRegressor`, `XGBClassifier`) and the core low-level functional API (`xgb.train()`, `xgb.DMatrix`).

---

## 1. Installation

### From Source via pip
```bash
# Clone the repository
git clone https://tea.ergotresearch.com/Ergot-Research/xgboost-webgpu.git
cd xgboost-webgpu

# Build and install into current Python environment
pip install .
```

### Editable Development with Maturin
```bash
pip install maturin
maturin develop --release
```

### Python Stable ABI (`abi3-py310`)
Because `xgboost-webgpu` compiles against the PEP 384 Stable ABI, a single compiled binary (`.so` or wheel) runs seamlessly across **Python 3.10, 3.11, 3.12, 3.13, and 3.14+** without segmentation faults, memory layout mismatches, or Python rebuild dependencies.

---

## 2. Scikit-Learn Estimator API

The estimators adhere to standard Scikit-Learn conventions and integrate into pipelines, cross-validation, and hyperparameter search tools.

### `XGBRegressor`

```python
import numpy as np
import xgboost_webgpu as xgb

# 1. Ingest Data
X = np.random.randn(1000, 8).astype(np.float32)
y = (X[:, 0] * 2.0 + X[:, 1] * -1.5 + np.random.randn(1000) * 0.1).astype(np.float32)

# 2. Instantiate with WebGPU acceleration
reg = xgb.XGBRegressor(
    n_estimators=50,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    device="webgpu",  # Acceleration on AMD, Apple Silicon, Intel, NVIDIA
)

# 3. Fit
reg.fit(X, y)

# 4. Predict & Score
preds = reg.predict(X)
r2 = reg.score(X, y)
print(f"R^2 Score: {r2:.4f}")

# 5. Feature Importances
print("Feature Importances:", reg.feature_importances_)

# 6. TreeSHAP Feature Attributions
shap_values = reg.predict_contributions(X)
print("TreeSHAP matrix shape:", np.array(shap_values).shape)

# 7. Model Persistence
reg.save_model("model.json")
```

---

### `XGBClassifier`

Supports both **binary logistic classification** and **multi-class classification** (with automatic class detection or explicit `num_class`).

```python
import numpy as np
import xgboost_webgpu as xgb

# Binary Classification
X_bin = np.random.randn(500, 4).astype(np.float32)
y_bin = (X_bin[:, 0] > 0).astype(np.float32)

clf_bin = xgb.XGBClassifier(n_estimators=30, learning_rate=0.1, device="webgpu")
clf_bin.fit(X_bin, y_bin)

print("Binary Accuracy:", clf_bin.score(X_bin, y_bin))
print("Probabilities (first 3):", clf_bin.predict_proba(X_bin[:3]))

# Multi-Class Classification (e.g. 3 classes)
X_multi = np.random.randn(600, 6).astype(np.float32)
y_multi = np.random.choice([0, 1, 2], size=600).astype(np.float32)

clf_multi = xgb.XGBClassifier(
    n_estimators=40,
    num_class=3,
    learning_rate=0.1,
    device="webgpu",
)
clf_multi.fit(X_multi, y_multi)

print("Multi-Class Accuracy:", clf_multi.score(X_multi, y_multi))
probs = clf_multi.predict_proba(X_multi[:3])
print("Class Probabilities (N x 3):", probs)
```

---

## 3. Low-Level Functional API

For maximum control matching native XGBoost workflows:

```python
import xgboost_webgpu as xgb
import numpy as np

# Create DMatrix from NumPy or nested Python lists
dtrain = xgb.DMatrix(X_train, labels=y_train, feature_names=["f0", "f1", "f2"])
deval = xgb.DMatrix(X_val, labels=y_val)

# Configure hyperparameters
params = {
    "objective": "reg:squarederror",
    "max_depth": 6,
    "learning_rate": 0.05,
    "reg_lambda": 1.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "grow_policy": "lossguide",
    "max_leaves": 31,
    "early_stopping_rounds": 10,
    "eval_metric": "rmse",
    "device": "webgpu",
}

# Train booster with evaluation tracking
bst = xgb.train(params, dtrain, num_boost_round=100, evals=[(deval, "val")])

# Inference
predictions = bst.predict(deval)
leaf_indices = bst.predict_leaf(deval)
shap_contribs = bst.predict_contributions(deval)
```
