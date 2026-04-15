"""
regression_model.py
-------------------
Linear Regression: Predict house prices from features.
Dataset: Synthetic housing data (size, bedrooms, age).
"""

import numpy as np
import json

# ── Reproducibility ──────────────────────────────────────
np.random.seed(42)

# ── Generate synthetic housing dataset ──────────────────
n = 200
size_sqft   = np.random.randint(500, 3500, n).astype(float)
bedrooms    = np.random.randint(1, 6, n).astype(float)
age_years   = np.random.randint(0, 40, n).astype(float)

# True relationship + noise
price = (
    120 * size_sqft
    + 15000 * bedrooms
    - 800 * age_years
    + 50000
    + np.random.normal(0, 20000, n)
)

X = np.column_stack([size_sqft, bedrooms, age_years])
y = price


# ── Min-max normalisation (no sklearn) ──────────────────
class MinMaxScaler:
    def fit(self, X):
        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)
        self.range_ = np.where(self.max_ - self.min_ == 0, 1, self.max_ - self.min_)
        return self

    def transform(self, X):
        return (X - self.min_) / self.range_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# ── Ordinary Least Squares regression (pure numpy) ──────
class LinearRegressionOLS:
    def fit(self, X, y):
        X_b = np.c_[np.ones(X.shape[0]), X]          # add bias column
        self.coef_ = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
        return self

    def predict(self, X):
        X_b = np.c_[np.ones(X.shape[0]), X]
        return X_b @ self.coef_

    @property
    def intercept_(self):
        return self.coef_[0]

    @property
    def weights_(self):
        return self.coef_[1:]


# ── Train / test split ───────────────────────────────────
split = int(0.8 * n)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

scaler = MinMaxScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

model = LinearRegressionOLS()
model.fit(X_train_s, y_train)

y_pred = model.predict(X_test_s)


# ── Metrics ──────────────────────────────────────────────
def mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot else 1.0

metrics = {
    "mse":  round(mse(y_test, y_pred), 2),
    "rmse": round(float(np.sqrt(mse(y_test, y_pred))), 2),
    "r2":   round(r2_score(y_test, y_pred), 4),
    "mae":  round(float(np.mean(np.abs(y_test - y_pred))), 2),
}

# ── Prediction function ──────────────────────────────────
def predict_price(size, bedrooms, age):
    x = np.array([[size, bedrooms, age]], dtype=float)
    x_s = scaler.transform(x)
    pred = model.predict(x_s)[0]
    return round(float(pred), 2)

# ── Export data for web app ──────────────────────────────
scatter_data = {
    "actual":    [round(float(v), 2) for v in y_test],
    "predicted": [round(float(v), 2) for v in y_pred],
}

feature_importance = {
    "features": ["Size (sqft)", "Bedrooms", "Age (years)"],
    "weights":  [round(float(w), 4) for w in model.weights_],
}

export = {
    "metrics": metrics,
    "scatter": scatter_data,
    "feature_importance": feature_importance,
    "intercept": round(float(model.intercept_), 2),
}

with open("static/regression_data.json", "w") as f:
    json.dump(export, f, indent=2)

# ── Quick demo ───────────────────────────────────────────
if __name__ == "__main__":
    print("=== Linear Regression: House Price Predictor ===")
    print(f"  R²   : {metrics['r2']}")
    print(f"  RMSE : ${metrics['rmse']:,.0f}")
    print(f"  MAE  : ${metrics['mae']:,.0f}")
    print()
    examples = [
        (1200, 2, 10),
        (2500, 4, 5),
        (800,  1, 30),
    ]
    for s, b, a in examples:
        p = predict_price(s, b, a)
        print(f"  {s} sqft, {b} bed, {a} yrs → ${p:,.0f}")
    print("\nData exported to static/regression_data.json")
