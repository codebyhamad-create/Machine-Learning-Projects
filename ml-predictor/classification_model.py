"""
classification_model.py
-----------------------
Logistic Regression Classifier: Predict customer churn (Yes / No).
Features: tenure, monthly_charges, support_calls, contract_length.
Pure numpy — no sklearn dependency.
"""

import numpy as np
import json

np.random.seed(7)

# ── Synthetic churn dataset ──────────────────────────────
n = 300

tenure          = np.random.randint(1, 72, n).astype(float)      # months
monthly_charges = np.random.uniform(20, 120, n)
support_calls   = np.random.randint(0, 10, n).astype(float)
contract_len    = np.random.choice([1, 12, 24], n).astype(float) # month/annual/2yr

# Churn probability driven by features
log_odds = (
    -0.05 * tenure
    + 0.03 * monthly_charges
    + 0.4  * support_calls
    - 0.1  * contract_len
    - 1.0
)
prob_churn = 1 / (1 + np.exp(-log_odds))

# Introduce stochasticity to make accuracy realistic (approx 90-95%)
y = (np.random.rand(n) < prob_churn).astype(int)

X = np.column_stack([tenure, monthly_charges, support_calls, contract_len])
feature_names = ["Tenure (months)", "Monthly Charges ($)", "Support Calls", "Contract Length"]


# ── Scaler ───────────────────────────────────────────────
class StandardScaler:
    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_  = np.where(X.std(axis=0) == 0, 1, X.std(axis=0))
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# ── Logistic Regression (gradient descent) ──────────────
class LogisticRegression:
    def __init__(self, lr=0.1, epochs=500):
        self.lr = lr
        self.epochs = epochs

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        X_b = np.c_[np.ones(X.shape[0]), X]
        self.weights = np.zeros(X_b.shape[1])
        self.loss_history = []

        for _ in range(self.epochs):
            z    = X_b @ self.weights
            pred = self.sigmoid(z)
            grad = X_b.T @ (pred - y) / len(y)
            self.weights -= self.lr * grad
            loss = -np.mean(y * np.log(pred + 1e-9) + (1-y) * np.log(1-pred + 1e-9))
            self.loss_history.append(round(float(loss), 6))
        return self

    def predict_proba(self, X):
        X_b = np.c_[np.ones(X.shape[0]), X]
        return self.sigmoid(X_b @ self.weights)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


# ── Train / test split ───────────────────────────────────
split = int(0.8 * n)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

clf = LogisticRegression(lr=0.1, epochs=500)
clf.fit(X_train_s, y_train)

y_pred      = clf.predict(X_test_s)
y_proba     = clf.predict_proba(X_test_s)


# ── Metrics ──────────────────────────────────────────────
def confusion_matrix(y_true, y_pred):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, fp, tn, fn

tp, fp, tn, fn = confusion_matrix(y_test, y_pred)
accuracy  = round((tp + tn) / len(y_test), 4)
precision = round(tp / (tp + fp), 4) if (tp + fp) else 0.0
recall    = round(tp / (tp + fn), 4) if (tp + fn) else 0.0
f1        = round(2 * precision * recall / (precision + recall), 4) if (precision + recall) else 0.0

metrics = {
    "accuracy":  accuracy,
    "precision": precision,
    "recall":    recall,
    "f1":        f1,
}

confusion = {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


# ── ROC curve (approx) ───────────────────────────────────
def roc_curve_approx(y_true, scores, steps=50):
    thresholds = np.linspace(0, 1, steps)
    tprs, fprs = [], []
    for t in thresholds:
        preds = (scores >= t).astype(int)
        tp_ = int(np.sum((y_true == 1) & (preds == 1)))
        fp_ = int(np.sum((y_true == 0) & (preds == 1)))
        tn_ = int(np.sum((y_true == 0) & (preds == 0)))
        fn_ = int(np.sum((y_true == 1) & (preds == 0)))
        tprs.append(round(tp_ / (tp_ + fn_), 4) if (tp_ + fn_) else 0.0)
        fprs.append(round(fp_ / (fp_ + tn_), 4) if (fp_ + tn_) else 0.0)
    return fprs, tprs

fprs, tprs = roc_curve_approx(y_test, y_proba)

# AUC via trapezoidal rule
auc = round(float(np.trapezoid(tprs, fprs)) * -1, 4)  # fprs descending → negative


# ── Prediction function ──────────────────────────────────
def predict_churn(tenure, monthly_charges, support_calls, contract_len):
    x = np.array([[tenure, monthly_charges, support_calls, contract_len]], dtype=float)
    x_s = scaler.transform(x)
    proba = float(clf.predict_proba(x_s)[0])
    label = "Churn" if proba >= 0.5 else "No Churn"
    return {"probability": round(proba, 4), "label": label}


# ── Export ───────────────────────────────────────────────
export = {
    "metrics":    metrics,
    "confusion":  confusion,
    "auc":        abs(auc),
    "roc": {
        "fpr": [round(v, 4) for v in fprs],
        "tpr": [round(v, 4) for v in tprs],
    },
    "loss_history": clf.loss_history[::10],  # every 10th epoch
    "feature_names": feature_names,
    "feature_weights": [round(float(w), 4) for w in clf.weights[1:]],
    "class_distribution": {
        "churn":    int(np.sum(y == 1)),
        "no_churn": int(np.sum(y == 0)),
    },
}

with open("static/classification_data.json", "w") as f:
    json.dump(export, f, indent=2)


if __name__ == "__main__":
    print("=== Logistic Regression: Customer Churn Classifier ===")
    print(f"  Accuracy  : {metrics['accuracy']:.2%}")
    print(f"  Precision : {metrics['precision']:.2%}")
    print(f"  Recall    : {metrics['recall']:.2%}")
    print(f"  F1 Score  : {metrics['f1']:.2%}")
    print(f"  AUC       : {abs(auc):.4f}")
    print()
    tests = [
        (2,  95, 8, 1,  "High risk"),
        (48, 45, 1, 24, "Low risk"),
        (12, 70, 4, 12, "Medium risk"),
    ]
    for t, m, s, c, desc in tests:
        r = predict_churn(t, m, s, c)
        print(f"  [{desc}] tenure={t}, charges=${m}, calls={s} → {r['label']} ({r['probability']:.1%})")
    print("\nData exported to static/classification_data.json")
