# 💳 Credit Card Fraud Detection System
## note: This projet is made by using Ai for practice purpose.
> End-to-end ML pipeline for real-time fraud detection | SMOTE | Random Forest | XGBoost | SHAP | Streamlit

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)

---

## 🎯 Overview

Detects fraudulent credit card transactions using ensemble ML with advanced imbalanced-data techniques. Trained on the industry-standard Kaggle dataset (284,807 transactions, 0.17% fraud rate).

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# 1. Download dataset → data/creditcard.csv
#    https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

# 2. Train model (run all cells)
jupyter notebook notebooks/fraud_detection.ipynb

# 3. Launch web app
streamlit run app/app.py
```

## 🧠 Key Features

- **SMOTE** oversampling to handle class imbalance (577:1 → 1:1)
- **Random Forest + XGBoost** ensemble models
- **Threshold tuning** for optimal precision/recall trade-off
- **SHAP explainability** — why was this transaction flagged?
- **Streamlit web app** — single & batch scoring
- **Business insights** — amount/time risk patterns

## 📊 Results

| Model | ROC-AUC | Recall |
|-------|---------|--------|
| Logistic Regression | ~0.97 | ~0.75 |
| Random Forest ⭐ | ~0.98 | ~0.87 |
| XGBoost | ~0.98 | ~0.86 |

## 📁 Structure

```
├── data/                   # Dataset (download from Kaggle)
├── notebooks/              # End-to-end ML notebook
├── models/                 # Saved model (model.pkl)
├── app/                    # Streamlit web app
├── reports/                # Figures + executive summary
└── requirements.txt
```

## 🔑 Why Recall > Accuracy?

With 99.83% legitimate transactions, a naive "always predict legit" model gets 99.83% accuracy — but catches **zero** fraud. We optimise for **recall** (catching fraud) using ROC-AUC and F1, with threshold tuning to balance business costs.
