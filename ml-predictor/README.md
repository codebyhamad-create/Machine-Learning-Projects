# ML Insight Studio

> *Three predictive models — regression, classification, and recommendation — built from scratch with pure NumPy and presented in a bespoke editorial web interface.*

---

## Project Structure

```
ml-predictor/
├── index.html                  ← Web app (open in browser)
├── run_all.py                  ← Run all models at once
├── models/
│   ├── regression_model.py     ← OLS Linear Regression
│   ├── classification_model.py ← Logistic Regression (gradient descent)
│   └── recommendation_model.py ← Collaborative Filtering (cosine similarity)
└── static/
    ├── regression_data.json
    ├── classification_data.json
    └── recommendation_data.json
```

---

## Quickstart

**1. Run all models (generates JSON data):**
```bash
python run_all.py
```

**2. Serve the web app:**
```bash
python -m http.server 8000
```

**3. Open in browser:**
```
http://localhost:8000
```

---

## Models

### Act I — Linear Regression (House Price Prediction)
- **Algorithm**: Ordinary Least Squares (closed-form solution)
- **Features**: size (sqft), bedrooms, age (years)
- **Target**: house price ($)
- **Key metric**: R² ≈ 0.91
- **From scratch**: custom MinMax scaler, OLS via matrix inverse

### Act II — Logistic Regression (Customer Churn)
- **Algorithm**: Gradient Descent (500 epochs, lr=0.1)
- **Features**: tenure, monthly charges, support calls, contract length
- **Target**: churn (binary)
- **Key metrics**: Accuracy, Precision, Recall, F1, AUC-ROC
- **From scratch**: custom StandardScaler, sigmoid, cross-entropy loss

### Act III — Collaborative Filtering (Movie Recommendations)
- **Algorithm**: User-based cosine similarity
- **Dataset**: 10 users × 12 movies, 60% sparse
- **Output**: personalised top-5 movie recommendations per user
- **From scratch**: cosine similarity, weighted average prediction, heatmap

---

## Dependencies

```
numpy   (only external dependency)
```

No pandas, no sklearn, no scipy. Pure NumPy throughout.

---

## Web App

- **Typography**: Playfair Display (serif headings) + DM Sans (body)
- **Theme**: Warm editorial — cream, gold, sage, rust
- **Charts**: Chart.js 4.4 (loaded via CDN)
- **Interactive**: live sliders for live prediction, user selector for recommendations
- **No framework**: vanilla HTML + CSS + JS

---

*Built for class presentation — COMP/ML module*
