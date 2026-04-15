"""
run_all.py
----------
Runs all three ML models and launches the web app.
Usage:  python run_all.py
"""

import os, sys, json, subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
os.makedirs("static", exist_ok=True)

print("=" * 54)
print("  ML Insight Studio — Model Runner")
print("=" * 54)

import numpy as np

# ── 1. Regression ────────────────────────────────────────
print("\n[1/3] Training Linear Regression (House Price)...")
np.random.seed(42)
n = 200
size_sqft   = np.random.randint(500, 3500, n).astype(float)
bedrooms    = np.random.randint(1, 6, n).astype(float)
age_years   = np.random.randint(0, 40, n).astype(float)

# More realistic relationship
price = (120 * size_sqft) + (15000 * bedrooms) - (800 * age_years) + 50000 + np.random.normal(0, 20000, n)
X = np.column_stack([size_sqft, bedrooms, age_years])
y = price

# 80/20 Train-Test Split
split = int(0.8 * n)
X_tr, X_te = X[:split], X[split:]
y_tr, y_te = y[:split], y[split:]

# Scaler (Fit on train only to avoid Data Leakage)
mn, mx = X_tr.min(0), X_tr.max(0)
rng = np.where(mx - mn == 0, 1, mx - mn)
Xs = (X_tr - mn) / rng
Xts = (X_te - mn) / rng

# OLS Solution
Xb = np.c_[np.ones(len(Xs)), Xs]
w = np.linalg.pinv(Xb.T @ Xb) @ Xb.T @ y_tr
Xbt = np.c_[np.ones(len(Xts)), Xts]
y_pred = Xbt @ w

ss_res = np.sum((y_te - y_pred)**2)
ss_tot = np.sum((y_te - np.mean(y_te))**2)
r2 = 1 - (ss_res / ss_tot)
rmse = np.sqrt(np.mean((y_te - y_pred)**2))

print(f"    R²={r2:.4f}  RMSE=${rmse:,.0f}")

export = {
    "metrics": {
        "r2": round(float(r2), 4),
        "mse": round(float(np.mean((y_te-y_pred)**2)), 2),
        "rmse": round(float(rmse), 2),
        "mae": round(float(np.mean(np.abs(y_te-y_pred))), 2)
    },
    "scatter": {
        "actual": [round(float(v), 2) for v in y_te],
        "predicted": [round(float(v), 2) for v in y_pred]
    },
    "feature_importance": {
        "features": ["Size (sqft)", "Bedrooms", "Age (years)"],
        "weights": [round(float(ww), 4) for ww in w[1:]]
    },
    "intercept": round(float(w[0]), 2)
}
with open("static/regression_data.json","w") as f: json.dump(export,f,indent=2)

# ── 2. Classification ─────────────────────────────────────
print("\n[2/3] Training Logistic Regression (Churn)...")
np.random.seed(7)
n = 300
tenure = np.random.randint(1, 72, n).astype(float)
monthly = np.random.uniform(20, 120, n)
calls = np.random.randint(0, 10, n).astype(float)
contract = np.random.choice([1, 12, 24], n).astype(float)

log_odds = -0.05*tenure + 0.03*monthly + 0.4*calls - 0.1*contract - 1.0
prob = 1 / (1 + np.exp(-log_odds))
# Sampling for realistic labels (prevents 100% accuracy)
y = (np.random.rand(n) < prob).astype(int)

X = np.column_stack([tenure, monthly, calls, contract])
split = int(0.8 * n)
X_tr, X_te = X[:split], X[split:]
y_tr, y_te = y[:split], y[split:]

mu, sd = X_tr.mean(0), np.where(X_tr.std(0) == 0, 1, X_tr.std(0))
Xs, Xts = (X_tr - mu) / sd, (X_te - mu) / sd
w = np.zeros(Xs.shape[1] + 1)

for _ in range(500):
    Xb = np.c_[np.ones(len(Xs)), Xs]
    z = Xb @ w
    p = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    w -= 0.1 * (Xb.T @ (p - y_tr) / len(y_tr))

Xbt = np.c_[np.ones(len(Xts)), Xts]
proba = 1 / (1 + np.exp(-np.clip(Xbt @ w, -500, 500)))
y_pred = (proba >= 0.5).astype(int)

tp=int(np.sum((y_te==1)&(y_pred==1))); fp=int(np.sum((y_te==0)&(y_pred==1))); tn=int(np.sum((y_te==0)&(y_pred==0))); fn=int(np.sum((y_te==1)&(y_pred==0)))
acc=(tp+tn)/len(y_te); prec=tp/(tp+fp) if tp+fp else 0; rec=tp/(tp+fn) if tp+fn else 0; f1=2*prec*rec/(prec+rec) if prec+rec else 0
thresholds=np.linspace(0,1,50); fprs=[]; tprs=[]
for t in thresholds:
    pp=(proba>=t).astype(int); tp_=int(np.sum((y_te==1)&(pp==1))); fp_=int(np.sum((y_te==0)&(pp==1))); tn_=int(np.sum((y_te==0)&(pp==0))); fn_=int(np.sum((y_te==1)&(pp==0)))
    tprs.append(round(tp_/(tp_+fn_),4) if tp_+fn_ else 0); fprs.append(round(fp_/(fp_+tn_),4) if fp_+tn_ else 0)

auc = round(abs(float(np.trapezoid(tprs, fprs))), 4)
print(f"    Accuracy={acc:.2%}  F1={f1:.4f}  AUC={auc:.4f}")

export={"metrics":{"accuracy":round(acc,4),"precision":round(prec,4),"recall":round(rec,4),"f1":round(f1,4)},"confusion":{"tp":tp,"fp":fp,"tn":tn,"fn":fn},"auc":auc,"roc":{"fpr":fprs,"tpr":tprs},"feature_names":["Tenure","Monthly Charges","Support Calls","Contract Length"],"feature_weights":[round(float(ww),4) for ww in w[1:]],"class_distribution":{"churn":int(np.sum(y==1)),"no_churn":int(np.sum(y==0))}}
with open("static/classification_data.json","w") as f: json.dump(export,f,indent=2)

# ── 3. Recommendation ─────────────────────────────────────
print("\n[3/3] Building Collaborative Filter (Movies)...")
np.random.seed(99)
movies=['Inception','The Dark Knight','Interstellar','Avengers','The Notebook','Pride & Prejudice','The Shining','Get Out','Parasite','Whiplash','La La Land','Toy Story']
genres=['sci-fi','action','sci-fi','action','romance','romance','horror','horror','thriller','drama','musical','animation']
ratings=np.array([[5,4,5,3,0,0,2,0,4,3,0,0],[4,5,3,5,0,0,0,3,0,0,0,0],[0,0,0,0,5,5,0,0,0,4,5,0],[0,0,0,0,0,0,5,5,4,0,0,0],[5,3,4,0,0,0,0,0,5,5,0,0],[0,0,0,3,5,4,0,0,0,0,5,4],[3,5,0,5,0,0,4,3,0,0,0,0],[0,0,5,0,4,0,0,0,5,4,0,0],[0,0,0,0,0,5,4,0,0,5,5,3],[4,0,4,0,0,0,3,4,4,0,0,0]],dtype=float)
n_users,n_movies=ratings.shape
def cos_sim(a,b):
    mask=(a>0)&(b>0)
    if mask.sum()==0: return 0.0
    a_,b_=a[mask],b[mask]; d=np.linalg.norm(a_)*np.linalg.norm(b_)
    return float(np.dot(a_,b_)/d) if d else 0.0
S=np.zeros((n_users,n_users))
for i in range(n_users):
    for j in range(n_users): S[i,j]=cos_sim(ratings[i],ratings[j])
all_recs={}
for uid in range(n_users):
    sims=S[uid].copy(); sims[uid]=0
    top4=np.argsort(sims)[::-1][:4]
    unrated=np.where(ratings[uid]==0)[0]; preds={}
    for m in unrated:
        num,den=0.0,0.0
        for u in top4:
            if ratings[u,m]>0: num+=sims[u]*ratings[u,m]; den+=abs(sims[u])
        if den>0: preds[int(m)]=round(num/den,3)
    sorted_preds=sorted(preds.items(),key=lambda x:x[1],reverse=True)[:5]
    all_recs[uid]=[{'movie_idx':idx,'movie':movies[idx],'genre':genres[idx],'score':score} for idx,score in sorted_preds]
print(f"    {n_users} users × {n_movies} movies  sparsity={1-int(np.sum(ratings>0))/(n_users*n_movies):.0%}")
export={'movies':movies,'genres':genres,'user_names':[f'User {i}' for i in range(n_users)],'ratings':ratings.tolist(),'similarity':[[round(float(v),3) for v in row] for row in S],'recommendations':{str(k):v for k,v in all_recs.items()},'stats':{'rated_items':int(np.sum(ratings>0)),'unrated_items':int(np.sum(ratings==0)),'avg_rating':round(float(np.mean(ratings[ratings>0])),3),'sparsity':round(1-int(np.sum(ratings>0))/(n_users*n_movies),4),'n_users':n_users,'n_movies':n_movies}}
with open("static/recommendation_data.json","w") as f: json.dump(export,f,indent=2)

print("\n" + "=" * 54)
print("  All models trained. JSON data saved to static/")
print("=" * 54)
print("\n  Open index.html in a browser, or run a local server:")
print("  python -m http.server 8000")
print("  then visit http://localhost:8000\n")
