"""
Movie Recommendation System — Step 5: Evaluation
=================================================
No scikit-surprise needed — uses the numpy/scipy SVD artifacts from 03b.

Metrics computed:
  - RMSE / MAE        : rating prediction accuracy (collaborative)
  - Precision@K       : fraction of top-K recs the user actually liked
  - Recall@K          : fraction of liked movies appearing in top-K
  - Catalog coverage  : % of movie catalog the system recommends
  - Intra-list diversity : avg pairwise dissimilarity within rec lists
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error

import matplotlib
matplotlib.use('Agg')

OUT_DIR = "./output"

print("Loading artifacts...")
movies       = pd.read_parquet(f"{OUT_DIR}/movies_master.parquet").reset_index(drop=True)
cosine_sim   = np.load(f"{OUT_DIR}/cosine_sim_matrix.npy")

with open(f"{OUT_DIR}/svd_model.pkl", "rb") as f:
    svd_artifacts = pickle.load(f)

predicted_df   = svd_artifacts["predicted_df"]
movieid_to_idx = svd_artifacts["movieid_to_idx"]
userid_to_idx  = svd_artifacts["userid_to_idx"]
ratings_sample = svd_artifacts["ratings_sample"]

title_to_idx = pd.Series(movies.index, index=movies["title_clean"].str.lower())
print(f"Loaded: {len(movies):,} movies | SVD matrix {predicted_df.shape}")

inverse_movieid_to_idx = {v: k for k, v in movieid_to_idx.items()}


# ── 1. RMSE / MAE on known ratings ───────────────────────────────────────────

print("\nComputing RMSE / MAE on held-out ratings...")

EVAL_SAMPLE = 10_000
eval_df = ratings_sample.sample(min(EVAL_SAMPLE, len(ratings_sample)), random_state=42)

actual_list = []
pred_list   = []

for _, row in eval_df.iterrows():
    u = int(row["user_idx"])
    m = int(row["movie_idx"])
    actual_list.append(row["rating"])
    pred_val = np.clip(float(predicted_df.iloc[u, m]), 0.5, 5.0)
    pred_list.append(pred_val)

actual_arr = np.array(actual_list)
pred_arr   = np.array(pred_list)

rmse = np.sqrt(mean_squared_error(actual_arr, pred_arr))
mae  = mean_absolute_error(actual_arr, pred_arr)

print(f"  RMSE : {rmse:.4f}")
print(f"  MAE  : {mae:.4f}")


# ── 2. Precision@K and Recall@K ──────────────────────────────────────────────

print("\nComputing Precision@K and Recall@K...")

THRESHOLD = 3.5
K_VALUES  = [5, 10]
N_USERS   = 1

sample_users = ratings_sample["userId"].drop_duplicates().sample(
    min(N_USERS, ratings_sample["userId"].nunique()), random_state=42
)

precision_scores = {k: [] for k in K_VALUES}
recall_scores    = {k: [] for k in K_VALUES}

for user_id in sample_users:
    if user_id not in userid_to_idx:
        continue

    u_idx        = userid_to_idx[user_id]
    user_ratings = ratings_sample[ratings_sample["userId"] == user_id]
    rated_ids    = set(user_ratings["movieId"].tolist())
    liked_ids    = set(user_ratings[user_ratings["rating"] >= THRESHOLD]["movieId"].tolist())

    if not liked_ids:
        continue

    user_preds = predicted_df.iloc[u_idx].values
    # Get top 1000 predicted movies for this user
    top_1000_indices = np.argsort(user_preds)[::-1][:1000]
    candidates = [
        (inverse_movieid_to_idx[m_idx], np.clip(user_preds[m_idx], 0.5, 5.0))
        for m_idx in top_1000_indices
        if inverse_movieid_to_idx[m_idx] not in rated_ids
    ]
    candidates_sorted = sorted(candidates, key=lambda x: x[1], reverse=True)

    for k in K_VALUES:
        top_k_ids = set(c[0] for c in candidates_sorted[:k])
        hits = len(top_k_ids & liked_ids)
        precision_scores[k].append(hits / k)
        recall_scores[k].append(hits / len(liked_ids))

p_at_5  = np.mean(precision_scores[5]) if precision_scores[5] else 0.0
p_at_10 = np.mean(precision_scores[10]) if precision_scores[10] else 0.0
r_at_5  = np.mean(recall_scores[5]) if recall_scores[5] else 0.0
r_at_10 = np.mean(recall_scores[10]) if recall_scores[10] else 0.0

print(f"  Precision@5  : {p_at_5:.4f}")
print(f"  Precision@10 : {p_at_10:.4f}")
print(f"  Recall@5     : {r_at_5:.4f}")
print(f"  Recall@10    : {r_at_10:.4f}")


# ── 3. Catalog coverage ───────────────────────────────────────────────────────

print("\nComputing catalog coverage...")

def catalog_coverage(sample_titles, n=10):
    recommended = set()
    for title_lower in sample_titles:
        if title_lower not in title_to_idx.index:
            continue
        idx_series = title_to_idx.loc[title_lower]
        if isinstance(idx_series, pd.Series):
            idx = int(idx_series.iloc[0])
        else:
            idx = int(idx_series)
        top_n = np.argsort(cosine_sim[idx])[-n-1:-1][::-1]
        # flatten to plain Python ints before adding to set
        recommended.update(int(i) for i in top_n)
    return len(recommended) / len(movies)

sample_titles = movies["title_clean"].str.lower().sample(200, random_state=42).tolist()
coverage = catalog_coverage(sample_titles, n=10)
print(f"  Catalog coverage : {coverage:.2%}")


# ── 4. Intra-list diversity ───────────────────────────────────────────────────

print("Computing intra-list diversity...")

def intra_list_diversity(title_lower, n=10):
    if title_lower not in title_to_idx.index:
        return np.nan
    idx_series = title_to_idx.loc[title_lower]
    if isinstance(idx_series, pd.Series):
        idx = int(idx_series.iloc[0])
    else:
        idx = int(idx_series)
    top_n = np.argsort(cosine_sim[idx])[-n-1:-1][::-1]
    top_n = [int(i) for i in top_n]             # plain Python ints
    sim_sub = cosine_sim[np.ix_(top_n, top_n)]
    np.fill_diagonal(sim_sub, 0)
    n_pairs = n * (n - 1)
    return round(float(1 - sim_sub.sum() / n_pairs), 4)

diversity_scores = [intra_list_diversity(t) for t in sample_titles[:100]]
diversity = float(np.nanmean(diversity_scores))
print(f"  Intra-list diversity : {diversity:.4f}  (1.0 = maximally diverse)")


# ── 5. Full summary ───────────────────────────────────────────────────────────

print(f"""
{'═'*45}
  Evaluation Summary
{'═'*45}
  RMSE              : {rmse:.4f}
  MAE               : {mae:.4f}
  Precision@5       : {p_at_5:.4f}
  Precision@10      : {p_at_10:.4f}
  Recall@5          : {r_at_5:.4f}
  Recall@10         : {r_at_10:.4f}
  Catalog coverage  : {coverage:.2%}
  Diversity         : {diversity:.4f}
{'═'*45}
""")


# ── 6. Metrics bar chart ──────────────────────────────────────────────────────

metric_labels = ["RMSE\n(↓better)", "MAE\n(↓better)",
                 "Precision\n@10", "Recall\n@10",
                 "Coverage", "Diversity"]
metric_values = [rmse, mae, p_at_10, r_at_10, coverage, diversity]
colors        = ["#D85A30", "#D85A30", "#1D9E75", "#1D9E75", "#185FA5", "#534AB7"]

fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(metric_labels, metric_values, color=colors, edgecolor="none", width=0.5)
ax.set_title("Recommendation System — Evaluation Metrics", fontsize=13, fontweight="bold")
ax.set_ylabel("Score")
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, max(metric_values) * 1.3)

for b, v in zip(bars, metric_values):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
            f"{v:.3f}", ha="center", va="bottom", fontsize=10)

legend_elements = [
    mpatches.Patch(facecolor="#D85A30", label="Error metrics (lower = better)"),
    mpatches.Patch(facecolor="#1D9E75", label="Ranking metrics (higher = better)"),
    mpatches.Patch(facecolor="#185FA5", label="Coverage (higher = better)"),
    mpatches.Patch(facecolor="#534AB7", label="Diversity (higher = better)"),
]
ax.legend(handles=legend_elements, fontsize=9, loc="upper right")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/evaluation_metrics.png", dpi=150, bbox_inches="tight")
print(f"Chart saved → {OUT_DIR}/evaluation_metrics.png")
plt.show()


# ── 7. Actual vs predicted scatter ────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("SVD Rating Prediction Quality", fontsize=13, fontweight="bold")

sample_idx = np.random.choice(len(actual_arr), min(1000, len(actual_arr)), replace=False)
ax = axes[0]
ax.scatter(actual_arr[sample_idx], pred_arr[sample_idx],
           alpha=0.25, s=8, color="#185FA5")
ax.plot([0.5, 5], [0.5, 5], color="#D85A30", linewidth=1.2,
        linestyle="--", label="Perfect prediction")
ax.set_xlabel("Actual rating")
ax.set_ylabel("Predicted rating")
ax.set_title(f"Actual vs Predicted  (RMSE={rmse:.3f})")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

errors = actual_arr - pred_arr
ax = axes[1]
ax.hist(errors, bins=60, color="#1D9E75", edgecolor="none")
ax.axvline(0, color="#D85A30", linewidth=1.2, linestyle="--", label="Zero error")
ax.set_xlabel("Error (actual − predicted)")
ax.set_ylabel("Count")
ax.set_title(f"Error distribution  (MAE={mae:.3f})")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/prediction_quality.png", dpi=150, bbox_inches="tight")
print(f"Chart saved → {OUT_DIR}/prediction_quality.png")
plt.show()