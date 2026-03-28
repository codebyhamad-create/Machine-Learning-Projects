"""
Movie Recommendation System — Step 3b: Collaborative Filtering (SVD)
=====================================================================
Pure numpy/scipy implementation — no scikit-surprise needed.
Uses Truncated SVD on the user-item ratings matrix to learn
latent factors, then predicts ratings for unseen movies.
"""

import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_absolute_error, mean_squared_error

OUT_DIR = "./output"
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading cleaned ratings...")
ratings      = pd.read_parquet(f"{OUT_DIR}/ratings_clean.parquet")
movies_index = pd.read_parquet(f"{OUT_DIR}/movies_index.parquet")

# ── 1. Subsample for speed ────────────────────────────────────────────────────
# Remove this block to train on all 138K users (slower but more accurate)

SAMPLE_USERS = 10_000
sampled_users  = ratings["userId"].drop_duplicates().sample(SAMPLE_USERS, random_state=42)
ratings_sample = ratings[ratings["userId"].isin(sampled_users)].copy()
print(f"Using {len(ratings_sample):,} ratings from {SAMPLE_USERS:,} sampled users")

# ── 2. Build user-item matrix ─────────────────────────────────────────────────

print("\nBuilding user-item matrix...")

ratings_sample["user_idx"]  = ratings_sample["userId"].astype("category").cat.codes
ratings_sample["movie_idx"] = ratings_sample["movieId"].astype("category").cat.codes

n_users  = ratings_sample["user_idx"].nunique()
n_movies = ratings_sample["movie_idx"].nunique()
print(f"Matrix size: {n_users} users × {n_movies} movies")

sparse_matrix = csr_matrix(
    (ratings_sample["rating"],
     (ratings_sample["user_idx"], ratings_sample["movie_idx"])),
    shape=(n_users, n_movies)
)

# Mean-centre per user before SVD
user_ratings_mean = np.array(sparse_matrix.mean(axis=1)).flatten()
matrix_demeaned   = sparse_matrix - user_ratings_mean.reshape(-1, 1)

# ── 3. Truncated SVD ──────────────────────────────────────────────────────────

print("\nRunning Truncated SVD (k=50 factors)...")
K = 50

U, sigma, Vt = svds(matrix_demeaned.astype(np.float32), k=K)
sigma_diag    = np.diag(sigma)

print("Reconstructing predicted ratings matrix...")
predicted_ratings = U @ sigma_diag @ Vt + user_ratings_mean.reshape(-1, 1)
predicted_df      = pd.DataFrame(
    predicted_ratings,
    columns=ratings_sample["movie_idx"].astype("category").cat.categories,
)
print(f"Predicted matrix shape: {predicted_df.shape}")

# ── 4. Evaluation ─────────────────────────────────────────────────────────────

print("\nEvaluating on known ratings (sample of 50K)...")

eval_sample    = ratings_sample.sample(min(50_000, len(ratings_sample)), random_state=42)
actual_ratings = []
pred_ratings   = []

for _, row in eval_sample.iterrows():
    u = row["user_idx"]
    m = row["movie_idx"]
    actual_ratings.append(row["rating"])
    pred_ratings.append(np.clip(predicted_df.iloc[u][m], 0.5, 5.0))

rmse = np.sqrt(mean_squared_error(actual_ratings, pred_ratings))
mae  = mean_absolute_error(actual_ratings, pred_ratings)

print(f"\n── SVD Evaluation Metrics ──")
print(f"  RMSE : {rmse:.4f}")
print(f"  MAE  : {mae:.4f}")

# ── 5. Lookup helpers ─────────────────────────────────────────────────────────

movieid_to_idx = dict(zip(
    ratings_sample["movieId"].astype("category").cat.categories,
    range(n_movies)
))
userid_to_idx = dict(zip(
    ratings_sample["userId"].astype("category").cat.categories,
    range(n_users)
))

# ── 6. Recommendation function ────────────────────────────────────────────────

def get_cf_recommendations(user_id: int, n: int = 10) -> pd.DataFrame:
    """
    Return top-N predicted-rating movies for a user,
    excluding movies they have already rated.
    """
    if user_id not in userid_to_idx:
        print(f"  User {user_id} not in training sample. Try another user ID.")
        return pd.DataFrame()

    u_idx = userid_to_idx[user_id]
    rated_movie_ids = set(
        ratings_sample[ratings_sample["userId"] == user_id]["movieId"].tolist()
    )

    user_preds = predicted_df.iloc[u_idx]
    results = []
    for movie_id, m_idx in movieid_to_idx.items():
        if movie_id in rated_movie_ids:
            continue
        pred_score = np.clip(user_preds.iloc[m_idx], 0.5, 5.0)
        results.append((movie_id, round(float(pred_score), 3)))

    results_sorted = sorted(results, key=lambda x: x[1], reverse=True)[:n]

    rec_ids      = [r[0] for r in results_sorted]
    pred_ratings_out = [r[1] for r in results_sorted]

    result_df = movies_index[movies_index["movieId"].isin(rec_ids)].copy()
    pred_map  = dict(zip(rec_ids, pred_ratings_out))
    result_df["predicted_rating"] = result_df["movieId"].map(pred_map)
    result_df = result_df.sort_values("predicted_rating", ascending=False).reset_index(drop=True)
    result_df.index += 1
    return result_df


# ── 7. Demo ───────────────────────────────────────────────────────────────────

sample_user = int(ratings_sample["userId"].iloc[0])
n_rated     = len(ratings_sample[ratings_sample["userId"] == sample_user])

print(f"\n{'═'*55}")
print(f"  SVD Recommendations for User {sample_user}")
print(f"  (has rated {n_rated} movies in sample)")
print(f"{'═'*55}")

recs = get_cf_recommendations(sample_user, n=10)
if not recs.empty:
    print(recs[["title_clean", "year", "genres", "predicted_rating"]].to_string())


# ── 8. Results plot ───────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("SVD Collaborative Filtering Results", fontsize=13, fontweight="bold")

sample_idx = np.random.choice(len(actual_ratings), 500, replace=False)
ax = axes[0]
ax.scatter(
    [actual_ratings[i] for i in sample_idx],
    [pred_ratings[i]   for i in sample_idx],
    alpha=0.3, s=8, color="#185FA5"
)
ax.plot([0.5, 5], [0.5, 5], color="#D85A30", linewidth=1.2, linestyle="--", label="Perfect")
ax.set_xlabel("Actual rating")
ax.set_ylabel("Predicted rating")
ax.set_title(f"Actual vs Predicted\nRMSE={rmse:.3f}  MAE={mae:.3f}")
ax.legend()
ax.grid(alpha=0.3)

errors = np.array(actual_ratings) - np.array(pred_ratings)
ax = axes[1]
ax.hist(errors, bins=60, color="#1D9E75", edgecolor="none")
ax.axvline(0, color="#D85A30", linewidth=1.2, linestyle="--")
ax.set_xlabel("Prediction error (actual − predicted)")
ax.set_ylabel("Count")
ax.set_title("Error distribution")
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/svd_results.png", dpi=150, bbox_inches="tight")
print(f"\nPlot saved → {OUT_DIR}/svd_results.png")


# ── 9. Save artifacts ─────────────────────────────────────────────────────────

artifacts = {
    "predicted_df"   : predicted_df,
    "movieid_to_idx" : movieid_to_idx,
    "userid_to_idx"  : userid_to_idx,
    "ratings_sample" : ratings_sample,
}
with open(f"{OUT_DIR}/svd_model.pkl", "wb") as f:
    pickle.dump(artifacts, f)

print(f"Artifacts saved → {OUT_DIR}/svd_model.pkl")