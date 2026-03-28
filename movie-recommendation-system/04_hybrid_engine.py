"""
Movie Recommendation System — Step 4: Hybrid Engine
=====================================================
Combines content-based similarity scores with SVD predicted ratings
using a weighted blend.

  - Content-based : works for any movie, no user needed
  - Collaborative : personalises based on user's taste
  - Hybrid        : weighted blend of both

NOTE: Uses the numpy/scipy SVD artifacts from 03b (no scikit-surprise).
"""

import pandas as pd
import numpy as np
import pickle
import os

OUT_DIR = "./output"

# ── Load artifacts ────────────────────────────────────────────────────────────

print("Loading model artifacts...")
cosine_sim   = np.load(f"{OUT_DIR}/cosine_sim_matrix.npy")
movies       = pd.read_parquet(f"{OUT_DIR}/movies_master.parquet").reset_index(drop=True)
movies_index = pd.read_parquet(f"{OUT_DIR}/movies_index.parquet")

with open(f"{OUT_DIR}/svd_model.pkl", "rb") as f:
    svd_artifacts = pickle.load(f)

predicted_df    = svd_artifacts["predicted_df"]
movieid_to_idx  = svd_artifacts["movieid_to_idx"]
userid_to_idx   = svd_artifacts["userid_to_idx"]
ratings_sample  = svd_artifacts["ratings_sample"]

title_to_idx = pd.Series(movies.index, index=movies["title_clean"].str.lower())

print(f"Loaded: {len(movies):,} movies | cosine matrix {cosine_sim.shape}")
print(f"SVD predicted matrix: {predicted_df.shape}")


# ── Helper: get SVD predicted rating ─────────────────────────────────────────

def svd_predict(user_id: int, movie_id: int) -> float:
    """Look up the SVD predicted rating for a user-movie pair."""
    if user_id not in userid_to_idx or movie_id not in movieid_to_idx:
        return np.nan
    u_idx = userid_to_idx[user_id]
    m_idx = movieid_to_idx[movie_id]
    score = float(predicted_df.iloc[u_idx, m_idx])
    return np.clip(score, 0.5, 5.0)


# ── Hybrid recommender ────────────────────────────────────────────────────────

def hybrid_recommend(
    title          : str,
    user_id        : int   = None,
    n              : int   = 10,
    content_weight : float = 0.5,
    collab_weight  : float = 0.5,
) -> pd.DataFrame:
    """
    Hybrid movie recommender.

    - If user_id is None  → pure content-based
    - If user_id provided → weighted blend of content similarity + SVD score

    Parameters
    ----------
    title          : str    Seed movie title (partial match OK)
    user_id        : int    Optional user ID for personalisation
    n              : int    Number of results to return
    content_weight : float  Weight for content similarity  (0–1)
    collab_weight  : float  Weight for SVD predicted rating (0–1)
                            content_weight + collab_weight must equal 1.0
    """
    assert abs(content_weight + collab_weight - 1.0) < 1e-6, \
        "content_weight + collab_weight must equal 1.0"

    title_lower = title.lower().strip()
    matches = [t for t in title_to_idx.index if title_lower in t]

    if not matches:
        print(f"  '{title}' not found. Try a different title.")
        return pd.DataFrame()

    matched = matches[0]
    idx = title_to_idx[matched]

    # Get content similarity scores for all movies vs the seed
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1 : n * 5]

    candidate_indices = [s[0] for s in sim_scores]
    candidates = movies.iloc[candidate_indices].copy()
    candidates["content_score"] = [s[1] for s in sim_scores]

    if user_id is not None and user_id in userid_to_idx:
        # Predict rating for each candidate and normalise to 0-1
        candidates["predicted_rating"] = candidates["movieId"].apply(
            lambda mid: svd_predict(user_id, mid)
        )
        # Normalise: rating scale 0.5-5.0 → 0.0-1.0
        candidates["collab_score"] = (candidates["predicted_rating"] - 0.5) / 4.5
        candidates["collab_score"] = candidates["collab_score"].fillna(0)

        candidates["hybrid_score"] = (
            content_weight * candidates["content_score"] +
            collab_weight  * candidates["collab_score"]
        )
    elif user_id is not None and user_id not in userid_to_idx:
        print(f"  User {user_id} not in training sample — falling back to content-based.")
        candidates["predicted_rating"] = np.nan
        candidates["collab_score"]     = np.nan
        candidates["hybrid_score"]     = candidates["content_score"]
    else:
        candidates["predicted_rating"] = np.nan
        candidates["collab_score"]     = np.nan
        candidates["hybrid_score"]     = candidates["content_score"]

    result = (
        candidates
        .sort_values("hybrid_score", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )
    result.index += 1

    display_cols = ["title_clean", "year", "genres", "avg_rating", "content_score", "hybrid_score"]
    if user_id is not None and user_id in userid_to_idx:
        display_cols.insert(-1, "predicted_rating")

    return result[display_cols].round(4)


# ── Demo ──────────────────────────────────────────────────────────────────────

# Pick a user that exists in the SVD training sample
sample_user = int(ratings_sample["userId"].iloc[0])

print(f"\n{'═'*60}")
print("  Pure Content-Based: 'The Dark Knight'")
print(f"{'═'*60}")
print(hybrid_recommend("The Dark Knight", n=8).to_string())

print(f"\n{'═'*60}")
print(f"  Hybrid (User {sample_user}): 'The Dark Knight'")
print(f"{'═'*60}")
print(hybrid_recommend(
    "The Dark Knight",
    user_id        = sample_user,
    n              = 8,
    content_weight = 0.4,
    collab_weight  = 0.6,
).to_string())

print(f"\n{'═'*60}")
print("  Pure Content-Based: 'Toy Story'")
print(f"{'═'*60}")
print(hybrid_recommend("Toy Story", n=8).to_string())

print(f"\n{'═'*60}")
print(f"  Hybrid (User {sample_user}): 'Toy Story'")
print(f"{'═'*60}")
print(hybrid_recommend(
    "Toy Story",
    user_id        = sample_user,
    n              = 8,
    content_weight = 0.4,
    collab_weight  = 0.6,
).to_string())