"""
recommendation_model.py
-----------------------
Collaborative Filtering Recommender using User-Based Cosine Similarity.
Dataset: Synthetic movie ratings (10 users × 12 movies).
Pure numpy — no sklearn dependency.
"""

import numpy as np
import json

np.random.seed(99)

# ── Dataset ──────────────────────────────────────────────
movies = [
    "Inception",        # 0  sci-fi
    "The Dark Knight",  # 1  action
    "Interstellar",     # 2  sci-fi
    "Avengers",         # 3  action
    "The Notebook",     # 4  romance
    "Pride & Prejudice",# 5  romance
    "The Shining",      # 6  horror
    "Get Out",          # 7  horror
    "Parasite",         # 8  thriller
    "Whiplash",         # 9  drama
    "La La Land",       # 10 musical
    "Toy Story",        # 11 animation
]

genres = ["sci-fi","action","sci-fi","action","romance",
          "romance","horror","horror","thriller","drama","musical","animation"]

# Ratings matrix: 10 users × 12 movies (0 = not rated)
ratings = np.array([
    [5, 4, 5, 3, 0, 0, 2, 0, 4, 3, 0, 0],  # User 0 – sci-fi fan
    [4, 5, 3, 5, 0, 0, 0, 3, 0, 0, 0, 0],  # User 1 – action fan
    [0, 0, 0, 0, 5, 5, 0, 0, 0, 4, 5, 0],  # User 2 – romance/drama
    [0, 0, 0, 0, 0, 0, 5, 5, 4, 0, 0, 0],  # User 3 – horror fan
    [5, 3, 4, 0, 0, 0, 0, 0, 5, 5, 0, 0],  # User 4 – thriller/sci-fi
    [0, 0, 0, 3, 5, 4, 0, 0, 0, 0, 5, 4],  # User 5 – family/romance
    [3, 5, 0, 5, 0, 0, 4, 3, 0, 0, 0, 0],  # User 6 – action/horror
    [0, 0, 5, 0, 4, 0, 0, 0, 5, 4, 0, 0],  # User 7 – mixed
    [0, 0, 0, 0, 0, 5, 4, 0, 0, 5, 5, 3],  # User 8 – drama/romance
    [4, 0, 4, 0, 0, 0, 3, 4, 4, 0, 0, 0],  # User 9 – sci-fi/horror
], dtype=float)

user_names = [f"User {i}" for i in range(10)]
n_users, n_movies = ratings.shape


# ── Cosine similarity (user-based) ──────────────────────
def cosine_similarity(a, b):
    """Similarity between two rating vectors (ignore unrated items)."""
    mask = (a > 0) & (b > 0)
    if mask.sum() == 0:
        return 0.0
    a_, b_ = a[mask], b[mask]
    denom = np.linalg.norm(a_) * np.linalg.norm(b_)
    return float(np.dot(a_, b_) / denom) if denom else 0.0


def compute_similarity_matrix(R):
    n = R.shape[0]
    S = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            S[i, j] = cosine_similarity(R[i], R[j])
    return S

sim_matrix = compute_similarity_matrix(ratings)


# ── Predict ratings for unrated items ───────────────────
def predict_ratings(user_idx, top_k=4):
    sims   = sim_matrix[user_idx].copy()
    sims[user_idx] = 0          # exclude self
    top_k_users = np.argsort(sims)[::-1][:top_k]

    unrated = np.where(ratings[user_idx] == 0)[0]
    predictions = {}
    for m in unrated:
        num, den = 0.0, 0.0
        for u in top_k_users:
            if ratings[u, m] > 0:
                num += sims[u] * ratings[u, m]
                den += abs(sims[u])
        if den > 0:
            predictions[int(m)] = round(num / den, 3)
    return dict(sorted(predictions.items(), key=lambda x: x[1], reverse=True))


# ── Generate recommendations for all users ──────────────
all_recs = {}
for uid in range(n_users):
    preds = predict_ratings(uid, top_k=4)
    top_recs = list(preds.items())[:5]
    all_recs[uid] = [
        {
            "movie_idx": idx,
            "movie":     movies[idx],
            "genre":     genres[idx],
            "score":     score,
        }
        for idx, score in top_recs
    ]


# ── Coverage metric ──────────────────────────────────────
rated_items   = int(np.sum(ratings > 0))
unrated_items = int(np.sum(ratings == 0))
avg_ratings   = round(float(np.mean(ratings[ratings > 0])), 3)
sparsity      = round(1 - rated_items / (n_users * n_movies), 4)


# ── Export ───────────────────────────────────────────────
sim_export = [[round(float(v), 3) for v in row] for row in sim_matrix]

export = {
    "movies":      movies,
    "genres":      genres,
    "user_names":  user_names,
    "ratings":     ratings.tolist(),
    "similarity":  sim_export,
    "recommendations": {str(k): v for k, v in all_recs.items()},
    "stats": {
        "rated_items":   rated_items,
        "unrated_items": unrated_items,
        "avg_rating":    avg_ratings,
        "sparsity":      sparsity,
        "n_users":       n_users,
        "n_movies":      n_movies,
    },
}

with open("static/recommendation_data.json", "w") as f:
    json.dump(export, f, indent=2)


if __name__ == "__main__":
    print("=== Collaborative Filtering Recommender ===")
    print(f"  Users: {n_users}  |  Movies: {n_movies}")
    print(f"  Sparsity: {sparsity:.1%}  |  Avg rating: {avg_ratings}")
    print()
    for uid in range(3):
        recs = all_recs[uid][:3]
        titles = ", ".join(f"{r['movie']} ({r['score']:.2f})" for r in recs)
        print(f"  User {uid}: {titles}")
    print("\nData exported to static/recommendation_data.json")
