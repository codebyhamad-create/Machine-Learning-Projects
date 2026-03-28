"""
Movie Recommendation System — Step 3a: Content-Based Filtering
===============================================================
Uses TF-IDF on (genres + user tags) to build a movie similarity
matrix, then recommends the top-N most similar movies.
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

OUT_DIR = "./output"
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading movies...")
movies = pd.read_parquet(f"{OUT_DIR}/movies_master.parquet")
movies = movies.reset_index(drop=True)

print(f"Working with {len(movies):,} movies")

# ── 1. TF-IDF on content soup ────────────────────────────────────────────────
# content_soup = genres + user tags (built in Step 1)

print("\nFitting TF-IDF vectorizer...")
tfidf = TfidfVectorizer(
    analyzer    = "word",
    ngram_range = (1, 2),    # unigrams + bigrams catch "sci fi", "based on novel" etc.
    min_df      = 3,         # ignore very rare terms
    max_features= 10_000,
    stop_words  = "english",
)

tfidf_matrix = tfidf.fit_transform(movies["content_soup"])
print(f"TF-IDF matrix: {tfidf_matrix.shape}  (movies × features)")

# ── 2. Cosine similarity matrix ───────────────────────────────────────────────
# For 10K+ movies this is large — we compute on demand in production,
# but precompute here for demo speed (requires ~1 GB RAM for full set).

print("\nComputing cosine similarity matrix (may take a moment)...")
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
print(f"Similarity matrix: {cosine_sim.shape}")

# Build title → index lookup (case-insensitive)
title_to_idx = pd.Series(movies.index, index=movies["title_clean"].str.lower())

# ── 3. Recommendation function ────────────────────────────────────────────────

def get_content_recommendations(title: str, n: int = 10) -> pd.DataFrame:
    """
    Return top-N content-similar movies for a given title.

    Parameters
    ----------
    title : str   Movie title (partial match supported)
    n     : int   Number of recommendations to return

    Returns
    -------
    pd.DataFrame  Ranked recommendations with similarity scores
    """
    title_lower = title.lower().strip()

    # Fuzzy match — find closest title
    matches = [t for t in title_to_idx.index if title_lower in t]
    if not matches:
        print(f"  '{title}' not found. Try a different title.")
        return pd.DataFrame()

    matched_title = matches[0]
    idx = title_to_idx[matched_title]

    # Similarity scores for this movie vs all others
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1 : n + 1]   # exclude itself

    movie_indices  = [s[0] for s in sim_scores]
    similarity_vals = [round(s[1], 4) for s in sim_scores]

    result = movies.iloc[movie_indices][
        ["title_clean", "year", "genres", "avg_rating", "rating_count"]
    ].copy()
    result["similarity"] = similarity_vals
    result = result.reset_index(drop=True)
    result.index += 1   # rank from 1

    return result


# ── 4. Demo ───────────────────────────────────────────────────────────────────

test_movies = ["The Dark Knight", "Toy Story", "Inception", "The Silence of the Lambs"]

for movie in test_movies:
    print(f"\n{'═'*60}")
    print(f"  Top 8 recommendations for: {movie}")
    print(f"{'═'*60}")
    recs = get_content_recommendations(movie, n=8)
    if not recs.empty:
        print(recs[["title_clean", "year", "genres", "avg_rating", "similarity"]].to_string())


# ── 5. Save model artifacts ───────────────────────────────────────────────────

with open(f"{OUT_DIR}/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

np.save(f"{OUT_DIR}/cosine_sim_matrix.npy", cosine_sim)
movies[["movieId", "title_clean", "year", "genres", "avg_rating", "rating_count"]]\
    .to_parquet(f"{OUT_DIR}/movies_index.parquet", index=False)

print(f"\nArtifacts saved:")
print(f"  {OUT_DIR}/tfidf_vectorizer.pkl")
print(f"  {OUT_DIR}/cosine_sim_matrix.npy")
print(f"  {OUT_DIR}/movies_index.parquet")
