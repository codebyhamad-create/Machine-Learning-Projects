"""
Movie Recommendation System — Step 1: Data Loading & Preprocessing
===================================================================
MovieLens 20M: 20M ratings, 27K movies, 138K users
Download: https://files.grouplens.org/datasets/movielens/ml-20m.zip

Expected files in ./data/ml-20m/:
  movies.csv | ratings.csv | tags.csv | genome-scores.csv | genome-tags.csv
"""

import pandas as pd
import numpy as np
import os
import re

DATA_DIR = "./data/ml-20m"
OUT_DIR  = "./output"
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Load ───────────────────────────────────────────────────────────────────

print("Loading raw files...")
movies  = pd.read_csv(f"{DATA_DIR}/movies.csv")
ratings = pd.read_csv(f"{DATA_DIR}/ratings.csv")
tags    = pd.read_csv(f"{DATA_DIR}/tags.csv")

print(f"  Movies  : {movies.shape}")
print(f"  Ratings : {ratings.shape}")
print(f"  Tags    : {tags.shape}")

# ── 2. Clean movies ───────────────────────────────────────────────────────────

movies["year"]        = movies["title"].str.extract(r"\((\d{4})\)").astype(float)
movies["title_clean"] = movies["title"].str.replace(r"\s*\(\d{4}\)", "", regex=True).str.strip()
movies["genres_list"] = movies["genres"].apply(
    lambda x: [] if x == "(no genres listed)" else x.split("|")
)
movies.drop_duplicates(subset="movieId", inplace=True)

print(f"\nYear range    : {int(movies['year'].min())} – {int(movies['year'].max())}")
print(f"Unique genres : {sorted(set(g for gl in movies['genres_list'] for g in gl))}")

# ── 3. Clean ratings ──────────────────────────────────────────────────────────

before = len(ratings)
ratings.drop_duplicates(subset=["userId", "movieId"], inplace=True)
ratings.dropna(inplace=True)
ratings["timestamp"] = pd.to_datetime(ratings["timestamp"], unit="s")
print(f"\nRemoved {before - len(ratings):,} duplicate/null ratings")
print(f"Rating scale  : {ratings['rating'].min()} – {ratings['rating'].max()}")
print(f"Unique users  : {ratings['userId'].nunique():,}")
print(f"Unique movies : {ratings['movieId'].nunique():,}")

# ── 4. Aggregate tags per movie ───────────────────────────────────────────────

tags["tag"] = tags["tag"].str.lower().str.strip()
movie_tags = (
    tags.groupby("movieId")["tag"]
    .apply(lambda x: " ".join(x.dropna().unique()))
    .reset_index()
    .rename(columns={"tag": "user_tags"})
)

# ── 5. Movie-level rating stats ───────────────────────────────────────────────

movie_stats = ratings.groupby("movieId").agg(
    avg_rating   = ("rating", "mean"),
    rating_count = ("rating", "count"),
    rating_std   = ("rating", "std"),
).reset_index().round(3)

# ── 6. Master dataframe ───────────────────────────────────────────────────────

movies_master = (
    movies
    .merge(movie_stats, on="movieId", how="left")
    .merge(movie_tags,  on="movieId", how="left")
)
movies_master["user_tags"]    = movies_master["user_tags"].fillna("")
movies_master["rating_count"] = movies_master["rating_count"].fillna(0)

# Keep only movies with enough ratings
MIN_RATINGS = 50
movies_filtered = movies_master[movies_master["rating_count"] >= MIN_RATINGS].copy()
print(f"\nMovies with >= {MIN_RATINGS} ratings: {len(movies_filtered):,}")

# ── 7. Content soup for TF-IDF ───────────────────────────────────────────────
# genres + user tags combined into one text field per movie

movies_filtered["content_soup"] = (
    movies_filtered["genres"].str.replace("|", " ", regex=False)
    + " " + movies_filtered["user_tags"]
).str.lower().str.strip()

# ── 8. Save ───────────────────────────────────────────────────────────────────

movies_filtered.to_parquet(f"{OUT_DIR}/movies_master.parquet", index=False)
ratings.to_parquet(f"{OUT_DIR}/ratings_clean.parquet", index=False)

print(f"\nSaved → {OUT_DIR}/movies_master.parquet  ({len(movies_filtered):,} movies)")
print(f"Saved → {OUT_DIR}/ratings_clean.parquet  ({len(ratings):,} ratings)")
