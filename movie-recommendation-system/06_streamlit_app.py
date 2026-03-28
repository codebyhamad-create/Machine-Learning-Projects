"""
Movie Recommendation System — Step 6: Streamlit App
====================================================
Interactive UI for exploring recommendations.
Run with: streamlit run 06_streamlit_app.py

Install: pip install streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

OUT_DIR = "./output"

st.set_page_config(
    page_title = "Movie Recommender",
    page_icon  = "🎬",
    layout     = "wide",
)

# ── Load artifacts (cached) ───────────────────────────────────────────────────

@st.cache_resource
def load_artifacts():
    movies       = pd.read_parquet(f"{OUT_DIR}/movies_master.parquet").reset_index(drop=True)
    cosine_sim   = np.load(f"{OUT_DIR}/cosine_sim_matrix.npy")
    movies_index = pd.read_parquet(f"{OUT_DIR}/movies_index.parquet")

    with open(f"{OUT_DIR}/svd_model.pkl", "rb") as f:
        svd = pickle.load(f)

    title_to_idx = pd.Series(movies.index, index=movies["title_clean"].str.lower())
    return movies, cosine_sim, movies_index, svd, title_to_idx

movies, cosine_sim, movies_index, svd, title_to_idx = load_artifacts()


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("🎬 Movie Recommender")
st.sidebar.markdown("**MovieLens 20M Dataset**")
st.sidebar.markdown(f"- {len(movies):,} movies")
st.sidebar.markdown("- Content-based + Collaborative SVD")

mode = st.sidebar.radio("Mode", ["Content-Based", "Hybrid (Personalised)"])
n_recs = st.sidebar.slider("Number of recommendations", 5, 20, 10)

if mode == "Hybrid (Personalised)":
    user_id = st.sidebar.number_input("User ID", min_value=1, max_value=138493, value=1)
    content_w = st.sidebar.slider("Content weight", 0.0, 1.0, 0.4, 0.1)
    collab_w  = round(1.0 - content_w, 1)
    st.sidebar.markdown(f"Collaborative weight: **{collab_w}**")

# ── Main UI ───────────────────────────────────────────────────────────────────

st.title("🎬 Movie Recommendation System")
st.markdown("Built on **MovieLens 20M** · TF-IDF Content Filtering + SVD Collaborative Filtering")

# Filters
col1, col2, col3 = st.columns([3, 1, 1])

with col1:
    all_titles = sorted(movies["title_clean"].dropna().tolist())
    seed_movie = st.selectbox("Seed movie", all_titles, index=all_titles.index("The Dark Knight")
                              if "The Dark Knight" in all_titles else 0)

with col2:
    genre_options = ["All"] + sorted(set(g for gl in movies["genres_list"] for g in gl))
    genre_filter  = st.selectbox("Filter genre", genre_options)

with col3:
    min_year, max_year = int(movies["year"].min()), int(movies["year"].max())
    year_range = st.slider("Release year", min_year, max_year, (1990, max_year))

if st.button("Get Recommendations", type="primary"):
    with st.spinner("Computing recommendations..."):

        title_lower = seed_movie.lower()
        idx = title_to_idx.get(title_lower)

        if idx is None:
            st.error(f"Movie not found: {seed_movie}")
        else:
            # Content scores
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1 : n_recs * 5]
            candidate_idxs = [s[0] for s in sim_scores]
            candidates = movies.iloc[candidate_idxs].copy()
            candidates["content_score"] = [s[1] for s in sim_scores]

            # Genre filter
            if genre_filter != "All":
                candidates = candidates[
                    candidates["genres_list"].apply(lambda gl: genre_filter in gl)
                ]

            # Year filter
            candidates = candidates[
                candidates["year"].between(year_range[0], year_range[1])
            ]

            if mode == "Hybrid (Personalised)":
                candidates["predicted_rating"] = candidates["movieId"].apply(
                    lambda mid: svd.predict(user_id, mid).est
                )
                collab_norm = (candidates["predicted_rating"] - 0.5) / 4.5
                candidates["score"] = (
                    content_w * candidates["content_score"] +
                    collab_w  * collab_norm
                )
            else:
                candidates["score"] = candidates["content_score"]

            recs = candidates.sort_values("score", ascending=False).head(n_recs)

            # ── Display results ────────────────────────────────────────────────

            st.markdown(f"### Recommendations for: **{seed_movie}**")

            seed_info = movies[movies["title_clean"].str.lower() == title_lower].iloc[0]
            st.markdown(
                f"📽 *{seed_info['genres']}* &nbsp;|&nbsp; "
                f"⭐ {seed_info['avg_rating']:.2f} &nbsp;|&nbsp; "
                f"🗓 {int(seed_info['year']) if not pd.isna(seed_info['year']) else 'N/A'}"
            )
            st.divider()

            for rank, (_, row) in enumerate(recs.iterrows(), 1):
                with st.container():
                    c1, c2, c3, c4, c5 = st.columns([0.4, 3, 2, 1.2, 1.2])
                    c1.markdown(f"**#{rank}**")
                    c2.markdown(f"**{row['title_clean']}**")
                    c3.markdown(f"*{row['genres']}*")
                    yr = int(row['year']) if not pd.isna(row.get('year', np.nan)) else "—"
                    c4.markdown(f"📅 {yr} · ⭐ {row['avg_rating']:.2f}")
                    c5.markdown(f"Score: `{row['score']:.3f}`")

            # ── Stats panel ────────────────────────────────────────────────────

            st.divider()
            s1, s2, s3 = st.columns(3)
            s1.metric("Movies searched", f"{len(candidates):,}")
            s2.metric("Top genre in recs",
                      pd.Series([g for gl in recs["genres_list"] for g in gl]).value_counts().index[0])
            s3.metric("Avg rating of recs", f"{recs['avg_rating'].mean():.2f}")

else:
    st.info("👆 Select a seed movie above and click **Get Recommendations**")

# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
st.markdown(
    "<small>Data: MovieLens 20M (GroupLens) · "
    "Models: TF-IDF cosine similarity + SVD (scikit-surprise)</small>",
    unsafe_allow_html=True
)
