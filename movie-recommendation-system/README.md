# Movie Recommendation System — MovieLens 20M

## Prepared by: Hamadullah Rajpar

## Before you start having a look at it make sure to know that, I have not uploaded Output folder along with Dataset Folder. They both are heavy in size that's the reason for not uploading it. In order to see them you must download the Dataset following the link given below this text and to see outputs you must run all the files in order to have them in your own pc. Thanks A lot.

End-to-end recommendation engine: data prep → EDA → content-based → collaborative SVD → hybrid → Streamlit UI.

**Dataset**: [MovieLens 20M](https://grouplens.org/datasets/movielens/20m/)
— 20M ratings · 27K movies · 138K users

---

## Setup

```bash
pip install -r requirements.txt
```

Download and unzip the dataset:
```bash
wget https://files.grouplens.org/datasets/movielens/ml-20m.zip
unzip ml-20m.zip -d data/
# Expected: data/ml-20m/movies.csv, ratings.csv, tags.csv, genome-scores.csv
```

---

## Run order

```bash
python 01_data_prep.py          # Clean + merge → output/movies_master.parquet
python 02_eda.py                # 6-panel EDA   → output/eda_plots.png
python 03a_content_based.py     # TF-IDF + cosine similarity
python 03b_collaborative_svd.py # SVD collaborative filtering
python 04_hybrid_engine.py      # Weighted content + SVD blend
python 05_evaluation.py         # RMSE, Precision@K, Coverage, Diversity
streamlit run 06_streamlit_app.py  # Interactive UI
```

---

## Architecture

```
MovieLens 20M
│
├── 01_data_prep.py
│   ├─ Clean movies, ratings, tags
│   ├─ Extract year, genres list
│   ├─ Aggregate user tags per movie
│   ├─ Compute avg_rating, rating_count per movie
│   └─ Build content_soup = genres + tags
│
├── 03a_content_based.py
│   ├─ TF-IDF (unigram + bigram, 10K features) on content_soup
│   ├─ Cosine similarity matrix (movies × movies)
│   └─ get_content_recommendations(title, n)
│
├── 03b_collaborative_svd.py
│   ├─ SVD: 100 latent factors, 20 epochs
│   ├─ 5-fold cross-validation → RMSE / MAE
│   └─ get_cf_recommendations(user_id, n)
│
├── 04_hybrid_engine.py
│   └─ hybrid_recommend(title, user_id, content_weight, collab_weight)
│
├── 05_evaluation.py
│   ├─ RMSE / MAE (collaborative)
│   ├─ Precision@K, Recall@K
│   ├─ Catalog coverage
│   └─ Intra-list diversity
│
└── 06_streamlit_app.py
    ├─ Movie selector with genre + year filters
    ├─ Content-based or Hybrid mode toggle
    └─ User ID input for personalised recommendations
```

---

## Output files

| File | Description |
|---|---|
| `output/movies_master.parquet` | Cleaned master movie table |
| `output/ratings_clean.parquet` | Cleaned ratings |
| `output/tfidf_vectorizer.pkl` | Fitted TF-IDF vectorizer |
| `output/cosine_sim_matrix.npy` | Precomputed cosine similarity |
| `output/svd_model.pkl` | Trained SVD model |
| `output/movies_index.parquet` | Movie lookup table |
| `output/eda_plots.png` | EDA visualisation |
| `output/evaluation_metrics.png` | Metrics bar chart |

---

## Key design decisions

- **TF-IDF over raw counts** — tf-idf downweights very common genre terms (Drama, Comedy) so rare meaningful tags drive similarity more.
- **Bigrams** — captures "sci fi", "based on novel", "kung fu" as single features.
- **MIN_RATINGS = 50** — removes cold-start noise from movies with very few ratings.
- **SVD over ALS** — lower memory footprint; well-tested on MovieLens; interpretable latent factors.
- **Hybrid blend** — content weight 0.4 + collab weight 0.6 works well in practice; tune via slider in the Streamlit app.
