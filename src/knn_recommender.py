import pandas as pd
import numpy as np
import os
import pickle

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


# ============================================================
# AI MOVIE RECOMMENDATION SYSTEM
# KNN MOVIE RECOMMENDER
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)


print("=" * 60)
print("KNN MOVIE RECOMMENDATION MODEL")
print("=" * 60)


# ------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------

print("\n[1/6] Loading movie data...")

metadata_file = os.path.join(
    DATA_DIR,
    "movie_metadata.csv"
)

features_file = os.path.join(
    DATA_DIR,
    "movie_features.csv"
)

metadata = pd.read_csv(metadata_file)
features = pd.read_csv(features_file)

print(f"Movies loaded: {len(metadata):,}")
print(f"Features loaded: {features.shape}")


# ------------------------------------------------------------
# 2. Select movies with reliable ratings
# ------------------------------------------------------------

print("\n[2/6] Selecting reliable movies...")

reliable_mask = metadata["reliable_rating"] == 1

metadata_knn = metadata[
    reliable_mask
].reset_index(drop=True)

features_knn = features.loc[
    reliable_mask
].reset_index(drop=True)

print(
    f"Movies used by KNN: "
    f"{len(metadata_knn):,}"
)


# ------------------------------------------------------------
# 3. Prepare feature matrix
# ------------------------------------------------------------

print("\n[3/6] Preparing feature matrix...")

# Features created during feature engineering
X = features_knn.values.astype(
    np.float32
)

print(
    f"KNN feature matrix: {X.shape}"
)


# ------------------------------------------------------------
# 4. Train KNN
# ------------------------------------------------------------

print("\n[4/6] Training KNN model...")

knn = NearestNeighbors(
    n_neighbors=11,
    metric="cosine",
    algorithm="brute",
    n_jobs=-1
)

knn.fit(X)

print("KNN model trained successfully.")


# ------------------------------------------------------------
# 5. Save model and metadata
# ------------------------------------------------------------

print("\n[5/6] Saving KNN model...")

model_file = os.path.join(
    MODEL_DIR,
    "knn_movie_model.pkl"
)

metadata_file_output = os.path.join(
    MODEL_DIR,
    "knn_movie_metadata.csv"
)

with open(model_file, "wb") as file:
    pickle.dump(knn, file)

metadata_knn.to_csv(
    metadata_file_output,
    index=False
)

print(f"Model saved: {model_file}")
print(
    f"Metadata saved: "
    f"{metadata_file_output}"
)


# ------------------------------------------------------------
# 6. Recommendation function
# ------------------------------------------------------------

def recommend_movies(movie_title, number_of_recommendations=10):

    # Search for movie title
    matches = metadata_knn[
        metadata_knn["primaryTitle"]
        .str.lower()
        .str.contains(
            movie_title.lower(),
            na=False
        )
    ]

    if matches.empty:
        print(
            f"\nMovie '{movie_title}' "
            "was not found."
        )
        return

    # Use first matching movie
    movie_index = matches.index[0]

    movie_name = metadata_knn.loc[
        movie_index,
        "primaryTitle"
    ]

    movie_vector = X[
        movie_index
    ].reshape(1, -1)

    distances, indices = knn.kneighbors(
        movie_vector,
        n_neighbors=number_of_recommendations + 1
    )

    recommendation_indices = indices[0][1:]
    recommendation_distances = distances[0][1:]

    recommendations = metadata_knn.iloc[
        recommendation_indices
    ].copy()

    # Convert cosine distance to similarity percentage
    recommendations["similarity"] = (
        1 - recommendation_distances
    ) * 100

    recommendations["similarity"] = (
        recommendations["similarity"]
        .round(2)
    )

    print("\n" + "=" * 60)
    print(
        f"RECOMMENDATIONS FOR: {movie_name}"
    )
    print("=" * 60)

    display_columns = [
        "primaryTitle",
        "startYear",
        "genres",
        "averageRating",
        "numVotes",
        "similarity"
    ]

    print(
        recommendations[
            display_columns
        ].to_string(index=False)
    )

    return recommendations


# ------------------------------------------------------------
# Test the recommender
# ------------------------------------------------------------

print("\n[6/6] Testing KNN recommender...")

recommend_movies(
    "The Dark Knight",
    10
)

print("\n" + "=" * 60)
print("KNN RECOMMENDER COMPLETED")
print("=" * 60)