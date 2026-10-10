import pandas as pd
import numpy as np
import os
import pickle


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("AI HYBRID MOVIE RECOMMENDATION ENGINE")
print("=" * 70)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n[1/7] Loading movie data...")

metadata_file = os.path.join(
    MODEL_DIR,
    "knn_movie_metadata.csv"
)

features_file = os.path.join(
    DATA_DIR,
    "movie_features.csv"
)

cluster_file = os.path.join(
    MODEL_DIR,
    "movie_clusters.csv"
)

metadata = pd.read_csv(metadata_file)

features = pd.read_csv(features_file)

clusters = pd.read_csv(cluster_file)

print(f"KNN movies: {len(metadata):,}")
print(f"Feature matrix: {features.shape}")
print(f"Cluster records: {len(clusters):,}")


# ============================================================
# 2. LOAD KNN MODEL
# ============================================================

print("\n[2/7] Loading KNN model...")

knn_model_file = os.path.join(
    MODEL_DIR,
    "knn_movie_model.pkl"
)

with open(knn_model_file, "rb") as file:
    knn = pickle.load(file)

print("KNN model loaded successfully.")


# ============================================================
# 3. PREPARE FEATURES
# ============================================================

print("\n[3/7] Preparing recommendation features...")

# movie_features.csv does not contain tconst.
# The feature rows are in the same order as the original
# processed movie dataset.
#
# knn_movie_metadata.csv contains only the 143,816 reliable
# movies used to train the KNN model.
#
# Therefore we use the reliable_rating information from the
# full movie metadata to select the same rows from features.

full_metadata_file = os.path.join(
    DATA_DIR,
    "movie_metadata.csv"
)

full_metadata = pd.read_csv(
    full_metadata_file
)

print(
    f"Full movie metadata: "
    f"{len(full_metadata):,}"
)

# Make sure the number of rows matches
if len(full_metadata) != len(features):

    raise ValueError(
        "Row count mismatch between "
        "movie_metadata.csv and movie_features.csv."
    )

# Identify reliable movies
reliable_mask = (
    full_metadata["reliable_rating"] == 1
)

# Select exactly the same reliable rows
features_knn = features[
    reliable_mask
].reset_index(drop=True)

# Remove non-feature columns if they exist
non_feature_columns = [
    "tconst",
    "reliable_rating"
]

feature_columns = [
    column
    for column in features_knn.columns
    if column not in non_feature_columns
]

X = features_knn[
    feature_columns
].values.astype(
    np.float32
)

print(
    f"KNN feature matrix: {X.shape}"
)

print(
    f"Expected KNN features: "
    f"{knn.n_features_in_}"
)

# Safety check
if X.shape[1] != knn.n_features_in_:

    raise ValueError(
        f"Feature mismatch! "
        f"KNN expects "
        f"{knn.n_features_in_} features "
        f"but received "
        f"{X.shape[1]}."
    )

print(
    "Feature dimensions match successfully."
)


# ============================================================
# 4. CONNECT K-MEANS CLUSTERS
# ============================================================

print("\n[4/7] Connecting movie clusters...")

cluster_information = clusters[
    ["tconst", "cluster"]
].copy()

metadata = metadata.merge(
    cluster_information,
    on="tconst",
    how="left"
)

print(
    "Cluster information connected successfully."
)


# ============================================================
# 5. HYBRID RECOMMENDER
# ============================================================

print(
    "\n[5/7] Creating hybrid recommendation engine..."
)


def recommend_movies(
    movie_title,
    number_of_recommendations=10
):

    print("\n" + "=" * 70)

    print(
        f"HYBRID RECOMMENDATIONS FOR: "
        f"{movie_title}"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # FIND MOVIE
    # --------------------------------------------------------

    exact_matches = metadata[
        metadata["primaryTitle"]
        .str.lower()
        .str.strip()
        == movie_title.lower().strip()
    ]


    if not exact_matches.empty:

        selected_movie = exact_matches.iloc[0]

    else:

        partial_matches = metadata[
            metadata["primaryTitle"]
            .str.lower()
            .str.contains(
                movie_title.lower(),
                na=False
            )
        ]

        if partial_matches.empty:

            print(
                f"\nMovie '{movie_title}' "
                f"was not found."
            )

            return None

        selected_movie = partial_matches.iloc[0]


    selected_tconst = selected_movie["tconst"]

    selected_title = selected_movie[
        "primaryTitle"
    ]

    selected_cluster = selected_movie[
        "cluster"
    ]


    print(
        f"\nSelected movie: {selected_title}"
    )

    print(
        f"Year: {selected_movie['startYear']}"
    )

    print(
        f"Genres: {selected_movie['genres']}"
    )

    print(
        f"IMDb Rating: "
        f"{selected_movie['averageRating']}"
    )

    print(
        f"Votes: "
        f"{selected_movie['numVotes']:,}"
    )

    print(
        f"K-Means Cluster: "
        f"{selected_cluster}"
    )


    # --------------------------------------------------------
    # FIND MOVIE INDEX
    # --------------------------------------------------------

    movie_indices = metadata.index[
        metadata["tconst"]
        == selected_tconst
    ]

    if len(movie_indices) == 0:

        print(
            "\nMovie index could not be found."
        )

        return None

    movie_index = movie_indices[0]


    # --------------------------------------------------------
    # CREATE MOVIE VECTOR
    # --------------------------------------------------------

    movie_vector = X[
        movie_index
    ].reshape(1, -1)


    # --------------------------------------------------------
    # KNN SEARCH
    # --------------------------------------------------------

    search_count = max(
        number_of_recommendations * 5,
        50
    )

    search_count = min(
        search_count,
        len(X)
    )


    distances, indices = knn.kneighbors(
        movie_vector,
        n_neighbors=search_count
    )


    recommendation_indices = indices[0]

    recommendation_distances = distances[0]


    # --------------------------------------------------------
    # CREATE RECOMMENDATION DATAFRAME
    # --------------------------------------------------------

    recommendations = metadata.iloc[
        recommendation_indices
    ].copy()


    recommendations[
        "knn_distance"
    ] = recommendation_distances


    recommendations[
        "knn_similarity"
    ] = (
        1
        - recommendations["knn_distance"]
    ) * 100


    # --------------------------------------------------------
    # REMOVE SELECTED MOVIE
    # --------------------------------------------------------

    recommendations = recommendations[
        recommendations["tconst"]
        != selected_tconst
    ]


    # --------------------------------------------------------
    # RATING SCORE
    # --------------------------------------------------------

    recommendations[
        "rating_score"
    ] = (
        recommendations["averageRating"]
        / 10
    ) * 100


    # --------------------------------------------------------
    # POPULARITY SCORE
    # --------------------------------------------------------

    log_votes = np.log1p(
        recommendations["numVotes"]
    )

    max_votes = log_votes.max()

    if max_votes > 0:

        recommendations[
            "popularity_score"
        ] = (
            log_votes
            / max_votes
        ) * 100

    else:

        recommendations[
            "popularity_score"
        ] = 0


    # --------------------------------------------------------
    # CLUSTER SCORE
    # --------------------------------------------------------

    recommendations[
        "cluster_score"
    ] = np.where(
        recommendations["cluster"]
        == selected_cluster,
        100,
        0
    )


    # --------------------------------------------------------
    # HYBRID SCORE
    # --------------------------------------------------------

    recommendations[
        "hybrid_score"
    ] = (

        recommendations[
            "knn_similarity"
        ] * 0.60

        +

        recommendations[
            "rating_score"
        ] * 0.20

        +

        recommendations[
            "popularity_score"
        ] * 0.10

        +

        recommendations[
            "cluster_score"
        ] * 0.10
    )


    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    recommendations = recommendations.sort_values(
        by="hybrid_score",
        ascending=False
    )


    # --------------------------------------------------------
    # TOP RECOMMENDATIONS
    # --------------------------------------------------------

    recommendations = recommendations.head(
        number_of_recommendations
    )


    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    output = recommendations[
        [
            "primaryTitle",
            "startYear",
            "genres",
            "averageRating",
            "numVotes",
            "cluster",
            "knn_similarity",
            "hybrid_score"
        ]
    ].copy()


    output[
        "knn_similarity"
    ] = output[
        "knn_similarity"
    ].round(2)


    output[
        "hybrid_score"
    ] = output[
        "hybrid_score"
    ].round(2)


    print("\n")
    print("-" * 100)

    print(
        output.to_string(
            index=False
        )
    )

    print("-" * 100)


    return recommendations


# ============================================================
# 6. TEST SYSTEM
# ============================================================

print(
    "\n[6/7] Testing hybrid recommender..."
)

recommend_movies(
    "The Dark Knight",
    10
)


# ============================================================
# 7. COMPLETION
# ============================================================

print("\n" + "=" * 70)

print(
    "HYBRID RECOMMENDATION ENGINE COMPLETED"
)

print("=" * 70)