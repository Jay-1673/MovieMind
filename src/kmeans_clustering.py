import pandas as pd
import numpy as np
import os
import pickle

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 60)
print("K-MEANS MOVIE CLUSTERING MODEL")
print("=" * 60)

print("\n[1/7] Loading movie data...")

metadata_file = os.path.join(DATA_DIR, "movie_metadata.csv")
features_file = os.path.join(DATA_DIR, "movie_features.csv")

metadata = pd.read_csv(metadata_file)
features = pd.read_csv(features_file)

print(f"Movies loaded: {len(metadata):,}")
print(f"Features loaded: {features.shape}")


# ============================================================
# 2. SELECT RELIABLE MOVIES
# ============================================================

print("\n[2/7] Selecting reliable movies...")

reliable_mask = metadata["reliable_rating"] == 1

metadata_cluster = metadata[reliable_mask].reset_index(drop=True)
features_cluster = features.loc[reliable_mask].reset_index(drop=True)

print(f"Movies used for clustering: {len(metadata_cluster):,}")


# ============================================================
# 3. PREPARE FEATURES
# ============================================================

print("\n[3/7] Preparing feature matrix...")

# Convert to NumPy
X = features_cluster.values.astype(np.float32)

print(f"Feature matrix: {X.shape}")


# ============================================================
# 4. SAMPLE DATA FOR SILHOUETTE SCORE
# ============================================================

print("\n[4/7] Preparing clustering sample...")

# Using a sample makes the silhouette calculation much faster.
sample_size = min(10000, len(X))

rng = np.random.RandomState(42)

sample_indices = rng.choice(
    len(X),
    size=sample_size,
    replace=False
)

X_sample = X[sample_indices]

print(f"Sample used for evaluation: {len(X_sample):,}")


# ============================================================
# 5. TRAIN K-MEANS
# ============================================================

print("\n[5/7] Training K-Means model...")

# Number of movie groups
NUMBER_OF_CLUSTERS = 10

kmeans = KMeans(
    n_clusters=NUMBER_OF_CLUSTERS,
    random_state=42,
    n_init=10
)

kmeans.fit(X)

print("K-Means model trained successfully.")


# ============================================================
# 6. EVALUATE MODEL
# ============================================================

print("\n[6/7] Evaluating clusters...")

sample_labels = kmeans.predict(X_sample)

silhouette = silhouette_score(
    X_sample,
    sample_labels
)

print(f"Silhouette Score: {silhouette:.4f}")


# ============================================================
# 7. SAVE MODEL AND RESULTS
# ============================================================

print("\n[7/7] Saving model and cluster results...")

# Add cluster number to metadata
metadata_cluster["cluster"] = kmeans.labels_

# Save model
model_file = os.path.join(
    MODEL_DIR,
    "kmeans_movie_model.pkl"
)

with open(model_file, "wb") as file:
    pickle.dump(kmeans, file)

print(f"Model saved: {model_file}")


# Save movie cluster information
cluster_file = os.path.join(
    MODEL_DIR,
    "movie_clusters.csv"
)

metadata_cluster.to_csv(
    cluster_file,
    index=False
)

print(f"Cluster data saved: {cluster_file}")


# ============================================================
# DISPLAY CLUSTER INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("MOVIES PER CLUSTER")
print("=" * 60)

cluster_counts = (
    metadata_cluster["cluster"]
    .value_counts()
    .sort_index()
)

for cluster, count in cluster_counts.items():
    print(f"Cluster {cluster}: {count:,} movies")


# ============================================================
# SHOW SAMPLE MOVIES FROM EACH CLUSTER
# ============================================================

print("\n" + "=" * 60)
print("SAMPLE MOVIES FROM EACH CLUSTER")
print("=" * 60)

for cluster_number in range(NUMBER_OF_CLUSTERS):

    cluster_movies = metadata_cluster[
        metadata_cluster["cluster"] == cluster_number
    ].head(5)

    print(f"\nCluster {cluster_number}:")

    if len(cluster_movies) == 0:
        print("No movies found.")
        continue

    for _, movie in cluster_movies.iterrows():
        print(
            f"  - {movie['primaryTitle']} "
            f"({movie['startYear']}) | "
            f"{movie['genres']}"
        )


print("\n" + "=" * 60)
print("K-MEANS CLUSTERING COMPLETED")
print("=" * 60)