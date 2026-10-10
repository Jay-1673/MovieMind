import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# AI MOVIE RECOMMENDATION SYSTEM
# Feature Engineering
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "movies_processed.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


print("=" * 60)
print("MOVIE FEATURE ENGINEERING")
print("=" * 60)


# ------------------------------------------------------------
# 1. Load processed dataset
# ------------------------------------------------------------

print("\n[1/7] Loading processed dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Movies loaded: {len(df):,}")


# ------------------------------------------------------------
# 2. Create reliable rating target
# ------------------------------------------------------------

print("\n[2/7] Creating reliable rating target...")

# Movies with very few votes can have unreliable ratings.
# We use 100 votes as a minimum threshold for the
# recommendation/classification dataset.

MIN_VOTES = 100

df["reliable_rating"] = (
    df["numVotes"] >= MIN_VOTES
).astype(int)

print(
    f"Movies with at least {MIN_VOTES} votes: "
    f"{df['reliable_rating'].sum():,}"
)


# ------------------------------------------------------------
# 3. Create rating categories
# ------------------------------------------------------------

print("\n[3/7] Creating rating categories...")


def rating_category(rating):
    if rating >= 8.0:
        return "Excellent"
    elif rating >= 7.0:
        return "Good"
    elif rating >= 5.0:
        return "Average"
    else:
        return "Low"


df["rating_category"] = df["averageRating"].apply(
    rating_category
)

print("\nRating category distribution:")

print(
    df["rating_category"]
    .value_counts()
)


# ------------------------------------------------------------
# 4. Process genres
# ------------------------------------------------------------

print("\n[4/7] Processing genres...")

df["genres"] = df["genres"].fillna("Unknown")

df["genre_list"] = df["genres"].apply(
    lambda x: [
        genre.strip()
        for genre in str(x).split(",")
        if genre.strip()
    ]
)

# One-hot encode genres
mlb = MultiLabelBinarizer()

genre_matrix = mlb.fit_transform(
    df["genre_list"]
)

genre_columns = [
    f"genre_{genre}"
    for genre in mlb.classes_
]

genre_df = pd.DataFrame(
    genre_matrix,
    columns=genre_columns,
    index=df.index
)

df = pd.concat(
    [df, genre_df],
    axis=1
)

print(f"Genres detected: {len(mlb.classes_)}")


# ------------------------------------------------------------
# 5. Create numerical ML features
# ------------------------------------------------------------

print("\n[5/7] Creating numerical features...")

# Log transformation reduces the huge range of vote counts.
df["log_numVotes"] = np.log1p(
    df["numVotes"]
)

# Normalize release year relative to the modern period.
df["movie_age"] = (
    2026 - df["startYear"]
)

# Prevent negative ages for future/recent titles.
df["movie_age"] = df["movie_age"].clip(
    lower=0
)

# Log-transformed popularity
df["log_popularity"] = np.log1p(
    df["popularity_score"]
)


# ------------------------------------------------------------
# 6. Create ML feature matrix
# ------------------------------------------------------------

print("\n[6/7] Creating numerical feature matrix...")

numeric_features = [
    "startYear",
    "runtimeMinutes",
    "log_numVotes",
    "movie_age",
    "log_popularity"
]

feature_columns = (
    numeric_features +
    genre_columns
)

X = df[feature_columns].copy()

# Standardize numerical features
scaler = StandardScaler()

X[numeric_features] = scaler.fit_transform(
    X[numeric_features]
)

# Save ML feature matrix
features_file = os.path.join(
    OUTPUT_DIR,
    "movie_features.csv"
)

X.to_csv(
    features_file,
    index=False
)


# ------------------------------------------------------------
# 7. Create TF-IDF content features
# ------------------------------------------------------------

print("\n[7/7] Creating TF-IDF movie content features...")

# Combine available IMDb metadata into a text representation.
#
# IMDb's official bulk dataset does not contain plot summaries,
# so we use genres + title metadata for content-based
# recommendation.

df["content"] = (
    df["primaryTitle"].fillna("") + " " +
    df["genres"].fillna("")
)

tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=5000,
    ngram_range=(1, 2)
)

tfidf_matrix = tfidf.fit_transform(
    df["content"]
)

print(
    f"TF-IDF matrix shape: {tfidf_matrix.shape}"
)


# ------------------------------------------------------------
# Save metadata
# ------------------------------------------------------------

metadata_columns = [
    "tconst",
    "primaryTitle",
    "originalTitle",
    "startYear",
    "runtimeMinutes",
    "genres",
    "averageRating",
    "numVotes",
    "reliable_rating",
    "rating_category",
    "high_rated"
]

metadata = df[metadata_columns].copy()

metadata_file = os.path.join(
    OUTPUT_DIR,
    "movie_metadata.csv"
)

metadata.to_csv(
    metadata_file,
    index=False
)


# Save TF-IDF matrix

tfidf_file = os.path.join(
    OUTPUT_DIR,
    "tfidf_features.npz"
)

from scipy.sparse import save_npz

save_npz(
    tfidf_file,
    tfidf_matrix
)


# ------------------------------------------------------------
# Final summary
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("FEATURE ENGINEERING COMPLETED")
print("=" * 60)

print(f"\nTotal movies: {len(df):,}")

print(
    f"Numerical + genre features: "
    f"{len(feature_columns)}"
)

print(
    f"TF-IDF features: "
    f"{tfidf_matrix.shape[1]}"
)

print("\nFiles created:")

print(
    "1.",
    features_file
)

print(
    "2.",
    metadata_file
)

print(
    "3.",
    tfidf_file
)

print("\n" + "=" * 60)