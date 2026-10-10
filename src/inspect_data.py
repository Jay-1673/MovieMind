import pandas as pd
import os

# ============================================================
# IMDb MOVIE RECOMMENDATION SYSTEM
# Dataset Inspection
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "movies_processed.csv"
)

print("=" * 60)
print("IMDb PROCESSED DATASET INSPECTION")
print("=" * 60)

# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------

df = pd.read_csv(DATA_FILE)

# ------------------------------------------------------------
# 1. Dataset size
# ------------------------------------------------------------

print("\n1. DATASET SIZE")
print("-" * 40)

print(f"Rows    : {df.shape[0]:,}")
print(f"Columns : {df.shape[1]}")

# ------------------------------------------------------------
# 2. Column information
# ------------------------------------------------------------

print("\n2. COLUMN INFORMATION")
print("-" * 40)

print(df.dtypes)

# ------------------------------------------------------------
# 3. Missing values
# ------------------------------------------------------------

print("\n3. MISSING VALUES")
print("-" * 40)

missing = df.isnull().sum()

print(missing)

# ------------------------------------------------------------
# 4. Duplicate records
# ------------------------------------------------------------

print("\n4. DUPLICATE RECORDS")
print("-" * 40)

duplicates = df.duplicated(subset=["tconst"]).sum()

print(f"Duplicate movie IDs: {duplicates:,}")

# ------------------------------------------------------------
# 5. Rating statistics
# ------------------------------------------------------------

print("\n5. RATING STATISTICS")
print("-" * 40)

print(df["averageRating"].describe())

# ------------------------------------------------------------
# 6. Runtime statistics
# ------------------------------------------------------------

print("\n6. RUNTIME STATISTICS")
print("-" * 40)

print(df["runtimeMinutes"].describe())

# ------------------------------------------------------------
# 7. Most popular movies
# ------------------------------------------------------------

print("\n7. TOP 10 MOST VOTED MOVIES")
print("-" * 40)

top_movies = df.nlargest(10, "numVotes")

print(
    top_movies[
        [
            "primaryTitle",
            "startYear",
            "genres",
            "averageRating",
            "numVotes"
        ]
    ].to_string(index=False)
)

# ------------------------------------------------------------
# 8. Highest rated movies
# ------------------------------------------------------------

print("\n8. TOP 10 HIGHEST RATED MOVIES")
print("-" * 40)

highest_rated = df.nlargest(10, "averageRating")

print(
    highest_rated[
        [
            "primaryTitle",
            "startYear",
            "genres",
            "averageRating",
            "numVotes"
        ]
    ].to_string(index=False)
)

# ------------------------------------------------------------
# 9. High-rated movie distribution
# ------------------------------------------------------------

print("\n9. HIGH-RATED MOVIE DISTRIBUTION")
print("-" * 40)

print(
    df["high_rated"]
    .value_counts()
    .sort_index()
)

print("\n0 = Rating below 7.0")
print("1 = Rating 7.0 or above")

# ------------------------------------------------------------
# 10. Genre distribution
# ------------------------------------------------------------

print("\n10. GENRE DISTRIBUTION")
print("-" * 40)

genre_counts = (
    df["genres"]
    .fillna("Unknown")
    .str.split(",")
    .explode()
    .str.strip()
    .value_counts()
)

print(genre_counts.head(15))

# ------------------------------------------------------------
# 11. Decade distribution
# ------------------------------------------------------------

print("\n11. MOVIES BY DECADE")
print("-" * 40)

print(
    df["decade"]
    .value_counts()
    .sort_index()
    .tail(15)
)

# ------------------------------------------------------------
# 12. Correlation analysis
# ------------------------------------------------------------

print("\n12. NUMERICAL CORRELATIONS")
print("-" * 40)

numeric_columns = [
    "startYear",
    "runtimeMinutes",
    "averageRating",
    "numVotes",
    "decade",
    "high_rated",
    "popularity_score"
]

correlation = df[numeric_columns].corr()

print(correlation.round(3))

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("DATASET INSPECTION COMPLETED")
print("=" * 60)

print("\nThe dataset is ready for AI/ML model development.")