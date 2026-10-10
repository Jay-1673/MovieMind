import pandas as pd
import os

# ============================================================
# AI MOVIE RECOMMENDATION SYSTEM
# IMDb Dataset Preprocessing
# ============================================================

# Project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

# Create processed folder if it doesn't exist
os.makedirs(PROCESSED_DIR, exist_ok=True)

print("=" * 60)
print("IMDb MOVIE DATA PREPROCESSING")
print("=" * 60)

# ------------------------------------------------------------
# File paths
# ------------------------------------------------------------

basics_file = os.path.join(RAW_DIR, "title.basics.tsv.gz")
ratings_file = os.path.join(RAW_DIR, "title.ratings.tsv.gz")

# ------------------------------------------------------------
# Check files
# ------------------------------------------------------------

if not os.path.exists(basics_file):
    raise FileNotFoundError(
        f"Could not find:\n{basics_file}\n\n"
        "Make sure title.basics.tsv.gz is inside data/raw/"
    )

if not os.path.exists(ratings_file):
    raise FileNotFoundError(
        f"Could not find:\n{ratings_file}\n\n"
        "Make sure title.ratings.tsv.gz is inside data/raw/"
    )

print("\n[1/5] IMDb files found successfully.")

# ------------------------------------------------------------
# Read title.basics
# ------------------------------------------------------------

print("\n[2/5] Reading title.basics.tsv.gz...")
print("This may take some time...")

basics = pd.read_csv(
    basics_file,
    sep="\t",
    compression="gzip",
    na_values="\\N",
    dtype={
        "tconst": "string",
        "titleType": "string",
        "primaryTitle": "string",
        "originalTitle": "string",
        "isAdult": "string",
        "startYear": "string",
        "endYear": "string",
        "runtimeMinutes": "string",
        "genres": "string"
}

)

print(f"Total IMDb title records: {len(basics):,}")

# ------------------------------------------------------------
# Keep only movies
# ------------------------------------------------------------

print("\n[3/5] Filtering movie records...")

movies = basics[
    (basics["titleType"] == "movie") &
    (basics["isAdult"] == "0")
].copy()

# Convert numeric columns
movies["startYear"] = pd.to_numeric(
    movies["startYear"],
    errors="coerce"
)

movies["runtimeMinutes"] = pd.to_numeric(
    movies["runtimeMinutes"],
    errors="coerce"
)

# Remove records without essential information
movies = movies.dropna(
    subset=["primaryTitle", "startYear", "runtimeMinutes"]
)

# Convert year and runtime to integer
movies["startYear"] = movies["startYear"].astype(int)
movies["runtimeMinutes"] = movies["runtimeMinutes"].astype(int)

# Keep sensible movie years
movies = movies[
    (movies["startYear"] >= 1900) &
    (movies["startYear"] <= 2026)
]

# Keep sensible runtime
movies = movies[
    (movies["runtimeMinutes"] >= 1) &
    (movies["runtimeMinutes"] <= 600)
]

print(f"Movies after cleaning: {len(movies):,}")

# ------------------------------------------------------------
# Read ratings
# ------------------------------------------------------------

print("\n[4/5] Reading title.ratings.tsv.gz...")

ratings = pd.read_csv(
    ratings_file,
    sep="\t",
    compression="gzip",
    dtype={
        "tconst": "string",
        "averageRating": "float32",
        "numVotes": "int32"
    }
)

print(f"Rating records: {len(ratings):,}")

# ------------------------------------------------------------
# Merge movies + ratings
# ------------------------------------------------------------

print("\n[5/5] Combining movie information with ratings...")

movies = movies.merge(
    ratings,
    on="tconst",
    how="inner"
)

# ------------------------------------------------------------
# Create useful features
# ------------------------------------------------------------

# Split genres into a clean text representation
movies["genres"] = movies["genres"].fillna("Unknown")

# Create decade feature
movies["decade"] = (movies["startYear"] // 10) * 10

# Create high-rated target
# This will later be useful for classification algorithms.
movies["high_rated"] = (
    movies["averageRating"] >= 7.0
).astype(int)

# Create popularity category
movies["popularity_score"] = (
    movies["averageRating"] * 
    __import__("numpy").log1p(movies["numVotes"])
)

# ------------------------------------------------------------
# Select final columns
# ------------------------------------------------------------

final_columns = [
    "tconst",
    "primaryTitle",
    "originalTitle",
    "startYear",
    "runtimeMinutes",
    "genres",
    "averageRating",
    "numVotes",
    "decade",
    "high_rated",
    "popularity_score"
]

movies = movies[final_columns]

# Remove duplicate movie IDs
movies = movies.drop_duplicates(
    subset=["tconst"]
)

# Sort by popularity
movies = movies.sort_values(
    by="numVotes",
    ascending=False
)

# ------------------------------------------------------------
# Save processed dataset
# ------------------------------------------------------------

output_file = os.path.join(
    PROCESSED_DIR,
    "movies_processed.csv"
)

movies.to_csv(
    output_file,
    index=False
)

# ------------------------------------------------------------
# Display results
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 60)

print(f"\nProcessed movies: {len(movies):,}")

print(f"\nSaved dataset:")
print(output_file)

print("\nDataset columns:")
print(list(movies.columns))

print("\nFirst 5 movies:")
print(
    movies[
        [
            "primaryTitle",
            "startYear",
            "genres",
            "averageRating",
            "numVotes"
        ]
    ].head()
)

print("\n" + "=" * 60)