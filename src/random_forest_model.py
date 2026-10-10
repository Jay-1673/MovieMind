import pandas as pd
import numpy as np
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("RANDOM FOREST MOVIE RATING CLASSIFICATION")
print("=" * 60)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n[1/8] Loading movie data...")

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


# ============================================================
# 2. SELECT RELIABLE MOVIES
# ============================================================

print("\n[2/8] Selecting reliable movies...")

reliable_mask = metadata["reliable_rating"] == 1

metadata_model = metadata[
    reliable_mask
].reset_index(drop=True)

features_model = features[
    reliable_mask
].reset_index(drop=True)

print(f"Reliable movies: {len(metadata_model):,}")


# ============================================================
# 3. PREPARE FEATURES
# ============================================================

print("\n[3/8] Preparing input features...")

# Do NOT use features derived from the rating itself.
excluded_columns = [
    "averageRating",
    "high_rated",
    "popularity_score",
    "log_popularity"
]

available_columns = [
    column
    for column in features_model.columns
    if column not in excluded_columns
]

X = features_model[
    available_columns
].astype(np.float32)

print(f"Features used: {X.shape[1]}")
print(f"Feature matrix: {X.shape}")


# ============================================================
# 4. CREATE TARGET
# ============================================================

print("\n[4/8] Creating target variable...")

# 1 = IMDb rating >= 7.0
# 0 = IMDb rating < 7.0

y = (
    metadata_model["averageRating"] >= 7.0
).astype(int)

print("\nTarget distribution:")

print(
    f"Not High-Rated (0): {(y == 0).sum():,}"
)

print(
    f"High-Rated (1):     {(y == 1).sum():,}"
)


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

print("\n[5/8] Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training samples: {len(X_train):,}")
print(f"Testing samples:  {len(X_test):,}")


# ============================================================
# 6. TRAIN RANDOM FOREST
# ============================================================

print("\n[6/8] Training Random Forest...")

random_forest = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

random_forest.fit(
    X_train,
    y_train
)

print("Random Forest trained successfully.")


# ============================================================
# 7. EVALUATE MODEL
# ============================================================

print("\n[7/8] Evaluating model...")

y_pred = random_forest.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

cm = confusion_matrix(
    y_test,
    y_pred
)


print("\n" + "=" * 60)
print("RANDOM FOREST PERFORMANCE")
print("=" * 60)

print(f"\nAccuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Not High-Rated",
            "High-Rated"
        ],
        zero_division=0
    )
)


# ============================================================
# 8. SAVE MODEL
# ============================================================

print("\n[8/8] Saving Random Forest model...")

model_file = os.path.join(
    MODEL_DIR,
    "random_forest_model.pkl"
)

with open(
    model_file,
    "wb"
) as file:

    pickle.dump(
        {
            "model": random_forest,
            "features": available_columns
        },
        file
    )

print(f"Model saved: {model_file}")


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 60)
print("TOP FEATURE IMPORTANCE")
print("=" * 60)

importance = pd.DataFrame({
    "feature": available_columns,
    "importance": random_forest.feature_importances_
})

importance = importance.sort_values(
    by="importance",
    ascending=False
)

print(
    importance.head(10).to_string(
        index=False
    )
)


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 60)
print("RANDOM FOREST MODEL COMPLETED")
print("=" * 60)