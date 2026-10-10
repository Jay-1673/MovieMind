import pandas as pd
import numpy as np
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
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
print("NAIVE BAYES MOVIE RATING CATEGORY CLASSIFICATION")
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

# Do not use rating-derived columns.
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
# 4. CREATE RATING CATEGORIES
# ============================================================

print("\n[4/8] Creating rating categories...")

# Rating categories:
#
# Below 5.0       = Low
# 5.0 - 6.9       = Average
# 7.0 - 7.9       = Good
# 8.0 and above   = Excellent

def rating_category(rating):

    if rating < 5.0:
        return "Low"

    elif rating < 7.0:
        return "Average"

    elif rating < 8.0:
        return "Good"

    else:
        return "Excellent"


y = metadata_model[
    "averageRating"
].apply(rating_category)

print("\nRating category distribution:")

category_counts = y.value_counts()

for category in [
    "Low",
    "Average",
    "Good",
    "Excellent"
]:

    print(
        f"{category:10s}: "
        f"{category_counts.get(category, 0):,}"
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
# 6. TRAIN NAIVE BAYES
# ============================================================

print("\n[6/8] Training Naive Bayes...")

naive_bayes = GaussianNB()

naive_bayes.fit(
    X_train,
    y_train
)

print("Naive Bayes trained successfully.")

# ============================================================
# 7. EVALUATE MODEL
# ============================================================

print("\n[7/8] Evaluating model...")

y_pred = naive_bayes.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[
        "Low",
        "Average",
        "Good",
        "Excellent"
    ]
)

print("\n" + "=" * 60)
print("NAIVE BAYES PERFORMANCE")
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
        labels=[
            "Low",
            "Average",
            "Good",
            "Excellent"
        ],
        zero_division=0
    )
)

# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

print("\n" + "=" * 60)
print("SAMPLE RATING CATEGORY PREDICTIONS")
print("=" * 60)

sample_size = min(10, len(X_test))

sample_X = X_test.iloc[:sample_size]
sample_actual = y_test.iloc[:sample_size]

sample_pred = naive_bayes.predict(sample_X)

sample_results = pd.DataFrame({
    "Actual Category": sample_actual.values,
    "Predicted Category": sample_pred
})

print(
    sample_results.to_string(index=False)
)

# ============================================================
# 8. SAVE MODEL
# ============================================================

print("\n[8/8] Saving Naive Bayes model...")

model_file = os.path.join(
    MODEL_DIR,
    "naive_bayes_model.pkl"
)

with open(
    model_file,
    "wb"
) as file:

    pickle.dump(
        {
            "model": naive_bayes,
            "features": available_columns,
            "categories": [
                "Low",
                "Average",
                "Good",
                "Excellent"
            ]
        },
        file
    )

print(f"Model saved: {model_file}")

# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 60)
print("NAIVE BAYES MODEL COMPLETED")
print("=" * 60)