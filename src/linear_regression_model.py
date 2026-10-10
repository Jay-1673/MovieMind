import pandas as pd
import numpy as np
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
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
print("LINEAR REGRESSION MOVIE RATING PREDICTION")
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
# 3. PREPARE INPUT FEATURES
# ============================================================

print("\n[3/8] Preparing input features...")

# These columns must NOT be used because they contain
# information derived from IMDb rating.

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

# Target = actual IMDb average rating

y = metadata_model[
    "averageRating"
].astype(np.float32)

print("\nRating statistics:")

print(f"Minimum rating: {y.min():.2f}")
print(f"Maximum rating: {y.max():.2f}")
print(f"Average rating: {y.mean():.2f}")


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

print("\n[5/8] Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print(f"Training samples: {len(X_train):,}")
print(f"Testing samples:  {len(X_test):,}")


# ============================================================
# 6. TRAIN LINEAR REGRESSION
# ============================================================

print("\n[6/8] Training Linear Regression...")

linear_model = LinearRegression()

linear_model.fit(
    X_train,
    y_train
)

print("Linear Regression trained successfully.")


# ============================================================
# 7. EVALUATE MODEL
# ============================================================

print("\n[7/8] Evaluating model...")

y_pred = linear_model.predict(X_test)

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

r2 = r2_score(
    y_test,
    y_pred
)


print("\n" + "=" * 60)
print("LINEAR REGRESSION PERFORMANCE")
print("=" * 60)

print(f"\nMAE : {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²  : {r2:.4f}")


# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

print("\n" + "=" * 60)
print("SAMPLE RATING PREDICTIONS")
print("=" * 60)

sample_indices = X_test.index[:10]

sample_results = pd.DataFrame({
    "Actual Rating": y_test.loc[sample_indices].values,
    "Predicted Rating": y_pred[:10]
})

sample_results["Actual Rating"] = (
    sample_results["Actual Rating"].round(2)
)

sample_results["Predicted Rating"] = (
    sample_results["Predicted Rating"].round(2)
)

print(
    sample_results.to_string(index=False)
)


# ============================================================
# 8. SAVE MODEL
# ============================================================

print("\n[8/8] Saving Linear Regression model...")

model_file = os.path.join(
    MODEL_DIR,
    "linear_regression_model.pkl"
)

with open(
    model_file,
    "wb"
) as file:

    pickle.dump(
        {
            "model": linear_model,
            "features": available_columns
        },
        file
    )

print(f"Model saved: {model_file}")


# ============================================================
# COEFFICIENT IMPORTANCE
# ============================================================

print("\n" + "=" * 60)
print("TOP MODEL COEFFICIENTS")
print("=" * 60)

coefficients = pd.DataFrame({
    "feature": available_columns,
    "coefficient": linear_model.coef_
})

coefficients["absolute_coefficient"] = (
    coefficients["coefficient"].abs()
)

coefficients = coefficients.sort_values(
    by="absolute_coefficient",
    ascending=False
)

print(
    coefficients[
        ["feature", "coefficient"]
    ].head(10).to_string(index=False)
)


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 60)
print("LINEAR REGRESSION MODEL COMPLETED")
print("=" * 60)