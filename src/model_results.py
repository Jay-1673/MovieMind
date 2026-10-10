# ============================================================
# MODEL PERFORMANCE RESULTS
# AI MOVIE RECOMMENDATION SYSTEM
# ============================================================


# ------------------------------------------------------------
# CLASSIFICATION MODELS
# ------------------------------------------------------------

classification_results = {

    "Decision Tree": {
        "Accuracy": 0.7396,
        "Precision": 0.4202,
        "Recall": 0.7356,
        "F1 Score": 0.5349
    },

    "Random Forest": {
        "Accuracy": 0.7728,
        "Precision": 0.4624,
        "Recall": 0.7146,
        "F1 Score": 0.5615
    },

    "Naive Bayes": {
        "Accuracy": 0.2368,
        "Precision": 0.5512,
        "Recall": 0.2368,
        "F1 Score": 0.1469
    }
}


# ------------------------------------------------------------
# REGRESSION MODEL
# ------------------------------------------------------------

regression_results = {

    "Linear Regression": {
        "MAE": 0.7694,
        "RMSE": 1.0156,
        "R2 Score": 0.3612
    }
}


# ------------------------------------------------------------
# CLUSTERING MODEL
# ------------------------------------------------------------

clustering_results = {

    "K-Means": {
        "Clusters": 10,
        "Silhouette Score": 0.1276
    }
}


# ------------------------------------------------------------
# RECOMMENDATION MODEL
# ------------------------------------------------------------

recommendation_results = {

    "KNN": {
        "Metric": "Cosine Similarity",
        "Neighbors": 10
    },

    "Hybrid AI": {
        "KNN Weight": 0.60,
        "Rating Weight": 0.20,
        "Popularity Weight": 0.10,
        "Cluster Weight": 0.10
    }
}


# ------------------------------------------------------------
# DATASET INFORMATION
# ------------------------------------------------------------

dataset_information = {

    "Source": "Official IMDb Non-Commercial Datasets",

    "Total Processed Movies": 310437,

    "Reliable Rated Movies": 143816,

    "TF-IDF Features": 5000,

    "Engineered Features": 32,

    "K-Means Clusters": 10
}


# ------------------------------------------------------------
# PRINT RESULTS
# ------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 70)
    print("AI MOVIE RECOMMENDATION SYSTEM")
    print("MODEL PERFORMANCE SUMMARY")
    print("=" * 70)

    print("\nCLASSIFICATION MODELS")
    print("-" * 70)

    for model, metrics in classification_results.items():

        print(f"\n{model}")

        for metric, value in metrics.items():

            print(
                f"{metric}: "
                f"{value:.4f}"
            )


    print("\nREGRESSION MODEL")
    print("-" * 70)

    for model, metrics in regression_results.items():

        print(f"\n{model}")

        for metric, value in metrics.items():

            print(
                f"{metric}: "
                f"{value:.4f}"
            )


    print("\nCLUSTERING MODEL")
    print("-" * 70)

    for model, metrics in clustering_results.items():

        print(f"\n{model}")

        for metric, value in metrics.items():

            print(
                f"{metric}: "
                f"{value}"
            )


    print("\nDATASET INFORMATION")
    print("-" * 70)

    for key, value in dataset_information.items():

        print(
            f"{key}: {value}"
        )

    print("\n" + "=" * 70)