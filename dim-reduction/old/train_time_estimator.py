def measure_runtime(model, X, y, subset_fraction=0.1):
    """
    Measures and predicts runtime for fit and predict, first using a subset of the data,
    then running on the full dataset to compare predictions to actual runtimes.

    Parameters:
    - model: Scikit-learn compatible model
    - X: Feature matrix
    - y: Target vector
    - subset_fraction: Fraction of data to use for initial timing

    Returns:
    - Dictionary with runtimes for the subset, predicted full runtime, and actual full runtime.
    """
    import time

    # Create a subset
    subset_size = int(len(X) * subset_fraction)
    X_subset = X[:subset_size]
    y_subset = y[:subset_size]

    # Measure runtime for subset
    start = time.time()
    model.fit(X_subset, y_subset)
    fit_time_subset = time.time() - start

    start = time.time()
    model.predict(X_subset)
    predict_time_subset = time.time() - start

    # Display runtime for subset
    print(f"Fit time for {subset_fraction*100:.0f}% of dataset: {fit_time_subset:.2f} seconds")
    print(f"Predict time for {subset_fraction*100:.0f}% of dataset: {predict_time_subset:.2f} seconds")

    # Predict full dataset runtime
    fit_time_pred_full = fit_time_subset / subset_fraction
    predict_time_pred_full = predict_time_subset / subset_fraction

    print(f"Predicted fit time for full dataset: {fit_time_pred_full:.2f} seconds")
    print(f"Predicted predict time for full dataset: {predict_time_pred_full:.2f} seconds")

    # Measure runtime for full dataset
    start = time.time()
    model.fit(X, y)
    fit_time_full = time.time() - start

    start = time.time()
    model.predict(X)
    predict_time_full = time.time() - start

    # Display runtime for full dataset
    print(f"Actual fit time for full dataset: {fit_time_full:.2f} seconds")
    print(f"Actual predict time for full dataset: {predict_time_full:.2f} seconds")

    # Return results
    return {
        "fit_time_subset": fit_time_subset,
        "predict_time_subset": predict_time_subset,
        "fit_time_pred_full": fit_time_pred_full,
        "predict_time_pred_full": predict_time_pred_full,
        "fit_time_full": fit_time_full,
        "predict_time_full": predict_time_full,
    }


# Example Usage
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

print("Creating data")
# Generate synthetic data
X, y = make_classification(n_samples=100000000, n_features=20)
print("Data created")


# Define model
model = LogisticRegression()

# Measure and compare runtimes
print("Calculating results")
runtime_results = measure_runtime(model, X, y, subset_fraction=0.1)
