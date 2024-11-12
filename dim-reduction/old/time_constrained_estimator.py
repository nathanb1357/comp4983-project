import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.base import clone
import time
import numpy as np

def time_constrained_grid_search_regression(model, param_grid, X, y, time_threshold, subset_fraction=0.1):
    """
    Perform a grid search while skipping grid points that exceed the time threshold for regression tasks.

    Parameters:
    - model: The base regression model to optimize.
    - param_grid: A dictionary of parameters to evaluate.
    - X: Feature matrix.
    - y: Target vector (continuous values, e.g., ClaimAmount).
    - time_threshold: Maximum allowed time (in seconds) per grid point.
    - subset_fraction: Fraction of data to use for time estimation.

    Returns:
    - Dictionary containing the best parameters, best score, and checked configurations.
    """
    from sklearn.model_selection import ParameterGrid

    # Split data for validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Subset for timing
    subset_size = int(len(X_train) * subset_fraction)
    X_subset = X_train[:subset_size]
    y_subset = y_train[:subset_size]

    best_score = np.inf  # For regression, lower scores are better
    best_params = None
    checked_points = []

    # Iterate over all parameter combinations
    for params in ParameterGrid(param_grid):
        # Clone the base model and set parameters
        current_model = clone(model)
        current_model.set_params(**params)

        # Measure runtime on the subset
        try:
            start = time.time()
            current_model.fit(X_subset, y_subset)
            fit_time_subset = time.time() - start

            start = time.time()
            y_pred_subset = current_model.predict(X_subset)
            predict_time_subset = time.time() - start
        except Exception as e:
            print(f"Skipping params {params} due to an error: {e}")
            continue

        # Estimate full runtime
        fit_time_full = fit_time_subset / subset_fraction
        predict_time_full = predict_time_subset / subset_fraction
        total_time_full = fit_time_full + predict_time_full

        # Skip this grid point if the estimated time exceeds the threshold
        if total_time_full > time_threshold:
            print(f"Skipping params {params} (estimated time {total_time_full:.2f}s exceeds {time_threshold}s)")
            continue

        # Evaluate on the validation set
        current_model.fit(X_train, y_train)
        y_pred_val = current_model.predict(X_val)
        score = mean_squared_error(y_val, y_pred_val)

        print(f"Evaluated params {params}: MSE = {score:.4f}, Time = {total_time_full:.2f}s")

        # Update the best score and parameters
        if score < best_score:
            best_score = score
            best_params = params

        # Log checked parameters
        checked_points.append({"params": params, "score": score, "estimated_time": total_time_full})

    return {
        "best_params": best_params,
        "best_score": best_score,
        "checked_points": checked_points
    }


# Load dataset
data_path = './original_data/trainingset.csv'
data = pd.read_csv(data_path)

# Drop the "rowIndex" column
data = data.drop(columns=['rowIndex'])

# Separate features and target
X = data.drop(columns=['ClaimAmount'])
y = data['ClaimAmount']

# Define model and parameter grid
model = RandomForestRegressor(random_state=42)
param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5]
}

# Perform time-constrained grid search for regression
results = time_constrained_grid_search_regression(model, param_grid, X, y, time_threshold=np.inf)

# Output best parameters and scores
print("Best Parameters:", results['best_params'])
print("Best Score (MSE):", results['best_score'])
print("Checked Points:", results['checked_points'])
