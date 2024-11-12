
import pandas as pd
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
import time


def optimize_regression_metrics_with_threshold(X, y, cv=5, subset_fraction=0.1):
    """
    Automatically selects the best model and parameters for a regression problem 
    by optimizing multiple metrics (MSE, MAE, R2) with time-based thresholding.

    Parameters:
    - X: Feature matrix.
    - y: Target vector (continuous values).
    - cv: Number of cross-validation folds.
    - subset_fraction: Fraction of data used for estimating runtime.

    Returns:
    - Dictionary containing the best models and scores for each metric.
    """
    # Define models and hyperparameter grids
    models = [
        {
            'name': 'Linear Regression',
            'pipeline': Pipeline([
                ('scaler', StandardScaler()),  # Scaling for linear models
                ('model', LinearRegression())
            ]),
            'params': {}
        },
        {
            'name': 'Random Forest',
            'pipeline': Pipeline([
                ('model', RandomForestRegressor(random_state=42))
            ]),
            'params': {
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': [5, 10, None],
                'model__min_samples_split': [2, 5, 10]
            }
        },
        {
            'name': 'Support Vector Regressor',
            'pipeline': Pipeline([
                ('scaler', StandardScaler()),  # Scaling for SVR
                ('model', SVR())
            ]),
            'params': {
                'model__C': [0.1, 1, 10],
                'model__kernel': ['linear', 'rbf']
            }
        },
        {
            'name': 'Decision Tree',
            'pipeline': Pipeline([
                ('model', DecisionTreeRegressor(random_state=42))
            ]),
            'params': {
                'model__max_depth': [5, 10, None],
                'model__min_samples_split': [2, 5, 10]
            }
        },
        {
            'name': 'K-Nearest Neighbors',
            'pipeline': Pipeline([
                ('scaler', StandardScaler()),  # Scaling for KNN
                ('model', KNeighborsRegressor())
            ]),
            'params': {
                'model__n_neighbors': [3, 5, 7],
                'model__weights': ['uniform', 'distance']
            }
        }
    ]

    # Define scorers
    scorers = {
        'MSE': make_scorer(mean_squared_error, greater_is_better=False),
        'MAE': make_scorer(mean_absolute_error, greater_is_better=False),
        'R2': make_scorer(r2_score)
    }

    results = {metric: {'best_model': None, 'best_params': None, 'best_score': -float('inf')} for metric in scorers}

    # Estimate time for each model's hyperparameters on a subset
    estimated_times = {}
    for model_info in models:
        print(f"Estimating time for {model_info['name']}...")
        param_grid = model_info['params']
        pipeline = model_info['pipeline']

        for params in ParameterGrid(param_grid):
            # Create a subset
            subset_size = int(len(X) * subset_fraction)
            X_subset = X[:subset_size]
            y_subset = y[:subset_size]

            current_model = pipeline.set_params(**params)

            # Measure fit and predict times on subset
            start = time.time()
            current_model.fit(X_subset, y_subset)
            fit_time = time.time() - start

            start = time.time()
            current_model.predict(X_subset)
            predict_time = time.time() - start

            # Estimate total time for full dataset
            total_time_estimate = (fit_time + predict_time) / subset_fraction

            estimated_times[(model_info['name'], tuple(params.items()))] = {
                'fit_time': fit_time,
                'predict_time': predict_time,
                'total_time': total_time_estimate
            }

    # Prompt user for time threshold
    print("\nEstimated times for hyperparameters:")
    for (model_name, params), time_info in estimated_times.items():
        print(f"Model: {model_name}, Params: {params}, Estimated Total Time: {time_info['total_time']:.2f} seconds")

    time_threshold = float(input("Enter the time threshold (seconds) for discarding points: "))

    # Filter points that exceed the threshold
    filtered_params = {
        model_name: [
            dict(params) for (m_name, params), time_info in estimated_times.items()
            if m_name == model_name and time_info['total_time'] <= time_threshold
        ]
        for model_name in {m_name for m_name, _ in estimated_times.keys()}
    }

    for model_name, params_list in filtered_params.items():
        print(f"\nUsing threshold {time_threshold}s, {len(params_list)} parameter combinations retained for {model_name}.")

    # Calculate estimated remaining time
    estimated_total_time_left = sum(
        time_info['total_time'] for (m_name, _), time_info in estimated_times.items()
        if m_name in filtered_params and time_info['total_time'] <= time_threshold
    )
    print(f"Estimated total time remaining for evaluation: {estimated_total_time_left:.2f} seconds.")

    # Evaluate remaining parameter combinations
    for model_info in models:
        model_name = model_info['name']
        pipeline = model_info['pipeline']

        if model_name not in filtered_params or not filtered_params[model_name]:
            continue

        print(f"\nOptimizing {model_name}...")
        for metric, scorer in scorers.items():
            for params in filtered_params[model_name]:
                current_model = pipeline.set_params(**params)

                # Measure actual time for fit and predict
                start = time.time()
                current_model.fit(X, y)
                actual_fit_time = time.time() - start

                start = time.time()
                y_pred = current_model.predict(X)
                actual_predict_time = time.time() - start

                # Calculate metric score
                score = scorer._score_func(y, y_pred)

                # Compare actual and estimated times
                print(
                    f"Params: {params}, Metric: {metric}, "
                    f"Score: {score:.4f}, "
                    f"Actual Fit Time: {actual_fit_time:.2f}s, "
                    f"Actual Predict Time: {actual_predict_time:.2f}s, "
                    f"Estimated Total Time: {estimated_times[(model_name, tuple(params.items()))]['total_time']:.2f}s"
                )

                # Update best model if this is better
                if metric == 'MSE' or metric == 'MAE':
                    is_better = score > results[metric]['best_score']
                else:
                    is_better = score > results[metric]['best_score']

                if is_better:
                    results[metric]['best_model'] = model_name
                    results[metric]['best_params'] = params
                    results[metric]['best_score'] = score

    return results


# Load dataset
data_path = './original_data/trainingset.csv'
data = pd.read_csv(data_path)

# Preprocess dataset
data = shuffle(data)  # Shuffle data
data = data.drop(columns=['rowIndex'])  # Drop unnecessary column
X = data.drop(columns=['ClaimAmount'])  # Features
y = data['ClaimAmount']  # Target

# Optimize for multiple metrics with time thresholding
results = optimize_regression_metrics_with_threshold(X, y, cv=5, subset_fraction=0.1)

# Output the best models and their details for each metric
for metric, result in results.items():
    print(f"\nBest Model for {metric}: {result['best_model']}")
    print(f"Best Parameters for {metric}: {result['best_params']}")
    print(f"Best Score for {metric}: {result['best_score']}")
