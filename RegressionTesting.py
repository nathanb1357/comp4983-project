import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.pipeline import Pipeline
from sklearnex import patch_sklearn

patch_sklearn()

# Load dataset
data = pd.read_csv("edited_data/non_zero_claims.csv")
X = data.drop(columns=["ClaimAmount"])
y = data["ClaimAmount"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Models and hyperparameters to test
models = {
    # "Linear Regression": {
    #     "model": LinearRegression(),
    #     "params": {}
    # },
    # "Ridge Regression": {
    #     "model": Ridge(),
    #     "params": {
    #         "model__alpha": [5000],
    #         "model__solver": ["sag"]
    #     }
    # },
    # "Lasso Regression": {
    #     "model": Lasso(),
    #     "params": {
    #         "model__alpha": [100, 500, 1000, 5000],
    #         "model__max_iter": [10000, 1000, 100000],
    #         "model__selection": ["cyclic", "random"]
    #     }
    # },
    # "Random Forest": {
    #     "model": RandomForestRegressor(),
    #     "params": {
    #         "model__n_estimators": [200, 500, 1000],
    #         "model__max_depth": [None, 30, 40, 50],
    #         "model__max_features": ["sqrt", "log2", 1.0],
    #         "model__bootstrap": [False]
    #     }
    # },
    # "KNN Regressor": {
    #     "model": KNeighborsRegressor(),
    #     "params": {
    #         "model__n_neighbors": [800],
    #         "model__weights": ["distance"],
    #         "model__p": [1],
    #         "model__algorithm": ["auto"]
    #     }
    # },
    # "SVR": {
    #     "model": SVR(cache_size=4000),
    #     "params": {
    #         "model__C": [1000],
    #         "model__kernel": ["rbf"],
    #         "model__gamma": ["scale"]
    #     }
    # }
    "Gradient Boosting Regressor": {
        "model": GradientBoostingRegressor(),
        "params": {
            "model__n_estimators": [200],
            "model__learning_rate": [0.01],
            "model__max_depth": [None],
            "model__max_features": ["log2"]
        }
    }
}

# Dimensionality reduction techniques
dimensionality_reduction = {
    # "None": None,
    "Select13Best": SelectKBest(f_classif, k=13),
    "Select14Best": SelectKBest(f_classif, k=14),
    "Select15Best": SelectKBest(f_classif, k=15)
}


def test_pipeline(apply_scaling):
    results = []
    for dr_name, dr_method in dimensionality_reduction.items():
        for model_name, model_info in models.items():
            steps = []

            # Add scaling if specified
            if apply_scaling:
                steps.append(("scaler", StandardScaler()))

            # Add dimensionality reduction if specified
            if dr_method is not None:
                steps.append(("reduction", dr_method))

            # Add model
            steps.append(("model", model_info["model"]))
            pipeline = Pipeline(steps)

            grid = GridSearchCV(
                pipeline,
                param_grid=model_info["params"],
                cv=5,
                scoring="neg_mean_absolute_error",
                verbose=10,
                n_jobs=-1,
            )
            grid.fit(X_train, y_train)

            # Evaluate on test set
            y_pred = grid.best_estimator_.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            results.append({
                "Scaling": "With Scaling" if apply_scaling else "Without Scaling",
                "Dimensionality Reduction": dr_name,
                "Model": model_name,
                "Best Params": grid.best_params_,
                "MAE Score": mae
            })
    return pd.DataFrame(results)


regression_results = test_pipeline(apply_scaling=False)

# Save the results to a CSV file
output_file = "regression_model_results.csv"
regression_results.to_csv(output_file, index=False)

print(f"Results have been saved to {output_file}")
