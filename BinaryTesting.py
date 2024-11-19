import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import f1_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearnex import patch_sklearn

patch_sklearn()

# Load dataset
data = pd.read_csv("original_data/trainingset.csv")
data["ClaimAmount"] = data["ClaimAmount"].apply(lambda x: 0 if x == 0 else 1)

X = data.drop(columns=["ClaimAmount"])
y = data["ClaimAmount"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Models and hyperparameters to test
models = {
    # "Logistic Regression": {
    #     "model": LogisticRegression(max_iter=10000),
    #     "params": {
    #         "model__C": [0.01],
    #         "model__solver": ["lbfgs"],
    #         "model__class_weight": ["balanced"]
    #     }
    # }
    # "Random Forest": {
    #     "model": RandomForestClassifier(max_depth=None),
    #     "params": {
    #         "model__n_estimators": [200],
    #         "model__max_features": ['sqrt'],
    #         "model__class_weight": ["balanced"]
    #     }
    # }
    # "KNN": {
    #     "model": KNeighborsClassifier(),
    #     "params": {
    #         "model__n_neighbors": [1],
    #         "model__weights": ['uniform']
    #     }
    # }
    # "SVM": {
    #     "model": SVC(cache_size=3000),
    #     "params": {
    #         "model__C": [10, 100, 1000],
    #         "model__kernel": ["rbf"],
    #         "model__gamma": ["scale"],
    #         "model__class_weight": ["balanced"]
    #     }
    # }
    "Gradient Boosting Classifier": {
        "model": GradientBoostingClassifier(),
        "params": {
            "model__n_estimators": [300, 400],
            "model__learning_rate": [0.2, 0.3, 0.5],
            "model__max_depth": [7, 10, 100, 500],
            "model__max_features": [None]
        }
    }
}

# Dimensionality reduction techniques
dimensionality_reduction = {
    # "None": None,
    # "Select13Best": SelectKBest(f_classif, k=13),
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

            grid = GridSearchCV(pipeline, param_grid=model_info["params"], cv=5, scoring="f1", verbose=10, n_jobs=-1)
            grid.fit(X_train, y_train)

            # Evaluate on test set
            y_pred = grid.best_estimator_.predict(X_test)
            f1 = f1_score(y_test, y_pred)
            results.append({
                "Scaling": "With Scaling" if apply_scaling else "Without Scaling",
                "Dimensionality Reduction": dr_name,
                "Model": model_name,
                "Best Params": grid.best_params_,
                "F1 Score": f1
            })
    return pd.DataFrame(results)


binary_results = test_pipeline(apply_scaling=False)

# Save the results to a CSV file
output_file = "binary_model_results.csv"
binary_results.to_csv(output_file, index=False)

print(f"Results have been saved to {output_file}")