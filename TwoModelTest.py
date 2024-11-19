import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR  # For Support Vector Regressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, f1_score

# Preprocessing step for class balancing with undersampling
class BalanceClasses:
    """Custom preprocessing step for balancing classes via undersampling."""
    def __init__(self, pos_ratio):
        self.pos_ratio = pos_ratio

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if y is None:
            raise ValueError("y must be provided for class balancing.")
        
        print("Balancing classes using undersampling...")
        
        # Separate positive and negative samples
        pos_indices = y[y == 1].index
        neg_indices = y[y == 0].index

        # Convert neg_indices to a Pandas Series for sampling
        neg_indices = pd.Series(neg_indices)

        # Calculate the number of negative samples to keep
        n_neg_samples = int(len(pos_indices) / self.pos_ratio)

        print(f"Positive samples: {len(pos_indices)}, Negative samples before: {len(neg_indices)}")
        print(f"Sampling {n_neg_samples} negative samples to achieve pos_ratio={self.pos_ratio}.")

        # Sample negative indices
        sampled_neg_indices = neg_indices.sample(n=n_neg_samples, random_state=42)

        # Combine positive and sampled negative indices
        balanced_indices = pos_indices.union(sampled_neg_indices)

        print(f"Balanced dataset size: {len(balanced_indices)}")
        return X.loc[balanced_indices], y.loc[balanced_indices]

# Example configurations
configurations = [
    {
        "classifier": {
            "model": KNeighborsClassifier(
                n_neighbors=20,
                weights="distance"
            ),
            "preprocessing": [
                BalanceClasses(pos_ratio=0.2),
                SelectKBest(f_classif, k=13),
                StandardScaler(),  # scaling=True
            ],
        },
        "regressor": {
            "model": RandomForestRegressor(
                bootstrap=False,
                n_estimators=500,
                max_depth=40,
                max_features="sqrt"
            ),
            "preprocessing": [
                SelectKBest(f_classif, k=15),
                # scaling=False (No StandardScaler)
            ],
        },
    },
    {
        "classifier": {
            "model": KNeighborsClassifier(
                n_neighbors=15,
                weights="distance"
            ),
            "preprocessing": [
                BalanceClasses(pos_ratio=0.15),
                SelectKBest(f_classif, k=13),
                StandardScaler(),  # scaling=True
            ],
        },
        "regressor": {
            "model": RandomForestRegressor(
                bootstrap=False,
                n_estimators=500,
                max_depth=40,
                max_features="sqrt"
            ),
            "preprocessing": [
                SelectKBest(f_classif, k=15),
                # scaling=False (No StandardScaler)
            ],
        },
    },
    {
        "classifier": {
            "model": KNeighborsClassifier(
                n_neighbors=1,
                weights="uniform"
            ),
            "preprocessing": [
                SelectKBest(f_classif, k=13),
                # scaling=False (No StandardScaler)
            ],
        },
        "regressor": {
            "model": RandomForestRegressor(
                bootstrap=False,
                n_estimators=500,
                max_depth=40,
                max_features="sqrt"
            ),
            "preprocessing": [
                SelectKBest(f_classif, k=15),
                # scaling=False (No StandardScaler)
            ],
        },
    },
    {
        "classifier": {
            "model": KNeighborsClassifier(
                n_neighbors=1,
                weights="uniform"
            ),
            "preprocessing": [
                SelectKBest(f_classif, k=13),
                # scaling=False (No StandardScaler)
            ],
        },
        "regressor": {
            "model": KNeighborsRegressor(
                n_neighbors=800,
                algorithm="auto",
                p=1,  # Manhattan distance
                weights="distance"
            ),
            "preprocessing": [
                SelectKBest(f_classif, k=15),
                StandardScaler(),  # scaling=True
            ],
        },
    },
    {
        "classifier": {
            "model": KNeighborsClassifier(
                n_neighbors=1,
                weights="uniform"
            ),
            "preprocessing": [
                SelectKBest(f_classif, k=13),
                # scaling=False (No StandardScaler)
            ],
        },
        "regressor": {
            "model": SVR(
                C=1000,
                kernel="rbf",
                gamma="scale"
            ),
            "preprocessing": [
                SelectKBest(f_classif, k=15),
                # scaling=False (No StandardScaler)
            ],
        },
    },
    {
        "classifier": {
            "model": KNeighborsClassifier(
                n_neighbors=1,
                weights="uniform"
            ),
            "preprocessing": [
                SelectKBest(f_classif, k=13),
                # scaling=False (No StandardScaler)
            ],
        },
        "regressor": {
            "model": GradientBoostingRegressor(
                n_estimators=200,
                max_depth=None,
                max_features="log2",
                learning_rate=0.01
            ),
            "preprocessing": [
                SelectKBest(f_classif, k=15),
                # scaling=False (No StandardScaler)
            ],
        },
    },
    {
        "classifier": {
            "model": RandomForestClassifier(
                class_weight="balanced",
                n_estimators=200,
                max_depth=None,
                max_features="sqrt",
                criterion="entropy"
            ),
            "preprocessing": [
                BalanceClasses(pos_ratio=0.1),
                SelectKBest(f_classif, k=13),
                # scaling=False (No StandardScaler)
            ],
        },
        "regressor": {
            "model": RandomForestRegressor(
                bootstrap=False,
                n_estimators=500,
                max_depth=40,
                max_features="sqrt"
            ),
            "preprocessing": [
                SelectKBest(f_classif, k=15),
                # scaling=False (No StandardScaler)
            ],
        },
    },
    {
        "classifier": {
            "model": RandomForestClassifier(
                class_weight="balanced",
                n_estimators=200,
                max_depth=None,
                max_features="sqrt"
            ),
            "preprocessing": [
                BalanceClasses(pos_ratio=0.1),
                SelectKBest(f_classif, k=13),
                StandardScaler(),  # scaling=True
            ],
        },
        "regressor": {
            "model": KNeighborsRegressor(
                n_neighbors=800,
                algorithm="auto",
                p=1,
                weights="distance"
            ),
            "preprocessing": [
                SelectKBest(f_classif, k=15),
                StandardScaler(),  # scaling=True
            ],
        },
    },
    {
        "classifier": {
            "model": RandomForestClassifier(
                class_weight="balanced",
                n_estimators=200,
                max_depth=None,
                max_features="sqrt"
            ),
            "preprocessing": [
                BalanceClasses(pos_ratio=0.1),
                SelectKBest(f_classif, k=13),
                # scaling=False (No StandardScaler)
            ],
        },
        "regressor": {
            "model": SVR(
                C=1000,
                kernel="rbf",
                gamma="scale"
            ),
            "preprocessing": [
                SelectKBest(f_classif, k=15),
                # scaling=False (No StandardScaler)
            ],
        },
    },
    {
        "classifier": {
            "model": RandomForestClassifier(
                class_weight="balanced",
                n_estimators=200,
                max_depth=None,
                max_features="sqrt"
            ),
            "preprocessing": [
                BalanceClasses(pos_ratio=0.1),
                SelectKBest(f_classif, k=13),
                # scaling=False (No StandardScaler)
            ],
        },
        "regressor": {
            "model": GradientBoostingRegressor(
                n_estimators=200,
                max_depth=None,
                max_features="log2",
                learning_rate=0.01
            ),
            "preprocessing": [
                SelectKBest(f_classif, k=15),
                # scaling=False (No StandardScaler)
            ],
        },
    }
]

# Train Classifier
def train_classifier(train_data, classifier_config):
    print("Training classifier...")
    X = train_data.drop(columns=["ClaimAmount", "rowIndex"])  # Exclude "rowIndex"
    y = (train_data["ClaimAmount"] > 0).astype(int)

    pipeline_steps = []

    # Add preprocessing steps
    for step in classifier_config["preprocessing"]:
        if isinstance(step, BalanceClasses):
            print("Applying class balancing step...")
            # Apply balancing as part of preprocessing
            step = step.fit(X, y)
            X, y = step.transform(X, y)
        else:
            pipeline_steps.append((f"step_{len(pipeline_steps)}", step))

    # Add classifier model
    pipeline_steps.append(("classifier", classifier_config["model"]))

    # Build and train the pipeline
    pipeline = Pipeline(pipeline_steps)
    pipeline.fit(X, y)
    print("Classifier training complete.")
    return pipeline

# Train Regressor
def train_regressor(train_data, regressor_config):
    print("Training regressor...")
    train_data = train_data[train_data["ClaimAmount"] > 0]  # Drop rows where ClaimAmount = 0
    print(f"Training regressor with {len(train_data)} rows (ClaimAmount > 0).")
    X = train_data.drop(columns=["ClaimAmount", "rowIndex"])  # Exclude "rowIndex"
    y = train_data["ClaimAmount"]

    pipeline_steps = [
        *[(f"step_{i}", step) for i, step in enumerate(regressor_config["preprocessing"])],
        ("regressor", regressor_config["model"])
    ]

    # Build and train the pipeline
    pipeline = Pipeline(pipeline_steps)
    pipeline.fit(X, y)
    print("Regressor training complete.")
    return pipeline

# Main workflow
for config in configurations:
    print("Loading and splitting training data...")
    train_data = pd.read_csv("original_data/trainingset.csv")

    # Split data into training and testing sets
    train_set, test_set = train_test_split(train_data, test_size=0.2, random_state=42)

    print("Starting classifier training...")
    classifier = train_classifier(train_set, config["classifier"])

    print("Starting regressor training...")
    regressor = train_regressor(train_set, config["regressor"])

    # Test set evaluation
    print("Evaluating on the test set...")
    X_test = test_set.drop(columns=["ClaimAmount", "rowIndex"])
    y_test = (test_set["ClaimAmount"] > 0).astype(int)

    # Classifier predictions and F1 score
    y_pred_class = classifier.predict(X_test)
    f1 = f1_score(y_test, y_pred_class)
    print(f"F1 Score (Classifier): {f1}")

    # Regression MAE
    reg_test_data = test_set[test_set["ClaimAmount"] > 0]  # Rows where ClaimAmount > 0
    if not reg_test_data.empty:
        X_test_reg = reg_test_data.drop(columns=["ClaimAmount", "rowIndex"])
        y_test_reg = reg_test_data["ClaimAmount"]

        y_pred_reg = regressor.predict(X_test_reg)
        mae = mean_absolute_error(y_test_reg, y_pred_reg)
        print(f"Mean Absolute Error (Regressor): {mae}")
    else:
        print("No positive ClaimAmount values in the test set for regression evaluation.")
