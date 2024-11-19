import pandas as pd
import os
import json
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR  # For Support Vector Regressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, f1_score


# File paths (adjust if necessary)
train_file = "original_data/trainingset.csv"
test_file = "original_data/testset.csv"
output_dir = "./out"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

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

    def __repr__(self):
        return f"BalanceClasses(pos_ratio={self.pos_ratio})"

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
                SelectKBest(f_classif, k=13),
                # scaling=False (No StandardScaler)
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
]

# Helper function to serialize configurations as string representations
def serialize_configuration(config):
    def clean_string(s):
        return " ".join(s.replace("\n", " ").split())

    return {
        "classifier": {
            "model": clean_string(repr(config["classifier"]["model"])),
            "preprocessing": [clean_string(repr(step)) for step in config["classifier"]["preprocessing"]],
        },
        "regressor": {
            "model": clean_string(repr(config["regressor"]["model"])),
            "preprocessing": [clean_string(repr(step)) for step in config["regressor"]["preprocessing"]],
        },
    }

# Train and evaluate on a train/test split
def evaluate_config_on_trainset(train_data, classifier_config, regressor_config):
    print("Evaluating configuration on train/test split...")
    
    # Split the training set into train and test subsets
    X = train_data.drop(columns=["ClaimAmount", "rowIndex"])
    y_classifier = (train_data["ClaimAmount"] > 0).astype(int)  # Binary target for classifier
    y_regressor = train_data["ClaimAmount"]  # Continuous target for regressor

    # Split for classifier evaluation
    X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X, y_classifier, test_size=0.2, random_state=42)

    # Split for regressor evaluation
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        train_data[train_data["ClaimAmount"] > 0].drop(columns=["ClaimAmount", "rowIndex"]),
        train_data[train_data["ClaimAmount"] > 0]["ClaimAmount"],
        test_size=0.2,
        random_state=42,
    )

    # Train classifier
    classifier_pipeline_steps = []
    for step in classifier_config["preprocessing"]:
        if isinstance(step, BalanceClasses):
            print("Applying class balancing during evaluation...")
            step = step.fit(X_train_cls, y_train_cls)
            X_train_cls, y_train_cls = step.transform(X_train_cls, y_train_cls)
        else:
            classifier_pipeline_steps.append((f"step_{len(classifier_pipeline_steps)}", step))
    classifier_pipeline_steps.append(("classifier", classifier_config["model"]))
    classifier_pipeline = Pipeline(classifier_pipeline_steps)
    classifier_pipeline.fit(X_train_cls, y_train_cls)

    # Evaluate classifier
    y_pred_cls = classifier_pipeline.predict(X_test_cls)
    f1 = f1_score(y_test_cls, y_pred_cls)
    print(f"Classifier F1 score: {f1}")

    # Train regressor
    regressor_pipeline_steps = [
        *[(f"step_{i}", step) for i, step in enumerate(regressor_config["preprocessing"])],
        ("regressor", regressor_config["model"]),
    ]
    regressor_pipeline = Pipeline(regressor_pipeline_steps)
    regressor_pipeline.fit(X_train_reg, y_train_reg)

    # Evaluate regressor
    y_pred_reg = regressor_pipeline.predict(X_test_reg)
    mae = mean_absolute_error(y_test_reg, y_pred_reg)
    print(f"Regressor MAE: {mae}")

    return f1, mae

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
map_json = {}
for config_index, config in enumerate(configurations, start=1):
    print(f"Processing configuration {config_index}...")

    # Load training data
    print("Loading training data...")
    train_data = pd.read_csv(train_file)

    # Evaluate on train/test split
    f1, mae = evaluate_config_on_trainset(train_data, config["classifier"], config["regressor"])

    # Train classifier
    print("Starting classifier training...")
    classifier = train_classifier(train_data, config["classifier"])

    # Train regressor
    print("Starting regressor training...")
    regressor = train_regressor(train_data, config["regressor"])

    # Load test data
    print("Loading test data...")
    test_data = pd.read_csv(test_file)

    # Save rowIndex for final ordering
    test_data_row_index = test_data["rowIndex"]

    print("Predicting ClaimAmount for test data...")
    # Predict ClaimAmount
    test_features = test_data.drop(columns=["rowIndex"])
    test_data["ClaimAmount"] = 0  # Initialize all rows as non-claim

    # Identify rows predicted as claims
    claim_indices = classifier.predict(test_features).astype(bool)

    if claim_indices.sum() > 0:
        # Predict continuous values for claims
        continuous_predictions = regressor.predict(test_features.loc[claim_indices])
        test_data.loc[claim_indices, "ClaimAmount"] = continuous_predictions

    print("Combining and sorting results...")
    # Restore original rowIndex order
    test_data = test_data.sort_values("rowIndex")

    # Save results to a CSV file
    output_filename = f"2_3_{config_index}.csv"
    output_filepath = os.path.join(output_dir, output_filename)
    test_data[["rowIndex", "ClaimAmount"]].to_csv(output_filepath, index=False)

    # Add serialized configuration and metrics to the map.json
    map_json[output_filename] = {
        **serialize_configuration(config),
        "F1_score": f1,
        "MAE": mae,
    }

    print(f"Results for configuration {config_index} saved to {output_filepath}.")

# Save the map.json file
map_filepath = os.path.join(output_dir, "map.json")
with open(map_filepath, "w") as f:
    json.dump(map_json, f, indent=4)

print(f"Configuration map saved to {map_filepath}.")
