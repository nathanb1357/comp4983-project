from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from preprocessdata import preprocess_data
import json
import pandas as pd
import numpy as np


def train_classifier(data, model_config):
    """
    Trains a classifier model based on the model configuration.

    Parameters:
    - data (pd.DataFrame): The raw input dataset
    - model_config (dict): Configuration dictionary containing model parameters and preprocessing settings

    Returns:
    - trained_classifier: The trained classifier model
    """
    # Extract classifier parameters and preprocessing settings
    classifier_type = (
        "KNN"
        if "n_neighbors" in model_config["parameters"]["classifier"]
        else "RandomForest"
    )
    classifier_params = model_config["parameters"]["classifier"]
    preprocessing_params = model_config.get("preprocessing", {}).get("classifier", {})

    # Preprocess the data
    processed_data = preprocess_data(data, preprocessing_params)

    # Separate features and target
    X = processed_data.drop(["target", "ClaimAmount"], axis=1, errors="ignore")
    y = processed_data["target"]

    # Store feature columns
    feature_columns = X.columns.tolist()

    # Initialize the appropriate classifier
    if classifier_type == "KNN":
        classifier = KNeighborsClassifier(
            n_neighbors=classifier_params.get("n_neighbors", 5),
            weights=classifier_params.get("weights", "uniform"),
        )
    elif classifier_type == "RandomForest":
        classifier = RandomForestClassifier(
            n_estimators=classifier_params.get("n_estimators", 100),
            max_depth=classifier_params.get("max_depth", None),
            max_features=classifier_params.get("max_features", "sqrt"),
            class_weight=classifier_params.get("class_weight", None),
            criterion=classifier_params.get("criterion", "gini"),
            random_state=42,
        )
    else:
        raise ValueError(f"Unsupported classifier type: {classifier_type}")

    # Train the classifier
    classifier.fit(X, y)

    return classifier, feature_columns


def predict_binary(classifier, feature_columns, test_data, model_config, model_name):
    """
    Predicts binary outcomes using the trained classifier.

    Parameters:
    - classifier: Trained classifier model
    - feature_columns: List of column names used during training
    - test_data (pd.DataFrame): Raw test dataset
    - model_config (dict): Configuration dictionary
    - model_name (str): Name of the model for saving predictions

    Returns:
    - predictions (np.array): Binary predictions
    """

    # Ensure test data has same columns as training data
    processed_test_data = test_data[feature_columns]

    # Make predictions
    predictions = classifier.predict(processed_test_data)

    # Create DataFrame with predictions
    predictions_df = pd.DataFrame(
        {
            "row_index": range(len(predictions)),
            f"{model_name}_binary_prediction": predictions,
        }
    )

    # Save predictions to CSV
    output_filename = f"predictions_{model_name}_binary.csv"
    predictions_df.to_csv(output_filename, index=False)

    # Also save just the predictions in a simple format
    np.savetxt(f"predictions_{model_name}_binary_simple.txt", predictions, fmt="%d")

    return predictions


def main():
    with open("./formattedParams.json") as file:
        config = json.load(file)

    train_df = pd.read_csv("./original_data/trainingset.csv")
    test_df = pd.read_csv("./original_data/testset.csv")

    all_predictions = {}

    for model in config["models"]:
        model_name = model.get("name", "unnamed_model")
        print(f"=======Processing {model_name}=======")

        # Train classifier and get feature columns
        classifier, feature_columns = train_classifier(train_df, model_config=model)

        # Make binary predictions
        predictions = predict_binary(
            classifier,
            feature_columns,
            test_df,
            model_config=model,
            model_name=model_name,
        )

        all_predictions[model_name] = predictions

        print(f"Predictions saved for {model_name}")
        print(f"Number of predictions: {len(predictions)}")
        print(f"Positive predictions: {sum(predictions)}")
        print(f"Negative predictions: {len(predictions) - sum(predictions)}")
        print("-------------------")

    print("All predictions completed and saved!")


if __name__ == "__main__":
    main()
