from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from preprocessdata import preprocess_data
import json
import pandas as pd
import numpy as np


def train_classifier(data, model_config):
    """
    Trains a classifier model and evaluates it on training data.

    Parameters:
    - data (pd.DataFrame): The raw input dataset
    - model_config (dict): Configuration dictionary containing model parameters and preprocessing settings

    Returns:
    - tuple: (trained_classifier, feature_columns, training_metrics)
        - trained_classifier: The trained classifier model
        - feature_columns: List of column names used for training
        - training_metrics: Dictionary containing training set performance metrics
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
    processed_data = preprocess_data(data, preprocessing_params, is_training=True)

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

    # Calculate training metrics
    y_pred_train = classifier.predict(X)

    training_metrics = {
        "accuracy": accuracy_score(y, y_pred_train),
        "error_rate": 1 - accuracy_score(y, y_pred_train),
        "confusion_matrix": confusion_matrix(y, y_pred_train).tolist(),
        "classification_report": classification_report(
            y, y_pred_train, output_dict=True
        ),
    }

    return classifier, feature_columns, training_metrics


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
    all_metrics = {}

    for model in config["models"]:
        model_name = model.get("name", "unnamed_model")
        print(f"\n=======Processing {model_name}=======")

        # Train classifier and get feature columns and metrics
        classifier, feature_columns, training_metrics = train_classifier(
            train_df, model_config=model
        )

        # Store metrics
        all_metrics[model_name] = training_metrics

        # Print training metrics
        print("\nTraining Set Metrics:")
        print(f"Accuracy: {training_metrics['accuracy']:.4f}")
        print(f"Error Rate: {training_metrics['error_rate']:.4f}")
        print("\nConfusion Matrix:")
        print(np.array(training_metrics["confusion_matrix"]))
        print("\nClassification Report:")
        for label, metrics in training_metrics["classification_report"].items():
            if isinstance(metrics, dict):
                print(f"\n{label}:")
                print(f"Precision: {metrics['precision']:.4f}")
                print(f"Recall: {metrics['recall']:.4f}")
                print(f"F1-score: {metrics['f1-score']:.4f}")
                print(f"Support: {metrics['support']}")

        # Make binary predictions on test set
        predictions = predict_binary(
            classifier,
            feature_columns,
            test_df,
            model_config=model,
            model_name=model_name,
        )

        all_predictions[model_name] = predictions

        print(f"\nTest Set Predictions:")
        print(f"Number of predictions: {len(predictions)}")
        print(f"Positive predictions: {sum(predictions)}")
        print(f"Negative predictions: {len(predictions) - sum(predictions)}")
        print("-------------------")

    # Optionally, save metrics to a JSON file
    with open("training_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=4)

    print("\nAll predictions completed and saved!")
    print("Training metrics have been saved to 'training_metrics.json'")


if __name__ == "__main__":
    main()
