from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
from preprocessdata import preprocess_data
import json
import pandas as pd
from classifier import train_classifier, predict_binary


def train_regressor(data, model_config):
    """
    Trains a regressor model and evaluates it on training data.

    Parameters:
    - data (pd.DataFrame): The raw input dataset
    - model_config (dict): Configuration dictionary containing model parameters and preprocessing settings

    Returns:
    - tuple: (trained_regressor, feature_columns, training_metrics)
        - trained_regressor: The trained regressor model
        - feature_columns: List of column names used for training
        - training_metrics: Dictionary containing training set performance metrics
    """
    # Extract regressor parameters and preprocessing settings
    regressor_params = model_config["parameters"]["regressor"]
    preprocessing_params = model_config.get("preprocessing", {}).get("regressor", {})

    # Preprocess the data
    processed_data = preprocess_data(data, preprocessing_params, is_training=True)

    # Filter only positive cases (where ClaimAmount > 0)
    positive_cases = processed_data[data["ClaimAmount"] > 0].copy()
    positive_cases["ClaimAmount"] = data["ClaimAmount"]

    # Separate features and target
    X = positive_cases.drop(["target", "ClaimAmount"], axis=1, errors="ignore")
    y = positive_cases["ClaimAmount"]

    # Store feature columns
    feature_columns = X.columns.tolist()

    # Initialize the regressor
    regressor = RandomForestRegressor(
        n_estimators=regressor_params.get("n_estimators", 100),
        max_depth=regressor_params.get("max_depth", None),
        max_features=regressor_params.get("max_features", "sqrt"),
        bootstrap=regressor_params.get("bootstrap", True),
        random_state=42,
    )

    # Train the regressor
    regressor.fit(X, y)

    # Calculate training metrics
    y_pred_train = regressor.predict(X)

    training_metrics = {
        "mae": mean_absolute_error(y, y_pred_train),
        "r2": r2_score(y, y_pred_train),
        "mean_prediction": np.mean(y_pred_train),
        "mean_actual": np.mean(y),
    }

    return regressor, feature_columns, training_metrics


def predict_regression(
    regressor,
    feature_columns,
    test_data,
    classifier_predictions,
    model_config,
    model_name,
):
    """
    Predicts claim amounts for cases predicted as positive by the classifier.

    Parameters:
    - regressor: Trained regressor model
    - feature_columns: List of column names used during training
    - test_data (pd.DataFrame): Raw test dataset
    - classifier_predictions: Binary predictions from classifier
    - model_config (dict): Configuration dictionary
    - model_name (str): Name of the model for saving predictions

    Returns:
    - predictions (np.array): Regression predictions (0 for negative cases)
    """
    # Preprocess test data using regressor preprocessing parameters
    preprocessing_params = model_config.get("preprocessing", {}).get("regressor", {})
    processed_test_data = preprocess_data(
        test_data, preprocessing_params, is_training=False
    )

    # Ensure test data has same columns as training data
    processed_test_data = processed_test_data[feature_columns]

    # Initialize predictions array with zeros
    final_predictions = np.zeros(len(test_data))

    # Get positive cases indices
    positive_indices = np.where(classifier_predictions == 1)[0]

    if len(positive_indices) > 0:
        # Make predictions only for positive cases
        positive_cases = processed_test_data.iloc[positive_indices]
        positive_predictions = regressor.predict(positive_cases)

        # Transform predictions back to original scale
        positive_predictions = np.expm1(positive_predictions)

        # Assign predictions to positive cases
        final_predictions[positive_indices] = positive_predictions

    # Save predictions
    predictions_df = pd.DataFrame(
        {
            "row_index": range(len(final_predictions)),
            f"{model_name}_regression_prediction": final_predictions,
        }
    )

    output_filename = f"predictions_{model_name}_regression.csv"
    predictions_df.to_csv(output_filename, index=False)

    # Also save in simple format
    np.savetxt(
        f"predictions_{model_name}_regression_simple.txt", final_predictions, fmt="%.2f"
    )

    return final_predictions


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

        # Train and evaluate classifier
        classifier, clf_features, clf_metrics = train_classifier(
            train_df, model_config=model
        )

        # Train and evaluate regressor
        regressor, reg_features, reg_metrics = train_regressor(
            train_df, model_config=model
        )

        # Store metrics
        all_metrics[model_name] = {
            "classifier_metrics": clf_metrics,
            "regressor_metrics": reg_metrics,
        }

        # Print training metrics
        print("\nClassifier Training Metrics:")
        print(f"Error Rate: {clf_metrics['error_rate']:.4f}")

        print("\nRegressor Training Metrics:")
        print(f"MAE: {reg_metrics['mae']:.2f}")
        print(f"R2 Score: {reg_metrics['r2']:.4f}")
        print(f"Mean Prediction: {reg_metrics['mean_prediction']:.2f}")
        print(f"Mean Actual: {reg_metrics['mean_actual']:.2f}")

        # Make predictions
        clf_predictions = predict_binary(
            classifier, clf_features, test_df, model_config=model, model_name=model_name
        )

        reg_predictions = predict_regression(
            regressor,
            reg_features,
            test_df,
            clf_predictions,
            model_config=model,
            model_name=model_name,
        )

        all_predictions[model_name] = {
            "binary": clf_predictions,
            "regression": reg_predictions,
        }

        print("\nPredictions Summary:")
        print(f"Positive cases: {sum(clf_predictions)}")
        print(f"Mean predicted amount: {np.mean(reg_predictions):.2f}")
        print("-------------------")

    # Save metrics
    with open("training_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=4)

    print("\nAll predictions completed and saved!")
    print("Training metrics have been saved to 'training_metrics.json'")


if __name__ == "__main__":
    main()
