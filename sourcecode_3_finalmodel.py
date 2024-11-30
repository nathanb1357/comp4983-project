import pandas as pd
import pickle
import os
from OneHotEndoding import OneHotEncoder

test_file = "competitionset.csv"
output_dir = "./"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)


class PipelineManager:
    @staticmethod
    def save_model(file_path, model):
        """
        Save a scikit-learn pipeline (or any Python object) to a file.

        :param file_path: Path to save the pipeline.
        :param model: The scikit-learn pipeline or Python object to save.
        :raises: Exception if there is an error during saving.
        """
        try:
            with open(file_path, "wb") as file:
                pickle.dump(model, file)
        except Exception as e:
            raise RuntimeError(f"Error saving model to {file_path}: {e}")

    @staticmethod
    def load_model(file_path):
        """
        Load a scikit-learn pipeline (or any Python object) from a file.

        :param file_path: Path to the file to load the pipeline from.
        :return: The loaded scikit-learn pipeline or Python object.
        :raises: FileNotFoundError if the file does not exist.
        :raises: Exception if there is an error during loading.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            with open(file_path, "rb") as file:
                return pickle.load(file)
        except Exception as e:
            raise RuntimeError(f"Error loading model from {file_path}: {e}")


# Preprocessing step for class balancing with undersampling


def main():
    # Main workflow
    pipeline_manager = PipelineManager()

    classifier = pipeline_manager.load_model("./FinalModelClassifier.pkl")
    regressor = pipeline_manager.load_model("FinalModelRegressor.pkl")

    # Load test data
    print("Loading data...")
    test_data = pd.read_csv(test_file)
    print("Done...\n")

    print("Predicting ClaimAmount for test data...")
    test_features = test_data.drop(columns=["rowIndex"], inplace=False)
    test_data["ClaimAmount"] = 0  # Initialize all rows as non-claim

    # Identify rows predicted as claims
    claim_indices = classifier.predict(test_features).astype(bool)

    if claim_indices.sum() > 0:
        # Predict continuous values for claims
        continuous_predictions = regressor.predict(test_features.loc[claim_indices])
        test_data.loc[claim_indices, "ClaimAmount"] = continuous_predictions
    print("Done...\n")

    # Restore original rowIndex order
    test_data = test_data.sort_values("rowIndex")

    # Save results to a CSV file
    output_filename = f"predictedClaimAmount.csv"
    output_filepath = os.path.join(output_dir, output_filename)
    test_data[["rowIndex", "ClaimAmount"]].to_csv(output_filepath, index=False)

    print(f"Results saved to {output_filepath}.")


if __name__ == "__main__":
    main()
