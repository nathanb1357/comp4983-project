import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.utils import shuffle

# Paths
data_dir = "./data"
models_dir = "./models"
output_dir = "./out"
map_file_path = os.path.join(output_dir, "map.json")

# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load datasets
train_data_path = os.path.join(data_dir, "trainingset.csv")
test_data_path = os.path.join(data_dir, "testset.csv")

# Autoincrementing file index
def get_next_index():
    existing_files = [f for f in os.listdir(output_dir) if f.endswith(".csv") and f.startswith("submission")]
    indices = [
        int(f.split("-")[0]) for f in existing_files if f.split("-")[0].isdigit()
    ]
    return max(indices, default=0) + 1

# Prepare output mapping
model_param_map = {}

# Load training data
train_df = pd.read_csv(train_data_path)
train_df.reset_index(drop=True, inplace=True)

# Drop rowIndex and separate features and label
X_train = train_df.drop(columns=["rowIndex", "ClaimAmount"])
y_train = train_df["ClaimAmount"]

# Load test data
test_df = pd.read_csv(test_data_path)
test_row_indices = test_df["rowIndex"]  # Save rowIndex for submission
X_test = test_df.drop(columns=["rowIndex"])

# Process each model
for model_file in os.listdir(models_dir):
    if model_file.endswith(".pkl"):
        model_path = os.path.join(models_dir, model_file)
        model_name = os.path.splitext(model_file)[0]

        print(f"Processing model: {model_name}")

        # Load the model
        model = joblib.load(model_path)

        # Train the model 3 times
        for run in range(3):  # Limit to 3 runs
            print(f"Run {run + 1} for {model_name}...")

            # Randomize training data order
            randomized_train_df = shuffle(train_df)
            X_train = randomized_train_df.drop(columns=["rowIndex", "ClaimAmount"])
            y_train = randomized_train_df["ClaimAmount"]

            # Train the model
            model.fit(X_train, y_train)

            # Predict on test data
            predictions = model.predict(X_test)

            # Prepare submission output
            submission_df = pd.DataFrame({
                "rowIndex": test_row_indices,
                "ClaimAmount": predictions
            })

            # Autoincrement file name
            submission_index = get_next_index()
            submission_file_name = f"{submission_index}-submission-{model_name}-run{run + 1}.csv"
            submission_file_path = os.path.join(output_dir, submission_file_name)

            # Save submission file
            submission_df.to_csv(submission_file_path, index=False)
            print(f"Saved submission to {submission_file_path}")

            # Save model parameters in the map
            model_params = model.get_params() if hasattr(model, "get_params") else "Unknown"
            model_param_map[submission_file_name] = {
                "params": model_params,
                "run": run + 1
            }

# Save the map.json file
from pprint import pprint
pprint(model_param_map)
with open(map_file_path, "w") as map_file:
    json.dump(model_param_map, map_file, indent=4)
    print(f"Saved parameter map to {map_file_path}")
