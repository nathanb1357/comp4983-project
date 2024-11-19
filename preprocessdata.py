from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
import pandas as pd


def preprocess_data(data, preprocessing_params, is_training=True):
    """
    Preprocesses the dataset according to the given preprocessing parameters.

    Parameters:
    - data (pd.DataFrame): The raw input dataset.
    - preprocessing_params (dict): A dictionary of preprocessing parameters with keys:
        - 'subset_selection': Number of features to select (int or None).
        - 'scaling': Whether to scale the data (bool).
        - 'pos_ratio': Desired positive ratio for balancing classes (float or None).
    - is_training (bool): Whether this is training data (with ClaimAmount) or test data

    Returns:
    - preprocessed_data (pd.DataFrame): The processed dataset.
    """
    # Create a copy to avoid modifying the original data
    data = data.copy()

    # Step 0: Create the target column only for training data
    if is_training:
        if "ClaimAmount" not in data.columns:
            raise ValueError("Training dataset must contain a 'ClaimAmount' column.")
        data["target"] = (data["ClaimAmount"] > 0).astype(int)

    # Step 1: Subset selection
    if preprocessing_params.get("subset_selection"):
        num_features = preprocessing_params["subset_selection"]
        selector = SelectKBest(score_func=f_classif, k=num_features)

        if is_training:
            features = data.drop(columns=["ClaimAmount", "target"], errors="ignore")
            target = data["target"]
            features_selected = selector.fit_transform(features, target)
            selected_feature_names = selector.get_feature_names_out(features.columns)
            data = pd.DataFrame(features_selected, columns=selected_feature_names)
            data["target"] = target  # Reattach target column
        else:
            # For test data, use the same feature selection
            features = (
                data.copy()
            )  # No need to drop target/ClaimAmount as they don't exist
            features_selected = selector.transform(features)
            selected_feature_names = selector.get_feature_names_out(features.columns)
            data = pd.DataFrame(features_selected, columns=selected_feature_names)

    # Step 2: Adjust positive ratio (only for training data)
    if is_training and preprocessing_params.get("pos_ratio"):
        pos_ratio = preprocessing_params["pos_ratio"]
        positive = data[data["target"] == 1]
        negative = data[data["target"] == 0]
        target_positive_count = int(pos_ratio * len(data))
        positive_sampled = positive.sample(
            n=target_positive_count, replace=True, random_state=42
        )
        negative_sampled = negative.sample(
            n=(len(data) - target_positive_count), replace=False, random_state=42
        )
        data = (
            pd.concat([positive_sampled, negative_sampled])
            .sample(frac=1, random_state=42)
            .reset_index(drop=True)
        )

    # Step 3: Scaling
    if preprocessing_params.get("scaling"):
        scaler = StandardScaler()
        if is_training:
            feature_columns = data.drop(
                columns=["ClaimAmount", "target"], errors="ignore"
            ).columns
        else:
            feature_columns = data.columns
        data[feature_columns] = scaler.fit_transform(data[feature_columns])

    return data
