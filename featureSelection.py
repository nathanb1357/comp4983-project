import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier


def interaction_feature_selection(sub_train, sub_test, bin_train_label, bin_test_label, top_k=10):
    """
    Perform feature interaction selection and return combined datasets with selected interactions.

    Parameters:
    - sub_train (pd.DataFrame): Training features (non-zero subset).
    - sub_test (pd.DataFrame): Test features (non-zero subset).
    - bin_train_label (pd.Series): Binary training labels.
    - bin_test_label (pd.Series): Binary test labels.
    - top_k (int): Number of top interactions to select.

    Returns:
    - X_train_combined (pd.DataFrame): Training data with selected interaction features.
    - X_test_combined (pd.DataFrame): Test data with selected interaction features.
    """
    # Step 1: Generate all pairwise interaction terms
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_train_interactions = poly.fit_transform(sub_train)
    X_test_interactions = poly.transform(sub_test)
    interaction_feature_names = poly.get_feature_names_out(sub_train.columns)

    # Step 2: Select potentially useful interactions using mutual information
    mi_scores = mutual_info_classif(X_train_interactions, bin_train_label, random_state=42)
    mi_sorted_indices = mi_scores.argsort()[::-1]
    important_interactions = [(interaction_feature_names[i], mi_scores[i]) for i in mi_sorted_indices[:top_k]]

    print(f"Top {top_k} interactions by mutual information:")
    for name, score in important_interactions:
        print(f"Feature: {name}, Score: {score:.4f}")

    # Step 3: Refine selection using Random Forest
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train_interactions, bin_train_label)
    importances = rf.feature_importances_
    important_indices = np.argsort(importances)[-top_k:][::-1]
    important_interactions_rf = [(interaction_feature_names[i], importances[i]) for i in important_indices]

    print(f"\nTop {top_k} interactions by Random Forest:")
    for name, score in important_interactions_rf:
        print(f"Feature: {name}, Importance: {score:.4f}")

    # Step 4: Create a subset with selected interactions
    selected_indices = np.array([i for i, _ in important_interactions_rf], dtype=int)  # Ensure indices are integers
    X_train_selected_interactions = X_train_interactions[:, selected_indices]
    X_test_selected_interactions = X_test_interactions[:, selected_indices]

    # Combine selected interaction features with the original features
    X_train_combined = pd.concat(
        [sub_train.reset_index(drop=True), pd.DataFrame(X_train_selected_interactions, columns=[f"int_{i}" for i in range(len(selected_indices))])],
        axis=1
    )
    X_test_combined = pd.concat(
        [sub_test.reset_index(drop=True), pd.DataFrame(X_test_selected_interactions, columns=[f"int_{i}" for i in range(len(selected_indices))])],
        axis=1
    )

    return X_train_combined, X_test_combined