""" code to reduce dimensionality without sacrificing performance

    contains methods to actually reduce dimensionality
    by removing features, collapsing features, applying transformations

    contains methods to score the performance of a model
    with those datasets with preprocessing methods applied

    data flow:
        original data -> preprocessing -> model fit/predict -> score

    in this flow:
        - original data stays the same within a single "run" but can be swapped between runs
        - many different preprocessing techniques in a "chain of responsibility" architecture
        - model stays the same within a single "run" but can be swapped between runs
        - scoring method is static but should be extensible in-code
"""
import os
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    make_scorer,
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


class PreprocessingStep(ABC):
    """Abstract base class for preprocessing steps in the chain."""

    def __init__(self):
        self._next_step = None

    def set_next(self, next_step):
        """Set the next preprocessing step in the chain."""
        self._next_step = next_step

    @abstractmethod
    def process(self, X):
        """Process the data and pass it to the next step."""
        pass

    def _process_next(self, X):
        if self._next_step:
            return self._next_step.process(X)
        else:
            return X


class ScalingStep(PreprocessingStep):
    """Scaling features using StandardScaler."""

    def __init__(self):
        super().__init__()
        self.scaler = StandardScaler()

    def process(self, X):
        X_scaled = self.scaler.fit_transform(X)
        return self._process_next(X_scaled)


class PCAStep(PreprocessingStep):
    """Apply PCA for dimensionality reduction."""

    def __init__(self, n_components):
        super().__init__()
        self.pca = PCA(n_components=n_components)

    def process(self, X):
        X_pca = self.pca.fit_transform(X)
        print(f"PCA applied: reduced to {self.pca.n_components_} components.")
        return self._process_next(X_pca)


class FeatureSelectionStep(PreprocessingStep):
    """Feature selection by removing low-variance features."""

    def __init__(self, threshold=0.1):
        super().__init__()
        self.threshold = threshold
        self.selector = None

    def process(self, X):
        from sklearn.feature_selection import VarianceThreshold

        self.selector = VarianceThreshold(threshold=self.threshold)
        X_selected = self.selector.fit_transform(X)
        print(f"Feature selection applied: reduced to {X_selected.shape[1]} features.")
        return self._process_next(X_selected)


class DataProcessor:
    """Handles preprocessing steps like scaling, dimensionality reduction, etc."""

    def __init__(self, data, target_column):
        self.data = data
        self.target_column = target_column
        self.X = None
        self.y = None
        self.preprocessor_chain = None

    def split_features_target(self):
        """Separate features (X) and target (y)."""
        self.X = self.data.drop(columns=[self.target_column])
        self.y = self.data[self.target_column]
        return self.X, self.y

    def set_preprocessor_chain(self, chain):
        """Set the chain of preprocessing steps."""
        self.preprocessor_chain = chain

    def process_features(self):
        """Process features through the chain of preprocessing steps."""
        if self.preprocessor_chain:
            self.X = self.preprocessor_chain.process(self.X)
        return self.X


class Evaluator(ABC):
    """Abstract base class for evaluators."""

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def get_metrics(self):
        pass


class ModelEvaluator(Evaluator):
    """Handles training, predictions, and metric evaluations."""

    def __init__(self, model, X_train, X_test, y_train, y_test):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_pred = None
        self.metrics = {}

    def evaluate(self):
        """Train the model and evaluate metrics."""
        self.train_model()
        self.predict()
        self.evaluate_metrics()

    def train_model(self):
        """Train the machine learning model."""
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        """Make predictions on the test set."""
        self.y_pred = self.model.predict(self.X_test)
        return self.y_pred

    def evaluate_metrics(self):
        """Evaluate and print performance metrics."""
        self.metrics["Mean Squared Error"] = mean_squared_error(
            self.y_test, self.y_pred
        )
        self.metrics["Mean Absolute Error"] = mean_absolute_error(
            self.y_test, self.y_pred
        )
        self.metrics["R2 Score"] = r2_score(self.y_test, self.y_pred)
        print("Performance Metrics:")
        for metric, value in self.metrics.items():
            print(f"{metric}: {value:.4f}")
        return self.metrics

    def add_metric(self, metric_name, metric_function):
        """Add a custom metric."""
        value = metric_function(self.y_test, self.y_pred)
        self.metrics[metric_name] = value

    def get_metrics(self):
        """Return the calculated metrics."""
        return self.metrics

    def residuals_plot(self, output_path=None):
        """Generate and save a residuals plot."""
        residuals = self.y_test - self.y_pred
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=self.y_pred, y=residuals)
        plt.axhline(y=0, color="r", linestyle="--")
        plt.title("Residuals Plot")
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        if output_path:
            plt.savefig(output_path)
            print(f"Residuals plot saved to {output_path}")
        plt.show()


class CrossValidator(Evaluator):
    """Handles cross-validation of the model."""

    def __init__(self, model, X, y, cv=5, scoring=None):
        self.model = model
        self.X = X
        self.y = y
        self.cv = cv
        self.scoring = scoring
        self.scores = {}

    def evaluate(self):
        """Perform cross-validation and compute metrics."""
        cv_results = cross_validate(
            self.model,
            self.X,
            self.y,
            cv=self.cv,
            scoring=self.scoring,
            return_train_score=False,
        )
        self.scores = cv_results

    def get_metrics(self):
        """Return the cross-validation metrics."""
        return self.scores

    def print_scores(self):
        """Print the cross-validation scores."""
        print("Cross-Validation Metrics:")
        for key in self.scores:
            mean = np.mean(self.scores[key])
            std = np.std(self.scores[key])
            print(f"{key}: {mean:.4f} ± {std:.4f}")


# Main Script
if __name__ == "__main__":
    # File path to the dataset (can be swapped between runs)
    file_path = "./original_data/trainingset.csv"

    # Load the dataset
    data = pd.read_csv(file_path)

    # Remove the 'rowIndex' column
    data = data.drop(columns=["rowIndex"])

    # Initialize DataProcessor with target column
    target_column = "ClaimAmount"  # Specify the target column
    processor = DataProcessor(data, target_column)

    # Split features and target
    X, y = processor.split_features_target()

    # Build preprocessing chain (can include multiple steps)
    scaling_step = ScalingStep()
    feature_selection_step = FeatureSelectionStep(threshold=0.1)
    pca_step = PCAStep(n_components=5)

    # Set up the chain of responsibility
    scaling_step.set_next(feature_selection_step)
    feature_selection_step.set_next(pca_step)

    # Assign the chain to the processor
    processor.set_preprocessor_chain(scaling_step)

    # Process features through the chain
    X = processor.process_features()

    # Initialize the model (can be swapped between runs)
    model = RandomForestRegressor(random_state=42)

    # Define scoring metrics for cross-validation
    scoring = {
        "neg_mean_squared_error": make_scorer(
            mean_squared_error, greater_is_better=False
        ),
        "neg_mean_absolute_error": make_scorer(
            mean_absolute_error, greater_is_better=False
        ),
        "r2": make_scorer(r2_score),
    }

    # Initialize CrossValidator
    cross_validator = CrossValidator(model, X, y, cv=5, scoring=scoring)

    # Perform cross-validation
    cross_validator.evaluate()

    # Get and print cross-validation metrics
    cross_validator.print_scores()

    # Optionally, perform a train/test split for additional evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize ModelEvaluator
    evaluator = ModelEvaluator(model, X_train, X_test, y_train, y_test)

    # Evaluate the model
    evaluator.evaluate()

    # Optionally add custom metrics
    # For example, Mean Absolute Percentage Error (MAPE)
    def mean_absolute_percentage_error(y_true, y_pred):
        """Custom metric: Mean Absolute Percentage Error."""
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    evaluator.add_metric("MAPE", mean_absolute_percentage_error)
    print(f"MAPE: {evaluator.metrics['MAPE']:.2f}%")

    # Plot residuals
    output_dir = "./out/"
    os.makedirs(output_dir, exist_ok=True)
    residuals_output_path = os.path.join(output_dir, "residuals_plot.png")
    evaluator.residuals_plot(output_path=residuals_output_path)
