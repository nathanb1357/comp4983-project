import random
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier



class Dataset:
    def __init__(self, DSpath, label):
        self.data = pd.read_csv(DSpath)
        self.data = self.data.drop(self.data.columns[0], axis=1)
        self.non_zero_data = None
        self.subsets = {}
        self.train_data = None
        self.test_data = None
        self.label = label
        self.ratio = Dataset.define_ds_ratio(self.data, self.label)
        self.train_features = None
        self.test_features = None
        self.train_label = None
        self.test_label = None
        self.bin_train_label = None
        self.bin_test_label = None
        self.nz_train_features = None
        self.nz_train_label = None
        self.sub_train_features = None
        self.sub_train_label = None
        self.sub_test_features = None
        self.sub_test_label = None

    def summary(self):
        """
        Prints a summary of the dataset, including class distribution and feature information.
        """
        print(f"Dataset Shape: {self.data.shape}")
        print(f"Label Distribution:\n{self.data[self.label].value_counts(normalize=True)}")

    def save_subset(self, subset_key, file_path):
        if subset_key in self.subsets:
            self.subsets[subset_key].to_csv(file_path, index=False)
        else:
            print(f"Subset with key {subset_key} does not exist.")

    def load_subset(self, file_path):
        return pd.read_csv(file_path)

    @staticmethod
    def define_ds_ratio(ds, label):
        class_1_df = ds[ds[label] > 0]
        class_0_df = ds[ds[label] == 0]


        return len(class_1_df)/ (len(class_0_df) + len(class_1_df))

    def improve_train_ratio(self, target_ratio):
        # Separate the positive and negative classes
        class_1_df = self.train_data[self.train_data[self.label] > 0]
        class_0_df = self.train_data[self.train_data[self.label] == 0]

        print(f"Ratio = {self.ratio}")

        # Calculate the required number of negative samples to achieve the target ratio
        target_num_class_0 = int(len(class_1_df) * (1 - target_ratio) / target_ratio)

        # Check if we need to reduce the negatives
        if len(class_0_df) > target_num_class_0:
            # Downsample the negative class to the target number
            class_0_df = class_0_df.sample(n=target_num_class_0, random_state=42)

        # Concatenate the class dataframes back together and shuffle
        self.train_data = pd.concat([class_0_df, class_1_df]).sample(frac=1, random_state=42).reset_index(drop=True)

        # Recalculate and print the new ratio
        new_ratio = len(class_1_df) / len(self.train_data)
        print(f"New ratio: {new_ratio:.2f}")

    def create_non_zero_data(self, column):
        self.non_zero_data = self.data[self.data[column] != 0]

    def scale_features(self, scaler=StandardScaler()):
        """
        Scales features using the provided scaler or StandardScaler by default.
        :param scaler: An instance of a scaler (e.g., StandardScaler or MinMaxScaler).
        """
        scaler = scaler
        self.data[self.data.columns] = scaler.fit_transform(self.data[self.data.columns])


    def create_subsets(self, ratios, seed=None):
        '''
        Populates the subset attribute with subsets of the data for each given ratio.
        :param ratios: a list of ratios of fraud data over total data
        '''
        class_1_df = self.train_data[self.train_data[self.label] == 1]
        class_0_df = self.train_data[self.train_data[self.label] == 0]

        # Generate subsets for each ratio
        for ratio in ratios:
            if self.ratio > ratio:  # Too much fraud, reduce fraud cases
                target_class_0_count = len(class_0_df)  # Use all non-fraud cases
                target_class_1_count = int(round((ratio * target_class_0_count)/(1 - ratio)))
                target_class_1_df = class_1_df.sample(n=target_class_1_count, random_state=seed)
                subset_df = pd.concat([target_class_1_df, class_0_df])

            elif self.ratio < ratio:  # Too little fraud, reduce non-fraud cases
                target_class_1_count = len(class_1_df)  # Use all fraud cases
                target_class_0_count = int(round(target_class_1_count * (1 - ratio) / ratio))
                target_class_0_df = class_0_df.sample(n=target_class_0_count, random_state=seed)
                subset_df = pd.concat([class_1_df, target_class_0_df])

            else:  # Fraud rate matches the desired ratio
                # No sampling needed, use the full dataset
                subset_df = self.train_data.copy()

            # Ensure the total rows in subset match expected rows
            subset_size = target_class_1_count + target_class_0_count
            subset_df = subset_df.sample(frac=1, random_state=seed).reset_index(drop=True)

            # Call function to split feature from labels and store as tuple.
            # Store the subset in the dictionary with the ratio as the key
            # self.subsets[ratio] = subset_df
            self.subsets[ratio] = Dataset.split_feature_label(subset_df)


    def create_train_test(self, ratio, seed=None):
        '''
        Divides the data into train and test subsets.
        :param ratio: The ratio between training set and test set
        :param seed: seed for random splitting
        '''

        self.ratio = ratio
        if 0 < ratio < 1:
            num_rows = self.data.shape[0]
            shuffled_indices = list(range(num_rows))
            if seed:
                random.seed(seed)
            random.shuffle(shuffled_indices)
            self.train_set_size = int(num_rows * ratio)
            train_indices = shuffled_indices[:self.train_set_size]
            test_indices = shuffled_indices[self.train_set_size:]
            self.train_data = self.data.iloc[train_indices]
            self.test_data = self.data.iloc[test_indices]
        elif ratio == 0:
            self.test_data = self.data
        else:
            self.train_data = self.data

    def define_label_features(self, features=None):
        '''
        Divides the data into features and labels for the whole training set.
        :param label: name of the label column
        :param features: names of the features column
        '''

        if self.train_data is not None:
            self.train_label = self.train_data[self.label]
            self.bin_train_label = self.train_label.apply(lambda x: 1 if x > 0 else 0)

            if features:
                self.train_features = self.train_data[features]
            else:
                self.train_features = self.train_data.drop(columns=[self.label])

        if self.test_data is not None:
            self.test_label = self.test_data[self.label]
            self.bin_test_label = self.test_label.apply(lambda x: 1 if x > 0 else 0)
            if features:
                self.test_features = self.test_data[features]
            else:
                self.test_features = self.test_data.drop(columns=[self.label])

    def get_train_test(self):
        """
        Returns the train/test split as (X_train, X_test, y_train, y_test).
        """
        return self.train_features, self.test_features, self.train_label, self.test_label

    def generate_non_zero_data(self):
        """
        Creates nz_train_features and nz_train_label by filtering out rows
        where the train_label is zero.
        """
        if self.train_features is None or self.train_label is None:
            raise ValueError("Training features and labels must be defined before filtering non-zero data.")

        # Filter non-zero rows
        non_zero_mask = self.train_label != 0
        self.nz_train_features = self.train_features[non_zero_mask]
        self.nz_train_label = self.train_label[non_zero_mask]

    @staticmethod
    def select_features(features, labels, num_features=None, threshold=None, binary=True, random_state=None):
        """
        Performs feature selection using Random Forest.

        Parameters:
            features (pd.DataFrame): Feature matrix.
            labels (pd.Series): Target labels (binary or continuous).
            num_features (int): Number of top features to select. If None, use threshold.
            threshold (float): Minimum importance score to retain a feature. Ignored if num_features is provided.
            binary (bool): Whether to use binary or continuous label for feature selection.
            random_state (int): Random state for reproducibility.

        Returns:
            List of selected feature names.
        """
        if binary:
            rf = RandomForestClassifier(random_state=random_state)
        else:
            rf = RandomForestRegressor(random_state=random_state)

        rf.fit(features, labels)

        # Get feature importances
        feature_importances = rf.feature_importances_
        feature_names = features.columns

        # Create a DataFrame to store feature names and importances
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)

        # Select top features based on the method
        if num_features:
            selected_features = importance_df.head(num_features)['Feature']
        elif threshold:
            selected_features = importance_df[importance_df['Importance'] >= threshold]['Feature']
        else:
            raise ValueError("Specify either num_features or threshold for feature selection.")

        print(f"Selected Features:\n{selected_features.tolist()}")
        return selected_features.tolist()

    def create_feature_subset(self, feature_list):
        """
        Creates a subset of training data with only the specified features.
        Stores the result in new attributes: sub_train_features and sub_train_label.
        :param feature_list: List of feature names to include in the subset.
        """
        if self.train_features is None or self.train_label is None:
            raise ValueError("Training features and labels must be defined before creating a subset.")

        missing_features_train = [feature for feature in feature_list if feature not in self.train_features.columns]
        missing_features_test = [feature for feature in feature_list if feature not in self.test_features.columns]
        missing_features = set(missing_features_train + missing_features_test)
        if missing_features:
            raise ValueError(f"The following features are not present in the dataset: {missing_features}")

        # Create the subsets
        self.sub_train_features = self.train_features[feature_list].copy()
        self.sub_train_label = self.train_label.copy()
        self.sub_test_features = self.test_features[feature_list].copy()
        self.sub_test_label = self.test_label.copy()

        print(f"Created feature subsets:")
        print(f" - Training subset: {len(feature_list)} features, {self.sub_train_features.shape[0]} samples.")
        print(f" - Testing subset: {len(feature_list)} features, {self.sub_test_features.shape[0]} samples.")


#### for our case study we have used below for cross-validation

def find_best_model(model, param_grid, cv, scoring, training_features, training_labels):
    """
    Perform cross-validation to find the best model.

    Parameters:
    model (estimator): The machine learning model to be optimized.
    param_grid (dict): The parameter grid to search over.
    cv (int): The number of folds for cross-validation. the k-fold cross validation is done for each parameter grid.
    scoring (str): The scoring method to evaluate the model.
    training_features: Training data features.
    training_labels: Training data labels.

    Returns:
    best_model (estimator): The best model found by GridSearchCV.
    best_params (dict): The best parameters found by GridSearchCV.
    best_score (float): The best score achieved by the best model.

    Example parameters:
    model = SVC()
    param_grid = {
        'C': [0.1, 1, 10, 100],
    }
    scoring = 'accuracy'
    cv = 5


    """


    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=2)
    grid_search.fit(training_features, training_labels)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    return best_model, best_params, best_score

