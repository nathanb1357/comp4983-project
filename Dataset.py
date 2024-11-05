import random
import pandas as pd
from sklearn.model_selection import GridSearchCV


class Dataset:
    def __init__(self, DSpath):
        self.data = pd.read_csv(DSpath)
        self.data = self.data.drop(self.data.columns[0], axis=1)
        self.subsets = {}
        self.train_data = None
        self.test_data = None
        self.ratio = None
        self.train_features = None
        self.test_features = None
        self.train_labels = None
        self.test_labels = None


    def create_subsets(self, ratios, seed=None):
        '''
        Populates the subset attribute with subsets of the data for each given ratio.
        :param ratios: a list of ratios of fraud data over total data
        '''
        class_1_df = self.train_data[self.train_data['Class'] == 1]
        class_0_df = self.train_data[self.train_data['Class'] == 0]

        # Generate subsets for each ratio
        for ratio in ratios:
            if self.fraud_rate > ratio:  # Too much fraud, reduce fraud cases
                target_class_0_count = len(class_0_df)  # Use all non-fraud cases
                target_class_1_count = int(round((ratio * target_class_0_count)/(1 - ratio)))
                target_class_1_df = class_1_df.sample(n=target_class_1_count, random_state=seed)
                subset_df = pd.concat([target_class_1_df, class_0_df])

            elif self.fraud_rate < ratio:  # Too little fraud, reduce non-fraud cases
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

    @staticmethod
    def split_feature_label(dataframe, label="Class"):
        '''

        :param dataframe: The dataframe to split between features and label
        :param label: the name of the label column in the dataframe
        :return: a dataframe of features and another of the frame.
        '''
        label_df = dataframe.loc[:, label]
        features = dataframe.drop(label, axis=1, inplace=False)
        return features, label_df

    def define_label_features(self, label, features=None):
        '''
        Divides the data into features and labels for the whole training set.
        :param label: name of the label column
        :param features: names of the features column
        '''
        if self.train_data is not None:
            self.train_label = self.train_data.loc[:,label]

            if features: # if we want to test specific features
                self.train_features = self.train_data.loc[:,features]
            else: # otherwise all features included
                self.train_features = self.train_data.drop(label, axis=1, inplace=False)

        if self.test_data is not None:
            self.test_label = self.test_data.loc[:, label]
            if features:
                self.test_features = self.test_data.loc[:, features]
            else:
                self.test_features = self.test_data.drop(label, axis=1, inplace=False)

    def define_test_only_no_label(self):
        self.test_features = self.data


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


    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring=scoring)
    grid_search.fit(training_features, training_labels)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    return best_model, best_params, best_score

