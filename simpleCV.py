from Dataset import Dataset, find_best_model
import pandas as pd
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif

def evaluate_models(models_with_params, cv, scoring, training_features, training_labels):
    results = []

    for model, param_grid in models_with_params:
        best_model, best_params, best_score = find_best_model(
            model=model,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            training_features=training_features,
            training_labels=training_labels
        )

        results.append({
            'model': model,
            'params': best_params,
            'score': best_score
        })

    return results


dataset_path = "./original_data/trainingset.csv"
ds = Dataset(dataset_path, "ClaimAmount")
train_test_ratio = 0.8
ds.create_train_test(ratio=train_test_ratio)
# ds.reduce_train_ratio(0.3)
ds.define_label_features()
X_train = ds.train_features
y_train = ds.bin_train_label #use ds.bin_train label for binary y, or ds.train_label for claim amount (for regression)
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)


scoring = "f1"

models_with_params = [
    (LogisticRegression(max_iter=10000), {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'saga'],
        'penalty': ['l1', 'l2'],
        'class_weight': ['balanced', None],
    }),
    # (LogisticRegression(max_iter=10000), {
    #     'C': [0.01, 0.1, 1, 10, 100],
    #     'solver': ['sag', 'lbfgs', 'newton-cg', 'saga'],
    #     'penalty': ['l2'],
    #     'multi_class': ['ovr', 'multinomial'],
    #     'class_weight': ['balanced', None],
    # }),
    # (SVC(), {
    #     'C': [0.1, 1, 10, 100],
    #     'kernel': ['rbf', 'sigmoid'],
    #     'gamma': ['scale', 0.001, 0.0001],
    # }),
    # (Pipeline([ ('feature_selection', SelectKBest(score_func=f_classif)), ('classifier', RandomForestClassifier()) ]),
    #     {'feature_selection__k': [i for i in range(1, 19, 2)],
    #      'classifier__n_estimators': [100, 200, 300],
    #      'classifier__max_depth': [None, 10, 30],
    #      'classifier__max_features': ['sqrt', 'log2'],
    #      'classifier__criterion': ['gini', 'entropy'],
    #      'classifier__class_weight': ['balanced', None],
    #      # 'classifier__bootstrap': [True, False],
    #      'classifier__min_samples_split': [2, 5, 7],
    # }),
    # (KNeighborsClassifier(), {
    #     'n_neighbors': [i for i in range(1, 52, 10)],
    #     'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski'],
    #     'weights': ['uniform', 'distance'],
    # }),
]



results = evaluate_models(
    models_with_params=models_with_params,
    cv=5,
    scoring=scoring,
    training_features=X_train,
    training_labels=y_train
)
for result in results:
    print("Best Model:", result['model'])
    print("Best Parameters:", result['params'])
    print("Best Score:", result['score'])
    print("---")
