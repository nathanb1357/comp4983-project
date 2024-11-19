# Dimensionality reduction techniques
from _typeshed import WriteableBuffer


dimensionality_reduction = {
    # "None": None,
    "Select13Best": SelectKBest(f_classif, k=13),
    "Select14Best": SelectKBest(f_classif, k=14),
    "Select15Best": SelectKBest(f_classif, k=15)
}


def test_pipeline(apply_scaling):
    results = []
    for dr_name, dr_method in dimensionality_reduction.items():
        for model_name, model_info in models.items():
            steps = []

            # Add scaling if specified
            if apply_scaling:
                steps.append(("scaler", StandardScaler()))

            # Add dimensionality reduction if specified
            if dr_method is not None:
                steps.append(("reduction", dr_method))

            # Add model
            steps.append(("model", model_info["model"]))
            pipeline = Pipeline(steps)

            grid = GridSearchCV(
                pipeline,
                param_grid=model_info["params"],
                cv=5,
                scoring="neg_mean_absolute_error",
                verbose=10,
                n_jobs=-1,
            )
            grid.fit(X_train, y_train)

            # Evaluate on test set
            y_pred = grid.best_estimator_.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            results.append({
                "Scaling": "With Scaling" if apply_scaling else "Without Scaling",
                "Dimensionality Reduction": dr_name,
                "Model": model_name,
                "Best Params": grid.best_params_,
                "MAE Score": mae
            })
    return pd.DataFrame(results)
