from sklearn.model_selection import GridSearchCV

def train_and_tune_model(pipeline, param_grid, X_train, y_train, scoring, cv=5):
    """Train and tune the model using GridSearchCV."""
    grid_search = GridSearchCV(pipeline, param_grid, scoring=scoring, cv=cv, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
