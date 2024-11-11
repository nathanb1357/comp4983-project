import joblib

def save_model(model, filepath):
    """Save a model to a file."""
    joblib.dump(model, filepath)

def load_model(filepath):
    """Load a model from a file."""
    return joblib.load(filepath)
