import pandas as pd
from sklearn.utils import shuffle

def load_and_preprocess_data(config):
    """Load and preprocess the dataset based on the configuration."""
    data = pd.read_csv(config['dataset']['path'])
    data = shuffle(data)
    if 'drop_columns' in config['dataset']:
        data = data.drop(columns=config['dataset']['drop_columns'])
    X = data.drop(columns=[config['dataset']['target_column']])
    y = data[config['dataset']['target_column']]
    return X, y
