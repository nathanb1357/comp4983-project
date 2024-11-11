from data_preprocessing import load_and_preprocess_data
from model_training import train_and_tune_model
from evaluation import plot_and_save_metrics
from utils import save_model
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.model_selection import ParameterGrid
import os
import numpy as np
from datetime import datetime
from simulation_configs import simulation_configs


def select_config():
    print("Available Configurations:")
    for i, key in enumerate(simulation_configs.keys(), start=1):
        print(f"{i}. {key}")
    choice = int(input("Enter the number of the configuration to run: ")) - 1
    selected_config = list(simulation_configs.keys())[choice]
    print(f"Selected Configuration: {selected_config}")
    return selected_config, simulation_configs[selected_config]


def get_output_directory(config_name):
    base_output_dir = "./out"
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)

    # Create directory for this configuration
    config_output_dir = os.path.join(base_output_dir, config_name)
    if not os.path.exists(config_output_dir):
        os.makedirs(config_output_dir)

    # Create unique simulation subdirectory
    simulation_count = len([name for name in os.listdir(config_output_dir) if os.path.isdir(os.path.join(config_output_dir, name))])
    datetime_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    simulation_dir = f"{simulation_count + 1}-{datetime_stamp}"
    simulation_output_dir = os.path.join(config_output_dir, simulation_dir)

    os.makedirs(simulation_output_dir)
    return simulation_output_dir


def log_progress(model_name, metric, param_index, total_params):
    print(f"Processing {model_name} - Metric: {metric}, Grid Search Progress: {param_index}/{total_params}")


# Main script
config_name, config = select_config()
output_dir = get_output_directory(config_name)

# Load and preprocess data
X, y = load_and_preprocess_data(config)

# Train and evaluate models
performance_data = {metric: [] for metric in config['metrics']}
for model_name, model_info in config['models'].items():
    print(f"Training {model_name}...")

    # Build pipeline
    if model_info['pipeline']['scaler']:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', eval(model_info['pipeline']['model'])(random_state=42 if model_name == "RandomForest" else None))
        ])
    else:
        pipeline = Pipeline([
            ('model', eval(model_info['pipeline']['model'])(random_state=42 if model_name == "RandomForest" else None))
        ])

    param_grid = model_info['params']

    for metric in config['metrics']:
        scorer = {
            "MSE": make_scorer(mean_squared_error, greater_is_better=False),
            "MAE": make_scorer(mean_absolute_error, greater_is_better=False),
            "R2": make_scorer(r2_score)
        }[metric]

        param_list = list(ParameterGrid(param_grid))
        total_params = len(param_list)

        for param_index, params in enumerate(param_list, start=1):
            # Log the progress of the grid search
            log_progress(model_name, metric, param_index, total_params)

            # Update pipeline with the current parameters
            pipeline.set_params(**params)

            # Train the model with current parameters
            pipeline.fit(X, y)

            # Evaluate the model
            y_pred = pipeline.predict(X)
            score = scorer._score_func(y, y_pred)

            print(f"Evaluated Params: {params}, Metric: {metric}, Score: {score:.4f}")

            if score > max(performance_data[metric], default=float('-inf')):
                performance_data[metric].append(score)

                # Save the best model with unique filename per metric
                model_path = os.path.join(output_dir, f"{model_name}_best_{metric}.pkl")
                save_model(pipeline, model_path)
                print(f"Updated best model saved to {model_path}")

# Plot and save metrics
pdf_path = os.path.join(output_dir, config['output_pdf'])
plot_and_save_metrics(performance_data, pdf_path)
print(f"Performance metrics saved to {pdf_path}")
