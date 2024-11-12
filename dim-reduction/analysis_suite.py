from data_preprocessing import load_and_preprocess_data
from evaluation import plot_and_save_metrics
from utils import save_model
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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


def log_progress(model_name, param_index, total_params, params, scores, log_file_path):
    metrics_log = ", ".join([f"{metric}: {score:.4f}" for metric, score in scores.items()])
    log_message = f"Processing {model_name}, Grid Search Progress: {param_index}/{total_params}\nParams: {params}\nMetrics: {metrics_log}"
    print(log_message)
    with open(log_file_path, "a") as log_file:
        log_file.write(log_message + "\n")


# Main script
config_name, config = select_config()
output_dir = get_output_directory(config_name)
log_file_path = os.path.join(output_dir, "progress.log")

# Load and preprocess data
X, y = load_and_preprocess_data(config)

# Train and evaluate models
best_models = {metric: None for metric in config['metrics']}  # Store best models for each metric
best_scores = {metric: float('inf') if metric in ["MSE", "MAE"] else float('-inf') for metric in config['metrics']}  # Track best scores for each metric
performance_data = {metric: {"params": [], "scores": []} for metric in config['metrics']}

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
    param_list = list(ParameterGrid(param_grid))

    total_params = len(param_list)

    for param_index, params in enumerate(param_list, start=1):
        # Set parameters for the current iteration
        pipeline.set_params(**params)

        # Train the model
        pipeline.fit(X, y)

        # Predict and calculate scores for all metrics
        y_pred = pipeline.predict(X)
        scores = {
            "MSE": mean_squared_error(y, y_pred),
            "MAE": mean_absolute_error(y, y_pred),
            "R2": r2_score(y, y_pred)
        }

        # Log metrics for the current parameter set
        log_progress(model_name, param_index, total_params, params, scores, log_file_path)

        # Save performance data for plotting
        for metric, score in scores.items():
            performance_data[metric]["params"].append(params)
            performance_data[metric]["scores"].append(score)

            # Update best model for this metric
            if ((metric in ["MSE", "MAE"] and score < best_scores[metric]) or
                (metric == "R2" and score > best_scores[metric])):  # Minimize MSE/MAE, maximize R2
                best_scores[metric] = score
                best_models[metric] = pipeline

                # Save the best model for this metric
                model_path = os.path.join(output_dir, f"{model_name}_best_{metric}.pkl")
                save_model(best_models[metric], model_path)
                log_message = f"Updated best model saved to {model_path}"
                print(log_message)
                with open(log_file_path, "a") as log_file:
                    log_file.write(log_message + "\n")

# Plot and save metrics
pdf_path = os.path.join(output_dir, config['output_pdf'])
plot_and_save_metrics(performance_data, param_list, pdf_path)
final_message = f"Performance metrics saved to {pdf_path}"
print(final_message)
with open(log_file_path, "a") as log_file:
    log_file.write(final_message + "\n")
