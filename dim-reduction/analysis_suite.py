from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
from data_preprocessing import load_and_preprocess_data
from evaluation import plot_and_save_metrics
from utils import save_model
from loss import insurance_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import ParameterGrid
import os
import copy
import traceback
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


def evaluate_pipeline(X, y, pipeline, params, metrics):
    """
    Train and evaluate the pipeline for given parameters.
    """
    pipeline.set_params(**params)
    pipeline.fit(X, y)
    y_pred = pipeline.predict(X)

    scores = {
        "MSE": mean_squared_error(y, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y, y_pred)),
        "MAE": mean_absolute_error(y, y_pred),
        "R2": r2_score(y, y_pred),
        "insurance_score": insurance_score(y, y_pred)
    }
    return {"params": params, "scores": scores, "pipeline": copy.deepcopy(pipeline)}


def summarize_findings(best_models, best_scores, X, y, log_file_path):
    """
    Summarize the best-performing models for each metric and evaluate their performance across all metrics.

    Parameters:
    - best_models: Dictionary of best models for each metric.
    - best_scores: Dictionary of best scores for each metric.
    - X: Features to evaluate the best models.
    - y: True labels to evaluate the best models.
    - log_file_path: File to append the summary.
    """
    summary = "\nFinal Results Summary:\n"
    for metric, best_model in best_models.items():
        if best_model is None:
            continue

        # Retrieve the best parameters and evaluate across all metrics
        best_params = best_model.named_steps['model'].get_params()  # Extract parameters of the model step
        y_pred = best_model.predict(X)
        scores = {
            "MSE": mean_squared_error(y, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y, y_pred)),
            "MAE": mean_absolute_error(y, y_pred),
            "R2": r2_score(y, y_pred),
            "insurance_score": insurance_score(y, y_pred)
        }

        summary += f"\nBest Model for {metric}:\n"
        summary += f"  Parameters: {best_params}\n"
        summary += f"  Performance:\n"
        for score_metric, score_value in scores.items():
            summary += f"    {score_metric}: {score_value:.4f}\n"

    print(summary)
    with open(log_file_path, "a") as log_file:
        log_file.write(summary)


# Main script
config_name, config = select_config()
output_dir = get_output_directory(config_name)
log_file_path = os.path.join(output_dir, "progress.log")

# Load and preprocess data
X, y = load_and_preprocess_data(config)

# Shared resources for managing best models and scores
manager = Manager()
best_models = manager.dict({metric: None for metric in config['metrics']})  # Shared best models
best_scores = manager.dict({metric: float('inf') if metric in ["MSE", "RMSE", "MAE"] else float('-inf') for metric in config['metrics']})  # Shared best scores
lock = manager.Lock()  # Lock to prevent race conditions

performance_data = {metric: {"params": [], "scores": []} for metric in config['metrics']}

for model_name, model_info in config['models'].items():
    print(f"Training {model_name}...")

    # Build pipeline
    if model_info['pipeline']['scaler']:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', eval(model_info['pipeline']['model'])())
        ])
    else:
        pipeline = Pipeline([
            ('model', eval(model_info['pipeline']['model'])())
        ])

    param_grid = model_info['params']
    param_list = list(ParameterGrid(param_grid))

    total_params = len(param_list)

    # Parallelize grid search
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {
            executor.submit(evaluate_pipeline, X, y, copy.deepcopy(pipeline), params, config['metrics']): params
            for params in param_list
        }

        for param_index, future in enumerate(as_completed(futures), start=1):
            try:
                result = future.result()
                params, scores, trained_pipeline = result["params"], result["scores"], result["pipeline"]

                # Log metrics for the current parameter set
                log_progress(model_name, param_index, total_params, params, scores, log_file_path)

                # Save performance data for plotting
                for metric, score in scores.items():
                    performance_data[metric]["params"].append(copy.deepcopy(params))
                    performance_data[metric]["scores"].append(score)

                    # Synchronize access to shared resources
                    with lock:
                        if ((metric in ["MSE", "RMSE", "MAE"] and score < best_scores[metric]) or
                            (metric in ["R2", "insurance_score"] and score > best_scores[metric])):  # Minimize MSE/RMSE/MAE, maximize R2
                            best_scores[metric] = score
                            best_models[metric] = copy.deepcopy(trained_pipeline)

                            # Save the best model for this metric
                            model_path = os.path.join(output_dir, f"best_{metric}.pkl")
                            save_model(best_models[metric], model_path)
                            log_message = f"Updated best model saved to {model_path}"
                            print(log_message)
                            with open(log_file_path, "a") as log_file:
                                log_file.write(log_message + "\n")

            except Exception as e:
                print(f"Error during evaluation: {e}")
                traceback.print_exc()  # Prints the traceback of the exception

# Plot and save metrics
pdf_path = os.path.join(output_dir, config['output_pdf'])
plot_and_save_metrics(performance_data, param_list, pdf_path)
final_message = f"Performance metrics saved to {pdf_path}"
print(final_message)
with open(log_file_path, "a") as log_file:
    log_file.write(final_message + "\n")

# Summarize findings
summarize_findings(best_models, best_scores, X, y, log_file_path)
