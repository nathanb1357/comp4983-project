import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def plot_and_save_metrics(performance_data, param_grid, output_pdf):
    """
    Plot metrics and save them to a PDF file, with breakdowns for individual parameter variations.

    Parameters:
    - performance_data: Dictionary containing scores for each parameter combination and metric.
    - param_grid: List of parameter combinations (from GridSearch).
    - output_pdf: Path to save the PDF file with plots.
    """
    with PdfPages(output_pdf) as pdf:
        for metric, results in performance_data.items():
            # Separate results for each parameter being varied
            for param_to_vary in results["params"][0].keys():  # Iterate over hyperparameter keys
                # Group results where all other parameters are fixed
                groups = group_results_by_fixed_params(results["params"], param_to_vary)

                for fixed_params, group in groups.items():
                    # Extract the values of the parameter being varied and corresponding metric scores
                    varied_values = [params[param_to_vary] for params in group["params"]]
                    scores = group["scores"]

                    # Create a meaningful title that includes the fixed parameter settings
                    fixed_params_text = "; ".join([f"{key}={value}" for key, value in fixed_params])
                    title = f"{metric} Performance - Varying {param_to_vary} (Fixed: {fixed_params_text})"

                    # Plot the metric performance
                    plt.figure(figsize=(10, 6))
                    plt.plot(varied_values, scores, marker='o', label=f"{metric}")
                    plt.xlabel(f"{param_to_vary}")
                    plt.ylabel(f"{metric} Score")
                    plt.title(title)
                    plt.legend()
                    plt.grid()
                    plt.tight_layout()

                    # Save each plot to the PDF
                    pdf.savefig()
                    plt.close()

    print(f"Performance plots saved to {output_pdf}.")


def group_results_by_fixed_params(param_combinations, param_to_vary):
    """
    Group parameter combinations by fixed parameters (excluding the one being varied).

    Parameters:
    - param_combinations: List of parameter dictionaries.
    - param_to_vary: The parameter being varied.

    Returns:
    - Dictionary where keys are fixed parameter settings (as tuples) and values are grouped results.
    """
    groups = {}
    for i, params in enumerate(param_combinations):
        # Extract fixed parameters
        fixed_params = {key: value for key, value in params.items() if key != param_to_vary}
        fixed_params_tuple = tuple(sorted(fixed_params.items()))  # Ensure immutability for dictionary keys

        # Initialize the group if it doesn't exist
        if fixed_params_tuple not in groups:
            groups[fixed_params_tuple] = {"params": [], "scores": []}

        # Append the current parameter set and its score
        groups[fixed_params_tuple]["params"].append(params)
        groups[fixed_params_tuple]["scores"].append(i)  # Placeholder for metric scores; replace with actual data

    return groups
