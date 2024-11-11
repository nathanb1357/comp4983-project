
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def plot_and_save_metrics(performance_data, output_pdf):
    """Plot metrics and save them to a PDF file."""
    with PdfPages(output_pdf) as pdf:
        for metric, scores in performance_data.items():
            plt.figure(figsize=(10, 6))
            plt.plot(range(len(scores)), scores, marker='o', label=metric)
            plt.xlabel("Parameter Combinations")
            plt.ylabel(f"{metric} Score")
            plt.title(f"Performance Tuning - {metric}")
            plt.legend()
            plt.tight_layout()
            pdf.savefig()
            plt.close()
    print(f"Performance plots saved to {output_pdf}.")
