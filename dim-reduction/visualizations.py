import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns  # Optional for better visuals


class DataLoader:
    """Handles loading data from a file."""
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        """Load the CSV data into a Pandas DataFrame."""
        if os.path.exists(self.file_path):
            self.data = pd.read_csv(self.file_path)
            print("Data loaded successfully.")
        else:
            raise FileNotFoundError(f"File not found: {self.file_path}")
        return self.data


class DataAnalyzer:
    """Analyzes data and provides visualizations to find relationships."""
    def __init__(self, data, pdf_writer):
        self.data = data
        self.pdf_writer = pdf_writer  # PDF writer to save plots
    
    def summarize_data(self):
        """Summarize data and write summary to PDF."""
        summary = self.data.describe()
        missing_values = self.data.isnull().sum()
        
        # Save text summary to PDF
        with self.pdf_writer as pdf:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.axis("off")
            summary_text = f"Data Summary:\n\n{summary}\n\nMissing Values:\n{missing_values}"
            ax.text(0, 1, summary_text, fontsize=10, ha='left', va='top', wrap=True)
            pdf.savefig(fig)
            plt.close(fig)
    
    def correlation_matrix(self):
        """Plot a correlation matrix to find relationships between features."""
        corr_matrix = self.data.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Matrix")
        self.pdf_writer.savefig()  # Save plot to PDF
        plt.close()

    def plot_feature_distributions(self):
        """Visualize the distributions of numeric features."""
        numeric_features = self.data.select_dtypes(include=[np.number]).columns
        for feature in numeric_features:
            plt.figure(figsize=(8, 4))
            sns.histplot(self.data[feature], kde=True, bins=30, color='blue')
            plt.title(f"Distribution of {feature}")
            plt.xlabel(feature)
            plt.ylabel("Frequency")
            self.pdf_writer.savefig()  # Save plot to PDF
            plt.close()

    def scatter_plot(self, feature_x, feature_y):
        """Scatter plot between two features."""
        if feature_x in self.data.columns and feature_y in self.data.columns:
            plt.figure(figsize=(8, 6))
            plt.scatter(self.data[feature_x], self.data[feature_y], alpha=0.6)
            plt.title(f"Scatter Plot: {feature_x} vs {feature_y}")
            plt.xlabel(feature_x)
            plt.ylabel(feature_y)
            plt.grid()
            self.pdf_writer.savefig()  # Save plot to PDF
            plt.close()
        else:
            print(f"Features {feature_x} and/or {feature_y} not found in data.")

    def box_plot(self, feature, target):
        """Create a box plot for a feature grouped by the target variable."""
        if feature in self.data.columns and target in self.data.columns:
            plt.figure(figsize=(8, 6))
            sns.boxplot(x=self.data[target], y=self.data[feature])
            plt.title(f"Box Plot of {feature} by {target}")
            plt.xlabel(target)
            plt.ylabel(feature)
            self.pdf_writer.savefig()  # Save plot to PDF
            plt.close()
        else:
            print(f"Feature {feature} and/or target {target} not found in data.")


# Example Usage
if __name__ == "__main__":
    # File path to the dataset
    file_path = './original_data/trainingset.csv'
    
    # Output PDF file path
    output_directory = "./out/"
    os.makedirs(output_directory, exist_ok=True)  # Create output directory if it doesn't exist
    output_file = os.path.join(output_directory, os.path.basename(__file__).replace('.py', '_writeup.pdf'))
    
    with PdfPages(output_file) as pdf_writer:
        # Load data
        loader = DataLoader(file_path)
        try:
            data = loader.load_data()
        except FileNotFoundError as e:
            print(e)
            exit()

        # Analyze data
        analyzer = DataAnalyzer(data, pdf_writer)
        
        # Summarize the data
        analyzer.summarize_data()
        
        # Correlation matrix
        analyzer.correlation_matrix()
        
        # Feature distributions
        analyzer.plot_feature_distributions()
        
        # Scatter plot for selected features
        analyzer.scatter_plot("feature1", "feature2")
        
        # Box plot for a feature by the target (ClaimAmount in this case)
        analyzer.box_plot("feature1", "ClaimAmount")

    print(f"Analysis writeup saved to: {output_file}")
