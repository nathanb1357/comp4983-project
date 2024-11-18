from typing import List, Optional, Dict, Union
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from enums import Columns, GraphType


class GraphGenerator:
    def __init__(self, filepath: str, graph_type: GraphType) -> None:
        self.filepath = filepath
        self.graph_type = graph_type
        self.df = self._load_data()
        self.columns = self._csv_to_dict()

    def _load_data(self) -> pd.DataFrame:
        """Load CSV data into DataFrame."""
        return pd.read_csv(self.filepath)

    def _csv_to_dict(self) -> Dict[str, np.ndarray]:
        """Convert CSV columns to dictionary of numpy arrays."""
        return {col: np.array(self.df[col].values[1:]) for col in self.df.columns}

    @staticmethod
    def _calculate_grid_dimensions(n_plots: int) -> tuple[int, int]:
        """Calculate grid dimensions for subplots."""
        dimension = int(math.ceil(math.sqrt(n_plots)))
        return dimension, dimension

    def _setup_subplots(self, n_plots: int) -> tuple[plt.Figure, np.ndarray]:
        """Setup subplot grid."""
        if n_plots <= 1:
            fig, ax = plt.subplots(figsize=(15, 15))
            return fig, np.array([ax])

        rows, cols = self._calculate_grid_dimensions(n_plots)
        fig, axs = plt.subplots(rows, cols, figsize=(15, 15))
        return fig, axs.flatten()

    def _plot_line(self, ax: plt.Axes, main_col: Columns, compare_col: Columns) -> None:
        """Create a single line plot."""
        ax.plot(self.columns[main_col.value], self.columns[compare_col.value])
        ax.set_xlabel(main_col.value)
        ax.set_ylabel(compare_col.value)
        ax.grid(True)

    def _plot_scatter(
        self,
        ax: plt.Axes,
        main_col: Columns,
        compare_col: Columns,
        color_by: Optional[Columns] = None,
        include_values: Optional[List[Union[int, float, str]]] = None,
    ) -> None:
        """
        Create a single scatter plot.

        Args:
            ax: The axis to plot on
            main_col: The column for x-axis
            compare_col: The column for y-axis
            color_by: The column to use for coloring points
            include_values: List of values to include from color_by column. If None, all values are included.
        """
        if color_by is not None:
            # Get the categorical values for coloring
            color_values = self.columns[color_by.value]

            # Get unique categories, filtered by include_values if specified
            if include_values is not None:
                unique_categories = sorted(set(color_values) & set(include_values))
            else:
                unique_categories = sorted(set(color_values))

            # Plot each category with different color
            for category in unique_categories:
                mask = color_values == category
                ax.scatter(
                    self.columns[main_col.value][mask],
                    self.columns[compare_col.value][mask],
                    label=str(category),
                    alpha=0.7,
                    edgecolors="k",
                    s=100,
                )
            ax.legend(title=color_by.value, bbox_to_anchor=(1.05, 1), loc="upper left")
        else:
            # Original plotting without categories
            ax.scatter(
                self.columns[main_col.value],
                self.columns[compare_col.value],
                color="blue",
                alpha=0.7,
                edgecolors="k",
                s=10,
                linewidths=0.1,
            )

        ax.set_xlabel(main_col.value)
        ax.set_ylabel(compare_col.value)
        ax.set_title(f"{main_col.value} vs {compare_col.value}")
        ax.grid(True)

    def _plot_bar(self, ax: plt.Axes, main_col: Columns, compare_col: Columns) -> None:
        """Create a single bar plot."""
        ax.bar(
            self.columns[main_col.value],
            self.columns[compare_col.value],
            color="skyblue",
            edgecolor="black",
        )
        ax.set_xlabel(main_col.value)
        ax.set_ylabel(compare_col.value)
        ax.set_title(f"{main_col.value} vs {compare_col.value}")
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.tick_params(axis="both", which="both", labelsize=10)

    def _plot_histogram(self, ax: plt.Axes, main_col: Columns) -> None:
        """Create a histogram on the given axis."""
        column_data = self.columns[main_col.value]
        bins = sorted(set(column_data))
        counts = [list(column_data).count(bin) for bin in bins]

        ax.bar(bins, counts, width=0.8, align="center")
        ax.set_xticks(bins)
        ax.set_xticklabels(bins, rotation=45, ha="center")
        ax.set_xlabel(main_col.value)
        ax.set_ylabel("Frequency")
        ax.set_title(f"Histogram of {main_col.value}")
        ax.grid(True)

    def make_graph(
        self,
        main_col: Union[Columns, List[Columns]],
        compare_cols: Optional[Union[Columns, List[Columns]]] = None,
        color_by: Optional[Columns] = None,
    ) -> None:
        """
        Generate the specified type of graph.
        Either main_col or compare_cols can be an array, but not both.
        """
        # Convert inputs to lists for consistent handling
        main_cols = main_col if isinstance(main_col, list) else [main_col]
        compare_cols = (
            compare_cols
            if isinstance(compare_cols, list)
            else [compare_cols] if compare_cols else []
        )

        # Handle histogram separately
        if self.graph_type == GraphType.HISTO:
            if compare_cols:
                raise ValueError("Histogram cannot be created with compare columns")

            # Setup subplots for histograms
            n_plots = len(main_cols)
            fig, axs = self._setup_subplots(n_plots)
            axs = axs if isinstance(axs, np.ndarray) else [axs]

            # Create histogram for each column
            for idx, main_col in enumerate(main_cols):
                if idx < len(axs):
                    self._plot_histogram(axs[idx], main_col)

            # Hide unused subplots
            for idx in range(n_plots, len(axs)):
                axs[idx].axis("off")

            plt.tight_layout()
            return
        if not compare_cols:
            raise ValueError("Compare columns required for non-histogram plots")

        # Determine the number of subplots needed
        n_plots = max(len(main_cols), len(compare_cols))
        fig, axs = self._setup_subplots(n_plots)
        axs = (
            axs if isinstance(axs, np.ndarray) else [axs]
        )  # Handle single subplot case

        plot_functions = {
            GraphType.LINE: self._plot_line,
            GraphType.SCATTER: lambda ax, main, compare: self._plot_scatter(
                ax, main, compare, color_by
            ),
            GraphType.BAR: self._plot_bar,
        }

        plot_func = plot_functions.get(self.graph_type)
        if not plot_func:
            raise ValueError(f"Unsupported graph type: {self.graph_type}")

        # Plot each combination
        for idx in range(n_plots):
            if idx < len(axs):
                main = main_cols[idx if len(main_cols) > 1 else 0]
                compare = compare_cols[idx if len(compare_cols) > 1 else 0]
                plot_func(axs[idx], main, compare)

        # Hide unused subplots
        for idx in range(n_plots, len(axs)):
            axs[idx].axis("off")

        plt.tight_layout()

    def save_plot(self, filename: str, dpi: int = 300, format: str = "png") -> None:
        """
        Save the current plot to the file system.

        Args:
            filename (str): Name of the file to save (without extension)
            dpi (int, optional): Resolution of the saved image. Defaults to 300.
            format (str, optional): Format of the saved image ('png', 'jpg', 'pdf', etc.). Defaults to 'png'.

        Example:
            graph_gen.make_graph(...)
            graph_gen.save_plot('my_plot', dpi=300, format='png')
        """
        try:
            # Ensure the filename has the correct extension
            if not filename.endswith(f".{format}"):
                filename = f"{filename}.{format}"

            plt.savefig(
                filename,
                dpi=dpi,
                bbox_inches="tight",  # Ensures all elements are included in the saved figure
                format=format,
            )
            print(f"Plot saved successfully as '{filename}'")
        except Exception as e:
            print(f"Error saving plot: {str(e)}")


def numbers_to_columns(numbers: list[int]) -> list[Columns]:
    """
    Convert a list of numbers to their corresponding Columns enum values.
    Numbers should be in range 1-18 to match feature1-feature18.

    Args:
        numbers (list[int]): List of numbers to convert

    Returns:
        list[Columns]: List of corresponding Columns enum values

    Example:
        numbers_to_columns([1, 2, 3]) -> [Columns.feature1, Columns.feature2, Columns.feature3]
    """
    column_map = {
        1: Columns.feature1,
        2: Columns.feature2,
        3: Columns.feature3,
        4: Columns.feature4,
        5: Columns.feature5,
        6: Columns.feature6,
        7: Columns.feature7,
        8: Columns.feature8,
        9: Columns.feature9,
        10: Columns.feature10,
        11: Columns.feature11,
        12: Columns.feature12,
        13: Columns.feature13,
        14: Columns.feature14,
        15: Columns.feature15,
        16: Columns.feature16,
        17: Columns.feature17,
        18: Columns.feature18,
    }

    return [column_map[n] for n in numbers if n in column_map]


# def main():
#     graph_gen = GraphGenerator("./onlynonezerodata.csv", GraphType.SCATTER)
#     graph_gen.make_graph(
#         numbers_to_columns(
#             [
#                 2,
#                 6,
#                 8,
#                 10,
#             ]
#         ),
#         Columns.feature2,
#         color_by=Columns.feature4,
#     )
#     plt.show()

#
# def main():
#     graph_gen = GraphGenerator("./onlynonezerodata.csv", GraphType.SCATTER)
#     graph_gen.make_graph(
#         numbers_to_columns(
#             [
#                 8,
#             ]
#         ),
#         Columns.feature2,
#         color_by=Columns.feature4,
#     )
#     graph_gen.save_plot("linearrelatoinshipF8-F2.png")
#     plt.show()


# 2, 6, 8, 10, 12, 17
# [1, 2, 6, 8, 10, 12, 17]
# 3, 4, 5, 7, 9, 11, 13, 14, 15, 16, 18
# [1, 2, 6, 8]
# if __name__ == "__main__":
#     main()
