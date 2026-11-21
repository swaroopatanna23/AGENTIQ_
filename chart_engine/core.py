"""
Core Chart Generation Engine (AgentIQ)
======================================
Provides the primary interface: `generate_chart(config, df)`.

This function:
1. Validates the configuration dictionary and DataFrame.
2. Initializes a clean Matplotlib Figure and Axes.
3. Delegates the rendering to the appropriate chart generator.
4. Applies universal chart styling (title, labels, grid, layout).
5. Returns the generated Matplotlib Figure object.
"""

from typing import Any, Dict
import matplotlib.figure
import matplotlib.pyplot as plt
import pandas as pd

from .validator import validate_inputs
from .generators import (
    render_bar,
    render_line,
    render_pie,
    render_scatter,
    render_histogram,
    render_area,
)

# Registry mapping chart type name to its generator function
_CHART_GENERATORS = {
    "bar": render_bar,
    "line": render_line,
    "pie": render_pie,
    "scatter": render_scatter,
    "histogram": render_histogram,
    "area": render_area,
}


def generate_chart(config: Dict[str, Any], df: pd.DataFrame) -> matplotlib.figure.Figure:
    """
    Generates a Matplotlib Figure based on the supplied configuration and DataFrame.

    Args:
        config: A dictionary containing chart settings:
            - chart_type (str, required): 'bar', 'line', 'pie', 'scatter', 'histogram', or 'area'.
            - x_column (str): Column name for X-axis (or pie labels / histogram column).
            - y_column (str): Column name for Y-axis (or pie values).
            - title (str, optional): Chart title.
            - x_label (str, optional): Label for X-axis (defaults to x_column).
            - y_label (str, optional): Label for Y-axis (defaults to y_column or 'Frequency').
            - color / colors (str or list, optional): Color hex code, name, or list of colors.
            - grid (bool, optional): Whether to display background grid (default: True).
            - figsize (tuple, optional): (width, height) in inches (default: (10, 6)).
            - alpha (float, optional): Opacity between 0.0 and 1.0.
            - bins (int, optional): Number of bins for histogram (default: 10).
        df: A pandas DataFrame containing the data to plot.

    Returns:
        matplotlib.figure.Figure: The generated Matplotlib Figure object.

    Raises:
        ChartValidationError: If the configuration or DataFrame fails validation.
    """
    # 1. Validate inputs and normalize config
    valid_config = validate_inputs(config, df)
    chart_type = valid_config["chart_type"]

    # 2. Setup Figure and Axes
    figsize = valid_config.get("figsize", (10, 6))
    fig, ax = plt.subplots(figsize=figsize)

    # 3. Dispatch to appropriate generator
    generator_fn = _CHART_GENERATORS[chart_type]
    generator_fn(ax, df, valid_config)

    # 4. Apply Chart Title
    title = valid_config.get("title")
    if title:
        ax.set_title(str(title), fontsize=14, fontweight="bold", pad=12)

    # 5. Apply Axis Labels & Grid (for non-pie charts)
    if chart_type != "pie":
        # Determine X-axis label
        x_label = valid_config.get("x_label")
        if x_label is None:
            x_label = valid_config.get("x_column", "")
        if x_label:
            ax.set_xlabel(str(x_label), fontsize=11, labelpad=8)

        # Determine Y-axis label
        y_label = valid_config.get("y_label")
        if y_label is None:
            if chart_type == "histogram":
                y_label = "Frequency"
            else:
                y_label = valid_config.get("y_column", "")
        if y_label:
            ax.set_ylabel(str(y_label), fontsize=11, labelpad=8)

        # Grid settings
        show_grid = valid_config.get("grid", True)
        if show_grid:
            ax.grid(True, linestyle="--", alpha=0.5)
        else:
            ax.grid(False)

        # Subtle aesthetic touch: clean up borders
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Ensure layout is neat and tight without overlapping elements
    fig.tight_layout()

    return fig
