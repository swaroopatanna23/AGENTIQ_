"""
Bar Chart Generator
===================
Renders vertical bar charts onto a Matplotlib Axes object.
"""

from typing import Any, Dict
import matplotlib.axes
import pandas as pd


def render_bar(ax: matplotlib.axes.Axes, df: pd.DataFrame, config: Dict[str, Any]) -> None:
    """
    Renders a bar chart on the given Axes.

    Args:
        ax: The Matplotlib Axes on which to draw.
        df: The pandas DataFrame containing the data.
        config: Validated configuration dictionary.
    """
    x_col = config["x_column"]
    y_col = config["y_column"]

    color = config.get("color") or config.get("colors", "#1f77b4")
    alpha = config.get("alpha", 0.85)

    # Convert X to string labels for categorical bar plotting
    x_data = df[x_col].astype(str)
    y_data = df[y_col]

    bars = ax.bar(
        x_data,
        y_data,
        color=color,
        alpha=alpha,
        edgecolor="#333333",
        linewidth=0.8,
    )

    # If category names are moderately long, rotate them slightly for readability
    if any(len(str(val)) > 4 for val in x_data):
        ax.tick_params(axis="x", labelrotation=30)
