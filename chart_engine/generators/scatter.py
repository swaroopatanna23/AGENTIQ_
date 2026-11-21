"""
Scatter Plot Generator
======================
Renders scatter plots onto a Matplotlib Axes object.
"""

from typing import Any, Dict
import matplotlib.axes
import pandas as pd


def render_scatter(ax: matplotlib.axes.Axes, df: pd.DataFrame, config: Dict[str, Any]) -> None:
    """
    Renders a scatter plot on the given Axes.

    Args:
        ax: The Matplotlib Axes on which to draw.
        df: The pandas DataFrame containing the data.
        config: Validated configuration dictionary.
    """
    x_col = config["x_column"]
    y_col = config["y_column"]

    color = config.get("color", "#1f77b4")
    alpha = config.get("alpha", 0.75)
    size = config.get("size", 60)
    edgecolor = config.get("edgecolor", "#333333")

    ax.scatter(
        df[x_col],
        df[y_col],
        c=color,
        s=size,
        alpha=alpha,
        edgecolors=edgecolor,
        linewidths=0.6,
    )
