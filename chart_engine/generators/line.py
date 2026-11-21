"""
Line Chart Generator
====================
Renders line charts onto a Matplotlib Axes object.
"""

from typing import Any, Dict
import matplotlib.axes
import pandas as pd


def render_line(ax: matplotlib.axes.Axes, df: pd.DataFrame, config: Dict[str, Any]) -> None:
    """
    Renders a line chart on the given Axes.

    Args:
        ax: The Matplotlib Axes on which to draw.
        df: The pandas DataFrame containing the data.
        config: Validated configuration dictionary.
    """
    x_col = config["x_column"]
    y_col = config["y_column"]

    color = config.get("color", "#1f77b4")
    alpha = config.get("alpha", 1.0)
    linewidth = config.get("linewidth", 2.0)
    marker = config.get("marker", "o")
    markersize = config.get("markersize", 5)

    x_data = df[x_col]
    y_data = df[y_col]

    ax.plot(
        x_data,
        y_data,
        color=color,
        alpha=alpha,
        linewidth=linewidth,
        marker=marker,
        markersize=markersize,
    )

    if any(isinstance(val, str) and len(val) > 4 for val in x_data):
        ax.tick_params(axis="x", labelrotation=30)
