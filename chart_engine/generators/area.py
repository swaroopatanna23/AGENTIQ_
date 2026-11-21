"""
Area Chart Generator
====================
Renders area charts (filled line plots) onto a Matplotlib Axes object.
"""

from typing import Any, Dict
import matplotlib.axes
import pandas as pd


def render_area(ax: matplotlib.axes.Axes, df: pd.DataFrame, config: Dict[str, Any]) -> None:
    """
    Renders an area chart on the given Axes.

    Args:
        ax: The Matplotlib Axes on which to draw.
        df: The pandas DataFrame containing the data.
        config: Validated configuration dictionary.
    """
    x_col = config["x_column"]
    y_col = config["y_column"]

    color = config.get("color", "#1f77b4")
    alpha = config.get("alpha", 0.35)
    linewidth = config.get("linewidth", 2.0)

    x_data = df[x_col]
    y_data = df[y_col]

    # If X values are non-numeric strings, use integer indices for fill_between
    # to maintain compatibility across all Matplotlib versions
    if not pd.api.types.is_numeric_dtype(x_data):
        indices = list(range(len(x_data)))
        ax.fill_between(indices, y_data, color=color, alpha=alpha)
        ax.plot(indices, y_data, color=color, linewidth=linewidth, marker="o", markersize=4)
        ax.set_xticks(indices)
        ax.set_xticklabels(x_data.astype(str))
    else:
        ax.fill_between(x_data, y_data, color=color, alpha=alpha)
        ax.plot(x_data, y_data, color=color, linewidth=linewidth)

    if any(isinstance(val, str) and len(val) > 4 for val in x_data):
        ax.tick_params(axis="x", labelrotation=30)
