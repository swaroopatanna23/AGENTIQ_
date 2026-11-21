"""
Histogram Generator
===================
Renders histograms onto a Matplotlib Axes object.
"""

from typing import Any, Dict
import matplotlib.axes
import pandas as pd


def render_histogram(ax: matplotlib.axes.Axes, df: pd.DataFrame, config: Dict[str, Any]) -> None:
    """
    Renders a histogram on the given Axes.

    Args:
        ax: The Matplotlib Axes on which to draw.
        df: The pandas DataFrame containing the data.
        config: Validated configuration dictionary.
    """
    x_col = config["x_column"]

    bins = config.get("bins", 10)
    color = config.get("color", "#1f77b4")
    alpha = config.get("alpha", 0.75)
    edgecolor = config.get("edgecolor", "#333333")

    # Drop NA values to prevent calculation errors in hist
    series_data = df[x_col].dropna()

    ax.hist(
        series_data,
        bins=bins,
        color=color,
        alpha=alpha,
        edgecolor=edgecolor,
        linewidth=0.8,
    )
