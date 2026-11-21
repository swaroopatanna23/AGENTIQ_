"""
Pie Chart Generator
===================
Renders pie charts onto a Matplotlib Axes object.
"""

from typing import Any, Dict
import matplotlib.axes
import pandas as pd


def render_pie(ax: matplotlib.axes.Axes, df: pd.DataFrame, config: Dict[str, Any]) -> None:
    """
    Renders a pie chart on the given Axes.

    Args:
        ax: The Matplotlib Axes on which to draw.
        df: The pandas DataFrame containing the data.
        config: Validated configuration dictionary.
    """
    y_col = config["y_column"]
    x_col = config.get("x_column")

    values = df[y_col]
    labels = df[x_col].astype(str) if x_col else None

    colors = config.get("colors") or config.get("color")
    autopct = config.get("autopct", "%1.1f%%")
    startangle = config.get("startangle", 140)

    pie_kwargs = {
        "x": values,
        "labels": labels,
        "autopct": autopct,
        "startangle": startangle,
        "wedgeprops": {"edgecolor": "white", "linewidth": 1.5},
    }

    if colors is not None:
        pie_kwargs["colors"] = colors if isinstance(colors, list) else [colors]

    ax.pie(**pie_kwargs)
    ax.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.
