"""
Generators Package
==================
Contains modular implementations for rendering each chart type:
- bar: Bar charts
- line: Line charts
- pie: Pie charts
- scatter: Scatter plots
- histogram: Histograms
- area: Area charts
"""

from .bar import render_bar
from .line import render_line
from .pie import render_pie
from .scatter import render_scatter
from .histogram import render_histogram
from .area import render_area

__all__ = [
    "render_bar",
    "render_line",
    "render_pie",
    "render_scatter",
    "render_histogram",
    "render_area",
]
