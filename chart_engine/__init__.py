"""
AgentIQ Chart Generation Engine
===============================
A modular, offline, lightweight chart generation engine built with
Python, Matplotlib, Pandas, and NumPy.

Public API:
- generate_chart(config, df): Generates and returns a Matplotlib Figure.
- export_chart(fig, filepath, format, dpi): Exports Figure to PNG, PDF, or SVG.
- ChartValidationError: Exception raised for invalid configurations or data.
"""

from .core import generate_chart
from .exporter import export_chart, SUPPORTED_EXPORT_FORMATS
from .validator import ChartValidationError, SUPPORTED_CHART_TYPES

__version__ = "1.0.0"

__all__ = [
    "generate_chart",
    "export_chart",
    "ChartValidationError",
    "SUPPORTED_CHART_TYPES",
    "SUPPORTED_EXPORT_FORMATS",
]
