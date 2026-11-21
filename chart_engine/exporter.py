"""
Export Module for Chart Generation Engine (AgentIQ)
===================================================
Handles saving Matplotlib Figure objects to disk in PNG, PDF, or SVG format.
Ensures output directories are created automatically and memory is managed properly.
"""

import os
from typing import Optional
import matplotlib.figure
import matplotlib.pyplot as plt

SUPPORTED_EXPORT_FORMATS = ["png", "pdf", "svg"]


def export_chart(
    fig: matplotlib.figure.Figure,
    filepath: str,
    format: Optional[str] = None,
    dpi: int = 300,
    close_fig: bool = False,
) -> str:
    """
    Exports a Matplotlib Figure to disk in PNG, PDF, or SVG format.

    Args:
        fig: The Matplotlib Figure object to export.
        filepath: Target output file path (e.g. 'output/sales.png').
        format: Export format ('png', 'pdf', 'svg'). If None, inferred from filepath.
        dpi: Resolution dots per inch (default: 300 for crisp quality).
        close_fig: If True, closes the Figure to free system memory (default: False).

    Returns:
        str: Absolute path of the exported file.

    Raises:
        ValueError: If the requested format is not supported.
        IOError: If an error occurs during file writing.
    """
    if not isinstance(fig, matplotlib.figure.Figure):
        actual_type = type(fig).__name__
        raise ValueError(f"Expected a Matplotlib Figure object, but received '{actual_type}'.")

    # Determine format: explicitly passed format takes precedence, otherwise infer from extension
    target_format = format.lower() if format else None
    base, ext = os.path.splitext(filepath)
    ext_format = ext.lstrip(".").lower()

    if not target_format:
        target_format = ext_format if ext_format else "png"

    if target_format not in SUPPORTED_EXPORT_FORMATS:
        raise ValueError(
            f"Unsupported export format '{target_format}'. "
            f"Supported formats are: {', '.join(SUPPORTED_EXPORT_FORMATS)}."
        )

    # Ensure filepath has the correct extension matching the target format
    if ext_format != target_format:
        filepath = f"{base}.{target_format}"

    # Ensure the parent directory exists
    abs_path = os.path.abspath(filepath)
    directory = os.path.dirname(abs_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    # Save the figure
    try:
        fig.savefig(abs_path, format=target_format, dpi=dpi, bbox_inches="tight")
    except Exception as err:
        raise IOError(f"Failed to export chart to '{abs_path}': {err}") from err
    finally:
        if close_fig:
            plt.close(fig)

    return abs_path
