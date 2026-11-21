"""
Validator Module for Chart Generation Engine (AgentIQ)
======================================================
This module validates the input DataFrame and configuration dictionary
before any chart is rendered. It provides clear, beginner-friendly error
messages to help users and teammates quickly identify mistakes.
"""

from typing import Any, Dict, List
import pandas as pd


class ChartValidationError(ValueError):
    """Custom exception raised when chart configuration or DataFrame fails validation."""
    pass


SUPPORTED_CHART_TYPES = ["bar", "line", "pie", "scatter", "histogram", "area"]


def _get_column_key(config: Dict[str, Any], primary_key: str, fallback_key: str) -> Any:
    """Helper to retrieve a column key, checking primary key first, then fallback."""
    if primary_key in config and config[primary_key] is not None:
        return config[primary_key]
    if fallback_key in config and config[fallback_key] is not None:
        return config[fallback_key]
    return None


def validate_dataframe(df: Any) -> None:
    """
    Validates that df is a non-empty pandas DataFrame.

    Raises:
        ChartValidationError: If df is not a DataFrame or is empty.
    """
    if not isinstance(df, pd.DataFrame):
        actual_type = type(df).__name__
        raise ChartValidationError(
            f"Invalid data input: Expected a pandas DataFrame, but received '{actual_type}'."
        )

    if df.empty:
        raise ChartValidationError(
            "Invalid data input: The provided DataFrame is empty. Cannot generate a chart from empty data."
        )


def validate_config(config: Any) -> Dict[str, Any]:
    """
    Validates that config is a valid dictionary and contains a supported chart_type.

    Returns:
        A normalized copy of the configuration dictionary.

    Raises:
        ChartValidationError: If config is invalid or chart_type is unsupported.
    """
    if not isinstance(config, dict):
        actual_type = type(config).__name__
        raise ChartValidationError(
            f"Invalid configuration: Expected a dictionary, but received '{actual_type}'."
        )

    if "chart_type" not in config or not config["chart_type"]:
        raise ChartValidationError(
            f"Missing required configuration key: 'chart_type'. "
            f"Supported chart types are: {', '.join(SUPPORTED_CHART_TYPES)}."
        )

    chart_type = str(config["chart_type"]).strip().lower()
    if chart_type not in SUPPORTED_CHART_TYPES:
        raise ChartValidationError(
            f"Unsupported chart_type '{chart_type}'. "
            f"Supported chart types are: {', '.join(SUPPORTED_CHART_TYPES)}."
        )

    # Validate optional styling parameters if present
    if "figsize" in config and config["figsize"] is not None:
        figsize = config["figsize"]
        if not (isinstance(figsize, (tuple, list)) and len(figsize) == 2 and all(isinstance(dim, (int, float)) and dim > 0 for dim in figsize)):
            raise ChartValidationError(
                f"Invalid 'figsize': Expected a tuple or list of two positive numbers (width, height), but received {figsize}."
            )

    if "grid" in config and config["grid"] is not None:
        if not isinstance(config["grid"], bool):
            raise ChartValidationError(
                f"Invalid 'grid': Expected a boolean (True or False), but received '{type(config['grid']).__name__}'."
            )

    if "alpha" in config and config["alpha"] is not None:
        alpha = config["alpha"]
        if not (isinstance(alpha, (int, float)) and 0.0 <= alpha <= 1.0):
            raise ChartValidationError(
                f"Invalid 'alpha': Expected a float between 0.0 and 1.0, but received {alpha}."
            )

    if "bins" in config and config["bins"] is not None:
        bins = config["bins"]
        if not (isinstance(bins, int) and bins > 0):
            raise ChartValidationError(
                f"Invalid 'bins': Expected a positive integer, but received {bins}."
            )

    # Return a normalized copy with normalized chart_type
    norm_config = dict(config)
    norm_config["chart_type"] = chart_type
    return norm_config


def validate_chart_requirements(config: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validates chart-specific requirements against the DataFrame columns and data types.

    Returns:
        Updated config with resolved 'x_column' and 'y_column'.

    Raises:
        ChartValidationError: If required columns are missing or of incorrect types.
    """
    chart_type = config["chart_type"]
    available_columns: List[str] = list(df.columns)

    resolved_config = dict(config)
    x_col = _get_column_key(config, "x_column", "x")
    y_col = _get_column_key(config, "y_column", "y")

    resolved_config["x_column"] = x_col
    resolved_config["y_column"] = y_col

    # Helper function to check numeric column
    def _check_numeric(column_name: str, purpose: str) -> None:
        if not pd.api.types.is_numeric_dtype(df[column_name]):
            raise ChartValidationError(
                f"Column '{column_name}' used for {purpose} must contain numeric data, "
                f"but has data type '{df[column_name].dtype}'."
            )

    # Helper function to check column existence
    def _check_exists(column_name: Any, key_name: str) -> None:
        if column_name is None:
            raise ChartValidationError(
                f"Missing required configuration key '{key_name}' for '{chart_type}' chart. "
                f"Available columns in data: {available_columns}."
            )
        if column_name not in df.columns:
            raise ChartValidationError(
                f"Column '{column_name}' specified for '{key_name}' was not found in DataFrame. "
                f"Available columns are: {available_columns}."
            )

    # Chart type specific validations
    if chart_type in ["bar", "line", "area"]:
        _check_exists(x_col, "x_column")
        _check_exists(y_col, "y_column")
        _check_numeric(y_col, f"'{chart_type}' chart values (y_column)")

    elif chart_type == "scatter":
        _check_exists(x_col, "x_column")
        _check_exists(y_col, "y_column")
        _check_numeric(x_col, "scatter plot X coordinates (x_column)")
        _check_numeric(y_col, "scatter plot Y coordinates (y_column)")

    elif chart_type == "histogram":
        # Histogram needs one column: x_column or y_column
        target_col = x_col or y_col
        if target_col is None:
            raise ChartValidationError(
                f"For a histogram, 'x_column' must be specified to indicate which column's distribution to plot. "
                f"Available columns are: {available_columns}."
            )
        _check_exists(target_col, "x_column")
        _check_numeric(target_col, "histogram values")
        resolved_config["x_column"] = target_col

    elif chart_type == "pie":
        # Pie needs values (y_column) and optionally slice labels (x_column)
        _check_exists(y_col, "y_column")
        _check_numeric(y_col, "pie chart slice values (y_column)")

        # Verify values are non-negative and sum > 0
        numeric_series = df[y_col].dropna()
        if (numeric_series < 0).any():
            raise ChartValidationError(
                f"Pie chart values in column '{y_col}' must be non-negative numbers."
            )
        if numeric_series.sum() <= 0:
            raise ChartValidationError(
                f"Pie chart values in column '{y_col}' must sum to a positive number."
            )

        if x_col is not None:
            _check_exists(x_col, "x_column")

    return resolved_config


def validate_inputs(config: Any, df: Any) -> Dict[str, Any]:
    """
    Main validation function that validates both DataFrame and configuration.

    Returns:
        The validated and normalized configuration dictionary.
    """
    validate_dataframe(df)
    norm_config = validate_config(config)
    return validate_chart_requirements(norm_config, df)
