"""
Shared helpers for the Data Processing and Data Validation modules (Member 4)
==============================================================================
Both modules need the same small tools (what kind of column is this? which
values are outliers? does this text look like a number or a date?).
They live here ONCE so processing.py and validation.py never duplicate code.
"""

import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# Text values that really mean "nothing here" (compared in lower case).
MISSING_PLACEHOLDERS = {
    "", "na", "n/a", "nan", "null", "none", "nil", "-", "--", "?", "missing", "undefined",
}

# Matches common date shapes: 2024-01-31, 31/01/2024, 31 Jan 2024, Jan 31, 2024
DATE_PATTERN = re.compile(
    r"^\s*("
    r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}"
    r"|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}"
    r"|\d{1,2}\s+[A-Za-z]{3,9}\.?,?\s+\d{2,4}"
    r"|[A-Za-z]{3,9}\.?\s+\d{1,2},?\s+\d{2,4}"
    r")([ T]\d{1,2}:\d{2}(:\d{2})?)?\s*$"
)

# Currency symbols / thousand separators we remove before reading numbers.
_NUMBER_NOISE = re.compile(r"[,\s$€£₹]")
# "00123" style values are identifiers (zip codes, ids) - never turn them into numbers.
_LEADING_ZERO = re.compile(r"^[+-]?0\d")


def column_kind(series: pd.Series) -> str:
    """Return a simple category for a column:
    'boolean', 'numeric', 'datetime', 'text' or 'other'."""
    dtype = series.dtype
    if pd.api.types.is_bool_dtype(dtype):
        return "boolean"
    if pd.api.types.is_numeric_dtype(dtype):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime"
    # pandas 2 stores text as 'object', pandas 3 as 'str' - handle both.
    if dtype == object or pd.api.types.is_string_dtype(dtype) or isinstance(dtype, pd.CategoricalDtype):
        return "text"
    return "other"


def text_to_numeric(series: pd.Series) -> Tuple[pd.Series, float]:
    """Try to read a text column as numbers (handles '1,200', '$45').

    Returns (converted_values, ratio) where ratio is the share of NON-EMPTY
    values that became numbers. Values that are not numbers become NaN.
    """
    non_null = series.dropna()
    if non_null.empty:
        return pd.Series(np.nan, index=series.index, dtype=float), 0.0
    as_text = non_null.astype(str).str.strip()
    if as_text.str.match(_LEADING_ZERO).any():
        return pd.Series(np.nan, index=series.index, dtype=float), 0.0
    cleaned = as_text.str.replace(_NUMBER_NOISE, "", regex=True)
    converted = pd.to_numeric(cleaned, errors="coerce")
    result = pd.Series(np.nan, index=series.index, dtype=float)
    result.loc[non_null.index] = converted.astype(float)
    ratio = float(converted.notna().mean())
    return result, ratio


def text_looks_like_date(series: pd.Series, threshold: float = 0.9) -> bool:
    """True when at least `threshold` of the non-empty text values look like dates."""
    non_null = series.dropna().astype(str)
    if non_null.empty:
        return False
    return bool(non_null.str.match(DATE_PATTERN).mean() >= threshold)


def iqr_bounds(series: pd.Series, factor: float = 1.5) -> Optional[Tuple[float, float]]:
    """Lower/upper limits using the IQR rule (the same rule Member 1's quality
    report used). Returns None when the rule cannot be applied
    (too few values, or all values nearly identical)."""
    values = series.dropna()
    if len(values) < 4:
        return None
    q1, q3 = values.quantile(0.25), values.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return None
    return float(q1 - factor * iqr), float(q3 + factor * iqr)


def outlier_mask(series: pd.Series, method: str = "iqr", factor: float = 1.5, z: float = 3.0) -> pd.Series:
    """Boolean Series: True where the value is an outlier ('iqr' or 'zscore')."""
    empty = pd.Series(False, index=series.index)
    if method == "zscore":
        values = series.dropna()
        if len(values) < 4 or values.std() == 0:
            return empty
        scores = (series - values.mean()) / values.std()
        return scores.abs() > z
    bounds = iqr_bounds(series, factor)
    if bounds is None:
        return empty
    return (series < bounds[0]) | (series > bounds[1])
