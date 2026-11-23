"""
Data Processing Module  (Member 4 - Amrutha)
============================================
Turns a raw uploaded file into a clean pandas DataFrame.

Main entry points
-----------------
load_dataset(source)                  -> DataFrame   (CSV / Excel / JSON)
process_dataset(data, ...)            -> DataFrame   (the full cleaning pipeline)
process_dataset_with_log(data, ...)   -> (DataFrame, list of messages)

Each cleaning step is its own small function so it can be tested and
explained on its own:
    standardize_column_names, clean_text_values, unify_text_case, convert_data_types,
    remove_duplicates, handle_missing_values, handle_outliers,
    extract_date_features, scale_numeric_columns, aggregate_data, aggregate_column

Everything here works fully offline (only pandas / numpy / openpyxl).
"""

import csv
import io
import json
import os
import re
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .common import (
    MISSING_PLACEHOLDERS,
    column_kind,
    iqr_bounds,
    outlier_mask,
    text_looks_like_date,
    text_to_numeric,
)

SUPPORTED_EXTENSIONS = (".csv", ".xlsx", ".xls", ".json")

# Words Qwen (or a user) may use for an aggregation -> the pandas name.
AGGREGATION_ALIASES = {
    "sum": "sum", "total": "sum",
    "average": "mean", "avg": "mean", "mean": "mean",
    "count": "count",
    "minimum": "min", "min": "min",
    "maximum": "max", "max": "max",
    "median": "median",
}


class DataProcessingError(Exception):
    """Raised when a processing step cannot be completed."""


class DataLoadError(DataProcessingError):
    """Raised when a file cannot be read."""


# =====================================================================
# 1. LOADING  (CSV, Excel, JSON)
# =====================================================================

def _source_name(source: Any, filename: Optional[str]) -> str:
    """Find the file name so we know the format (works for paths and uploads)."""
    if filename:
        return str(filename)
    name = getattr(source, "name", None)
    return str(name) if name else str(source)


def _rewind(source: Any) -> None:
    """Go back to the start of an uploaded file object before reading again."""
    if hasattr(source, "seek"):
        source.seek(0)


def load_csv(source: Any) -> pd.DataFrame:
    """Read a CSV. Tries several encodings and detects the separator
    automatically (comma, semicolon, tab, ...)."""
    last_error: Optional[Exception] = None
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            _rewind(source)
            try:
                return pd.read_csv(source, sep=None, engine="python", encoding=encoding)
            except (csv.Error, pd.errors.ParserError):
                _rewind(source)  # e.g. single-column files: no separator to detect
                return pd.read_csv(source, encoding=encoding)
        except UnicodeDecodeError as err:
            last_error = err
    raise DataLoadError(f"Could not decode the CSV file with common encodings: {last_error}")


def load_excel(source: Any, sheet_name: Union[int, str] = 0) -> pd.DataFrame:
    """Read one sheet of an Excel file (.xlsx needs openpyxl, .xls needs xlrd)."""
    _rewind(source)
    return pd.read_excel(source, sheet_name=sheet_name)


def _json_to_dataframe(obj: Any) -> pd.DataFrame:
    """Convert the common JSON layouts into a table."""
    if isinstance(obj, list):
        if all(isinstance(item, dict) for item in obj):
            return pd.json_normalize(obj, sep="_")  # flattens nested objects
        return pd.DataFrame({"value": obj})
    if isinstance(obj, dict):
        # {"data": [ {...}, {...} ]}  -> use the inner list
        list_values = [v for v in obj.values() if isinstance(v, list) and v and isinstance(v[0], dict)]
        if len(list_values) == 1:
            return pd.json_normalize(list_values[0], sep="_")
        # {"col1": [..], "col2": [..]}
        if obj and all(isinstance(v, list) for v in obj.values()):
            return pd.DataFrame(obj)
        # {"a": 1, "b": 2}  -> one row
        return pd.json_normalize(obj, sep="_")
    raise DataLoadError("JSON must contain a list of records or an object of columns.")


def load_json(source: Any) -> pd.DataFrame:
    """Read JSON (list of records, dict of columns, {'data': [...]}) or JSON-Lines."""
    if isinstance(source, (str, Path)) and os.path.exists(str(source)):
        text = Path(source).read_text(encoding="utf-8-sig")
    else:
        _rewind(source)
        raw = source.read()
        text = raw.decode("utf-8-sig") if isinstance(raw, bytes) else raw
    try:
        return _json_to_dataframe(json.loads(text))
    except json.JSONDecodeError:
        try:  # JSON Lines: one record per line
            return pd.read_json(io.StringIO(text), lines=True)
        except ValueError as err:
            raise DataLoadError(f"The file is not valid JSON: {err}")


def load_dataset(source: Any, filename: Optional[str] = None, sheet_name: Union[int, str] = 0) -> pd.DataFrame:
    """Load a CSV, Excel or JSON dataset into a DataFrame.

    `source` can be a file path or an uploaded-file object (e.g. from Streamlit).
    Raises DataLoadError with a clear message when the file cannot be used.
    """
    name = _source_name(source, filename)
    extension = Path(name).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise DataLoadError(
            f"Unsupported file type '{extension or name}'. "
            f"Supported types: {', '.join(SUPPORTED_EXTENSIONS)}."
        )
    if isinstance(source, (str, Path)) and not os.path.exists(str(source)):
        raise DataLoadError(f"File not found: {source}")

    try:
        if extension == ".csv":
            return load_csv(source)
        if extension in (".xlsx", ".xls"):
            return load_excel(source, sheet_name)
        return load_json(source)
    except DataLoadError:
        raise
    except pd.errors.EmptyDataError:
        raise DataLoadError("The file is empty - there is no data to load.")
    except ImportError as err:
        raise DataLoadError(f"A required reader library is missing: {err}. Run: pip install -r requirements.txt")
    except Exception as err:  # corrupt files, bad sheet names, ...
        raise DataLoadError(f"Could not read '{Path(name).name}': {err}")


def load_dataset_from_text(text: str) -> pd.DataFrame:
    """Read pasted text (used by the 'Manual Data Entry' box): JSON or CSV."""
    stripped = (text or "").strip()
    if not stripped:
        raise DataLoadError("No data was entered.")
    try:
        if stripped[0] in "[{":
            return load_json(io.StringIO(stripped))
        return load_csv(io.StringIO(stripped))
    except DataLoadError:
        raise
    except Exception as err:
        raise DataLoadError(f"Could not understand the entered data: {err}")


# =====================================================================
# 2. CLEANING STEPS
#    Every step takes a DataFrame and returns a NEW DataFrame.
#    `log` (optional list) collects human-readable messages.
# =====================================================================

def _note(log: Optional[List[str]], message: str) -> None:
    if log is not None:
        log.append(message)


def _clean_one_name(name: Any) -> str:
    text = str(name).strip()
    text = text.replace("%", " pct ")
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)   # CustomerID -> Customer_ID
    text = re.sub(r"[^0-9A-Za-z]+", "_", text)             # spaces, symbols -> _
    return re.sub(r"_+", "_", text).strip("_").lower()


def standardize_column_names(df: pd.DataFrame, log: Optional[List[str]] = None) -> pd.DataFrame:
    """Make column names lower_snake_case: 'Total Sales ($)' -> 'total_sales'.
    Blank names become column_N and duplicates get _2, _3, ..."""
    new_names: List[str] = []
    seen: Dict[str, int] = {}
    for position, old in enumerate(df.columns, start=1):
        name = _clean_one_name(old) or f"column_{position}"
        seen[name] = seen.get(name, 0) + 1
        new_names.append(name if seen[name] == 1 else f"{name}_{seen[name]}")
    changed = [f"'{o}' -> '{n}'" for o, n in zip(df.columns, new_names) if str(o) != n]
    if changed:
        _note(log, f"Standardized {len(changed)} column name(s): " + ", ".join(changed))
    result = df.copy()
    result.columns = new_names
    return result


def drop_unnamed_index_columns(df: pd.DataFrame, log: Optional[List[str]] = None) -> pd.DataFrame:
    """Remove the extra 'Unnamed: 0' column that appears when a CSV was saved with its index."""
    drop = []
    for col in df.columns:
        if str(col).startswith("Unnamed:") and column_kind(df[col]) == "numeric":
            values = df[col].dropna()
            if len(values) == len(df) and (
                values.tolist() == list(range(len(df))) or values.tolist() == list(range(1, len(df) + 1))
            ):
                drop.append(col)
    if drop:
        _note(log, f"Removed index column(s): {drop}")
        return df.drop(columns=drop)
    return df


def clean_text_values(df: pd.DataFrame, log: Optional[List[str]] = None) -> pd.DataFrame:
    """Strip extra spaces from text and turn placeholders ('N/A', 'null', '-', '')
    into real missing values."""
    result = df.copy()
    placeholders_found = 0

    def fix(value: Any) -> Any:
        if isinstance(value, str):
            stripped = value.strip()
            return np.nan if stripped.lower() in MISSING_PLACEHOLDERS else stripped
        return value

    for col in result.columns:
        if column_kind(result[col]) == "text" and not isinstance(result[col].dtype, pd.CategoricalDtype):
            before = result[col].isna().sum()
            result[col] = result[col].map(fix)
            placeholders_found += int(result[col].isna().sum() - before)
    if placeholders_found:
        _note(log, f"Converted {placeholders_found} placeholder value(s) (like 'N/A', 'null', '-') to missing.")
    return result


def unify_text_case(df: pd.DataFrame, log: Optional[List[str]] = None) -> pd.DataFrame:
    """Make 'north', 'NORTH' and 'North' the same value (the most common spelling wins)."""
    result = df.copy()
    for col in result.columns:
        if column_kind(result[col]) != "text" or isinstance(result[col].dtype, pd.CategoricalDtype):
            continue
        texts = result[col].dropna()
        if texts.empty or not texts.map(lambda v: isinstance(v, str)).all():
            continue
        key = texts.str.lower()
        if texts.nunique() <= key.nunique():
            continue  # no spelling variants in this column
        # Most common spelling wins; on a tie prefer the one that is not all lowercase ('North' over 'north').
        def best_spelling(group: pd.Series) -> str:
            counts = group.value_counts()
            return sorted(counts.index, key=lambda v: (-counts[v], v == v.lower(), v))[0]
        best = texts.groupby(key).agg(best_spelling)
        result.loc[texts.index, col] = key.map(best)
        _note(log, f"Unified spelling/case of values in '{col}'.")
    return result


def drop_empty_rows_and_columns(df: pd.DataFrame, log: Optional[List[str]] = None) -> pd.DataFrame:
    """Remove columns and rows that contain no data at all."""
    if len(df) == 0:
        return df  # a header-only table: keep its columns so validation can report 'no rows'
    result = df.dropna(axis=1, how="all")
    removed_cols = [c for c in df.columns if c not in result.columns]
    rows_before = len(result)
    result = result.dropna(axis=0, how="all")
    if removed_cols:
        _note(log, f"Removed {len(removed_cols)} completely empty column(s): {removed_cols}")
    if rows_before - len(result):
        _note(log, f"Removed {rows_before - len(result)} completely empty row(s).")
    return result


def convert_data_types(
    df: pd.DataFrame,
    date_columns: Optional[Iterable[str]] = None,
    threshold: float = 0.9,
    dayfirst: bool = False,
    log: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Give every column its correct type.

    * Text that is really numbers ('1,200', '$45') -> numeric
    * Text that is really dates                    -> datetime
    A column is converted only when at least `threshold` (90%) of its values
    fit; values that do not fit become missing and are reported.
    """
    result = df.copy()
    forced_dates = set(date_columns or [])
    for col in result.columns:
        if column_kind(result[col]) != "text":
            continue
        series = result[col]
        if col in forced_dates or text_looks_like_date(series, threshold):
            non_null = series.dropna()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = pd.to_datetime(non_null.astype(str), errors="coerce", dayfirst=dayfirst, format="mixed")
            if col in forced_dates or (len(non_null) and parsed.notna().mean() >= threshold):
                full = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
                full.loc[non_null.index] = parsed.astype("datetime64[ns]")
                result[col] = full
                bad = int(parsed.isna().sum())
                _note(log, f"Column '{col}' converted to dates" + (f" ({bad} invalid value(s) set to missing)." if bad else "."))
                continue
        numbers, ratio = text_to_numeric(series)
        if ratio >= threshold:
            bad = int(series.notna().sum() - numbers.notna().sum())
            result[col] = numbers
            _note(log, f"Column '{col}' converted to numbers" + (f" ({bad} invalid value(s) set to missing)." if bad else "."))
    return result


def _tidy_integer_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Float columns that hold only whole numbers (and no gaps) become integers."""
    result = df.copy()
    for col in result.columns:
        s = result[col]
        if column_kind(s) == "numeric" and pd.api.types.is_float_dtype(s.dtype):
            if s.notna().all() and len(s) and np.isfinite(s).all() and (s == s.round()).all():
                result[col] = s.astype("int64")
    return result


def remove_duplicates(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
    keep: str = "first",
    log: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Remove repeated rows (keeping the first copy) and renumber the rows."""
    try:
        result = df.drop_duplicates(subset=subset, keep=keep)
    except TypeError:  # cells containing lists/dicts cannot be compared
        _note(log, "Skipped duplicate removal: some cells hold lists or objects.")
        return df
    removed = len(df) - len(result)
    if removed:
        _note(log, f"Removed {removed} duplicate row(s).")
    return result.reset_index(drop=True)


def handle_missing_values(
    df: pd.DataFrame,
    method: str = "fill",
    numeric_method: str = "median",
    log: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Deal with empty cells.

    method='fill' : numbers -> median (or mean/zero), text -> most common value
                    (or 'Unknown'). Dates are never invented; they stay empty.
    method='drop' : delete rows that have any missing value.
    method='none' : leave the data as it is.
    """
    if method not in ("fill", "drop", "none"):
        raise DataProcessingError("missing-value method must be 'fill', 'drop' or 'none'.")
    if numeric_method not in ("median", "mean", "zero"):
        raise DataProcessingError("numeric_method must be 'median', 'mean' or 'zero'.")
    if method == "none" or df.empty:
        return df
    if method == "drop":
        result = df.dropna().reset_index(drop=True)
        if len(df) - len(result):
            _note(log, f"Dropped {len(df) - len(result)} row(s) that had missing values.")
        return result

    result = df.copy()
    for col in result.columns:
        missing = int(result[col].isna().sum())
        if not missing:
            continue
        kind = column_kind(result[col])
        if kind == "numeric":
            if numeric_method == "median":
                value = result[col].median()
            elif numeric_method == "mean":
                value = result[col].mean()
            else:
                value = 0
            if pd.isna(value):
                continue
            result[col] = result[col].fillna(value)
            _note(log, f"Filled {missing} missing value(s) in '{col}' with the {numeric_method}.")
        elif kind in ("text", "boolean"):
            modes = result[col].mode(dropna=True)
            value = modes.iloc[0] if not modes.empty else "Unknown"
            result[col] = result[col].fillna(value)
            _note(log, f"Filled {missing} missing value(s) in '{col}' with '{value}'.")
        else:
            _note(log, f"Left {missing} missing value(s) in '{col}' unfilled (dates are not invented).")
    return result


def handle_outliers(
    df: pd.DataFrame,
    action: str = "cap",
    method: str = "iqr",
    columns: Optional[List[str]] = None,
    log: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Treat extreme numeric values.
    action='cap'    : pull them back to the IQR limits
    action='remove' : delete the rows that contain them
    action='none'   : do nothing"""
    if action not in ("cap", "remove", "none"):
        raise DataProcessingError("outlier action must be 'cap', 'remove' or 'none'.")
    if method not in ("iqr", "zscore"):
        raise DataProcessingError("outlier method must be 'iqr' or 'zscore'.")
    if action == "none" or df.empty:
        return df
    result = df.copy()
    targets = columns or [c for c in result.columns if column_kind(result[c]) == "numeric"]
    remove_rows = pd.Series(False, index=result.index)
    for col in targets:
        if col not in result.columns or column_kind(result[col]) != "numeric":
            continue
        mask = outlier_mask(result[col], method)
        count = int(mask.sum())
        if not count:
            continue
        if action == "remove":
            remove_rows |= mask
        else:
            lower, upper = iqr_bounds(result[col]) or (result[col].min(), result[col].max())
            if method == "zscore":
                mean, std = result[col].mean(), result[col].std()
                lower, upper = mean - 3 * std, mean + 3 * std
            if pd.api.types.is_integer_dtype(result[col].dtype):
                lower, upper = int(np.ceil(lower)), int(np.floor(upper))
            result[col] = result[col].clip(lower=lower, upper=upper)
            _note(log, f"Capped {count} outlier(s) in '{col}' to [{lower:.4g}, {upper:.4g}].")
    if action == "remove" and remove_rows.any():
        _note(log, f"Removed {int(remove_rows.sum())} row(s) containing outliers.")
        result = result.loc[~remove_rows].reset_index(drop=True)
    return result


# =====================================================================
# 3. TRANSFORMATION
# =====================================================================

def extract_date_features(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    parts: Tuple[str, ...] = ("year", "month", "day"),
    log: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Add columns such as order_date_year / order_date_month from date columns."""
    result = df.copy()
    targets = columns or [c for c in result.columns if column_kind(result[c]) == "datetime"]
    for col in targets:
        if col not in result.columns or column_kind(result[col]) != "datetime":
            continue
        for part in parts:
            if part not in ("year", "month", "day", "weekday", "quarter"):
                raise DataProcessingError(f"Unsupported date part '{part}'.")
            name = f"{col}_{part}"
            if name not in result.columns:
                result[name] = getattr(result[col].dt, part)
        _note(log, f"Added date parts {list(parts)} from '{col}'.")
    return result


def scale_numeric_columns(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "minmax",
    log: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Add scaled copies of numeric columns ('<name>_scaled').
    minmax -> values between 0 and 1;  zscore -> mean 0, standard deviation 1."""
    if method not in ("minmax", "zscore"):
        raise DataProcessingError("scaling method must be 'minmax' or 'zscore'.")
    result = df.copy()
    targets = columns or [c for c in result.columns if column_kind(result[c]) == "numeric"]
    for col in targets:
        if col not in result.columns or column_kind(result[col]) != "numeric":
            continue
        s = result[col].astype(float)
        spread = (s.max() - s.min()) if method == "minmax" else s.std()
        if not spread or pd.isna(spread):
            continue  # constant column cannot be scaled
        result[f"{col}_scaled"] = (s - s.min()) / spread if method == "minmax" else (s - s.mean()) / spread
        _note(log, f"Added '{col}_scaled' ({method}).")
    return result


def aggregate_data(
    df: pd.DataFrame,
    group_by: str,
    target: Optional[str] = None,
    aggregation: str = "sum",
) -> pd.DataFrame:
    """Group rows and summarise them, e.g. average sales per region.

    `aggregation` accepts pandas names or Qwen's words (average, minimum, ...).
    With target=None and aggregation='count' it counts rows per group.
    Returns a small table: [group_by, target] (or [group_by, 'count']).
    """
    agg = AGGREGATION_ALIASES.get(str(aggregation).strip().lower())
    if agg is None:
        raise DataProcessingError(f"Unsupported aggregation '{aggregation}'. Use: {sorted(set(AGGREGATION_ALIASES))}.")
    if group_by not in df.columns:
        raise DataProcessingError(f"Column '{group_by}' not found.")
    if target is None:
        if agg != "count":
            raise DataProcessingError("A target column is needed for this aggregation.")
        out = df.groupby(group_by, dropna=True).size().reset_index(name="count")
        return out
    if target not in df.columns:
        raise DataProcessingError(f"Column '{target}' not found.")
    if agg != "count" and column_kind(df[target]) != "numeric":
        raise DataProcessingError(f"Cannot calculate the {agg} of '{target}' because it is not a numeric column.")
    out = df.groupby(group_by, dropna=True)[target].agg(agg).reset_index()
    return out


def aggregate_column(df: pd.DataFrame, target: str, aggregation: str = "sum") -> float:
    """One summary number for a whole column (e.g. the total of 'sales')."""
    agg = AGGREGATION_ALIASES.get(str(aggregation).strip().lower())
    if agg is None:
        raise DataProcessingError(f"Unsupported aggregation '{aggregation}'.")
    if target not in df.columns:
        raise DataProcessingError(f"Column '{target}' not found.")
    if agg != "count" and column_kind(df[target]) != "numeric":
        raise DataProcessingError(f"Cannot calculate the {agg} of '{target}' because it is not a numeric column.")
    return getattr(df[target], agg)()


# =====================================================================
# 4. THE FULL PIPELINE
# =====================================================================

def process_dataset_with_log(
    data: Any,
    standardize_columns: bool = True,
    convert_types: bool = True,
    drop_duplicates: bool = True,
    missing: str = "fill",
    numeric_fill: str = "median",
    outliers: str = "none",
    outlier_method: str = "iqr",
    date_columns: Optional[List[str]] = None,
    add_date_parts: bool = False,
    scale: Optional[str] = None,
    dayfirst: bool = False,
    filename: Optional[str] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Run every cleaning step in a sensible order. Returns (clean_df, log).

    `data` is a DataFrame, or a file path / uploaded file (it is loaded first).
    Order matters: names -> text -> empty rows/cols -> types -> duplicates ->
    missing values -> outliers -> transformations.
    (Types are fixed BEFORE duplicates/missing values so '5' and 5 match and
    medians can be calculated.)
    """
    log: List[str] = []
    df = data if isinstance(data, pd.DataFrame) else load_dataset(data, filename=filename)
    if not isinstance(df, pd.DataFrame):
        raise DataProcessingError("Input must be a DataFrame or a supported file.")

    df = df.copy()
    df = drop_unnamed_index_columns(df, log)
    if standardize_columns:
        df = standardize_column_names(df, log)
    df = clean_text_values(df, log)
    df = unify_text_case(df, log)
    df = drop_empty_rows_and_columns(df, log)
    if convert_types:
        df = convert_data_types(df, date_columns=date_columns, dayfirst=dayfirst, log=log)
    if drop_duplicates:
        df = remove_duplicates(df, log=log)
    df = handle_missing_values(df, method=missing, numeric_method=numeric_fill, log=log)
    df = handle_outliers(df, action=outliers, method=outlier_method, log=log)
    if add_date_parts:
        df = extract_date_features(df, log=log)
    if scale:
        df = scale_numeric_columns(df, method=scale, log=log)
    df = _tidy_integer_columns(df).reset_index(drop=True)
    if not log:
        log.append("No changes were needed.")
    return df, log


def process_dataset(data: Any, **options: Any) -> pd.DataFrame:
    """Same as process_dataset_with_log but returns only the clean DataFrame."""
    return process_dataset_with_log(data, **options)[0]
