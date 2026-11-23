"""
Data Validation Module  (Member 4 - Amrutha)
============================================
Checks whether a dataset is healthy and READY FOR ANALYSIS, and writes a report.

This is about the quality of the DATA (missing values, duplicates, wrong types,
bad column names, invalid values, consistency).  It is separate from Member 3's
chart_engine/validator.py, which only checks a *chart request*
(is chart_type supported? does the x_column exist?).

Main entry point
----------------
validate_dataset(df, ...) -> ValidationReport

Rules of the report
-------------------
* ERROR   = a real problem. The dataset is NOT ready (status FAILED).
* WARNING = something to look at, but analysis can continue
            (status PASSED_WITH_WARNINGS).
* no errors and no warnings -> status PASSED.

Validation never changes the data and never crashes on bad data - it reports.
"""

import difflib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .common import MISSING_PLACEHOLDERS, column_kind, outlier_mask, text_looks_like_date, text_to_numeric

STATUS_PASSED = "PASSED"
STATUS_WARNINGS = "PASSED_WITH_WARNINGS"
STATUS_FAILED = "FAILED"

# Words in a column name that suggest the values should never be negative.
NON_NEGATIVE_HINTS = {
    "age", "price", "quantity", "qty", "amount", "salary", "revenue", "sales",
    "cost", "units", "count", "population", "height", "weight", "marks",
}
VALID_SCHEMA_KINDS = {"numeric", "integer", "float", "text", "datetime", "boolean", "any"}


class DataValidationError(Exception):
    """Raised by assert_valid() when a dataset has errors."""


@dataclass
class ValidationReport:
    """Everything we learned about the dataset."""
    total_rows: int = 0
    total_columns: int = 0
    missing_total: int = 0
    missing_by_column: Dict[str, int] = field(default_factory=dict)
    duplicate_count: int = 0
    data_types: Dict[str, str] = field(default_factory=dict)
    outliers_by_column: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """True when there are no errors = ready for analysis."""
        return not self.errors

    @property
    def status(self) -> str:
        if self.errors:
            return STATUS_FAILED
        return STATUS_WARNINGS if self.warnings else STATUS_PASSED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_rows": self.total_rows,
            "total_columns": self.total_columns,
            "missing_total": self.missing_total,
            "missing_by_column": dict(self.missing_by_column),
            "duplicate_count": self.duplicate_count,
            "data_types": dict(self.data_types),
            "outliers_by_column": dict(self.outliers_by_column),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "status": self.status,
            "ready_for_analysis": self.is_valid,
        }

    def to_text(self) -> str:
        """A plain-text report (used for the download button)."""
        lines = ["AGENTIQ - DATA VALIDATION REPORT", "=" * 40,
                 f"Status: {self.status}",
                 f"Ready for analysis: {'YES' if self.is_valid else 'NO'}",
                 f"Total rows: {self.total_rows}",
                 f"Total columns: {self.total_columns}",
                 f"Missing values: {self.missing_total}",
                 f"Duplicate rows: {self.duplicate_count}",
                 "", "DATA TYPES"]
        lines += [f"  {col}: {dtype}" for col, dtype in self.data_types.items()] or ["  (none)"]
        lines += ["", "MISSING VALUES BY COLUMN"]
        lines += [f"  {c}: {n}" for c, n in self.missing_by_column.items() if n] or ["  none"]
        lines += ["", "POSSIBLE OUTLIERS"]
        lines += [f"  {c}: {n}" for c, n in self.outliers_by_column.items()] or ["  none"]
        lines += ["", f"ERRORS ({len(self.errors)})"] + [f"  - {e}" for e in self.errors]
        lines += ["", f"WARNINGS ({len(self.warnings)})"] + [f"  - {w}" for w in self.warnings]
        return "\n".join(lines)


# ---------------------------------------------------------------------
# Individual checks. Each returns (errors, warnings) so it can be tested alone.
# ---------------------------------------------------------------------

Issues = Tuple[List[str], List[str]]


def check_structure(df: Any) -> Issues:
    """Is it a real, non-empty table?"""
    errors: List[str] = []
    warnings: List[str] = []
    if not isinstance(df, pd.DataFrame):
        return [f"Input is not a table (DataFrame); got '{type(df).__name__}'."], warnings
    if df.shape[1] == 0:
        errors.append("The dataset has no columns.")
    if df.shape[0] == 0:
        errors.append("The dataset is empty: it has no rows.")
    elif df.shape[1] and df.isna().all().all():
        errors.append("Every value in the dataset is missing.")
    elif df.shape[0] == 1:
        warnings.append("The dataset has only 1 row; analysis will be very limited.")
    return errors, warnings


def check_column_names(df: pd.DataFrame) -> Issues:
    """Duplicate, blank, unnamed or non-standard column names."""
    errors: List[str] = []
    warnings: List[str] = []
    names = list(df.columns)
    duplicated = sorted({str(n) for n in names if names.count(n) > 1})
    if duplicated:
        errors.append(f"Duplicate column names: {duplicated}.")
    blank = [n for n in names if str(n).strip() == ""]
    if blank:
        errors.append("One or more columns have a blank name.")
    non_text = [n for n in names if not isinstance(n, str)]
    if non_text:
        warnings.append(f"Column names that are not text: {non_text}.")
    unnamed = [str(n) for n in names if str(n).startswith("Unnamed")]
    if unnamed:
        warnings.append(f"Unnamed columns found (often a saved index): {unnamed}.")
    messy = [str(n) for n in names if str(n).strip() != "" and not re.fullmatch(r"[a-z0-9]+(_[a-z0-9]+)*", str(n))
             and not str(n).startswith("Unnamed")]
    if messy:
        warnings.append(f"Column names are not standardized (use lower_snake_case): {messy[:10]}.")
    return errors, warnings


def check_required_columns(df: pd.DataFrame, required_columns: Optional[Iterable[str]]) -> Issues:
    """Every required column must exist (with a 'did you mean' hint)."""
    errors: List[str] = []
    for col in (required_columns or []):
        if col not in df.columns:
            close = difflib.get_close_matches(str(col), [str(c) for c in df.columns], n=1, cutoff=0.6)
            hint = f" Did you mean '{close[0]}'?" if close else ""
            errors.append(f"Required column '{col}' is missing.{hint}")
    return errors, []


def _kind_matches(series: pd.Series, expected: str) -> bool:
    kind = column_kind(series)
    if expected == "any":
        return True
    if expected == "integer":
        return kind == "numeric" and pd.api.types.is_integer_dtype(series.dtype)
    if expected == "float":
        return kind == "numeric" and pd.api.types.is_float_dtype(series.dtype)
    return kind == expected


def check_schema(df: pd.DataFrame, schema: Optional[Dict[str, str]], strict: bool = False) -> Issues:
    """Compare the table with an expected schema, e.g. {'sales': 'numeric', 'date': 'datetime'}.
    Allowed types: numeric, integer, float, text, datetime, boolean, any."""
    errors: List[str] = []
    warnings: List[str] = []
    if not schema:
        return errors, warnings
    for col, expected in schema.items():
        if expected not in VALID_SCHEMA_KINDS:
            errors.append(f"Schema problem: '{expected}' is not a valid type for '{col}'. Use {sorted(VALID_SCHEMA_KINDS)}.")
        elif col not in df.columns:
            errors.append(f"Schema column '{col}' is missing from the dataset.")
        elif not _kind_matches(df[col], expected):
            errors.append(f"Column '{col}' should be {expected} but is {df[col].dtype} ({column_kind(df[col])}).")
    if strict:
        extra = [c for c in df.columns if c not in schema]
        if extra:
            warnings.append(f"Columns not in the schema: {extra}.")
    return errors, warnings


def check_missing_values(df: pd.DataFrame) -> Tuple[Dict[str, int], Issues]:
    """Count empty cells per column."""
    errors: List[str] = []
    warnings: List[str] = []
    by_column = {str(c): int(df[c].isna().sum()) for c in df.columns}
    total = sum(by_column.values())
    rows = len(df)
    if total:
        affected = {c: n for c, n in by_column.items() if n}
        warnings.append(f"{total} missing value(s) in {len(affected)} column(s): {affected}.")
    for col, n in by_column.items():
        if rows and n == rows:
            errors.append(f"Column '{col}' has no data at all (every value is missing).")
        elif rows and n / rows > 0.5:
            warnings.append(f"Column '{col}' is more than 50% empty ({n} of {rows}).")
    return by_column, (errors, warnings)


def check_duplicates(df: pd.DataFrame) -> Tuple[int, Issues]:
    """Count repeated rows."""
    warnings: List[str] = []
    try:
        count = int(df.duplicated().sum())
    except TypeError:
        return 0, ([], ["Could not check duplicates: some cells contain lists or objects."])
    if count:
        warnings.append(f"{count} duplicate row(s) found ({count / max(len(df), 1):.1%} of the data).")
    return count, ([], warnings)


def check_data_types(df: pd.DataFrame) -> Issues:
    """Columns whose stored type does not match their content."""
    warnings: List[str] = []
    for col in df.columns:
        s = df[col]
        if column_kind(s) != "text" or s.dropna().empty:
            continue
        if pd.api.types.infer_dtype(s, skipna=True) in ("mixed", "mixed-integer"):
            warnings.append(f"Column '{col}' mixes different kinds of values (text and numbers).")
            continue
        _, ratio = text_to_numeric(s)
        if ratio >= 0.9:
            warnings.append(f"Column '{col}' looks numeric but is stored as text; convert it to numbers.")
        elif text_looks_like_date(s):
            warnings.append(f"Column '{col}' looks like dates but is stored as text; convert it to dates.")
    return [], warnings


def check_invalid_values(
    df: pd.DataFrame,
    non_negative_columns: Optional[Iterable[str]] = None,
    value_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    allowed_values: Optional[Dict[str, Iterable[Any]]] = None,
) -> Issues:
    """Impossible or suspicious values.
    Explicit rules you pass in give ERRORS; built-in name-based guesses give WARNINGS."""
    errors: List[str] = []
    warnings: List[str] = []
    explicit_non_negative = set(non_negative_columns or [])

    for col in df.columns:
        s = df[col]
        kind = column_kind(s)
        if kind == "numeric" and not pd.api.types.is_bool_dtype(s.dtype):
            infinite = int(np.isinf(s.astype(float)).sum())
            if infinite:
                errors.append(f"Column '{col}' contains {infinite} infinite value(s).")
            negatives = int((s < 0).sum())
            if negatives and col in explicit_non_negative:
                errors.append(f"Column '{col}' must not be negative but has {negatives} negative value(s).")
            elif negatives and NON_NEGATIVE_HINTS & set(str(col).split("_")):
                warnings.append(f"Column '{col}' has {negatives} negative value(s); check if that is valid.")
            tokens = set(str(col).split("_"))
            if "age" in tokens and int((s > 120).sum()):
                warnings.append(f"Column '{col}' has {int((s > 120).sum())} value(s) above 120.")
            if tokens & {"percent", "percentage", "pct"} and int(((s < 0) | (s > 100)).sum()):
                warnings.append(f"Column '{col}' looks like a percentage but has values outside 0-100.")
        elif kind == "text":
            placeholders = int(s.dropna().astype(str).str.strip().str.lower().isin(MISSING_PLACEHOLDERS).sum())
            if placeholders:
                warnings.append(f"Column '{col}' contains {placeholders} placeholder value(s) like 'N/A' or 'null'.")

    for col, (low, high) in (value_ranges or {}).items():
        if col not in df.columns:
            errors.append(f"Range rule refers to a missing column '{col}'.")
        elif column_kind(df[col]) != "numeric":
            errors.append(f"Range rule needs a numeric column, but '{col}' is {column_kind(df[col])}.")
        else:
            bad = int(((df[col] < low) | (df[col] > high)).sum())
            if bad:
                errors.append(f"Column '{col}' has {bad} value(s) outside the allowed range {low} to {high}.")

    for col, allowed in (allowed_values or {}).items():
        if col not in df.columns:
            errors.append(f"Allowed-values rule refers to a missing column '{col}'.")
            continue
        bad_values = sorted({str(v) for v in df[col].dropna().unique() if v not in set(allowed)})
        if bad_values:
            errors.append(f"Column '{col}' has values that are not allowed: {bad_values[:10]}.")
    return errors, warnings


def check_consistency(df: pd.DataFrame, check_outliers: bool = True) -> Tuple[Dict[str, int], Issues]:
    """Does the data agree with itself?
    constant columns, 'North' vs 'north ' variants, start date after end date, outliers."""
    warnings: List[str] = []
    outliers: Dict[str, int] = {}
    rows = len(df)

    for col in df.columns:
        s = df[col]
        kind = column_kind(s)
        try:
            if rows > 1 and s.nunique(dropna=True) == 1:
                warnings.append(f"Column '{col}' has the same value in every row.")
        except TypeError:
            pass
        if kind == "text":
            texts = s.dropna().astype(str)
            if not texts.empty and texts.nunique() > texts.str.strip().str.lower().nunique():
                warnings.append(f"Column '{col}' has inconsistent spelling/case of the same value (e.g. 'North' vs 'north').")
        if check_outliers and kind == "numeric" and not pd.api.types.is_bool_dtype(s.dtype):
            count = int(outlier_mask(s.astype(float)).sum())
            if count:
                outliers[str(col)] = count
                warnings.append(f"Column '{col}' has {count} possible outlier(s).")

    for col in df.columns:  # start_x must not be after end_x
        if "start" in str(col) and column_kind(df[col]) == "datetime":
            partner = str(col).replace("start", "end")
            if partner in df.columns and column_kind(df[partner]) == "datetime":
                bad = int((df[col] > df[partner]).sum())
                if bad:
                    warnings.append(f"{bad} row(s) have '{col}' later than '{partner}'.")

    if rows and not any(column_kind(df[c]) == "numeric" for c in df.columns):
        warnings.append("There are no numeric columns, so numeric summaries and most charts are not possible.")
    return outliers, ([], warnings)


# ---------------------------------------------------------------------
# The main function
# ---------------------------------------------------------------------

def validate_dataset(
    df: Any,
    required_columns: Optional[Iterable[str]] = None,
    schema: Optional[Dict[str, str]] = None,
    strict_schema: bool = False,
    non_negative_columns: Optional[Iterable[str]] = None,
    value_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    allowed_values: Optional[Dict[str, Iterable[Any]]] = None,
    check_outliers: bool = True,
) -> ValidationReport:
    """Run every check and return one ValidationReport.

    Optional rules (all default to 'not used'):
      required_columns      -> list of column names that must exist
      schema                -> {'column': 'numeric'|'integer'|'float'|'text'|'datetime'|'boolean'|'any'}
      non_negative_columns  -> columns that must not contain negatives
      value_ranges          -> {'age': (0, 120)}
      allowed_values        -> {'region': ['North', 'South']}
    """
    report = ValidationReport()

    errors, warnings = check_structure(df)
    report.errors += errors
    report.warnings += warnings
    if not isinstance(df, pd.DataFrame):
        return report

    report.total_rows, report.total_columns = int(df.shape[0]), int(df.shape[1])
    report.data_types = {str(c): str(df[c].dtype) for c in df.columns} if not df.columns.duplicated().any() else {
        f"{c}#{i}": str(t) for i, (c, t) in enumerate(zip(df.columns, df.dtypes))}

    if df.shape[1] == 0 or df.shape[0] == 0:
        return report  # nothing more to check on an empty table

    def add(issues: Issues) -> None:
        report.errors.extend(issues[0])
        report.warnings.extend(issues[1])

    add(check_column_names(df))
    if df.columns.duplicated().any():
        report.errors.append("Further checks skipped: duplicate column names make column-level checks unreliable.")
        return report
    add(check_required_columns(df, required_columns))
    add(check_schema(df, schema, strict_schema))
    report.missing_by_column, issues = check_missing_values(df)
    report.missing_total = int(sum(report.missing_by_column.values()))
    add(issues)
    report.duplicate_count, issues = check_duplicates(df)
    add(issues)
    add(check_data_types(df))
    add(check_invalid_values(df, non_negative_columns, value_ranges, allowed_values))
    report.outliers_by_column, issues = check_consistency(df, check_outliers)
    add(issues)
    return report


def assert_valid(report: ValidationReport) -> None:
    """Raise DataValidationError if the report contains errors."""
    if not report.is_valid:
        raise DataValidationError("Dataset is not ready for analysis: " + " | ".join(report.errors))
