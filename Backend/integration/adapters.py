"""
Integration Adapters  (Member 4 - Amrutha)
==========================================
The four members' modules speak slightly different "languages":

  Data pipeline  -> a clean pandas DataFrame
  Qwen (Member 2)-> needs TEXT describing the columns;  returns a JSON PLAN
                    {intent, target_column, group_by, aggregation, chart_type, explanation}
  Chart engine   -> needs (config dict, DataFrame) with keys chart_type / x_column / y_column
                    and only knows bar, line, pie, scatter, histogram, area
  Frontend       -> shows a figure, a table, or a single number

These functions translate between those formats. No module is rewritten;
the adapters sit in the middle.
"""

import difflib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from Backend.data.common import column_kind
from Backend.data.processing import (
    AGGREGATION_ALIASES,
    DataProcessingError,
    aggregate_column,
    aggregate_data,
)
from .errors import IntegrationError

MAX_BAR_CATEGORIES = 20   # more bars than this are unreadable
MAX_PIE_SLICES = 8
MAX_TABLE_ROWS = 20
CHART_TYPES_WITH_AXES = ("bar", "line", "pie", "area")
_AGG_WORDS = {"sum": "Total", "mean": "Average", "count": "Count of", "min": "Minimum", "max": "Maximum", "median": "Median"}


@dataclass
class ChartRequest:
    """What the chart engine / frontend should show for one question."""
    kind: str                                  # 'chart', 'table' or 'value'
    data: pd.DataFrame
    config: Optional[Dict[str, Any]] = None    # chart-engine config (only for kind='chart')
    summary: str = ""
    notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------
# Dataset  ->  text for Qwen
# ---------------------------------------------------------------------

def pretty_name(column: str) -> str:
    """'total_sales' -> 'Total Sales' (for chart titles and labels)."""
    return str(column).replace("_", " ").strip().title()


def _plain(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    return value


def build_dataset_info(df: pd.DataFrame, max_examples: int = 3, max_columns: int = 60) -> Dict[str, Any]:
    """Describe the dataset: row count, and for each column its name, type and examples."""
    columns = []
    for col in list(df.columns)[:max_columns]:
        examples = [_plain(v) for v in df[col].dropna().unique()[:max_examples]]
        columns.append({"name": str(col), "type": column_kind(df[col]), "examples": examples})
    return {"rows": int(len(df)), "columns": columns}


def dataset_info_to_text(info: Dict[str, Any]) -> str:
    """Turn the description into the short text that is added to Qwen's prompt."""
    lines = [f"The dataset has {info['rows']} rows and these columns:"]
    for col in info["columns"]:
        examples = ", ".join(str(e) for e in col["examples"])
        lines.append(f"- {col['name']} ({col['type']}); example values: {examples}")
    return "\n".join(lines)


# ---------------------------------------------------------------------
# Qwen's column names  ->  real DataFrame columns
# ---------------------------------------------------------------------

def _squash(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def resolve_column(name: Any, columns: List[str], cutoff: float = 0.75) -> Optional[str]:
    """Match a column name from Qwen to a real column.
    Exact match first, then ignoring case/spaces/underscores, then the closest spelling.
    Returns None when Qwen gave no column (null)."""
    if name is None or str(name).strip().lower() in ("", "null", "none", "n/a"):
        return None
    name = str(name)
    if name in columns:
        return name
    squashed = {_squash(c): c for c in columns}
    if _squash(name) in squashed:
        return squashed[_squash(name)]
    close = difflib.get_close_matches(_squash(name), list(squashed), n=1, cutoff=cutoff)
    if close:
        return squashed[close[0]]
    raise IntegrationError(
        "adapter",
        f"The AI mentioned a column called '{name}', but the dataset has no such column. "
        f"Available columns: {columns}. Try rephrasing your question with one of these names.",
    )


def normalize_aggregation(aggregation: Any) -> Optional[str]:
    """Qwen word ('average', 'maximum') -> pandas name ('mean', 'max'). 'none' -> None."""
    word = str(aggregation or "none").strip().lower()
    if word in ("none", "", "null"):
        return None
    if word not in AGGREGATION_ALIASES:
        raise IntegrationError("adapter", f"The AI chose an unsupported calculation '{aggregation}'.")
    return AGGREGATION_ALIASES[word]


# ---------------------------------------------------------------------
# Qwen's plan  ->  (DataFrame, chart-engine config)
# ---------------------------------------------------------------------

def _title(agg: Optional[str], target: Optional[str], group: Optional[str]) -> str:
    if agg == "count" and not target:
        head = "Number of Records"
    else:
        word, name = _AGG_WORDS.get(agg or "", ""), pretty_name(target)
        head = name if name.lower().startswith(word.lower()) else f"{word} {name}".strip()
    return f"{head} by {pretty_name(group)}" if group else head


def plan_to_chart_request(plan: Dict[str, Any], df: pd.DataFrame) -> ChartRequest:
    """Convert Qwen's JSON plan into something the chart engine / frontend can show."""
    columns = [str(c) for c in df.columns]
    chart_type = str(plan.get("chart_type", "none")).strip().lower()
    target = resolve_column(plan.get("target_column"), columns)
    group = resolve_column(plan.get("group_by"), columns)
    agg = normalize_aggregation(plan.get("aggregation"))
    summary = str(plan.get("explanation") or "").strip()
    notes: List[str] = []

    try:
        # ---- no chart wanted: table or single number --------------------------------
        if chart_type in ("table", "none"):
            return _table_or_value(df, target, group, agg, summary, notes)

        # ---- histogram: distribution of ONE numeric column --------------------------
        if chart_type == "histogram":
            column = target or group
            if column is None:
                raise IntegrationError("adapter", "A histogram needs a numeric column. Mention one in your question.")
            return ChartRequest("chart", df[[column]].dropna(), {
                "chart_type": "histogram", "x_column": column,
                "title": f"Distribution of {pretty_name(column)}", "x_label": pretty_name(column),
            }, summary, notes)

        # ---- scatter: two numeric columns, no aggregation ---------------------------
        if chart_type == "scatter":
            if target is None or group is None:
                raise IntegrationError("adapter", "A scatter plot needs two numeric columns. Mention both in your question.")
            return ChartRequest("chart", df[[group, target]].dropna(), {
                "chart_type": "scatter", "x_column": group, "y_column": target,
                "title": f"{pretty_name(target)} vs {pretty_name(group)}",
                "x_label": pretty_name(group), "y_label": pretty_name(target),
            }, summary, notes)

        # ---- bar / line / pie / area: group rows, then summarise --------------------
        if chart_type not in CHART_TYPES_WITH_AXES:
            raise IntegrationError("adapter", f"Chart type '{chart_type}' is not supported.")
        if group is None:
            notes.append("No grouping column was found in the question, so a table is shown instead of a chart.")
            return _table_or_value(df, target, group, agg, summary, notes)

        if target is None:
            if agg not in (None, "count"):
                raise IntegrationError("adapter", f"To calculate the {agg}, name the numeric column in your question.")
            agg, value_col = "count", "count"
            data = aggregate_data(df, group, None, "count")
        elif agg is None and df[group].is_unique:
            value_col, data = target, df[[group, target]].dropna()   # already one row per group
        else:
            if agg is None:
                agg = "sum"
                notes.append("No calculation was specified, so the values were added up (sum).")
            value_col, data = target, aggregate_data(df, group, target, agg)

        if column_kind(data[group]) == "datetime" or chart_type in ("line", "area"):
            data = data.sort_values(group)
        else:
            data = data.sort_values(value_col, ascending=False)
            limit = MAX_PIE_SLICES if chart_type == "pie" else MAX_BAR_CATEGORIES
            if len(data) > limit:
                notes.append(f"Only the top {limit} of {len(data)} groups are shown to keep the chart readable.")
                data = data.head(limit)
        return ChartRequest("chart", data.reset_index(drop=True), {
            "chart_type": chart_type, "x_column": group, "y_column": value_col,
            "title": _title(agg, target, group),
            "x_label": pretty_name(group), "y_label": pretty_name(value_col),
        }, summary, notes)

    except DataProcessingError as err:
        raise IntegrationError("adapter", str(err))


def _table_or_value(df, target, group, agg, summary, notes) -> ChartRequest:
    """Answer without a chart: grouped table, one number, or a preview of the columns."""
    if group and (target or agg == "count"):
        table = aggregate_data(df, group, target, agg or ("count" if target is None else "sum"))
        return ChartRequest("table", table, None, summary, notes)
    if target and agg:
        value = aggregate_column(df, target, agg)
        label = f"{_AGG_WORDS.get(agg, agg)} {pretty_name(target)}".strip()
        return ChartRequest("value", pd.DataFrame({"metric": [label], "value": [_plain(value)]}), None, summary, notes)
    shown = [c for c in (group, target) if c] or list(df.columns)
    notes.append(f"Showing the first {MAX_TABLE_ROWS} rows.")
    return ChartRequest("table", df[shown].head(MAX_TABLE_ROWS), None, summary, notes)
