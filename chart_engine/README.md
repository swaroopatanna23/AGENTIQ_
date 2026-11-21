# AgentIQ - Chart Generation Engine Module

The **Chart Generation Engine** is an offline, lightweight, and modular Python module designed for **AgentIQ**. It generates production-ready Matplotlib charts from a pandas DataFrame and a configuration dictionary.

---

## Features

- **100% Offline & Local**: No external cloud APIs, internet connectivity, or AI model dependencies required.
- **6 Supported Chart Types**:
  - `bar` (Bar Chart)
  - `line` (Line Chart)
  - `pie` (Pie Chart)
  - `scatter` (Scatter Plot)
  - `histogram` (Histogram)
  - `area` (Area Chart)
- **Customizable**: Full control over title, X/Y axis labels, colors, grid visibility, opacity (`alpha`), and figure size (`figsize`).
- **Input Validation**: Beginner-friendly error messages that catch invalid data, missing columns, or bad configurations before attempting to render.
- **Multi-Format Exporter**: Export charts directly to **PNG**, **PDF**, or **SVG**.
- **Object-Oriented**: Returns a standard `matplotlib.figure.Figure` object so your teammates can easily embed it in GUIs, PDFs, or web viewers.

---

## Directory & File Structure

```text
chart_engine/
│
├── __init__.py               # Package root: exposes generate_chart(), export_chart(), ChartValidationError
├── core.py                   # Main engine: implements generate_chart(config, df)
├── validator.py              # Validation module: verifies DataFrames, config keys, and types
├── exporter.py               # Export module: saves figures to PNG, PDF, and SVG
│
├── generators/               # Dedicated renderers for each chart type
│   ├── __init__.py           # Exports all rendering functions
│   ├── bar.py                # Vertical bar chart generator
│   ├── line.py               # Line chart generator with marker points
│   ├── pie.py                # Pie chart generator with percentage labels
│   ├── scatter.py            # Scatter plot generator for bivariate data
│   ├── histogram.py          # Distribution histogram generator
│   └── area.py               # Area chart (filled line plot) generator
│
├── tests/                    # Unit tests and sample datasets
│   ├── __init__.py
│   ├── sample_data.py        # Realistic sample DataFrames for each chart type
│   ├── test_engine.py        # Automated test suite (6 chart types, formats, and validations)
│   └── output/               # Generated test charts (PNG, PDF, SVG)
│
├── requirements.txt          # Python dependencies (matplotlib, pandas, numpy)
└── README.md                 # Complete documentation and quickstart guide
```

---

## File Explanations

| File | Purpose |
| :--- | :--- |
| `core.py` | Contains the central `generate_chart(config, df)` function. Validates input, sets up the Matplotlib canvas, routes rendering to the appropriate generator, applies styling, and returns the figure. |
| `validator.py` | Validates that `df` is a non-empty DataFrame, that `config` contains valid keys and supported chart types, and that columns exist with proper data types. Raises `ChartValidationError`. |
| `exporter.py` | Contains `export_chart(fig, filepath, format, dpi)`. Creates directories automatically and writes high-resolution files in PNG, PDF, or SVG format. |
| `generators/*.py` | Each file is responsible for drawing one specific chart type onto a Matplotlib `Axes` object. |
| `tests/sample_data.py` | Provides test datasets (e.g., product revenue, monthly trends, browser shares, test distributions). |
| `tests/test_engine.py` | Executes 15 comprehensive unit tests to ensure all chart types, export options, and validation rules work properly. |

---

## Installation

Ensure you have Python 3.10+ installed. Install the required dependencies:

```bash
pip install -r chart_engine/requirements.txt
```

---

## Quickstart & Code Examples

### 1. Generating a Bar Chart

```python
import pandas as pd
from chart_engine import generate_chart, export_chart

# 1. Prepare your DataFrame
data = {
    "product": ["Laptops", "Smartphones", "Monitors", "Keyboards"],
    "revenue": [54000, 72000, 31000, 12400]
}
df = pd.DataFrame(data)

# 2. Define the chart configuration
config = {
    "chart_type": "bar",
    "x_column": "product",
    "y_column": "revenue",
    "title": "Quarterly Revenue by Product",
    "x_label": "Product Category",
    "y_label": "Revenue (USD)",
    "color": "#1f77b4",
    "grid": True,
    "figsize": (10, 6)
}

# 3. Generate the chart (returns a Matplotlib Figure)
fig = generate_chart(config, df)

# 4. Export the chart to disk
export_chart(fig, "output/revenue_bar_chart.png", format="png", dpi=300)
```

---

### 2. Configuration Dictionary Reference

| Key | Type | Required? | Description | Example |
| :--- | :--- | :--- | :--- | :--- |
| `chart_type` | `str` | **Yes** | Type of chart (`'bar'`, `'line'`, `'pie'`, `'scatter'`, `'histogram'`, `'area'`) | `"bar"` |
| `x_column` | `str` | Conditional | Column for X-axis (or pie labels, or histogram column) | `"month"` |
| `y_column` | `str` | Conditional | Column for Y-axis (or pie values) | `"revenue"` |
| `title` | `str` | No | Title displayed on the top of the chart | `"Monthly Revenue"` |
| `x_label` | `str` | No | Label for the X-axis (defaults to `x_column`) | `"Time (Months)"` |
| `y_label` | `str` | No | Label for the Y-axis (defaults to `y_column`) | `"USD"` |
| `color` | `str` | No | Color name or hex string | `"#2ca02c"` |
| `colors` | `list` | No | List of colors (ideal for pie slices or bar series) | `["#4285F4", "#EA4335"]` |
| `grid` | `bool` | No | Whether to show subtle grid lines (default: `True`) | `True` |
| `figsize` | `tuple` | No | Figure dimensions `(width, height)` in inches | `(10, 6)` |
| `alpha` | `float` | No | Opacity between `0.0` (transparent) and `1.0` (opaque) | `0.7` |
| `bins` | `int` | No | Number of bins for `histogram` (default: `10`) | `15` |

---

### 3. All Six Chart Types at a Glance

#### Line Chart
```python
config = {
    "chart_type": "line",
    "x_column": "month",
    "y_column": "active_users",
    "title": "User Growth Trend",
    "color": "#1f77b4"
}
```

#### Pie Chart
```python
config = {
    "chart_type": "pie",
    "x_column": "browser",          # Slices labels
    "y_column": "market_share_pct", # Slices values (must be positive numbers)
    "title": "Browser Market Share"
}
```

#### Scatter Plot
```python
config = {
    "chart_type": "scatter",
    "x_column": "study_hours",      # Numeric
    "y_column": "exam_score",       # Numeric
    "title": "Study Hours vs Score",
    "color": "#d62728"
}
```

#### Histogram
```python
config = {
    "chart_type": "histogram",
    "x_column": "exam_score",       # Single numeric column
    "bins": 12,
    "title": "Exam Score Distribution"
}
```

#### Area Chart
```python
config = {
    "chart_type": "area",
    "x_column": "week",
    "y_column": "page_views",
    "title": "Weekly Page Views",
    "alpha": 0.4
}
```

---

### 4. Exporting Charts

The `export_chart()` function supports **PNG**, **PDF**, and **SVG**:

```python
from chart_engine import export_chart

# Export to PNG (high-resolution image)
export_chart(fig, "charts/sales.png", format="png", dpi=300)

# Export to PDF (vector format for reports and printing)
export_chart(fig, "charts/sales.pdf", format="pdf")

# Export to SVG (scalable vector format for web apps)
export_chart(fig, "charts/sales.svg", format="svg")
```

---

### 5. Error Handling

When something is wrong, `ChartValidationError` is raised with a descriptive message:

```python
from chart_engine import generate_chart, ChartValidationError

try:
    fig = generate_chart({"chart_type": "bar", "x_column": "product"}, df)
except ChartValidationError as err:
    print(f"Validation error: {err}")
    # Output: Validation error: Missing required configuration key 'y_column' for 'bar' chart. Available columns in data: ['product', 'revenue'].
```

---

## Running the Test Suite

From your project root, run:

```bash
python chart_engine/tests/test_engine.py
```

Or using Python's built-in `unittest` runner:

```bash
python -m unittest chart_engine/tests/test_engine.py
```

All generated sample test charts will be saved to `chart_engine/tests/output/` where you can view them.
