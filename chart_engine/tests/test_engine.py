"""
Comprehensive Test Suite for Chart Generation Engine
=====================================================
Tests all six chart types, custom styling, export formats (PNG, PDF, SVG),
and error validation handling.

Run this test suite via:
    python -m unittest chart_engine/tests/test_engine.py
or directly:
    python chart_engine/tests/test_engine.py
"""

import os
import sys
import unittest

# Ensure the root workspace directory is in sys.path so 'chart_engine' is discoverable
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

import matplotlib.figure
import pandas as pd

from chart_engine.core import generate_chart
from chart_engine.exporter import export_chart, SUPPORTED_EXPORT_FORMATS
from chart_engine.validator import ChartValidationError
from chart_engine.tests.sample_data import (
    get_sales_data,
    get_monthly_trend_data,
    get_market_share_data,
    get_student_study_data,
    get_exam_distribution_data,
    get_website_traffic_data,
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


class TestChartEngine(unittest.TestCase):
    """Test suite covering chart generation, exports, and validations."""

    @classmethod
    def setUpClass(cls):
        """Prepare output directory for sample generated charts."""
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    def test_01_bar_chart_generation_and_export(self):
        """Test bar chart generation and PNG export."""
        df = get_sales_data()
        config = {
            "chart_type": "bar",
            "x_column": "product",
            "y_column": "revenue_usd",
            "title": "Revenue by Product Category",
            "x_label": "Product",
            "y_label": "Revenue (USD)",
            "color": "#2ca02c",
            "grid": True,
            "figsize": (9, 5),
        }

        fig = generate_chart(config, df)
        self.assertIsInstance(fig, matplotlib.figure.Figure)

        output_path = os.path.join(OUTPUT_DIR, "test_bar_chart.png")
        saved_path = export_chart(fig, output_path, format="png", close_fig=True)
        self.assertTrue(os.path.exists(saved_path))
        self.assertGreater(os.path.getsize(saved_path), 0)

    def test_02_line_chart_generation_and_export(self):
        """Test line chart generation and PDF export."""
        df = get_monthly_trend_data()
        config = {
            "chart_type": "line",
            "x_column": "month",
            "y_column": "active_users",
            "title": "Monthly Active Users Growth",
            "x_label": "Month",
            "y_label": "Active Users",
            "color": "#1f77b4",
            "grid": True,
            "figsize": (10, 5),
        }

        fig = generate_chart(config, df)
        self.assertIsInstance(fig, matplotlib.figure.Figure)

        output_path = os.path.join(OUTPUT_DIR, "test_line_chart.pdf")
        saved_path = export_chart(fig, output_path, format="pdf", close_fig=True)
        self.assertTrue(os.path.exists(saved_path))
        self.assertGreater(os.path.getsize(saved_path), 0)

    def test_03_pie_chart_generation_and_export(self):
        """Test pie chart generation and SVG export."""
        df = get_market_share_data()
        config = {
            "chart_type": "pie",
            "x_column": "browser",
            "y_column": "market_share_pct",
            "title": "Global Browser Market Share",
            "colors": ["#4285F4", "#34A853", "#FBBC05", "#EA4335", "#9C27B0"],
            "figsize": (7, 7),
        }

        fig = generate_chart(config, df)
        self.assertIsInstance(fig, matplotlib.figure.Figure)

        output_path = os.path.join(OUTPUT_DIR, "test_pie_chart.svg")
        saved_path = export_chart(fig, output_path, format="svg", close_fig=True)
        self.assertTrue(os.path.exists(saved_path))
        self.assertGreater(os.path.getsize(saved_path), 0)

    def test_04_scatter_chart_generation_and_export(self):
        """Test scatter plot generation with numeric X and Y columns."""
        df = get_student_study_data()
        config = {
            "chart_type": "scatter",
            "x_column": "study_hours",
            "y_column": "exam_score",
            "title": "Study Hours vs Exam Score Correlation",
            "x_label": "Weekly Study Hours",
            "y_label": "Exam Score (out of 100)",
            "color": "#d62728",
            "grid": True,
            "figsize": (8, 5),
        }

        fig = generate_chart(config, df)
        self.assertIsInstance(fig, matplotlib.figure.Figure)

        output_path = os.path.join(OUTPUT_DIR, "test_scatter_chart.png")
        saved_path = export_chart(fig, output_path, format="png", close_fig=True)
        self.assertTrue(os.path.exists(saved_path))
        self.assertGreater(os.path.getsize(saved_path), 0)

    def test_05_histogram_chart_generation_and_export(self):
        """Test histogram distribution chart generation."""
        df = get_exam_distribution_data()
        config = {
            "chart_type": "histogram",
            "x_column": "student_score",
            "bins": 12,
            "title": "Exam Score Distribution",
            "x_label": "Score Range",
            "y_label": "Student Count",
            "color": "#9467bd",
            "grid": True,
            "figsize": (8, 5),
        }

        fig = generate_chart(config, df)
        self.assertIsInstance(fig, matplotlib.figure.Figure)

        output_path = os.path.join(OUTPUT_DIR, "test_histogram_chart.png")
        saved_path = export_chart(fig, output_path, format="png", close_fig=True)
        self.assertTrue(os.path.exists(saved_path))
        self.assertGreater(os.path.getsize(saved_path), 0)

    def test_06_area_chart_generation_and_export(self):
        """Test area chart generation with fill."""
        df = get_website_traffic_data()
        config = {
            "chart_type": "area",
            "x_column": "week",
            "y_column": "page_views",
            "title": "Weekly Page Views Growth",
            "x_label": "Week",
            "y_label": "Total Views",
            "color": "#17becf",
            "alpha": 0.4,
            "grid": True,
            "figsize": (9, 5),
        }

        fig = generate_chart(config, df)
        self.assertIsInstance(fig, matplotlib.figure.Figure)

        output_path = os.path.join(OUTPUT_DIR, "test_area_chart.png")
        saved_path = export_chart(fig, output_path, format="png", close_fig=True)
        self.assertTrue(os.path.exists(saved_path))
        self.assertGreater(os.path.getsize(saved_path), 0)

    # -------------------------------------------------------------
    # Validation and Error Handling Tests
    # -------------------------------------------------------------

    def test_validation_empty_dataframe(self):
        """Ensures an empty DataFrame raises a clear ChartValidationError."""
        empty_df = pd.DataFrame()
        config = {"chart_type": "bar", "x_column": "a", "y_column": "b"}
        with self.assertRaises(ChartValidationError) as ctx:
            generate_chart(config, empty_df)
        self.assertIn("DataFrame is empty", str(ctx.exception))

    def test_validation_invalid_data_type(self):
        """Ensures non-DataFrame data input raises ChartValidationError."""
        invalid_data = [{"product": "A", "revenue": 100}]
        config = {"chart_type": "bar", "x_column": "product", "y_column": "revenue"}
        with self.assertRaises(ChartValidationError) as ctx:
            generate_chart(config, invalid_data)  # type: ignore
        self.assertIn("Expected a pandas DataFrame", str(ctx.exception))

    def test_validation_missing_chart_type(self):
        """Ensures missing chart_type raises ChartValidationError."""
        df = get_sales_data()
        config = {"x_column": "product", "y_column": "revenue_usd"}
        with self.assertRaises(ChartValidationError) as ctx:
            generate_chart(config, df)
        self.assertIn("Missing required configuration key: 'chart_type'", str(ctx.exception))

    def test_validation_unsupported_chart_type(self):
        """Ensures unsupported chart_type raises ChartValidationError."""
        df = get_sales_data()
        config = {"chart_type": "radar", "x_column": "product", "y_column": "revenue_usd"}
        with self.assertRaises(ChartValidationError) as ctx:
            generate_chart(config, df)
        self.assertIn("Unsupported chart_type 'radar'", str(ctx.exception))

    def test_validation_missing_columns(self):
        """Ensures missing required column raises ChartValidationError listing available columns."""
        df = get_sales_data()
        config = {"chart_type": "bar", "x_column": "product"}  # missing y_column
        with self.assertRaises(ChartValidationError) as ctx:
            generate_chart(config, df)
        self.assertIn("Missing required configuration key 'y_column'", str(ctx.exception))

    def test_validation_non_existent_column(self):
        """Ensures specifying a non-existent column gives a helpful error message."""
        df = get_sales_data()
        config = {"chart_type": "bar", "x_column": "non_existent_col", "y_column": "revenue_usd"}
        with self.assertRaises(ChartValidationError) as ctx:
            generate_chart(config, df)
        self.assertIn("Column 'non_existent_col' specified for 'x_column' was not found", str(ctx.exception))
        self.assertIn("Available columns are", str(ctx.exception))

    def test_validation_non_numeric_y_column(self):
        """Ensures non-numeric column passed for Y-axis raises ChartValidationError."""
        df = get_sales_data()
        # 'product' is string, not numeric
        config = {"chart_type": "line", "x_column": "units_sold", "y_column": "product"}
        with self.assertRaises(ChartValidationError) as ctx:
            generate_chart(config, df)
        self.assertIn("must contain numeric data", str(ctx.exception))

    def test_validation_scatter_requires_numeric_x(self):
        """Ensures scatter plot requires numeric X column."""
        df = get_sales_data()
        # 'product' is string
        config = {"chart_type": "scatter", "x_column": "product", "y_column": "revenue_usd"}
        with self.assertRaises(ChartValidationError) as ctx:
            generate_chart(config, df)
        self.assertIn("scatter plot X coordinates", str(ctx.exception))

    def test_exporter_unsupported_format(self):
        """Ensures exporting with unsupported format raises ValueError."""
        df = get_sales_data()
        config = {"chart_type": "bar", "x_column": "product", "y_column": "revenue_usd"}
        fig = generate_chart(config, df)
        with self.assertRaises(ValueError) as ctx:
            export_chart(fig, "test.docx", format="docx", close_fig=True)
        self.assertIn("Unsupported export format 'docx'", str(ctx.exception))


def run_all_tests():
    """Helper function to run tests and print a clean summary."""
    print("=" * 70)
    print(" AgentIQ - Chart Generation Engine Test Suite")
    print("=" * 70)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestChartEngine)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("=" * 70)
    if result.wasSuccessful():
        print(f" ALL {result.testsRun} TESTS PASSED SUCCESSFULLY!")
        print(f"Sample generated chart files saved in: {os.path.abspath(OUTPUT_DIR)}")
    else:
        print(f"FAILED: {len(result.failures)} failures, {len(result.errors)} errors.")
    print("=" * 70)
    return result.wasSuccessful()


if __name__ == "__main__":
    run_all_tests()
