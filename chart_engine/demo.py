"""
AgentIQ Chart Generation Engine - Quick Demo
============================================
A simple walkthrough showing how beginner-friendly it is to:
1. Create a pandas DataFrame
2. Define a configuration dictionary with `x_column` and `y_column`
3. Generate a Matplotlib Figure using `generate_chart()`
4. Export to PNG, PDF, or SVG using `export_chart()`
"""

import os
import sys

# Ensure chart_engine is discoverable
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

import pandas as pd
from chart_engine import generate_chart, export_chart

def main():
    print("=" * 60)
    print("AgentIQ Chart Generation Engine - Demo")
    print("=" * 60)

    # 1. Prepare sample data
    data = {
        "product": ["Laptops", "Smartphones", "Headphones", "Monitors", "Keyboards"],
        "sales": [54000, 72000, 18500, 31000, 12400]
    }
    df = pd.DataFrame(data)
    print("\n1. Sample DataFrame:")
    print(df)

    # 2. Define chart configuration using x_column and y_column
    config = {
        "chart_type": "bar",
        "x_column": "product",
        "y_column": "sales",
        "title": "Quarterly Product Sales (AgentIQ Demo)",
        "x_label": "Product",
        "y_label": "Revenue (USD)",
        "color": "#2ca02c",
        "grid": True,
        "figsize": (8, 5)
    }

    # 3. Generate chart (returns Matplotlib Figure)
    print("\n2. Generating chart...")
    fig = generate_chart(config, df)
    print(f"   Success! Generated Figure: {fig}")

    # 4. Export chart to output directory
    output_path = os.path.join(os.path.dirname(__file__), "demo_chart.png")
    exported_path = export_chart(fig, output_path, format="png", close_fig=True)
    print(f"3. Exported chart saved to:\n   {exported_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
