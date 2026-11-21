"""
Sample Data Module for Chart Generation Engine
==============================================
Provides sample DataFrames for testing and demonstrating all six chart types.
Built using pandas and numpy without any external network calls.
"""

import numpy as np
import pandas as pd


def get_sales_data() -> pd.DataFrame:
    """Returns sample categorical sales data suitable for Bar charts."""
    return pd.DataFrame({
        "product": ["Laptops", "Smartphones", "Headphones", "Monitors", "Keyboards"],
        "revenue_usd": [54000, 72000, 18500, 31000, 12400],
        "units_sold": [54, 120, 210, 85, 310],
    })


def get_monthly_trend_data() -> pd.DataFrame:
    """Returns sample time-series data suitable for Line charts."""
    return pd.DataFrame({
        "month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        "active_users": [1200, 1450, 1800, 2100, 2050, 2400, 2900, 3100, 3400, 3600, 4100, 4500],
    })


def get_market_share_data() -> pd.DataFrame:
    """Returns sample proportions suitable for Pie charts."""
    return pd.DataFrame({
        "browser": ["Chrome", "Safari", "Edge", "Firefox", "Opera"],
        "market_share_pct": [65.2, 18.5, 5.4, 3.2, 2.7],
    })


def get_student_study_data() -> pd.DataFrame:
    """Returns bivariate correlation data suitable for Scatter plots."""
    np.random.seed(42)
    study_hours = np.array([1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5])
    # Scores roughly correlate with study hours with some random variance
    noise = np.random.normal(0, 4, len(study_hours))
    exam_score = np.clip(50 + study_hours * 5.5 + noise, 0, 100)

    return pd.DataFrame({
        "study_hours": study_hours,
        "exam_score": np.round(exam_score, 1),
    })


def get_exam_distribution_data() -> pd.DataFrame:
    """Returns sample scores suitable for Histograms."""
    np.random.seed(42)
    # Generate 150 normally distributed student test scores around mean 72, std 12
    scores = np.random.normal(loc=72.0, scale=12.0, size=150)
    scores = np.clip(scores, 35.0, 100.0)

    return pd.DataFrame({
        "student_score": np.round(scores, 1),
    })


def get_website_traffic_data() -> pd.DataFrame:
    """Returns cumulative or trend data suitable for Area charts."""
    return pd.DataFrame({
        "week": [f"Wk {i}" for i in range(1, 11)],
        "page_views": [12000, 14500, 13800, 17200, 19100, 22400, 25100, 27800, 31200, 34500],
    })
