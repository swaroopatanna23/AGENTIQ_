"""Shared test setup: makes the project root importable and provides small fixtures."""
import io
import os
import sys

import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

SAMPLE_DIR = os.path.join(ROOT, "sample_data")


@pytest.fixture
def sample_dir():
    return SAMPLE_DIR


@pytest.fixture
def messy_df():
    """A small table with every kind of problem."""
    return pd.DataFrame({
        "Order ID": [1, 2, 3, 3, 4, 5],
        "Total Sales ($)": ["$1,000", "2,000", "N/A", "N/A", "4000", "5000"],
        "Region ": ["north ", "North", "South", "South", None, "East"],
        "Order Date": ["2024-01-01", "2024-02-01", "15/03/2024", "15/03/2024", "2024-04-01", "2024-05-01"],
    })


class FakeUpload(io.BytesIO):
    """Behaves like a Streamlit UploadedFile: bytes + a .name attribute."""
    def __init__(self, data: bytes, name: str):
        super().__init__(data)
        self.name = name
