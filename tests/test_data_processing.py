"""Tests for Backend/data/processing.py  (Member 4: Data Processing)."""
import json

import numpy as np
import pandas as pd
import pytest

from Backend.data.processing import (
    DataLoadError, DataProcessingError, aggregate_data, clean_text_values, convert_data_types,
    extract_date_features, handle_missing_values, handle_outliers, load_dataset,
    load_dataset_from_text, process_dataset, process_dataset_with_log, remove_duplicates,
    scale_numeric_columns, standardize_column_names, unify_text_case,
)
from conftest import FakeUpload


# ---------- loading ----------------------------------------------------------
class TestLoading:
    def test_csv_from_path(self, sample_dir):
        df = load_dataset(f"{sample_dir}/messy_sales.csv")
        assert df.shape == (20, 6)

    def test_excel_from_path(self, sample_dir):
        df = load_dataset(f"{sample_dir}/messy_sales.xlsx")
        assert df.shape == (20, 6) and "Total Sales ($)" in df.columns

    def test_json_from_path(self, sample_dir):
        df = load_dataset(f"{sample_dir}/messy_sales.json")
        assert df.shape == (20, 6)

    def test_all_three_formats_give_same_table(self, sample_dir):
        a, b, c = (load_dataset(f"{sample_dir}/messy_sales.{e}") for e in ("csv", "xlsx", "json"))
        assert list(a.columns) == list(b.columns) == list(c.columns)

    def test_csv_from_upload_object(self):
        df = load_dataset(FakeUpload(b"a,b\n1,2\n3,4\n", "x.csv"))
        assert df.shape == (2, 2)

    def test_csv_semicolon_separator_detected(self):
        df = load_dataset(FakeUpload(b"a;b\n1;2\n3;4\n", "x.csv"))
        assert list(df.columns) == ["a", "b"]

    def test_csv_latin1_encoding(self):
        df = load_dataset(FakeUpload("name,city\nJosé,Málaga\n".encode("latin-1"), "x.csv"))
        assert df.loc[0, "name"] == "José"

    def test_excel_from_upload_object(self, sample_dir):
        data = open(f"{sample_dir}/messy_sales.xlsx", "rb").read()
        assert load_dataset(FakeUpload(data, "up.xlsx")).shape[0] == 20

    @pytest.mark.parametrize("payload", [
        [{"a": 1, "b": 2}, {"a": 3, "b": 4}],                 # list of records
        {"a": [1, 3], "b": [2, 4]},                            # dict of columns
        {"data": [{"a": 1, "b": 2}, {"a": 3, "b": 4}]},        # wrapped records
    ])
    def test_json_layouts(self, payload):
        df = load_dataset(FakeUpload(json.dumps(payload).encode(), "x.json"))
        assert df.shape == (2, 2)

    def test_json_nested_is_flattened(self):
        df = load_dataset(FakeUpload(json.dumps([{"a": 1, "p": {"q": 2}}]).encode(), "x.json"))
        assert "p_q" in df.columns

    def test_json_lines(self):
        df = load_dataset(FakeUpload(b'{"a":1}\n{"a":2}\n', "x.json"))
        assert df["a"].tolist() == [1, 2]

    def test_unsupported_extension(self):
        with pytest.raises(DataLoadError, match="Unsupported"):
            load_dataset(FakeUpload(b"hello", "notes.txt"))

    def test_missing_file(self):
        with pytest.raises(DataLoadError, match="not found"):
            load_dataset("does_not_exist.csv")

    def test_empty_csv_file(self, sample_dir):
        with pytest.raises(DataLoadError, match="empty"):
            load_dataset(f"{sample_dir}/empty.csv")

    def test_header_only_csv_gives_empty_frame(self, sample_dir):
        df = load_dataset(f"{sample_dir}/header_only.csv")
        assert df.shape == (0, 3)

    def test_corrupt_excel(self):
        with pytest.raises(DataLoadError):
            load_dataset(FakeUpload(b"this is not an excel file", "bad.xlsx"))

    def test_invalid_json(self):
        with pytest.raises(DataLoadError):
            load_dataset(FakeUpload(b"{not json at all", "bad.json"))

    def test_text_entry_csv_and_json(self):
        assert load_dataset_from_text("a,b\n1,2").shape == (1, 2)
        assert load_dataset_from_text('[{"a": 1}]').shape == (1, 1)
        with pytest.raises(DataLoadError):
            load_dataset_from_text("   ")


# ---------- column names -----------------------------------------------------
class TestColumnNames:
    def test_standardization(self):
        df = pd.DataFrame(columns=["Total Sales ($)", " Order-Date ", "CustomerID", "Growth %"])
        assert list(standardize_column_names(df).columns) == ["total_sales", "order_date", "customer_id", "growth_pct"]

    def test_duplicates_and_blanks_are_made_unique(self):
        df = pd.DataFrame([[1, 2, 3]], columns=["A b", "a_b", ""])
        assert list(standardize_column_names(df).columns) == ["a_b", "a_b_2", "column_3"]

    def test_original_not_modified(self):
        df = pd.DataFrame({"Big Name": [1]})
        standardize_column_names(df)
        assert list(df.columns) == ["Big Name"]


# ---------- cleaning ---------------------------------------------------------
class TestCleaning:
    def test_whitespace_and_placeholders(self):
        df = pd.DataFrame({"x": [" a ", "N/A", "null", "-", "", "b"]})
        out = clean_text_values(df)
        assert out["x"].tolist()[0] == "a" and out["x"].isna().sum() == 4

    def test_case_variants_unified(self):
        out = unify_text_case(pd.DataFrame({"r": ["North", "north", "North", "South"]}))
        assert out["r"].tolist() == ["North", "North", "North", "South"]

    def test_duplicates_removed_and_index_reset(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
        out = remove_duplicates(df)
        assert len(out) == 2 and out.index.tolist() == [0, 1]

    def test_duplicates_subset(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "y", "z"]})
        assert len(remove_duplicates(df, subset=["a"])) == 2

    def test_duplicates_with_unhashable_cells_do_not_crash(self):
        df = pd.DataFrame({"a": [[1], [1]]})
        assert len(remove_duplicates(df)) in (1, 2)   # older pandas cannot compare lists; it must not crash

    def test_unnamed_index_column_removed(self):
        df = pd.DataFrame({"Unnamed: 0": [0, 1, 2], "v": [5, 6, 7]})
        assert list(process_dataset(df).columns) == ["v"]

    def test_empty_rows_and_columns_removed(self):
        df = pd.DataFrame({"a": [1, None, 3], "b": [None, None, None]})
        out = process_dataset(df)
        assert list(out.columns) == ["a"] and len(out) == 2


# ---------- data types -------------------------------------------------------
class TestTypes:
    def test_text_numbers_become_numeric(self):
        out = convert_data_types(pd.DataFrame({"p": ["$1,200", "300", "45.5"]}))
        assert pd.api.types.is_numeric_dtype(out["p"]) and out["p"].iloc[0] == 1200

    def test_dates_converted_including_day_first_values(self):
        out = convert_data_types(pd.DataFrame({"d": ["2024-01-05", "15/06/2024", "2024-03-01"]}))
        assert pd.api.types.is_datetime64_any_dtype(out["d"])
        assert out["d"].iloc[1] == pd.Timestamp("2024-06-15")

    def test_text_stays_text(self):
        out = convert_data_types(pd.DataFrame({"city": ["Pune", "Delhi", "Goa"]}))
        assert not pd.api.types.is_numeric_dtype(out["city"])

    def test_ids_with_leading_zeros_stay_text(self):
        out = convert_data_types(pd.DataFrame({"zip": ["00123", "00456", "01234"]}))
        assert out["zip"].iloc[0] == "00123"

    def test_mostly_text_column_is_not_forced_to_numbers(self):
        out = convert_data_types(pd.DataFrame({"m": ["a", "b", "c", "5"]}))
        assert not pd.api.types.is_numeric_dtype(out["m"])

    def test_bad_values_in_numeric_column_become_missing(self):
        out = convert_data_types(pd.DataFrame({"n": ["1"] * 19 + ["oops"]}))
        assert out["n"].isna().sum() == 1


# ---------- missing values ---------------------------------------------------
class TestMissing:
    def test_fill_numeric_median_and_text_mode(self):
        df = pd.DataFrame({"n": [1.0, 2.0, np.nan, 100.0], "t": ["a", "a", None, "b"]})
        out = handle_missing_values(df)
        assert out["n"].iloc[2] == 2.0 and out["t"].iloc[2] == "a"
        assert out.isna().sum().sum() == 0

    def test_fill_mean_and_zero(self):
        df = pd.DataFrame({"n": [2.0, 4.0, np.nan]})
        assert handle_missing_values(df, numeric_method="mean")["n"].iloc[2] == 3.0
        assert handle_missing_values(df, numeric_method="zero")["n"].iloc[2] == 0

    def test_all_text_missing_uses_unknown(self):
        out = handle_missing_values(pd.DataFrame({"t": [None, None]}, dtype=object))
        assert out["t"].tolist() == ["Unknown", "Unknown"]

    def test_drop_rows(self):
        out = handle_missing_values(pd.DataFrame({"n": [1, None, 3]}), method="drop")
        assert len(out) == 2

    def test_none_leaves_data(self):
        assert handle_missing_values(pd.DataFrame({"n": [1, None]}), method="none")["n"].isna().sum() == 1

    def test_dates_are_not_invented(self):
        df = pd.DataFrame({"d": pd.to_datetime(["2024-01-01", None])})
        assert handle_missing_values(df)["d"].isna().sum() == 1

    def test_invalid_method(self):
        with pytest.raises(DataProcessingError):
            handle_missing_values(pd.DataFrame({"a": [1]}), method="magic")


# ---------- outliers ---------------------------------------------------------
class TestOutliers:
    df = pd.DataFrame({"v": [10, 11, 12, 11, 10, 12, 11, 500]})

    def test_cap(self):
        out = handle_outliers(self.df, action="cap")
        assert out["v"].max() < 500 and len(out) == 8

    def test_remove(self):
        out = handle_outliers(self.df, action="remove")
        assert len(out) == 7 and out["v"].max() == 12

    def test_none_changes_nothing(self):
        assert handle_outliers(self.df, action="none").equals(self.df)

    def test_zscore_method_runs(self):
        assert len(handle_outliers(self.df, action="remove", method="zscore")) <= 8

    def test_bad_action(self):
        with pytest.raises(DataProcessingError):
            handle_outliers(self.df, action="explode")


# ---------- transformation ---------------------------------------------------
class TestTransform:
    def test_date_parts(self):
        df = pd.DataFrame({"d": pd.to_datetime(["2024-03-05", "2025-12-31"])})
        out = extract_date_features(df)
        assert out["d_year"].tolist() == [2024, 2025] and out["d_month"].tolist() == [3, 12]

    def test_minmax_scaling(self):
        out = scale_numeric_columns(pd.DataFrame({"v": [0, 5, 10]}))
        assert out["v_scaled"].tolist() == [0.0, 0.5, 1.0]

    def test_zscore_scaling_mean_zero(self):
        out = scale_numeric_columns(pd.DataFrame({"v": [1, 2, 3, 4]}), method="zscore")
        assert abs(out["v_scaled"].mean()) < 1e-9

    def test_constant_column_not_scaled(self):
        assert "v_scaled" not in scale_numeric_columns(pd.DataFrame({"v": [5, 5, 5]})).columns

    def test_aggregate_with_qwen_words(self):
        df = pd.DataFrame({"r": ["a", "a", "b"], "v": [1, 3, 10]})
        assert aggregate_data(df, "r", "v", "average")["v"].tolist() == [2.0, 10.0]
        assert aggregate_data(df, "r", "v", "maximum")["v"].tolist() == [3, 10]
        assert aggregate_data(df, "r", None, "count")["count"].tolist() == [2, 1]

    def test_aggregate_errors(self):
        df = pd.DataFrame({"r": ["a"], "t": ["x"]})
        with pytest.raises(DataProcessingError):
            aggregate_data(df, "r", "t", "sum")          # text column cannot be summed
        with pytest.raises(DataProcessingError):
            aggregate_data(df, "nope", "t", "count")
        with pytest.raises(DataProcessingError):
            aggregate_data(df, "r", "t", "banana")


# ---------- whole pipeline ---------------------------------------------------
class TestFullProcessing:
    def test_messy_frame_becomes_clean(self, messy_df):
        out, log = process_dataset_with_log(messy_df)
        assert list(out.columns) == ["order_id", "total_sales", "region", "order_date"]
        assert pd.api.types.is_numeric_dtype(out["total_sales"])
        assert pd.api.types.is_datetime64_any_dtype(out["order_date"])
        assert len(out) == 5                       # one duplicate removed
        assert out.isna().sum().sum() == 0         # everything filled
        assert set(out["region"]) == {"North", "South", "East"}   # 'north ' -> 'North'
        assert any("duplicate" in line for line in log)

    def test_returns_dataframe(self, messy_df):
        assert isinstance(process_dataset(messy_df), pd.DataFrame)

    def test_options_off_keep_duplicates_and_missing(self, messy_df):
        out = process_dataset(messy_df, drop_duplicates=False, missing="none")
        assert len(out) == 6 and out.isna().sum().sum() > 0

    def test_input_is_not_modified(self, messy_df):
        before = messy_df.copy()
        process_dataset(messy_df)
        assert messy_df.equals(before)

    def test_accepts_file_path(self, sample_dir):
        assert isinstance(process_dataset(f"{sample_dir}/messy_sales.csv"), pd.DataFrame)

    def test_empty_frames_do_not_crash(self):
        assert process_dataset(pd.DataFrame()).shape == (0, 0)
        assert process_dataset(pd.DataFrame(columns=["a", "b"])).shape == (0, 2)

    def test_rejects_wrong_input(self):
        with pytest.raises(DataLoadError):
            process_dataset(12345)

    def test_integer_columns_restored_after_fill(self):
        out = process_dataset(pd.DataFrame({"q": [1, 2, None, 4, 5]}))
        assert out["q"].isna().sum() == 0

    def test_transform_options(self, messy_df):
        out = process_dataset(messy_df, add_date_parts=True, scale="minmax")
        assert "order_date_year" in out.columns and "total_sales_scaled" in out.columns
