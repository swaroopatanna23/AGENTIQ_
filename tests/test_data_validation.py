"""Tests for Backend/data/validation.py  (Member 4: Data Validation)."""
import numpy as np
import pandas as pd
import pytest

from Backend.data.processing import process_dataset
from Backend.data.validation import (
    DataValidationError, STATUS_FAILED, STATUS_PASSED, STATUS_WARNINGS, ValidationReport,
    assert_valid, check_column_names, check_duplicates, check_missing_values, check_schema,
    validate_dataset,
)


def good_df():
    return pd.DataFrame({
        "region": ["North", "South", "East", "West", "North", "South"],
        "sales": [100, 200, 150, 120, 130, 170],
        "date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01", "2024-05-01", "2024-06-01"]),
    })


def has(messages, text):
    return any(text.lower() in m.lower() for m in messages)


class TestStructure:
    def test_empty_dataframe(self):
        r = validate_dataset(pd.DataFrame())
        assert r.status == STATUS_FAILED and has(r.errors, "empty")

    def test_header_only_has_no_rows(self):
        r = validate_dataset(pd.DataFrame(columns=["a", "b"]))
        assert r.status == STATUS_FAILED and has(r.errors, "no rows") and r.total_columns == 2

    def test_all_values_missing(self):
        r = validate_dataset(pd.DataFrame({"a": [None, None], "b": [None, None]}))
        assert r.status == STATUS_FAILED and has(r.errors, "missing")

    def test_not_a_dataframe(self):
        r = validate_dataset([1, 2, 3])
        assert r.status == STATUS_FAILED and has(r.errors, "not a table")

    def test_single_row_warns(self):
        r = validate_dataset(good_df().head(1))
        assert has(r.warnings, "only 1 row") and r.is_valid


class TestHealthyData:
    def test_clean_data_passes(self):
        r = validate_dataset(good_df())
        assert r.status == STATUS_PASSED and r.is_valid and not r.errors and not r.warnings

    def test_counts_and_types_in_report(self):
        r = validate_dataset(good_df())
        assert (r.total_rows, r.total_columns) == (6, 3)
        assert set(r.data_types) == {"region", "sales", "date"}
        assert r.missing_total == 0 and r.duplicate_count == 0


class TestMissingAndDuplicates:
    def test_missing_values_counted(self):
        df = good_df()
        df.loc[0, "sales"] = np.nan
        df.loc[1, "region"] = None
        r = validate_dataset(df)
        assert r.missing_total == 2 and r.missing_by_column["sales"] == 1
        assert r.status == STATUS_WARNINGS and r.is_valid

    def test_column_with_no_data_is_error(self):
        df = good_df()
        df["empty"] = np.nan
        r = validate_dataset(df)
        assert has(r.errors, "no data at all") and not r.is_valid

    def test_mostly_empty_column_warns(self):
        df = good_df()
        df["sparse"] = [1, np.nan, np.nan, np.nan, np.nan, np.nan]
        assert has(validate_dataset(df).warnings, "more than 50% empty")

    def test_duplicates_counted(self):
        df = pd.concat([good_df(), good_df().head(2)], ignore_index=True)
        r = validate_dataset(df)
        assert r.duplicate_count == 2 and has(r.warnings, "duplicate row")

    def test_check_functions_work_alone(self):
        assert check_duplicates(pd.DataFrame({"a": [1, 1]}))[0] == 1
        assert check_missing_values(pd.DataFrame({"a": [1, None]}))[0] == {"a": 1}


class TestColumns:
    def test_duplicate_column_names_are_errors(self):
        df = pd.DataFrame([[1, 2]], columns=["a", "a"])
        r = validate_dataset(df)
        assert has(r.errors, "duplicate column")

    def test_blank_column_name(self):
        assert has(check_column_names(pd.DataFrame([[1]], columns=[" "]))[0], "blank")

    def test_unstandardized_names_warn(self):
        r = validate_dataset(pd.DataFrame({"Total Sales": [1, 2]}))
        assert has(r.warnings, "not standardized")

    def test_unnamed_column_warns(self):
        assert has(check_column_names(pd.DataFrame({"Unnamed: 0": [1]}))[1], "Unnamed")

    def test_required_columns(self):
        r = validate_dataset(good_df(), required_columns=["region", "profit"])
        assert has(r.errors, "'profit' is missing") and not has(r.errors, "'region'")

    def test_required_column_suggests_close_name(self):
        r = validate_dataset(good_df(), required_columns=["sale"])
        assert has(r.errors, "Did you mean 'sales'")


class TestSchema:
    schema = {"region": "text", "sales": "numeric", "date": "datetime"}

    def test_matching_schema_passes(self):
        assert validate_dataset(good_df(), schema=self.schema).is_valid

    def test_wrong_type_is_error(self):
        df = good_df()
        df["sales"] = df["sales"].astype(str)
        r = validate_dataset(df, schema=self.schema)
        assert has(r.errors, "'sales' should be numeric")

    def test_missing_schema_column(self):
        r = validate_dataset(good_df(), schema={"profit": "numeric"})
        assert has(r.errors, "'profit' is missing")

    def test_invalid_schema_type_name(self):
        assert has(check_schema(good_df(), {"sales": "banana"})[0], "not a valid type")

    def test_integer_vs_float(self):
        assert validate_dataset(good_df(), schema={"sales": "integer"}).is_valid
        assert not validate_dataset(good_df(), schema={"sales": "float"}).is_valid

    def test_strict_mode_flags_extra_columns(self):
        r = validate_dataset(good_df(), schema={"sales": "numeric"}, strict_schema=True)
        assert has(r.warnings, "not in the schema")


class TestDataTypes:
    def test_numbers_stored_as_text(self):
        r = validate_dataset(pd.DataFrame({"price": ["10", "20", "30"]}))
        assert has(r.warnings, "numeric but is stored as text")

    def test_dates_stored_as_text(self):
        r = validate_dataset(pd.DataFrame({"when": ["2024-01-01", "2024-02-01"], "v": [1, 2]}))
        assert has(r.warnings, "dates but is stored as text")

    def test_mixed_types(self):
        r = validate_dataset(pd.DataFrame({"m": ["a", 1, "b", 2], "v": [1, 2, 3, 4]}))
        assert has(r.warnings, "mixes different kinds")


class TestInvalidValues:
    def test_infinite_values_are_errors(self):
        r = validate_dataset(pd.DataFrame({"v": [1.0, np.inf, 3.0]}))
        assert has(r.errors, "infinite")

    def test_explicit_non_negative_rule_is_error(self):
        r = validate_dataset(pd.DataFrame({"discount": [1, -5, 3]}), non_negative_columns=["discount"])
        assert has(r.errors, "must not be negative")

    def test_name_based_negative_guess_is_only_warning(self):
        r = validate_dataset(pd.DataFrame({"price": [10, -5, 30]}))
        assert has(r.warnings, "negative") and r.is_valid

    def test_age_above_120(self):
        assert has(validate_dataset(pd.DataFrame({"age": [20, 30, 200]})).warnings, "above 120")

    def test_percentage_out_of_range(self):
        assert has(validate_dataset(pd.DataFrame({"pass_pct": [50, 120, 70]})).warnings, "outside 0-100")

    def test_value_range_rule(self):
        r = validate_dataset(pd.DataFrame({"age": [10, 20, 999]}), value_ranges={"age": (0, 120)})
        assert has(r.errors, "outside the allowed range")

    def test_range_rule_on_missing_or_text_column(self):
        assert has(validate_dataset(good_df(), value_ranges={"nope": (0, 1)}).errors, "missing column")
        assert has(validate_dataset(good_df(), value_ranges={"region": (0, 1)}).errors, "numeric")

    def test_allowed_values_rule(self):
        r = validate_dataset(good_df(), allowed_values={"region": ["North", "South"]})
        assert has(r.errors, "not allowed") and "East" in " ".join(r.errors)

    def test_placeholders_left_in_text(self):
        r = validate_dataset(pd.DataFrame({"c": ["a", "N/A", "b"]}))
        assert has(r.warnings, "placeholder")


class TestConsistency:
    def test_constant_column(self):
        r = validate_dataset(pd.DataFrame({"a": [1, 1, 1], "b": [1, 2, 3]}))
        assert has(r.warnings, "same value in every row")

    def test_inconsistent_case(self):
        r = validate_dataset(pd.DataFrame({"r": ["North", "north", "South"], "v": [1, 2, 3]}))
        assert has(r.warnings, "inconsistent spelling")

    def test_outliers_reported(self):
        r = validate_dataset(pd.DataFrame({"v": [10, 11, 12, 11, 10, 12, 11, 900]}))
        assert r.outliers_by_column == {"v": 1}

    def test_outlier_check_can_be_switched_off(self):
        r = validate_dataset(pd.DataFrame({"v": [10, 11, 12, 11, 10, 12, 11, 900]}), check_outliers=False)
        assert r.outliers_by_column == {}

    def test_start_after_end(self):
        df = pd.DataFrame({
            "start_date": pd.to_datetime(["2024-05-01", "2024-01-01"]),
            "end_date": pd.to_datetime(["2024-04-01", "2024-02-01"]),
        })
        assert has(validate_dataset(df).warnings, "later than")

    def test_no_numeric_columns(self):
        assert has(validate_dataset(pd.DataFrame({"a": ["x", "y"]})).warnings, "no numeric columns")


class TestReport:
    def test_report_contents(self):
        d = validate_dataset(good_df()).to_dict()
        for key in ("total_rows", "total_columns", "missing_total", "duplicate_count",
                    "data_types", "errors", "warnings", "status", "ready_for_analysis"):
            assert key in d
        assert d["ready_for_analysis"] is True

    def test_text_report(self):
        df = good_df()
        df.loc[0, "sales"] = np.nan
        text = validate_dataset(df).to_text()
        assert "DATA VALIDATION REPORT" in text and "Total rows: 6" in text and "PASSED_WITH_WARNINGS" in text

    def test_assert_valid(self):
        assert_valid(validate_dataset(good_df()))
        with pytest.raises(DataValidationError):
            assert_valid(validate_dataset(pd.DataFrame()))

    def test_validation_does_not_modify_data(self):
        df = good_df()
        before = df.copy()
        validate_dataset(df)
        assert df.equals(before)

    def test_never_crashes_on_odd_data(self):
        weird = pd.DataFrame({"a": [[1], [2]], "b": [{"x": 1}, {"x": 2}]})
        assert isinstance(validate_dataset(weird), ValidationReport)


class TestProcessingAndValidationTogether:
    def test_cleaning_fixes_what_validation_reports(self, messy_df):
        before = validate_dataset(messy_df)
        after = validate_dataset(process_dataset(messy_df))
        assert before.duplicate_count > 0 and after.duplicate_count == 0
        assert before.missing_total > 0 and after.missing_total == 0
        assert has(before.warnings, "not standardized") and not has(after.warnings, "not standardized")
        assert has(before.warnings, "stored as text") and not has(after.warnings, "stored as text")
        assert after.is_valid

    def test_sample_dataset_end_to_end(self, sample_dir):
        from Backend.data.processing import load_dataset
        raw = load_dataset(f"{sample_dir}/messy_sales.csv")
        assert validate_dataset(raw).warnings
        final = validate_dataset(process_dataset(raw))
        assert final.is_valid and final.duplicate_count == 0 and final.missing_total == 0
