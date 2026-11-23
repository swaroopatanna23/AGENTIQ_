"""
Tests for Backend/integration  (Member 4: Data Integration)
===========================================================
Checks the connections:  processed data -> Qwen,  validated data -> Chart Engine,
output -> Frontend, and the whole pipeline.

NOTE: Qwen itself (the Ollama model) is NOT started here. Two stand-ins are used:
  * a fake `nlp_fn`               - to test the adapters / chart engine quickly
  * a patched `ollama.chat`       - to run Member 2's REAL understand_query() code
    (prompt building, JSON reading, validation) without a model.
The real model is checked by scripts/check_ollama.py on your own computer.
"""
import json
import os
import re
import socket

import matplotlib.figure
import numpy as np
import ollama
import pandas as pd
import pytest

import Backend.integration.pipeline as pipeline
from Backend.integration.adapters import (
    build_dataset_info, dataset_info_to_text, normalize_aggregation, plan_to_chart_request,
    pretty_name, resolve_column,
)
from Backend.integration.errors import IntegrationError
from Backend.integration.pipeline import (
    PreparedData, analyze_question, check_ollama_status, figure_to_png_bytes, get_plan, prepare_dataset,
)
from conftest import FakeUpload, ROOT


def plan(**kw):
    base = {"intent": "aggregation", "target_column": None, "group_by": None,
            "aggregation": "none", "chart_type": "none", "explanation": "test"}
    base.update(kw)
    return base


@pytest.fixture
def prepared(sample_dir):
    return prepare_dataset(f"{sample_dir}/messy_sales.csv")


def fake_nlp(**kw):
    return lambda question, info: plan(**kw)


# ===================== dataset -> Qwen ======================================
class TestDatasetToQwen:
    def test_dataset_info_lists_every_column_and_type(self, prepared):
        info = build_dataset_info(prepared.data)
        names = {c["name"]: c["type"] for c in info["columns"]}
        assert names == {"order_id": "numeric", "order_date": "datetime", "region": "text",
                         "product": "text", "quantity": "numeric", "total_sales": "numeric"}
        assert info["rows"] == len(prepared.data)

    def test_info_text_has_names_and_examples(self, prepared):
        text = dataset_info_to_text(build_dataset_info(prepared.data))
        assert "total_sales (numeric)" in text and "North" in text and "19 rows" in text

    def test_info_is_json_friendly(self, prepared):
        json.dumps(build_dataset_info(prepared.data))   # numpy / Timestamp values must be converted

    def test_qwen_prompt_contains_the_standardized_column_names(self, prepared, monkeypatch):
        """Runs Member 2's REAL understand_query with a patched ollama.chat and inspects the prompt."""
        seen = {}

        def fake_chat(model, messages, format=None, **kw):
            seen["model"], seen["prompt"], seen["format"] = model, messages[0]["content"], format
            return {"message": {"content": json.dumps(plan(target_column="total_sales", group_by="region",
                                                           aggregation="sum", chart_type="bar"))}}

        monkeypatch.setattr(ollama, "chat", fake_chat)
        result = analyze_question(prepared, "Show total sales by region")
        for column in ("order_id", "order_date", "region", "product", "quantity", "total_sales"):
            assert column in seen["prompt"]
        assert "Show total sales by region" in seen["prompt"]
        assert seen["model"] == "qwen2.5:7b-instruct" and seen["format"] == "json"
        assert result.kind == "chart"

    def test_original_query_function_still_works_without_dataset_info(self, monkeypatch):
        from Backend.nlp.query_understanding import understand_query
        monkeypatch.setattr(ollama, "chat", lambda **kw: {"message": {"content": json.dumps(plan())}})
        assert understand_query("hello")["intent"] == "aggregation"


# ===================== Qwen's names -> real columns =========================
class TestColumnResolution:
    cols = ["order_id", "total_sales", "region", "order_date"]

    def test_exact_case_space_and_typo(self):
        assert resolve_column("total_sales", self.cols) == "total_sales"
        assert resolve_column("Total Sales", self.cols) == "total_sales"
        assert resolve_column("TOTALSALES", self.cols) == "total_sales"
        assert resolve_column("totl_sales", self.cols) == "total_sales"

    def test_null_values(self):
        for value in (None, "", "null", "None"):
            assert resolve_column(value, self.cols) is None

    def test_invented_column_is_a_clear_error(self):
        with pytest.raises(IntegrationError) as err:
            resolve_column("profit_margin", self.cols)
        assert err.value.stage == "adapter" and "profit_margin" in err.value.message and "total_sales" in err.value.message

    def test_aggregation_names(self):
        assert [normalize_aggregation(a) for a in ("average", "minimum", "maximum", "sum", "count", "none", None)] == \
               ["mean", "min", "max", "sum", "count", None, None]
        with pytest.raises(IntegrationError):
            normalize_aggregation("median-ish")

    def test_pretty_name(self):
        assert pretty_name("total_sales") == "Total Sales"


# ===================== plan -> chart request ================================
class TestPlanToChartRequest:
    def test_bar_sum(self, prepared):
        req = plan_to_chart_request(plan(target_column="total_sales", group_by="region", aggregation="sum", chart_type="bar"), prepared.data)
        assert req.kind == "chart" and req.config["chart_type"] == "bar"
        assert req.config["x_column"] == "region" and req.config["y_column"] == "total_sales"
        assert set(req.data.columns) == {"region", "total_sales"} and len(req.data) == 4
        assert req.data["total_sales"].is_monotonic_decreasing

    def test_average_is_mean(self, prepared):
        req = plan_to_chart_request(plan(target_column="quantity", group_by="product", aggregation="average", chart_type="bar"), prepared.data)
        expected = prepared.data.groupby("product")["quantity"].mean()
        for _, row in req.data.iterrows():
            assert row["quantity"] == pytest.approx(expected[row["product"]])

    def test_line_is_sorted_by_group(self, prepared):
        req = plan_to_chart_request(plan(target_column="total_sales", group_by="order_date", aggregation="sum", chart_type="line"), prepared.data)
        assert req.data["order_date"].is_monotonic_increasing

    def test_pie_limits_slices(self):
        df = pd.DataFrame({"c": [f"k{i}" for i in range(30)], "v": range(1, 31)})
        req = plan_to_chart_request(plan(target_column="v", group_by="c", aggregation="sum", chart_type="pie"), df)
        assert len(req.data) == 8 and any("top 8" in n for n in req.notes)

    def test_count_rows_per_group(self, prepared):
        req = plan_to_chart_request(plan(group_by="product", aggregation="count", chart_type="bar"), prepared.data)
        assert req.config["y_column"] == "count" and req.data["count"].sum() == len(prepared.data)

    def test_missing_aggregation_defaults_to_sum_with_note(self, prepared):
        req = plan_to_chart_request(plan(target_column="total_sales", group_by="region", chart_type="bar"), prepared.data)
        assert any("sum" in n for n in req.notes)

    def test_histogram_and_scatter(self, prepared):
        h = plan_to_chart_request(plan(target_column="total_sales", chart_type="histogram"), prepared.data)
        assert h.config == {"chart_type": "histogram", "x_column": "total_sales",
                            "title": "Distribution of Total Sales", "x_label": "Total Sales"}
        s = plan_to_chart_request(plan(target_column="total_sales", group_by="quantity", chart_type="scatter"), prepared.data)
        assert s.config["x_column"] == "quantity" and s.config["y_column"] == "total_sales"

    def test_table_and_single_value(self, prepared):
        t = plan_to_chart_request(plan(target_column="total_sales", group_by="region", aggregation="sum", chart_type="table"), prepared.data)
        assert t.kind == "table" and len(t.data) == 4
        v = plan_to_chart_request(plan(target_column="quantity", aggregation="maximum", chart_type="none"), prepared.data)
        assert v.kind == "value" and v.data.iloc[0]["value"] == prepared.data["quantity"].max()

    def test_chart_without_grouping_falls_back_to_table(self, prepared):
        req = plan_to_chart_request(plan(target_column="quantity", chart_type="bar"), prepared.data)
        assert req.kind == "table" and req.notes

    def test_errors(self, prepared):
        with pytest.raises(IntegrationError, match="scatter"):
            plan_to_chart_request(plan(target_column="quantity", chart_type="scatter"), prepared.data)
        with pytest.raises(IntegrationError, match="histogram"):
            plan_to_chart_request(plan(chart_type="histogram"), prepared.data)
        with pytest.raises(IntegrationError, match="not a numeric"):
            plan_to_chart_request(plan(target_column="product", group_by="region", aggregation="sum", chart_type="bar"), prepared.data)
        with pytest.raises(IntegrationError, match="no such column"):
            plan_to_chart_request(plan(target_column="ghost", chart_type="histogram"), prepared.data)


# ===================== validated data -> chart engine =======================
class TestChartEngineConnection:
    @pytest.mark.parametrize("chart,kw", [
        ("bar", dict(target_column="total_sales", group_by="region", aggregation="sum")),
        ("line", dict(target_column="total_sales", group_by="order_date", aggregation="sum")),
        ("pie", dict(target_column="total_sales", group_by="region", aggregation="sum")),
        ("scatter", dict(target_column="total_sales", group_by="quantity")),
        ("histogram", dict(target_column="total_sales")),
    ])
    def test_real_chart_engine_draws_every_type(self, prepared, chart, kw):
        result = analyze_question(prepared, "q", nlp_fn=fake_nlp(chart_type=chart, **kw))
        assert result.kind == "chart" and isinstance(result.figure, matplotlib.figure.Figure)
        assert figure_to_png_bytes(result.figure)[:8] == b"\x89PNG\r\n\x1a\n"

    def test_area_chart_via_adapter(self, prepared):
        """Member 2's validator does not allow 'area' (Qwen never suggests it), but the engine supports it,
        so it is tested here without Qwen's validator."""
        from chart_engine import generate_chart
        req = plan_to_chart_request(plan(target_column="total_sales", group_by="order_date",
                                         aggregation="sum", chart_type="area"), prepared.data)
        assert isinstance(generate_chart(req.config, req.data), matplotlib.figure.Figure)

    def test_engine_receives_the_validated_dataframe_and_config(self, prepared, monkeypatch):
        captured = {}

        def spy(config, df):
            captured["config"], captured["df"] = config, df
            return matplotlib.figure.Figure()

        monkeypatch.setattr(pipeline, "generate_chart", spy)
        analyze_question(prepared, "q", nlp_fn=fake_nlp(target_column="total_sales", group_by="region",
                                                         aggregation="sum", chart_type="bar"))
        assert isinstance(captured["df"], pd.DataFrame)
        assert captured["config"]["x_column"] in captured["df"].columns
        assert captured["config"]["y_column"] in captured["df"].columns
        assert captured["df"].isna().sum().sum() == 0

    def test_chart_engine_error_becomes_integration_error(self, monkeypatch):
        df = pd.DataFrame({"c": ["a", "b"], "v": [-5, -3]})
        p = prepare_dataset(df)
        with pytest.raises(IntegrationError) as err:
            analyze_question(p, "q", nlp_fn=fake_nlp(target_column="v", group_by="c", aggregation="sum", chart_type="pie"))
        assert err.value.stage == "chart" and "non-negative" in err.value.message


# ===================== prepare_dataset ======================================
class TestPrepareDataset:
    @pytest.mark.parametrize("ext", ["csv", "xlsx", "json"])
    def test_every_format_reaches_the_same_clean_result(self, sample_dir, ext):
        p = prepare_dataset(f"{sample_dir}/messy_sales.{ext}")
        assert p.ready and list(p.data.columns) == ["order_id", "order_date", "region", "product", "quantity", "total_sales"]
        assert len(p.data) == 19 and p.report.duplicate_count == 0 and p.report.missing_total == 0

    def test_accepts_upload_object_and_dataframe(self, sample_dir):
        data = open(f"{sample_dir}/messy_sales.csv", "rb").read()
        assert prepare_dataset(FakeUpload(data, "up.csv")).ready
        assert prepare_dataset(pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})).ready

    def test_before_and_after_reports(self, prepared):
        assert prepared.raw_report.duplicate_count == 1 and prepared.report.duplicate_count == 0
        assert prepared.raw_report.missing_total > 0 and prepared.log

    def test_bad_files_give_data_errors(self, sample_dir):
        for source in (f"{sample_dir}/empty.csv", FakeUpload(b"x", "notes.txt"), "nope.csv"):
            with pytest.raises(IntegrationError) as err:
                prepare_dataset(source)
            assert err.value.stage == "data"

    def test_header_only_file_is_not_ready(self, sample_dir):
        p = prepare_dataset(f"{sample_dir}/header_only.csv")
        assert not p.ready and any("no rows" in e for e in p.report.errors)

    def test_required_columns_and_schema_are_applied_to_clean_data(self, sample_dir):
        p = prepare_dataset(f"{sample_dir}/messy_sales.csv", required_columns=["total_sales", "profit"],
                            schema={"total_sales": "numeric"})
        assert not p.ready and any("'profit'" in e for e in p.report.errors)

    def test_cleaning_options_are_passed_through(self, sample_dir):
        p = prepare_dataset(f"{sample_dir}/messy_sales.csv", drop_duplicates=False, missing="none")
        assert p.report.duplicate_count == 1 and p.report.missing_total > 0 and p.ready

    def test_unready_data_cannot_reach_qwen(self, sample_dir):
        p = prepare_dataset(f"{sample_dir}/header_only.csv")
        called = []
        with pytest.raises(IntegrationError) as err:
            analyze_question(p, "anything", nlp_fn=lambda q, i: called.append(1))
        assert err.value.stage == "validation" and not called

    def test_empty_question(self, prepared):
        with pytest.raises(IntegrationError) as err:
            analyze_question(prepared, "   ", nlp_fn=fake_nlp())
        assert err.value.stage == "input"

    def test_needs_prepared_data_not_raw_frame(self):
        with pytest.raises(IntegrationError):
            analyze_question(pd.DataFrame({"a": [1]}), "q", nlp_fn=fake_nlp())


# ===================== Qwen / Ollama error handling =========================
class TestQwenErrors:
    def test_ollama_not_running(self, prepared, monkeypatch):
        def boom(**kw):
            raise ConnectionError("Failed to connect to Ollama")
        monkeypatch.setattr(ollama, "chat", boom)
        with pytest.raises(IntegrationError) as err:
            analyze_question(prepared, "total sales by region")
        assert err.value.stage == "ollama" and "Start Ollama" in err.value.message

    def test_model_not_installed_and_never_downloaded(self, prepared, monkeypatch):
        def missing(**kw):
            raise ollama.ResponseError("model 'qwen2.5:7b-instruct' not found", 404)

        def forbidden(*a, **kw):
            raise AssertionError("AgentIQ must never download a model!")
        monkeypatch.setattr(ollama, "chat", missing)
        monkeypatch.setattr(ollama, "pull", forbidden)
        with pytest.raises(IntegrationError) as err:
            analyze_question(prepared, "total sales by region")
        assert err.value.stage == "ollama" and "ollama pull qwen2.5:7b-instruct" in err.value.message

    def test_invalid_json_is_retried_then_reported(self, monkeypatch):
        calls = []
        def bad(q, i):
            calls.append(1)
            raise json.JSONDecodeError("bad", "x", 0)
        with pytest.raises(IntegrationError) as err:
            get_plan("q", "info", bad)
        assert err.value.stage == "nlp" and len(calls) == 2

    def test_retry_can_succeed(self):
        answers = iter([json.JSONDecodeError("bad", "x", 0), plan(chart_type="table")])
        def flaky(q, i):
            a = next(answers)
            if isinstance(a, Exception):
                raise a
            return a
        assert get_plan("q", "info", flaky)["chart_type"] == "table"

    def test_plan_with_unsupported_chart_is_rejected(self):
        with pytest.raises(IntegrationError) as err:
            get_plan("q", "info", lambda q, i: plan(chart_type="radar"))
        assert err.value.stage == "nlp" and "chart" in err.value.message.lower()

    def test_plan_with_missing_field_is_rejected(self):
        with pytest.raises(IntegrationError):
            get_plan("q", "info", lambda q, i: {"intent": "x"})

    def test_remote_ollama_host_is_refused(self, prepared, monkeypatch):
        monkeypatch.setenv("OLLAMA_HOST", "https://some-cloud.example.com")
        with pytest.raises(IntegrationError, match="not this computer"):
            analyze_question(prepared, "q")
        assert not check_ollama_status()["ok"]

    def test_local_ollama_host_is_accepted(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:11434")
        pipeline._ensure_local_ollama()


class TestOllamaStatus:
    class Listing:
        def __init__(self, names):
            self.models = [type("M", (), {"model": n})() for n in names]

    def test_ready(self, monkeypatch):
        monkeypatch.setattr(ollama, "list", lambda: self.Listing(["qwen2.5:7b-instruct", "llama3:8b"]))
        status = check_ollama_status()
        assert status["ok"] and "qwen2.5:7b-instruct" in status["models"]

    def test_model_missing(self, monkeypatch):
        monkeypatch.setattr(ollama, "list", lambda: self.Listing(["llama3:8b"]))
        status = check_ollama_status()
        assert not status["ok"] and "ollama pull qwen2.5:7b-instruct" in status["message"]

    def test_not_running(self, monkeypatch):
        def boom():
            raise ConnectionError("nope")
        monkeypatch.setattr(ollama, "list", boom)
        assert "Start Ollama" in check_ollama_status()["message"]

    def test_dict_style_listing(self, monkeypatch):
        monkeypatch.setattr(ollama, "list", lambda: {"models": [{"name": "qwen2.5:7b-instruct"}]})
        assert check_ollama_status()["ok"]


# ===================== the complete pipeline ================================
class TestCompletePipeline:
    def test_file_to_figure_with_real_qwen_code_path(self, sample_dir, monkeypatch):
        """Raw file -> processing -> validation -> Qwen code (ollama.chat patched) -> adapters -> chart engine."""
        answer = plan(target_column="Total Sales", group_by="Region", aggregation="sum", chart_type="bar",
                      explanation="Total sales for each region")
        monkeypatch.setattr(ollama, "chat", lambda **kw: {"message": {"content": json.dumps(answer)}})
        prepared = prepare_dataset(f"{sample_dir}/messy_sales.xlsx")
        result = analyze_question(prepared, "Show total sales by region")
        assert result.kind == "chart" and result.summary == "Total sales for each region"
        north = prepared.data.loc[prepared.data["region"] == "North", "total_sales"].sum()
        assert result.data.set_index("region").loc["North", "total_sales"] == north
        assert result.figure.axes[0].get_title() == "Total Sales by Region"

    def test_pipeline_works_with_internet_blocked(self, sample_dir, monkeypatch):
        """Any attempt to open a non-local network connection fails the test."""
        real_connect = socket.socket.connect

        def guarded(self, address, *a, **kw):
            host = address[0] if isinstance(address, tuple) else str(address)
            if host not in ("127.0.0.1", "localhost", "::1"):
                raise AssertionError(f"network access attempted: {address}")
            return real_connect(self, address, *a, **kw)

        monkeypatch.setattr(socket.socket, "connect", guarded)
        prepared = prepare_dataset(f"{sample_dir}/messy_sales.json")
        result = analyze_question(prepared, "q", nlp_fn=fake_nlp(target_column="quantity", chart_type="histogram"))
        assert result.kind == "chart"

    def test_no_online_services_in_project_code(self):
        """Static check: our own code must not import web clients or contain web addresses."""
        banned_imports = re.compile(r"^\s*(import|from)\s+(requests|urllib|openai|anthropic|httpx|aiohttp|socket)\b", re.M)
        folders = ["Backend", "AgnetIQ", "chart_engine"]
        offenders = []
        for folder in folders:
            for root, _, files in os.walk(os.path.join(ROOT, folder)):
                for name in files:
                    if name.endswith(".py"):
                        text = open(os.path.join(root, name), encoding="utf-8").read()
                        if banned_imports.search(text) or re.search(r"https?://(?!localhost|127\.0\.0\.1)", text):
                            offenders.append(os.path.join(root, name))
        assert offenders == []
