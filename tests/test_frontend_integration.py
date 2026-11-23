"""
Frontend integration tests (Member 4)
=====================================
Runs Member 1's real Streamlit app headlessly with Streamlit's AppTest and checks that
the data pipeline output reaches the screen. Qwen is replaced by a patched `ollama.chat`
(the real model is not available in the test environment).
LIMITATION: AppTest does not open a browser, so the look of the page is not tested here.
"""
import json

import ollama
import pytest
from streamlit.testing.v1 import AppTest

from conftest import ROOT, SAMPLE_DIR

APP = f"{ROOT}/AgnetIQ/frontend/app.py"


def start():
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception
    return at


def upload(at, filename):
    content = open(f"{SAMPLE_DIR}/{filename}", "rb").read()
    at.file_uploader[0].upload(filename, content).run()
    return at


def click(at, label):
    [b for b in at.button if b.label == label][0].click().run()


def ask(at, question):
    [t for t in at.text_area if "Enter your question" in (t.label or "")][0].set_value(question)
    click(at, "Generate")


def patch_qwen(monkeypatch, **plan_fields):
    plan = {"intent": "aggregation", "target_column": None, "group_by": None,
            "aggregation": "none", "chart_type": "none", "explanation": "ok"}
    plan.update(plan_fields)
    monkeypatch.setattr(ollama, "chat", lambda **kw: {"message": {"content": json.dumps(plan)}})


def texts(elements):
    return " ".join(str(e.value) for e in elements)


def test_app_starts_with_no_data():
    at = start()
    assert len(at.tabs) == 3 and "prepared" not in at.session_state


@pytest.mark.parametrize("filename", ["messy_sales.csv", "messy_sales.xlsx", "messy_sales.json"])
def test_upload_runs_processing_and_validation(filename):
    at = upload(start(), filename)
    assert not at.exception
    prepared = at.session_state["prepared"]
    assert prepared.ready and len(prepared.data) == 20 and "total_sales" in prepared.data.columns
    assert "Data Validation Report" in texts(at.subheader)
    assert any("Dataset uploaded" in s.value for s in at.success)


def test_clean_button_applies_user_choices():
    at = upload(start(), "messy_sales.csv")
    before = at.session_state["prepared"]
    assert before.report.duplicate_count == 1          # safe defaults keep duplicates
    [c for c in at.checkbox if "duplicate" in c.label][0].check()
    [c for c in at.checkbox if "missing" in c.label][0].check()
    click(at, "Clean Dataset")
    after = at.session_state["prepared"]
    assert not at.exception
    assert after.report.duplicate_count == 0 and after.report.missing_total == 0
    assert at.session_state["cleaned_data"].equals(after.data)


def test_new_upload_replaces_old_cleaned_data():
    at = upload(start(), "messy_sales.csv")
    click(at, "Clean Dataset")
    assert "cleaned_data" in at.session_state
    upload(at, "clean_students.csv")
    assert "cleaned_data" not in at.session_state
    assert "student_name" in at.session_state["prepared"].data.columns


def test_bad_file_shows_error_not_crash():
    at = start()
    at.file_uploader[0].upload("empty.csv", b"").run()
    assert not at.exception and any("empty" in e.value.lower() for e in at.error)


def test_question_reaches_chart_on_screen(monkeypatch):
    patch_qwen(monkeypatch, target_column="total_sales", group_by="region", aggregation="sum",
               chart_type="bar", explanation="Total sales for each region")
    at = upload(start(), "messy_sales.csv")
    ask(at, "Show total sales by region")
    assert not at.exception
    result = at.session_state["analysis_result"]
    assert result.kind == "chart" and result.config["x_column"] == "region"
    assert len(at.get("image")) >= 1                      # the matplotlib chart was rendered
    assert "Total sales for each region" in texts(at.info)


def test_single_value_answer_shown(monkeypatch):
    patch_qwen(monkeypatch, target_column="quantity", aggregation="maximum", chart_type="none")
    at = upload(start(), "messy_sales.csv")
    ask(at, "What is the highest quantity?")
    assert not at.exception and at.session_state["analysis_result"].kind == "value"
    assert len(at.metric) >= 1


def test_ollama_down_shows_clear_error_and_fallback_summary(monkeypatch):
    def boom(**kw):
        raise ConnectionError("Failed to connect")
    monkeypatch.setattr(ollama, "chat", boom)
    at = upload(start(), "messy_sales.csv")
    ask(at, "Show total sales by region")
    assert not at.exception
    assert any("Start Ollama" in e.value for e in at.error)
    assert "Dataset Summary" in texts(at.subheader)      # Member 1's summary still shown
    assert "analysis_result" not in at.session_state


def test_model_missing_message_on_screen(monkeypatch):
    def missing(**kw):
        raise ollama.ResponseError("model not found", 404)
    monkeypatch.setattr(ollama, "chat", missing)
    at = upload(start(), "messy_sales.csv")
    ask(at, "Show total sales by region")
    assert any("ollama pull qwen2.5:7b-instruct" in e.value for e in at.error)


def test_quality_question_uses_validation_report():
    at = upload(start(), "messy_sales.csv")
    ask(at, "Are there any data quality problems?")
    assert not at.exception
    assert "Dataset Quality Report" in texts(at.subheader)
    assert any("duplicate" in w.value for w in at.warning)


def test_question_without_dataset_warns():
    at = start()
    ask(at, "anything")
    assert any("upload a dataset" in w.value.lower() for w in at.warning)


def test_status_button_reports_missing_ollama(monkeypatch):
    def boom():
        raise ConnectionError("nope")
    monkeypatch.setattr(ollama, "list", boom)
    at = start()
    click(at, "Check Qwen / Ollama status")
    assert any("Start Ollama" in e.value for e in at.error)


def test_dashboard_receives_cleaned_data():
    at = upload(start(), "messy_sales.csv")
    assert not at.exception
    labels = [s.label for s in at.selectbox]
    assert "Choose X-axis" in labels
    x = [s for s in at.selectbox if s.label == "Choose X-axis"][0]
    assert "total_sales" in x.options          # standardized column names reach the dashboard


def test_manual_csv_text_becomes_dataset():
    at = start()
    [t for t in at.text_area if "Enter your data" in (t.label or "")][0].set_value("A Col,Value\nx,1\ny,2")
    click(at, "Save Manual Data")
    assert list(at.session_state["prepared"].data.columns) == ["a_col", "value"]


def test_plain_notes_are_saved_without_breaking():
    at = start()
    [t for t in at.text_area if "Enter your data" in (t.label or "")][0].set_value("just a note")
    click(at, "Save Manual Data")
    assert not at.exception and at.session_state["manual_text"] == "just a note"
