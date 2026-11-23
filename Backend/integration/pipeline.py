"""
Data Integration Pipeline  (Member 4 - Amrutha)
===============================================
Connects everything, in this order:

  Raw dataset
   -> Member 4  Data Processing   (Backend/data/processing.py)
   -> Member 4  Data Validation   (Backend/data/validation.py)
   -> Validated dataset  (PreparedData)
   -> Member 2  Qwen NLP          (Backend/nlp/query_understanding.py, via Ollama)
   -> adapters  (adapters.py: Qwen's plan -> chart config + chart data)
   -> Member 3  Chart Engine      (chart_engine/)
   -> Member 1  Frontend          (AgnetIQ/frontend/)

Two public functions do the work:
    prepare_dataset(source, ...)           -> PreparedData
    analyze_question(prepared, question)   -> AnalysisResult

All problems are raised as IntegrationError(stage, message) with a message
that is safe to show to the user.   100% offline: Qwen runs in LOCAL Ollama.
"""

import json
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # draw charts in memory (no desktop window) - needed for Streamlit

import pandas as pd

from chart_engine import ChartValidationError, export_chart, generate_chart
from Backend.data.processing import DataProcessingError, load_dataset, process_dataset_with_log
from Backend.data.validation import ValidationReport, validate_dataset
from .adapters import ChartRequest, build_dataset_info, dataset_info_to_text, plan_to_chart_request
from .errors import IntegrationError

LOCAL_HOSTS = ("localhost", "127.0.0.1", "0.0.0.0", "[::1]", "::1")


# =====================================================================
# Stage 1 + 2:  processing and validation
# =====================================================================

@dataclass
class PreparedData:
    """A dataset after processing + validation (this is what Qwen and the charts receive)."""
    data: pd.DataFrame                 # the clean DataFrame
    raw_report: ValidationReport       # problems found in the ORIGINAL data
    report: ValidationReport           # validation of the CLEAN data
    log: List[str] = field(default_factory=list)   # what processing changed

    @property
    def ready(self) -> bool:
        """True when the clean data has no validation errors."""
        return self.report.is_valid


def prepare_dataset(
    source: Any,
    filename: Optional[str] = None,
    required_columns: Optional[Iterable[str]] = None,
    schema: Optional[Dict[str, str]] = None,
    non_negative_columns: Optional[Iterable[str]] = None,
    value_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    allowed_values: Optional[Dict[str, Iterable[Any]]] = None,
    **process_options: Any,
) -> PreparedData:
    """Load (if needed) -> validate the raw data -> clean it -> validate the clean data.

    `source` is a file path, an uploaded file, or a DataFrame.
    `process_options` go straight to process_dataset_with_log
    (drop_duplicates, missing, outliers, ...).
    Raises IntegrationError('data', ...) when the file cannot be read or processed.
    """
    try:
        raw = source if isinstance(source, pd.DataFrame) else load_dataset(source, filename=filename)
        raw_report = validate_dataset(raw)
        clean, log = process_dataset_with_log(raw, **process_options)
    except DataProcessingError as err:   # DataLoadError is a DataProcessingError
        raise IntegrationError("data", str(err))
    report = validate_dataset(
        clean,
        required_columns=required_columns,
        schema=schema,
        non_negative_columns=non_negative_columns,
        value_ranges=value_ranges,
        allowed_values=allowed_values,
    )
    return PreparedData(data=clean, raw_report=raw_report, report=report, log=log)


# =====================================================================
# Local Ollama / Qwen helpers (offline)
# =====================================================================

def _model_name() -> str:
    from Backend.nlp.query_understanding import MODEL_NAME
    return MODEL_NAME


def _ollama_down_message() -> str:
    return ("Cannot reach Ollama on this computer. Start Ollama (open the Ollama app, or run "
            "'ollama serve' in a terminal) and try again.")


def _model_missing_message(model: str) -> str:
    return (f"The Qwen model '{model}' is not installed in Ollama. AgentIQ never downloads models "
            f"by itself. While you still have internet, run:  ollama pull {model}")


def _ensure_local_ollama() -> None:
    """AgentIQ must stay offline: refuse an OLLAMA_HOST that points to another machine."""
    host = os.environ.get("OLLAMA_HOST", "").strip().lower()
    if host and not any(local in host for local in LOCAL_HOSTS):
        raise IntegrationError("ollama", f"OLLAMA_HOST is set to '{host}', which is not this computer. "
                                         "AgentIQ only uses a local Ollama so it can work offline.")


def check_ollama_status() -> Dict[str, Any]:
    """Is Ollama running and is the Qwen model installed? Never downloads anything.
    Returns {'ok': bool, 'message': str, 'models': [...]}."""
    try:
        _ensure_local_ollama()
        import ollama
        model = _model_name()
    except IntegrationError as err:
        return {"ok": False, "message": err.message, "models": []}
    except ImportError:
        return {"ok": False, "models": [],
                "message": "The Python package 'ollama' is not installed. Run: pip install -r requirements.txt"}
    try:
        listing = ollama.list()
    except Exception:
        return {"ok": False, "message": _ollama_down_message(), "models": []}
    items = listing["models"] if isinstance(listing, dict) else getattr(listing, "models", [])
    names = [(m.get("model") or m.get("name")) if isinstance(m, dict) else getattr(m, "model", None) for m in items]
    names = [n for n in names if n]
    if not any(n == model or n.split(":")[0] == model for n in names):
        return {"ok": False, "message": _model_missing_message(model), "models": names}
    return {"ok": True, "message": f"Ollama is running and '{model}' is installed.", "models": names}


def _ask_qwen(question: str, dataset_text: str) -> Dict[str, Any]:
    """Call Member 2's understand_query with the dataset description and
    turn every failure into a clear IntegrationError."""
    _ensure_local_ollama()
    try:
        import ollama
        from Backend.nlp.query_understanding import understand_query
    except ImportError:
        raise IntegrationError("ollama", "The Python package 'ollama' is not installed. Run: pip install -r requirements.txt")
    try:
        return understand_query(question, dataset_info=dataset_text)
    except json.JSONDecodeError:
        raise
    except ollama.ResponseError as err:
        if getattr(err, "status_code", None) == 404:
            raise IntegrationError("ollama", _model_missing_message(_model_name()))
        raise IntegrationError("ollama", f"Ollama reported an error: {err}")
    except ConnectionError:
        raise IntegrationError("ollama", _ollama_down_message())
    except ValueError:
        raise
    except Exception as err:
        if "connect" in str(err).lower() or "refused" in str(err).lower():
            raise IntegrationError("ollama", _ollama_down_message())
        raise IntegrationError("nlp", f"Qwen could not process the question: {err}")


def get_plan(question: str, dataset_text: str, nlp_fn: Optional[Callable] = None, max_attempts: int = 2) -> Dict[str, Any]:
    """Ask Qwen for an analysis plan and check it with Member 2's validate_result.
    Small local models sometimes return bad JSON, so we try again once."""
    try:
        from Backend.nlp.query_understanding import validate_result
    except ImportError:
        raise IntegrationError("ollama", "The Python package 'ollama' is not installed. Run: pip install -r requirements.txt")
    nlp_fn = nlp_fn or _ask_qwen
    last_problem = ""
    for _ in range(max_attempts):
        try:
            plan = nlp_fn(question, dataset_text)
            validate_result(plan)
            return plan
        except json.JSONDecodeError:
            last_problem = "Qwen did not return valid JSON."
        except (ValueError, TypeError) as err:
            last_problem = f"Qwen returned an unusable plan: {err}"
    raise IntegrationError("nlp", f"{last_problem} Please rephrase the question and try again.")


# =====================================================================
# Stage 3-5:  Qwen -> adapters -> chart engine -> frontend output
# =====================================================================

@dataclass
class AnalysisResult:
    """Everything the frontend needs to show the answer to one question."""
    question: str
    plan: Dict[str, Any]                    # Qwen's validated JSON plan
    kind: str                               # 'chart', 'table' or 'value'
    data: pd.DataFrame                      # table to show / the data that was charted
    config: Optional[Dict[str, Any]] = None # chart-engine configuration (kind == 'chart')
    figure: Any = None                      # matplotlib Figure from Member 3's engine
    summary: str = ""                       # Qwen's explanation
    notes: List[str] = field(default_factory=list)


def analyze_question(prepared: PreparedData, question: str, nlp_fn: Optional[Callable] = None) -> AnalysisResult:
    """Answer one question about a VALIDATED dataset.

    nlp_fn(question, dataset_text) -> plan   can replace the real Qwen call (used by tests).
    """
    if not isinstance(prepared, PreparedData):
        raise IntegrationError("data", "analyze_question needs the result of prepare_dataset().")
    if not prepared.ready:
        raise IntegrationError("validation", "The dataset is not ready for analysis: " + " | ".join(prepared.report.errors))
    if not question or not question.strip():
        raise IntegrationError("input", "Please enter a question.")

    dataset_text = dataset_info_to_text(build_dataset_info(prepared.data))
    plan = get_plan(question.strip(), dataset_text, nlp_fn)
    request: ChartRequest = plan_to_chart_request(plan, prepared.data)

    figure = None
    if request.kind == "chart":
        try:
            figure = generate_chart(request.config, request.data)
        except ChartValidationError as err:
            raise IntegrationError("chart", f"The chart could not be drawn: {err}")
        except Exception as err:
            raise IntegrationError("chart", f"The chart engine failed: {err}")
    return AnalysisResult(question.strip(), plan, request.kind, request.data, request.config,
                          figure, request.summary, request.notes)


def figure_to_png_bytes(figure: Any, dpi: int = 150) -> bytes:
    """PNG bytes of a chart (for the frontend's download button) using Member 3's exporter."""
    with tempfile.TemporaryDirectory() as folder:
        path = export_chart(figure, os.path.join(folder, "chart.png"), format="png", dpi=dpi)
        with open(path, "rb") as handle:
            return handle.read()
