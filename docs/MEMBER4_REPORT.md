# Member 4 Report - Amrutha (Data Processing, Validation, Integration)

## 1. Project status BEFORE my work (checked by reading AND running the code)

| Area | Owner | State found |
|---|---|---|
| Streamlit frontend (3 tabs) | M1 | Works. Loads CSV/XLSX with plain `pandas`, has its own small cleaning, a Plotly dashboard. **Not connected to Qwen or the chart engine** ("Qwen integration will be added later"). |
| Qwen NLP | M2 | `understand_query()` works as a standalone script. Receives **only the question**, never the column names. Allows chart types bar/line/pie/scatter/histogram/table/none (no "area"). |
| Chart engine | M3 | Complete, 15/15 tests pass. Offline. Connected to nothing. |
| Data processing | M4 | Only a few lines inside `app.py` (drop duplicates, fill median/mode). No JSON, no types, no names. |
| Data validation | M4 | None (only a rule-based report inside `app.py`). |
| Integration | M4 | None. |

Gaps I found: no JSON loading; stale `cleaned_data` after uploading a new file; `.describe()` crashes the table display once dates are real dates; Qwen can guess column names; Qwen's words (`average`, `maximum`) are not pandas names; chart engine needs `chart_type/x_column/y_column`; Qwen may answer `table`/`none` which the engine cannot draw.

## 2. What I built

**New files (mine)**
* `Backend/data/common.py` - shared helpers (column kind, outlier rule, number/date detection) used by both modules (no duplicate code)
* `Backend/data/processing.py` - Data Processing
* `Backend/data/validation.py` - Data Validation
* `Backend/integration/errors.py`, `adapters.py`, `pipeline.py` - Data Integration
* `AgnetIQ/frontend/data_views.py` - small display helpers (no logic)
* `tests/` (4 test files + conftest), `sample_data/`, `scripts/`, `docs/`, `.streamlit/config.toml`, `requirements.txt`, `offline_wheels/`, `run_agentiq.*`, `README.md`
* empty `__init__.py` in `Backend/` and `Backend/nlp/` (so Python can import them)

**Other members' files I had to modify (smallest possible change)**
| File | Change | Why it was unavoidable |
|---|---|---|
| `Backend/nlp/query_understanding.py` (M2) | `MODEL_NAME` constant; optional `dataset_info` argument added to the prompt | Qwen must see the real column names. Old calls still work. |
| `AgnetIQ/frontend/app.py` (M1) | import path line; upload -> `load_dataset`; Clean button -> `prepare_dataset`; validation report display; AI tab -> `analyze_question`; dashboard gets cleaned data; stale-state fix; `describe(include="number")` | The frontend was not connected to anything. Layout, tabs, widgets and download buttons are kept. |
| `AgnetIQ/frontend/components.py` (M1) | file types now also `xls`, `json` | JSON/xls support |
Chart engine (M3): **not modified.** Dashboard (M1): **not modified** (it now simply receives the cleaned data).

## 3. How Data Processing works (`Backend/data/processing.py`)
`process_dataset(df_or_file)` runs small steps in a deliberate order:
1. drop "Unnamed: 0" index column -> 2. `standardize_column_names` (`Total Sales ($)` -> `total_sales`)
3. `clean_text_values` (trim spaces, `N/A`/`null`/`-` -> missing) -> 4. `unify_text_case` (`north ` -> `North`)
5. drop fully empty rows/columns -> 6. `convert_data_types` (`"$1,200"` -> 1200, text dates -> real dates)
7. `remove_duplicates` -> 8. `handle_missing_values` (median / mode / "Unknown"; dates never invented)
9. `handle_outliers` (off by default: cap/remove) -> 10. optional `extract_date_features`, `scale_numeric_columns`.
Types are fixed BEFORE duplicates and missing values, otherwise "5" and 5 would look different and a median could not be calculated.
`load_dataset` handles CSV (auto separator + encodings), Excel (.xlsx/.xls), JSON (records, columns, `{"data":[...]}`, JSON-lines) from a path or an uploaded file.
`aggregate_data` / `aggregate_column` do the group-and-summarise step that Qwen's plan asks for.
Errors: `DataLoadError` / `DataProcessingError` with plain messages.

## 4. How Data Validation works (`Backend/data/validation.py`)
`validate_dataset(df, required_columns, schema, non_negative_columns, value_ranges, allowed_values)` returns a `ValidationReport`
(rows, columns, missing total + per column, duplicates, data types, outliers, errors, warnings, status).
* **Error** = real problem -> status `FAILED`, not ready (empty data, duplicate column names, missing required column, wrong schema type, infinite values, rule violations, a column with no data).
* **Warning** = look at it -> `PASSED_WITH_WARNINGS` (missing values, duplicates, numbers/dates stored as text, mixed types, placeholders, constant column, inconsistent spelling, outliers, non-standard names).
It never changes data and never crashes. It is separate from Member 3's `chart_engine/validator.py` (that checks a *chart request*, mine checks the *data*).

## 5. How Data Integration works (`Backend/integration/`)
* `prepare_dataset(file)` -> load -> validate raw -> process -> validate clean -> `PreparedData(data, raw_report, report, log)`.
* `analyze_question(prepared, question)`:
  1. refuses data that failed validation (so bad data never reaches Qwen or charts)
  2. `build_dataset_info` -> text listing real column names, types, examples -> **sent to Qwen** with the question
  3. Qwen's JSON plan is checked with Member 2's own `validate_result` (one retry if the JSON is bad)
  4. `plan_to_chart_request` (adapter): matches Qwen's column names to real columns (case/typo tolerant), converts `average->mean`, groups the data, sorts, limits to readable size, builds the chart-engine config
  5. **Chart engine** `generate_chart(config, df)` draws it; `table`/`none` answers become a table or one number
  6. `AnalysisResult` goes to the **frontend**, which shows chart/table/number, plan, PNG download.
* Every failure is an `IntegrationError(stage, message)` - e.g. "Start Ollama", "run: ollama pull qwen2.5:7b-instruct", "the AI mentioned a column that does not exist".
* Offline guards: only local Ollama (a remote `OLLAMA_HOST` is refused); never calls `ollama pull`; Matplotlib uses the in-memory `Agg` backend.

## 6. Test results (what was actually run)
* 184 tests of mine + Member 3's 15 = **199 passed**, on pandas 2.3.3, 3.0.6 and 3.1.1 (Python 3.12, Linux).
* Real Streamlit server started and answered (health `ok`).
* **NOT executed:** the real Qwen model (no Ollama/model in my environment) - run `python scripts/check_ollama.py`; a real browser view; running on Windows (wheels were checked to resolve with `--no-index`, but not installed on Windows).

## 7. Open items for the other members (I did not change these)
* M2: `validate_result` does not allow `area`, so Qwen can never request an area chart although M3 supports it.
* M2: the prompt text starts with the rule bullets before "User question" - works, but could be tidied.
* M1: `use_container_width` is deprecated in new Streamlit (still works, prints a warning).
* M1: Dashboard still uses Plotly, not Member 3's engine (left as designed).

## 8. Viva cheat sheet
**One sentence:** "I built the data layer: it loads and cleans the dataset, validates that it is trustworthy, and connects it to Qwen, the chart engine and the frontend."
**Demo (3 min):** upload `sample_data/messy_sales.csv` -> show the *original problems* expander (duplicate, missing, text numbers, 'north ') -> click Clean -> validation PASSED -> Tab 2 ask "total sales by region" -> chart. Then stop Ollama and ask again to show the clear error.
**Likely questions**
* *Why clean before validate?* I validate both: raw (to report problems) and clean (to decide readiness).
* *Error vs warning?* Error blocks analysis; warning does not.
* *Why does Qwen get column names?* Otherwise it invents them. The adapter also corrects small spelling differences.
* *How is it offline?* Local Ollama only, no downloads, no web libraries (a test scans the code), no CDN.
* *Why median?* Not distorted by outliers (e.g. the 108,000 sale).
* *What if Qwen is wrong?* Its plan is validated, columns are checked, the chart engine validates again; failures show a message.
* *What did you not change?* Chart engine untouched; Qwen/frontend only minimal, listed above.
