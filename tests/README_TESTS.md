# Tests
Run all: `python -m pytest tests chart_engine/tests -q`

| File | What it covers |
|---|---|
| test_data_processing.py | CSV/Excel/JSON loading, names, types, missing values, duplicates, outliers, transformation |
| test_data_validation.py | empty data, missing, duplicates, types, columns, schema, invalid values, consistency, report |
| test_data_integration.py | data -> Qwen prompt, Qwen plan -> chart engine, error handling, offline checks, full pipeline |
| test_frontend_integration.py | the real Streamlit app run headlessly (upload, clean, ask, errors) |

## NOT tested automatically (needs your computer)
* The real Qwen model answering (needs Ollama + the model). Run `python scripts/check_ollama.py`.
  Tests use a patched `ollama.chat`, which tests Member 2's real prompt/JSON/validation code but not model quality.
* Looking at the page in a real browser (colours, layout).
* Disconnecting your Wi-Fi: tests block non-local sockets and scan the code for web clients instead.
