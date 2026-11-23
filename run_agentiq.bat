@echo off
REM Starts AgentIQ (works offline). Double-click, or run from the project folder.
cd /d "%~dp0"
start "" http://localhost:8501
python -m streamlit run AgnetIQ/frontend/app.py
