#!/usr/bin/env bash
# Starts AgentIQ (works offline). Run:  bash run_agentiq.sh
cd "$(dirname "$0")"
echo "Open http://localhost:8501 in your browser"
python3 -m streamlit run AgnetIQ/frontend/app.py
