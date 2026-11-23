"""
Run this ON YOUR OWN COMPUTER to check the real Qwen model (it cannot be tested without Ollama):

    python scripts/check_ollama.py

Step 1: is Ollama running and is the model installed?  (never downloads anything)
Step 2: sends a real question about sample_data/messy_sales.csv through the FULL pipeline
        (processing -> validation -> Qwen -> adapters -> chart engine) and saves the chart.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Backend.integration.errors import IntegrationError
from Backend.integration.pipeline import analyze_question, check_ollama_status, figure_to_png_bytes, prepare_dataset

status = check_ollama_status()
print("Ollama check:", "OK" if status["ok"] else "PROBLEM", "-", status["message"])
if not status["ok"]:
    sys.exit(1)

question = "Show total sales by region as a bar chart"
try:
    prepared = prepare_dataset(os.path.join("sample_data", "messy_sales.csv"))
    result = analyze_question(prepared, question)
except IntegrationError as err:
    print("Pipeline problem:", err)
    sys.exit(1)

print("Question :", question)
print("Qwen plan:", result.plan)
print("Result   :", result.kind)
print(result.data)
if result.figure is not None:
    with open("check_ollama_chart.png", "wb") as handle:
        handle.write(figure_to_png_bytes(result.figure))
    print("Chart saved as check_ollama_chart.png")
