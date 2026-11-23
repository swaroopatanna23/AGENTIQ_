"""
Creates the sample datasets in sample_data/ (run: python scripts/generate_sample_datasets.py).
The "messy" files contain problems ON PURPOSE so Data Processing / Validation can be demonstrated:
  - messy column names ("Order ID", "Total Sales ($)")  - numbers stored as text ("$1,200")
  - missing values and placeholders ("N/A", "null")      - duplicate rows
  - dates in different formats                           - extra spaces and different spelling ("north ")
  - one extreme outlier
"""
import json
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "sample_data"
OUT.mkdir(exist_ok=True)

rows = [
    # Order ID, Order Date, Region, Product, Quantity, Total Sales ($)
    (1001, "2024-01-05", "North", "Laptop", 2, "$2,400"),
    (1002, "2024-01-12", "South", "Phone", 5, "$3,000"),
    (1003, "2024-02-03", "East", "Tablet", 3, "$1,500"),
    (1004, "2024-02-14", "West", "Laptop", 1, "$1,200"),
    (1005, "2024-03-01", "north ", "Monitor", 4, "$1,600"),
    (1006, "2024-03-09", "South", "Phone", 2, "$1,200"),
    (1007, "2024-03-21", "East", "Laptop", 3, "$3,600"),
    (1008, "2024-04-02", "West", "Tablet", 6, "$3,000"),
    (1009, "2024-04-15", "North", "Phone", 4, "N/A"),
    (1010, "2024-05-07", "South", "Monitor", None, "$2,000"),
    (1011, "2024-05-19", "East", "Phone", 3, "$1,800"),
    (1012, "2024-06-01", "West", "Laptop", 2, "$2,400"),
    (1012, "2024-06-01", "West", "Laptop", 2, "$2,400"),      # duplicate row
    (1013, "15/06/2024", "North", "Tablet", 5, "$2,500"),      # different date format
    (1014, "2024-06-28", None, "Phone", 3, "$1,800"),          # missing region
    (1015, "2024-07-04", "South", "Laptop", 1, "$1,200"),
    (1016, "2024-07-22", "East", "Monitor", 2, "$800"),
    (1017, "2024-08-10", "West", "Phone", 4, "$2,400"),
    (1018, "2024-08-25", "North", "Laptop", 90, "$108,000"),   # outlier
    (1019, "2024-09-02", "South", "Tablet", 3, "null"),
]
messy = pd.DataFrame(rows, columns=["Order ID", "Order Date", "Region ", "Product", "Quantity", "Total Sales ($)"])
messy.to_csv(OUT / "messy_sales.csv", index=False)
messy.to_excel(OUT / "messy_sales.xlsx", index=False)
(OUT / "messy_sales.json").write_text(json.dumps(json.loads(messy.to_json(orient="records")), indent=2))

clean = pd.DataFrame({
    "student_name": ["Asha", "Bharat", "Chitra", "Deepak", "Esha", "Farhan", "Gita", "Harsh"],
    "department": ["CSE", "ECE", "CSE", "IT", "ECE", "CSE", "IT", "ECE"],
    "study_hours": [3.5, 2.0, 5.0, 4.0, 1.5, 6.0, 4.5, 2.5],
    "exam_score": [72, 55, 88, 79, 48, 93, 82, 60],
})
clean.to_csv(OUT / "clean_students.csv", index=False)

# Bad inputs for error-handling demos
(OUT / "empty.csv").write_text("")
(OUT / "header_only.csv").write_text("a,b,c\n")
print("Sample datasets written to", OUT)
