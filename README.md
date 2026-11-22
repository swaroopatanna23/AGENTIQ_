# AGENTIQ_

AI-powered data analysis project with modular agents for data processing, analysis, visualization, and integration.

---

## Member 2: Qwen Model and Natural Language Processing

### Project Overview

The NLP module understands natural-language analytics questions and converts them into structured JSON analysis plans.

### Technologies Used
- Python 3.12
- Ollama
- Qwen 2.5 7B Instruct
- JSON

### Implemented Features
- Connected the Qwen model using Ollama and Python.
- Accepts natural-language analytics questions.
- Converts user questions into structured JSON output.
- Validates required JSON fields and allowed chart types and aggregation operations.

### Project Structure
```text
Backend/
└── nlp/
    ├── qwen_test.py
    └── query_understanding.py
```

### Setup Requirements
1. Install Python 3.12.
2. Install Ollama.
3. Download the model:
   `ollama pull qwen2.5:7b-instruct`
4. Install Python dependencies:
   `pip install -r requirements.txt`

### Run the Module
```bash
python Backend/nlp/query_understanding.py
```

### Example Input
`Show average sales by region`

The module generates a structured JSON analysis plan and validates its basic structure.
