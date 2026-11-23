
import ollama
import json

# [Member 4 integration] The model name is now one constant so the integration
# layer can check that exactly this model is installed in Ollama (offline use).
MODEL_NAME = "qwen2.5:7b-instruct"


# 1. Validate the JSON output
def validate_result(result):

    required_fields = [
        "intent",
        "target_column",
        "group_by",
        "aggregation",
        "chart_type",
        "explanation"
    ]

    for field in required_fields:
        if field not in result:
            raise ValueError(f"Missing field: {field}")

    allowed_charts = [
        "bar", "line", "pie", "scatter",
        "histogram", "table", "none"
    ]

    if result["chart_type"] not in allowed_charts:
        raise ValueError("Invalid chart type")

    allowed_operations = [
        "sum", "average", "count",
        "minimum", "maximum", "min", "max", "none"
    ]

    if result["aggregation"] not in allowed_operations:
        raise ValueError("Invalid aggregation")

    return True


# 2. Understand the user's analytics question
# [Member 4 integration] New OPTIONAL argument `dataset_info`: a short text that
# lists the real column names (and types) of the validated dataset. Without it
# Qwen can only guess column names. Old calls understand_query("...") still work.
def understand_query(user_query, dataset_info=None):

    dataset_section = ""
    if dataset_info:
        dataset_section = f"""
    Dataset information (use ONLY these exact column names):
    {dataset_info}
    """

    prompt = f"""
   - Use "maximum" for highest or largest values.
- Use "minimum" for lowest or smallest values.
- Use only these aggregation values:
  sum, average, count, minimum, maximum, none.
    {dataset_section}
    User question: {user_query}

    JSON structure:
    {{
        "intent": "",
        "target_column": null,
        "group_by": null,
        "aggregation": "none",
        "chart_type": "none",
        "explanation": ""
    }}

    Rules:
    - Intent must be a category such as aggregation,
      comparison, trend, or distribution.
    - Identify the target data column.
    - Identify the grouping column if applicable.
    - Identify the aggregation operation.
    - Suggest an appropriate chart type.
    - Do not invent column names.
    - Use null when a column is not specified.
    - Return JSON only.
    """

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        format="json"
    )

    result = json.loads(response["message"]["content"])

    return result


# 3. Main program
if __name__ == "__main__":

    question = input("Enter your analytics question: ")

    try:
        result = understand_query(question)

        if validate_result(result):
            print("\nValidation Successful!")
            print("\nAgentIQ NLP Output:\n")
            print(json.dumps(result, indent=4))

    except (ValueError, json.JSONDecodeError) as error:
        print("\nValidation Failed:", error)

    except Exception as error:
        print("\nError:", error)
