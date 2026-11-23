import ollama

response = ollama.chat(
    model="qwen2.5:7b-instruct",
    messages=[
        {
            "role": "user",
            "content": "Explain data analytics in simple words."
        }
    ]
)

print(response["message"]["content"])