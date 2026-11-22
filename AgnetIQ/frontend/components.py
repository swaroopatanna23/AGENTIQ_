
import streamlit as st

def show_prompt():
    st.subheader("Ask Your Data Agent")

    prompt = st.text_area(
        "Enter your question:",
        placeholder="Ask something about your dataset..."
    )

    generate = st.button("Generate")

    return prompt, generate


def show_file_upload():
    st.subheader("Upload Your Dataset")

    uploaded_file = st.file_uploader(
        "Choose CSV or Excel file",
        type=["csv", "xlsx"]
    )

    return uploaded_file


def show_manual_entry():
    st.subheader("Manual Data Entry")

    manual_text = st.text_area(
        "Enter your data here:"
    )

    return manual_text
