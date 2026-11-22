
import streamlit as st

def apply_styles():

    st.markdown("""
        <style>

        .stApp {
            background-color: #F4F7FC;
        }

        h1 {
            color: #173B70;
            font-weight: bold;
        }

        h2, h3 {
            color: #24558A;
        }

        .stButton button {
            background-color: #2563EB;
            color: white;
            border-radius: 8px;
            border: none;
            padding: 8px 20px;
        }

        .stButton button:hover {
            background-color: #1746A2;
            color: white;
        }

        </style>
    """, unsafe_allow_html=True)
