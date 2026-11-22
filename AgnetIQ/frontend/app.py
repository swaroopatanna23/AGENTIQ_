
import streamlit as st
import pandas as pd

from components import show_prompt, show_file_upload, show_manual_entry
from styles import apply_styles
from dashboard import show_dashboard

st.set_page_config(
    page_title="AgentIQ",
    page_icon="🤖",
    layout="wide"
)

apply_styles()

st.title("🤖 AgentIQ")
st.subheader("Your AI-Powered Data Analytics Assistant")
st.write("Upload your dataset, ask questions and explore data through charts.")

st.divider()

# Create three tabs
tab1, tab2, tab3 = st.tabs([
    "📁 Data Upload",
    "🤖 AI Assistant",
    "📊 Dashboard"
])

# TAB 1: DATA UPLOAD
with tab1:
    st.header("Upload Your Dataset")

    uploaded_file = show_file_upload()

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)

            st.session_state["data"] = data
            st.success("Dataset uploaded successfully!")

        except Exception as e:
            st.error(f"File error: {e}")

    data = st.session_state.get("data")

    if data is not None:
        st.subheader("Dataset Preview")
        st.dataframe(data, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Total Rows", data.shape[0])

        with col2:
            st.metric("Total Columns", data.shape[1])

        
        st.divider()
        st.subheader("🧹 Data Cleaning")

        remove_duplicates = st.checkbox(
            "Remove duplicate rows"
        )

        fill_missing = st.checkbox(
            "Fill missing values"
        )

        if st.button("Clean Dataset"):

            cleaned_data = data.copy()

            if remove_duplicates:
                cleaned_data = cleaned_data.drop_duplicates()

            if fill_missing:
                for col in cleaned_data.columns:

                    if pd.api.types.is_numeric_dtype(cleaned_data[col]):
                        median_value = cleaned_data[col].median()

                        if pd.notna(median_value):
                            cleaned_data[col] = cleaned_data[col].fillna(
                                median_value
                            )

                    else:
                        modes = cleaned_data[col].mode()

                        fill_value = (
                            modes.iloc[0] if not modes.empty
                            else "Unknown"
                        )

                        cleaned_data[col] = cleaned_data[col].fillna(
                            fill_value
                        )

            st.session_state["cleaned_data"] = cleaned_data

            st.success("Dataset cleaned successfully!")

        if "cleaned_data" in st.session_state:

            cleaned_data = st.session_state["cleaned_data"]

            st.subheader("Cleaned Dataset Preview")

            st.dataframe(cleaned_data, use_container_width=True)

            st.write("Original Rows:", data.shape[0])
            st.write("Cleaned Rows:", cleaned_data.shape[0])

            st.download_button(
                "Download Cleaned Dataset",
                data=cleaned_data.to_csv(index=False),
                file_name="AgentIQ_Cleaned_Dataset.csv",
                mime="text/csv"
            )


    st.divider()

    manual_data = show_manual_entry()

    if st.button("Save Manual Data"):
        st.session_state["manual_text"] = manual_data
        st.success("Manual input saved!")

# TAB 2: AI ASSISTANT

# TAB 2: AI ASSISTANT
with tab2:
    st.header("🤖 Ask Your Data Agent")

    prompt, generate = show_prompt()
    data = st.session_state.get("data")

    if generate:
        if not prompt.strip():
            st.warning("Please enter a question.")

        elif data is None:
            st.warning("Please upload a dataset first.")

        elif any(word in prompt.lower() for word in
                 ["irregular", "issue", "error", "problem",
                  "abnormal", "anomal", "quality"]):

            st.subheader("🔍 Dataset Quality Report")

            report = []
            report.append("AGENTIQ - DATASET QUALITY REPORT")
            report.append("=" * 40)
            report.append(f"Total Rows: {data.shape[0]}")
            report.append(f"Total Columns: {data.shape[1]}")

            # Missing values
            missing = data.isnull().sum()
            missing = missing[missing > 0]

            report.append("\nMISSING VALUES")
            if len(missing) > 0:
                st.write("Missing Values:")
                st.write(missing)
                for col, count in missing.items():
                    report.append(f"{col}: {count} missing values")
            else:
                st.success("No missing values found.")
                report.append("No missing values found.")

            # Duplicate rows
            duplicates = data.duplicated().sum()

            report.append("\nDUPLICATE ROWS")
            st.write("Duplicate Rows:", duplicates)
            report.append(f"Duplicate rows: {duplicates}")

            # Numerical outliers
            outliers = {}
            numeric_columns = data.select_dtypes(
                include="number"
            ).columns

            for col in numeric_columns:
                q1 = data[col].quantile(0.25)
                q3 = data[col].quantile(0.75)
                iqr = q3 - q1

                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr

                count = (
                    (data[col] < lower) |
                    (data[col] > upper)
                ).sum()

                if count > 0:
                    outliers[col] = int(count)

            report.append("\nPOSSIBLE NUMERICAL OUTLIERS")

            if outliers:
                st.write("Possible Outliers:")
                st.write(outliers)
                for col, count in outliers.items():
                    report.append(f"{col}: {count} possible outliers")
            else:
                st.success("No possible numerical outliers found.")
                report.append("No possible numerical outliers found.")

            st.divider()

            # Download report
            st.download_button(
                "📥 Download Quality Report",
                data="\n".join(report),
                file_name="AgentIQ_Quality_Report.txt",
                mime="text/plain"
            )

            st.info(
                "These are basic rule-based checks. "
                "Qwen AI integration will be added later."
            )

        else:
            st.subheader("Dataset Summary")

            st.write("Total Rows:", data.shape[0])
            st.write("Total Columns:", data.shape[1])
            st.write("Column Names:", list(data.columns))
            st.write("Missing Values:", data.isnull().sum())
            st.write("Numerical Summary:")
            st.write(data.describe())

# TAB 3: DASHBOARD
with tab3:
    data = st.session_state.get("data")
    show_dashboard(data)

st.divider()
st.caption("AgentIQ | AI-Powered Data Analytics")
