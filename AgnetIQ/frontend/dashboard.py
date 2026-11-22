
import streamlit as st
import plotly.express as px

def show_dashboard(data):
    st.header("📊 AgentIQ Dashboard")

    if data is None:
        st.info("Upload a dataset first.")
        return

    columns = data.columns.tolist()
    numeric_columns = data.select_dtypes(include="number").columns.tolist()

    if len(numeric_columns) == 0:
        st.warning("No numerical columns found.")
        return

    x = st.selectbox("Choose X-axis", columns)
    y = st.selectbox("Choose Y-axis", numeric_columns)

    chart = st.selectbox(
        "Choose Chart",
        ["Bar", "Line", "Scatter"]
    )

    st.subheader("Customize Your Chart")

    chart_title = st.text_input(
        "Chart Title",
        "My Data Analysis Chart"
    )

    x_label = st.text_input("X-axis Label", x)
    y_label = st.text_input("Y-axis Label", y)

    chart_color = st.color_picker(
        "Choose Chart Colour",
        "#2563EB"
    )

    if st.button("Generate Chart"):

        with st.spinner("Generating chart..."):

            if chart == "Bar":
                fig = px.bar(
                    data, x=x, y=y,
                    color_discrete_sequence=[chart_color]
                )

            elif chart == "Line":
                fig = px.line(
                    data, x=x, y=y,
                    color_discrete_sequence=[chart_color]
                )

            else:
                fig = px.scatter(
                    data, x=x, y=y,
                    color_discrete_sequence=[chart_color]
                )

            fig.update_layout(
                title=chart_title,
                xaxis_title=x_label,
                yaxis_title=y_label
            )

            st.plotly_chart(fig, use_container_width=True)

            st.download_button(
                "Download Chart",
                data=fig.to_html(),
                file_name="AgentIQ.html",
                mime="text/html"
            )
