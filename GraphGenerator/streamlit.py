import streamlit as st
import matplotlib.pyplot as plt
import glob
from enums import Columns, GraphType
from graphGenerator import GraphGenerator, numbers_to_columns


def create_streamlit_app():
    st.title("Data Visualization Tool")

    # Use glob to find all .csv files in the current directory
    csv_files = glob.glob("*.csv")

    # Display the selectbox with the dynamically fetched .csv files
    uploaded_file = st.selectbox(
        "Select dataset",
        options=csv_files,
    )

    # # File uploader
    # uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Initialize GraphGenerator with uploaded file
        graph_type = st.selectbox(
            "Select Graph Type",
            options=[GraphType.SCATTER, GraphType.BAR, GraphType.LINE, GraphType.HISTO],
            format_func=lambda x: x.name,
        )

        graph_gen = GraphGenerator(uploaded_file, graph_type)

        # Create two columns for main and comparison features
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Main Features")
            # Multiple selection for main columns
            main_features = st.multiselect(
                "Select Main Feature(s)",
                options=list(range(1, 19)) + ["ClaimAmount"],
                help="Select one or more features (1-18)",
            )

        with col2:
            st.subheader("Comparison Features")
            if graph_type != GraphType.HISTO:
                # Single or multiple selection for comparison columns
                compare_feature = st.selectbox(
                    "Select Comparison Feature",
                    options=[f"feature{i}" for i in range(1, 19)] + ["ClaimAmount"],
                    help="Select a feature to compare against",
                )

                # Color by feature (for scatter plots)
                if graph_type == GraphType.SCATTER:
                    color_by = st.selectbox(
                        "Color by Feature",
                        options=[None]
                        + [f"feature{i}" for i in range(1, 19)]
                        + ["ClaimAmount"],
                        help="Select a feature to color the points by",
                    )
                else:
                    color_by = None
            else:
                compare_feature = None
                color_by = None

        # Generate button
        if st.button("Generate Graph"):
            try:
                # Convert inputs to appropriate format
                main_cols = numbers_to_columns(main_features)
                compare_col = Columns(compare_feature) if compare_feature else None
                color_by_col = Columns(color_by) if color_by else None

                # Create figure with larger size
                plt.figure(figsize=(12, 8))

                # Generate the graph
                graph_gen.make_graph(main_cols, compare_col, color_by=color_by_col)

                # Display the plot in Streamlit
                st.pyplot(plt)

                # Add download button
                if st.button("Download Plot"):
                    plt.savefig("temp_plot.png", dpi=300, bbox_inches="tight")
                    with open("temp_plot.png", "rb") as file:
                        st.download_button(
                            label="Download Plot",
                            data=file,
                            file_name="plot.png",
                            mime="image/png",
                        )

            except Exception as e:
                st.error(f"Error generating graph: {str(e)}")

        # Add additional options
        with st.expander("Advanced Options"):
            st.write("Plot Customization Options:")
            plot_title = st.text_input("Plot Title", "")
            x_label = st.text_input("X-axis Label", "")
            y_label = st.text_input("Y-axis Label", "")

            if plot_title:
                plt.title(plot_title)
            if x_label:
                plt.xlabel(x_label)
            if y_label:
                plt.ylabel(y_label)


def main():
    create_streamlit_app()


if __name__ == "__main__":
    main()
