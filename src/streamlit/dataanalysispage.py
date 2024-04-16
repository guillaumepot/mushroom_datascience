import streamlit as st
import pandas as pd


# Variables
from streamlit_vars import cleaned_dataset_with_features_dimensions_url, num_unique_values
# Functions
from streamlit_vars import displayDataframeInformations, displayCharts, displayRandomImages, displayFeaturesCharts

pd.set_option('display.max_columns', None)


# Load DF
cleaned_dataset_with_features_dimensions = pd.read_csv(cleaned_dataset_with_features_dimensions_url)



def load_data_visualization_page():
    st.title("Data information & Visualization")
    st.write("This page shows somes graphs and datas after original dataset treatments.")
    st.markdown("*Check the sidebar to choose a category.*")

    # Add sidebar element
    page = st.sidebar.selectbox("Choose a category", ["Dataset Information", "Data Visualization"])
    
    # Part 1 - Dataset Information
    if page == "Dataset Information":
        st.subheader("Dataset Information")
        displayDataframeInformations(cleaned_dataset_with_features_dimensions, display_classification_repartition=True)

        st.write('\n Label repartition by % :')
        st.write('\n', (cleaned_dataset_with_features_dimensions['label'].value_counts(normalize = True)*100).round(2))



    # Part 2 - Data Visualization
    elif page == "Data Visualization":
        st.subheader("Data Visualization")
        displayCharts(cleaned_dataset_with_features_dimensions, num_unique_values)