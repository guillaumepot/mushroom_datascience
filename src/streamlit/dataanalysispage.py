import streamlit as st
import pandas as pd


# Variables
from streamlit_vars import cleaned_dataset_url, num_unique_values
# Functions
from streamlit_vars import displayDataframeInformations, displayCharts



pd.set_option('display.max_columns', None)




def load_data_visualization_page():
    st.title("Data information & Visualization")
    st.write("This page shows somes graphs after original dataset treatments.")
    st.markdown("*Check the sidebar to choose a category.*")

    # Load Dataset
    cleaned_dataset = pd.read_csv(cleaned_dataset_url)

    # Add sidebar element
    page = st.sidebar.selectbox("Choose a category", ["Dataset Information", "Data Visualization"])
    
    # Part 1 - Dataset Information
    if page == "Dataset Information":
        st.subheader("Dataset Information")
        displayDataframeInformations(cleaned_dataset, display_classification_repartition=True)

        st.write('\n Label repartition by % :')
        st.write('\n', (cleaned_dataset['label'].value_counts(normalize = True)*100).round(2))



    # Part 2 - Data Visualization
    elif page == "Data Visualization":
        st.subheader("Data Visualization")
        displayCharts(cleaned_dataset, num_unique_values)