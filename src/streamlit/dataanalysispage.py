import streamlit as st
import pandas as pd


# Variables
from streamlit_vars import cleaned_dataset_url, features_dataset_url, num_unique_values
# Functions
from streamlit_vars import displayDataframeInformations, displayCharts, displayRandomImages, displayFeaturesCharts



pd.set_option('display.max_columns', None)


def load_data_visualization_page():
    st.title("Data information & Visualization")
    st.write("This page shows somes graphs and datas after original dataset treatments.")
    st.markdown("*Check the sidebar to choose a category.*")

    # Load Dataset
    cleaned_dataset = pd.read_csv(cleaned_dataset_url)
    features_dataset = pd.read_csv(features_dataset_url)

    # Add sidebar element
    page = st.sidebar.selectbox("Choose a category", ["Dataset Information", "Data Visualization", "Imgs Visualization"])
    
    # Part 1 - Dataset Information
    if page == "Dataset Information":
        st.subheader("Dataset Information")
        displayDataframeInformations(features_dataset, display_classification_repartition=True)

        st.write('\n Label repartition by % :')
        st.write('\n', (features_dataset['label'].value_counts(normalize = True)*100).round(2))



    # Part 2 - Data Visualization
    elif page == "Data Visualization":
        st.subheader("Data Visualization")
        displayCharts(features_dataset, num_unique_values)


    
    # Part 3 - Imgs Visualization
    elif page == "Imgs Visualization":
        st.subheader("Images Visualization")
        n = st.selectbox('Number of images to display', list(range(1, 6)))
        if st.button('Show random images from Dataset'):
            displayRandomImages(features_dataset, n=n)

        displayFeaturesCharts(features_dataset)