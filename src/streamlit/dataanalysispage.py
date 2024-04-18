import streamlit as st
import pandas as pd


# Variables
from streamlit_vars import cleaned_dataset_top10_species_url, cleaned_dataset_with_features_dimensions_url, num_unique_values
# Functions
from streamlit_vars import displayDataframeInformations, displayCharts

pd.set_option('display.max_columns', None)


# Load DF
cleaned_dataset_with_features_dimensions = pd.read_csv(cleaned_dataset_with_features_dimensions_url)
cleaned_dataset_top10_species = pd.read_csv(cleaned_dataset_top10_species_url)


def load_data_visualization_page():
    st.title("Data information & Visualization")
    st.write("This page shows somes graphs and comparison after original dataset treatments.")
    st.markdown("*Check the sidebar to choose a category.*")

    # Add sidebar element
    page = st.sidebar.selectbox("Choose a category", ["Dataset Information", "Data Visualization"])
    
    # Part 1 - Dataset Information
    if page == "Dataset Information":
        choice =  st.radio('Select Dataset', ['Cleaned Dataset', 'Top10 species Dataset'])

        if choice == 'Cleaned Dataset':
            st.write('\n Label repartition by % :')
            st.write('\n', (cleaned_dataset_with_features_dimensions['species'].value_counts(normalize = True)*100).round(2))

            st.subheader("Cleaned Dataset Information")
            displayDataframeInformations(cleaned_dataset_with_features_dimensions, display_classification_repartition=True)



        else:
            st.write('\n Label repartition by % :')
            st.write('\n', (cleaned_dataset_top10_species['species'].value_counts(normalize = True)*100).round(2))

            st.subheader("Top10 species Dataset")
            displayDataframeInformations(cleaned_dataset_top10_species, display_classification_repartition=True)




    # Part 2 - Data Visualization
    elif page == "Data Visualization":

        # Create two columns for chart comparison
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Cleaned Dataset")
            displayCharts(cleaned_dataset_with_features_dimensions, num_unique_values)

        with col2:
            st.subheader("Top10 species Dataset")
            displayCharts(cleaned_dataset_top10_species, num_unique_values)