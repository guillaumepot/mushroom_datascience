"""
This page is DEPRECATED
"""


import streamlit as st
import pandas as pd

# Variables
from streamlit_vars import cleaned_dataset_top10_species_url, cleaned_dataset_with_features_dimensions_url
# Functions
from streamlit_vars import displayDataframeInformations, displayRandomImages


pd.set_option('display.max_columns', None)


def load_classification_choice():
    st.title("Classification choice")
    st.write("This page show different choices about datas.")
    st.markdown("*Check the sidebar to choose a category.*")

    # Load Datasets
    df_cleaned_dataset_with_features = pd.read_csv(cleaned_dataset_with_features_dimensions_url)
    df_cleaned_dataset_with_features_top_10_species = pd.read_csv(cleaned_dataset_top10_species_url)

    # Comparison table
    st.header("Comparison table")
    
    data = {
        "Cleaned Dataset": [
            df_cleaned_dataset_with_features.shape,
            df_cleaned_dataset_with_features["species"].nunique()
        ],
        "Top 10 Species": [
            df_cleaned_dataset_with_features_top_10_species.shape,
            df_cleaned_dataset_with_features_top_10_species["species"].nunique()
        ]
    }

    df_comparison = pd.DataFrame(data, index=["Shape", "Unique Species"])
    st.table(df_comparison)

    # Display Unique Species
    st.subheader("Unique Species for classification")
    unique_species = pd.DataFrame(df_cleaned_dataset_with_features_top_10_species['species'].unique(), columns=['Species'])
    st.dataframe(unique_species)

    # Display random images from df
    st.subheader("Species Visualization")
    n = st.selectbox('Number of images to display', list(range(1, 4)))
    s = st.selectbox('Specy', unique_species['Species'])
    df_img_to_show = df_cleaned_dataset_with_features_top_10_species[df_cleaned_dataset_with_features_top_10_species['species'] == s]
    if st.button('Show random images from Dataset'):
        displayRandomImages(df_img_to_show, n=n)


    # Display DF informations
    displayDataframeInformations(df_cleaned_dataset_with_features_top_10_species, display_classification_repartition=True)