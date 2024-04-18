import streamlit as st
import pandas as pd

from streamlit_vars import original_dataset_url, cleaned_dataset, cleaned_dataset_with_features_dimensions_url, cleaned_dataset_top10_species_url
from streamlit_vars import displayDataframeInformations, displayDataframeComparison


pd.set_option('display.max_columns', None)

# Load Datasets
original_dataset = pd.read_csv(original_dataset_url)
cleaned_dataset = pd.read_csv(cleaned_dataset)
cleaned_dataset_with_features_dimensions_url = pd.read_csv(cleaned_dataset_with_features_dimensions_url)




def load_dataset_csv_page():
    st.title("Original Dataset")
    st.markdown("This page shows the original dataset used for the project and a comparison with the cleaned datasets")
    
    # Add sidebar element
    page = st.sidebar.selectbox("Choose a category", ["Original Dataset Information", "Cleaning comparison"])
    
    # Part 1 - Original Dataset Information
    if page == "Original Dataset Information":
        st.subheader("Original Dataset Information")

        # Display Dataset Informations
        displayDataframeInformations(original_dataset, display_classification_repartition=False)



    # Part 2 - Cleaning comparison
    elif page == "Cleaning comparison":
        st.subheader("Cleaning comparison")


        # Display comparison board
        displayDataframeComparison(df1 = original_dataset,
                                   df2 = cleaned_dataset,
                                   df3 = cleaned_dataset_with_features_dimensions_url,
                                   df4 = cleaned_dataset_top10_species_url)