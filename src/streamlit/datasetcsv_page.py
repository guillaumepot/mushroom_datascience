import streamlit as st
import pandas as pd

from streamlit_vars import original_dataset_url, cleaned_dataset, cleaned_dataset_with_features_dimensions_url, cleaned_dataset_top10_species_url
from streamlit_vars import displayDataframeInformations, displayDataframeComparison


pd.set_option('display.max_columns', None)

# Load Datasets
original_dataset = pd.read_csv(original_dataset_url)
cleaned_dataset = pd.read_csv(cleaned_dataset)
cleaned_dataset_with_features_dimensions = pd.read_csv(cleaned_dataset_with_features_dimensions_url)
cleaned_dataset_top10_species = pd.read_csv(cleaned_dataset_top10_species_url)



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
        st.markdown("The table below shows a comparison between the different steps of the dataset cleaning process. Here are some details about the datasets:")
        st.markdown("* The original dataset is the dataset used for the project without any processing.")
        st.markdown("* The cleaned dataset is the original dataset after some cleaning processes like removing duplicates, NaN values, etc.")
        st.markdown("* The cleaned dataset with feature dimensions is the cleaned dataset with the addition of features and dimensions. Images with width and/or height < 200px have been removed.")
        st.markdown("* The cleaned dataset top 10 species is the cleaned dataset with the top 10 species. Other species have been added to help the model classify other species. A manual sort has been done to remove 'bad images'. The choice of species to keep is explained on the classification choice page. Also, you can visualize some sorted images on the 'Image Visualization' page.")

        # Display comparison board
        displayDataframeComparison(df1 = original_dataset,
                                   df2 = cleaned_dataset,
                                   df3 = cleaned_dataset_with_features_dimensions,
                                   df4 = cleaned_dataset_top10_species)