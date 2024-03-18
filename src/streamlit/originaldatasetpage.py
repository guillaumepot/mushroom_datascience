import streamlit as st
import pandas as pd

from streamlit_vars import original_dataset_url, displayDataframeInformations


pd.set_option('display.max_columns', None)


def load_original_dataset_page():
    st.title("Original Dataset")
    st.markdown("This page shows the original dataset used for the project. Some informations are displayed below.")
    st.markdown("__Cleaned dataset is saved as 'cleaned_dataset.csv' in the storage/datas/csv/cleaned/ folder.__")

    # Load original dataset
    original_dataset = pd.read_csv(original_dataset_url)
    displayDataframeInformations(original_dataset, display_classification_repartition=False)