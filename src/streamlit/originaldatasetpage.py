import streamlit as st
import pandas as pd
import os

pd.set_option('display.max_columns', None)


def load_original_dataset_page():
    st.title("Original Dataset Page")
    st.markdown("This page shows the original dataset used for the project. Some informations are displayed below.")
    st.markdown("__Cleaned dataset is saved as 'cleaned_dataset.csv' in the storage/datas/csv/cleaned/ folder.__")

    # Load original dataset
    original_dataset_url = os.getenv("ORIGINAL_DATASET_URL", "../../storage/datas/csv/raw/observations_mushroom.csv")
    original_dataset = pd.read_csv(original_dataset_url)

    # Display Dataset
    st.subheader("Original Dataset (5 first lines)")
    st.dataframe(original_dataset.head(5))

    # Dataset Information
    st.subheader("Dataset Information")
    dataset_observation_count = original_dataset.shape[0]
    st.text(f"Original observation count: {dataset_observation_count}")

    dataset_columns = original_dataset.columns.tolist()
    dataset_column_nb = len(dataset_columns)
    st.text(f"Column count: {dataset_column_nb}")
    st.text(f"Columns: \n {dataset_columns}")

    dataset_duplicates_nb = original_dataset.duplicated().sum()
    st.text(f"Number of duplicates: {dataset_duplicates_nb}")

    st.subheader("Null values count by column:")
    dataset_null_values = original_dataset.isnull().sum().to_frame().reset_index()
    dataset_null_values.columns = ['Column', 'Null Values']
    st.table(dataset_null_values)