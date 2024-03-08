import streamlit as st
import pandas as pd
import os

pd.set_option('display.max_columns', None)


def load_original_dataset_page():
    st.write("### Original Dataset Page")
    st.write("This page shows the original dataset used for the project. Some informations are displayed below.")

    # Load original dataset
    original_dataset_url = os.getenv("ORIGINAL_DATASET_URL", "../../storage/datas/raw/observations_mushroom.csv")
    original_dataset = pd.read_csv(original_dataset_url)

    # Show original dataset (5 first lines)
    original_dataset_shortened = original_dataset.head(5)
    st.write(original_dataset_shortened)

    # Show some informations about the dataset
    dataset_observation_count = original_dataset.shape[0]
    st.write(f"Original observation count: {dataset_observation_count}")

    dataset_columns = original_dataset.columns.tolist()
    dataset_column_nb = len(dataset_columns)
    st.write(f"Column count: {dataset_column_nb}")
    st.write(f"Columns: \n {dataset_columns}")

    dataset_duplicates_nb = original_dataset.duplicated().sum()
    st.write(f"Number of duplicates: {dataset_duplicates_nb}")

    st.write("Null values count by column:")
    dataset_null_values = original_dataset.isnull().sum().to_frame().reset_index()
    dataset_null_values.columns = ['Column', 'Null Values']
    st.dataframe(dataset_null_values)
