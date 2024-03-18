"""
Libs
"""
import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


"""
Common vars
"""
storage_cleaned_datasets_base_url = os.getenv("CLEANED_DATASETS_BASE_URL", "../../storage/datas/csv/clean")
storage_raw_datasets_base_url = os.getenv("RAW_DATASETS_BASE_URL", "../../storage/datas/csv/raw")

#img_url = '/home/guillaume/Téléchargements/mushroom-dataset/dataset_images/'


# Original Dataset Page
original_dataset_url = os.path.join(storage_raw_datasets_base_url, "observations_mushroom.csv")


# Data analysis page
cleaned_dataset_url = os.path.join(storage_cleaned_datasets_base_url, "cleaned_dataset.csv")


num_unique_values = {'phylum': None,       # Num unique values to display on each graph (sorted by descending order)
                     'class': None,
                     'order': 15,
                     'family': 30,
                     'genus': 30,
                     'species': 50}



"""
Functions
"""
# Sidebar Footer
def addSidebarFooter():
    st.markdown("""
        <style>
        .reportview-container .main footer {visibility: hidden;}
        </style>
        """, unsafe_allow_html=True)

    footer="""
     <footer style="margin-top: 350px;">
        <p>Author: Guillaume Pot</p>
        <p>Email: <a href="mailto:guillaumepot.pro@outlook.com">guillaumepot.pro@outlook.com</a></p>
        <p>LinkedIn: <a href="https://www.linkedin.com/in/062guillaumepot/" target="_blank">Click Here</a></p>
    </footer>
    """
    st.sidebar.markdown(footer, unsafe_allow_html=True)



# Display DF
def displayDataframeInformations(df:pd.DataFrame, display_classification_repartition=False) -> None:
    """
    Display various information about the given DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to display information for.

    Returns:
    None
    """

    # Display Dataset
    st.subheader("Cleaned Dataset (5 first lines)")
    st.dataframe(df.head(5))

    # Dataset Information
    st.subheader("Dataset Information")
    dataset_observation_count = df.shape[0]
    st.text(f"Observation count: {dataset_observation_count}")

    dataset_columns = df.columns.tolist()
    dataset_column_nb = len(dataset_columns)
    st.text(f"Column count: {dataset_column_nb}")
    st.text(f"Columns: \n {dataset_columns}")

    dataset_duplicates_nb = df.duplicated().sum()
    st.text(f"Number of duplicates: {dataset_duplicates_nb}")

    st.subheader("Null values count by column:")
    dataset_null_values = df.isnull().sum().to_frame().reset_index()
    dataset_null_values.columns = ['Column', 'Null Values']
    st.table(dataset_null_values)

    if display_classification_repartition:
        st.subheader("Classification repartition:")
        st.text("Kingdom (1) is 'Fungi' and not kept as values.")
        st.text(f"phylum: {df['phylum'].nunique()}")
        st.text(f"class: {df['class'].nunique()}")
        st.text(f"order: {df['order'].nunique()}")
        st.text(f"family: {df['family'].nunique()}")
        st.text(f"genus: {df['genus'].nunique()}")
        st.text(f"species: {df['species'].nunique()}")



# Display Charts
def displayCharts(df,num_unique_values:dict) -> None:
    """
    Display charts for analyzing the given DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame to be analyzed.

    Returns:
    None
    """

    def create_subplot(df, column, ax, num):
        """
        Create a subplot for visualizing the count of observations in a column of a DataFrame.

        Parameters:
        - df (pandas.DataFrame): The DataFrame containing the data.
        - column (str): The name of the column to visualize.
        - ax (matplotlib.axes.Axes): The subplot to plot on.
        - num (int): The number of unique values to consider.

        Returns:
        None
        """
        # Get the most frequent values and filter the dataset
        top_values = df[column].value_counts().index[:num]
        filtered_dataset = df[df[column].isin(top_values)]

        # Plot the count of observations for each value
        sns.countplot(data=filtered_dataset,
                      x=column,
                      hue=column,
                      order=filtered_dataset[column].value_counts().index,
                      palette='deep',
                      legend=False,
                      ax=ax)
        ax.set_title(column.capitalize())
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.set_ylabel("Observations")

        if column == 'phylum' or column == 'class':
            ax.set_yscale("log")

    # Define the number of unique values to consider for each column
    columns = num_unique_values.keys()
    fig, axs = plt.subplots(len(columns), 1, figsize=(10, 6*len(columns)))

    # Create a subplot for each value to visualize
    for ax, column in zip(axs, columns):
        create_subplot(df, column, ax, num_unique_values[column])


    # Display the plots
    plt.tight_layout()
    st.pyplot(fig)