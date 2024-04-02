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

img_url = os.getenv("IMAGE_URL", "/home/guillaume/Téléchargements/mushroom-dataset/clean_dataset/")


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

features_dataset_url = os.path.join(storage_cleaned_datasets_base_url, "cleaned_dataset_with_features.csv")

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



# Display random imgs
def displayRandomImages(df:pd.DataFrame, img_url:str = img_url, n:int=5) -> None:
    """
    Display random images from the dataset.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the image paths.
    img_url (str): The base URL of the images.
    n (int): The number of images to display.

    Returns:
    None
    """
    # Get random images
    random_images = df.sample(n)
    cols = st.columns(n)

    for i, (index, row) in enumerate(random_images.iterrows()):
        img = plt.imread(os.path.join(img_url, row['image_lien']))
        # Display image in the corresponding column
        cols[i].image(img, caption=row['label'], use_column_width=True)


# Display Features charts


def displayFeaturesCharts(df:pd.DataFrame) -> None:
    """
    Display various charts and histograms for analyzing the features of a given DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data to be analyzed.

    Returns:
    None
    """

    def plotHist(ax, x, data, color, title):
        """
        Plot a histogram on the given axes.

        Parameters:
        ax (matplotlib.axes.Axes): The axes on which to plot the histogram.
        x (str): The column name of the data to be plotted on the x-axis.
        data (pandas.DataFrame): The data to be used for plotting.
        color (str): The color of the histogram bars.
        title (str): The title of the histogram.

        Returns:
        None
        """
        ax.hist(x=x, data=data, color=color)
        ax.set_title(title)
        ax.set_xlabel(x)
        ax.set_ylabel('Nb')
        ax.grid(True, linestyle='--')


    # Features Box Plot
    fig, ax = plt.subplots(figsize=(16,12))
    sns.catplot(df, kind='boxen')
    plt.grid(False)
    plt.title('Features Box plot')
    plt.xticks(rotation = 90)
    st.pyplot(fig)

    # Img dimensions comparison
    df['dimensions'] = df['high'] * df['width']
    fig, axes = plt.subplots(1,3, figsize=(16, 6))

    attributes = ['width', 'high', 'dimensions']
    colors = ['yellow', 'green', 'red']
    titles = ['Width distribution', 'High distribution', 'Dimensions distribution']

    for ax, attr, color, title in zip(axes, attributes, colors, titles):
        plotHist(ax, attr, df, color, title)
    st.pyplot(fig)

    # RGB mean comparison
    fig, axes = plt.subplots(2,2, figsize=(10,8))
    plt.subplots_adjust(wspace=1, hspace=1)

    attributes = ['red mean', 'green mean', 'blue mean', 'color mean']
    colors = ['red', 'green', 'blue', 'purple']
    titles = ['Red mean distribution', 'Green mean distribution', 'Blue mean distribution', 'Color mean distribution']

    for ax, attr, color, title in zip(axes.flatten(), attributes, colors, titles):
        plotHist(ax, attr, df, color, title)
    st.pyplot(fig)