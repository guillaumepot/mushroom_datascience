"""
Libs
"""
import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import random
import cv2
from keras.preprocessing import image



"""
Common vars
"""
storage_cleaned_datasets_base_url = os.getenv("CLEANED_DATASETS_BASE_URL", "../../storage/datas/csv/clean")
storage_raw_datasets_base_url = os.getenv("RAW_DATASETS_BASE_URL", "../../storage/datas/csv/raw")

top10_species_img_url = os.getenv("TOP10_SPECIES_IMAGE_URL", "/home/guillaume/Téléchargements/mushroom_images_dataset/cleaned_dataset/")
bad_img_url = os.getenv("BAD_IMAGE_URL", "../../storage/datas/imgs/bad_images/")

train_dataset_path = "../../storage/datas/tf_datasets/train_dataset"



# CSV url
original_dataset_url = os.path.join(storage_raw_datasets_base_url, "observations_mushroom.csv")
cleaned_dataset = os.path.join(storage_cleaned_datasets_base_url, "cleaned_dataset.csv")
cleaned_dataset_with_features_dimensions_url = os.path.join(storage_cleaned_datasets_base_url, "cleaned_dataset_with_features_and_dimensions.csv")
cleaned_dataset_top10_species_url = os.path.join(storage_cleaned_datasets_base_url, "cleaned_dataset_with_features_and_dimensions_top_10_species.csv")
bad_images_df = os.path.join(storage_cleaned_datasets_base_url, "bad_images_moved.csv")

# Dataset CSV Page
##


# Data analysis page
num_unique_values = {'phylum': None,       # Num unique values to display on each graph (sorted by descending order)
                     'class': None,
                     'order': 15,
                     'family': 30,
                     'genus': 30,
                     'species': 50}


# Modelization page
available_models = ["efficientnetv2_21k with custom top layers - V1", "efficientnetv2_21k with custom top layers - V1 - Fine Tuned"]

builded_model_url_dict = {
    "efficientnetv2_21k with custom top layers - V1" : "../../storage/models/builded/generated_model_efficientnetv2_21k_finetuned_1k_v1.keras",
    "efficientnetv2_21k with custom top layers - V1 - Fine Tuned" : "../../storage/models/builded/generated_model_efficientnetv2_21k_finetuned_1k_v1_fine_tuned.keras"  
}


# Evaluation page
trained_model_url_dict = {
    "efficientnetv2_21k with custom top layers - V1" : "../../storage/models/trained/generated_model_efficientnetv2_21k_finetuned_1k_v1_trained.keras",
    "efficientnetv2_21k with custom top layers - V1 - Fine Tuned" : "../../storage/models/trained/generated_model_efficientnetv2_21k_finetuned_1k_v1_trained_fine_tuned.keras"    
}

history_model_url_dict = {
    "efficientnetv2_21k with custom top layers - V1" : "../../storage/models/histories/generated_model_efficientnetv2_21k_finetuned_1k_v1_trained_history.pickle",
    "efficientnetv2_21k with custom top layers - V1 - Fine Tuned" : "../../storage/models/histories/generated_model_efficientnetv2_21k_finetuned_1k_v1_trained_fine_tuned_history.pickle"
}

test_dataset_path = "../../storage/datas/tf_datasets/test_dataset"
encoded_labels_json_path = "../../storage/datas/json/encoded_labels.json"


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
    st.subheader("Dataset (5 first lines)")
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



# Display DF Comparison
def displayDataframeComparison(df1: pd.DataFrame, df2: pd.DataFrame, df3: pd.DataFrame = None, df4: pd.DataFrame = None) -> None:
    """
    Display a comparison table of information about the input dataframes.

    Parameters:
    - df1 (pd.DataFrame): The first dataframe to compare.
    - df2 (pd.DataFrame): The second dataframe to compare.
    - df3 (pd.DataFrame, optional): An optional third dataframe to compare. Default is None.
    - df4 (pd.DataFrame, optional): An optional fourth dataframe to compare. Default is None.

    Returns:
    None
    """

    def getDfInfo(df: pd.DataFrame) -> pd.Series:
        """
        Get information about the input dataframe.

        Parameters:
        - df (pd.DataFrame): The dataframe to get information from.

        Returns:
        pd.Series: A series containing information about the dataframe.
        """
        info = {
            'shape (rows)': df.shape[0],
            'shape (cols)': df.shape[1],
            'duplicated': df.duplicated(subset = 'image_lien').sum(),
            'NAN': df.isna().sum().sum(),
            'phylum': df['phylum'].nunique() if 'phylum' in df.columns else df['gbif_info/phylum'].nunique(),
            'class': df['class'].nunique() if 'class' in df.columns else df['gbif_info/class'].nunique(),
            'order': df['order'].nunique() if 'order' in df.columns else df['gbif_info/order'].nunique(),
            'family': df['family'].nunique() if 'family' in df.columns else df['gbif_info/family'].nunique(),
            'genus': df['genus'].nunique() if 'genus' in df.columns else df['gbif_info/genus'].nunique(),
            'species': df['species'].nunique() if 'species' in df.columns else df['gbif_info/species'].nunique()
        }

        return pd.Series(info)

    df1_info = getDfInfo(df1)
    df2_info = getDfInfo(df2)
    df3_info = getDfInfo(df3) if df3 is not None else None
    df4_info = getDfInfo(df4) if df4 is not None else None

    comparison = pd.DataFrame({'Original': df1_info, 'Cleaned (with files missing)': df2_info})
    if df3 is not None:
        df3_info = getDfInfo(df3)
        comparison['Cleaned (without missing files)'] = df3_info

    if df4 is not None:
        df4_info = getDfInfo(df4)
        comparison["Top 10 species + 1 'other' class ; with manual imgs sort"] = df4_info
    
    st.table(comparison)



# Display Charts
def displayCharts(df : pd.DataFrame , num_unique_values:dict) -> None:
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
    fig, axs = plt.subplots(len(columns), 1, figsize=(18, 10*len(columns)))

    # Create a subplot for each value to visualize
    for ax, column in zip(axs, columns):
        create_subplot(df, column, ax, num_unique_values[column])


    # Display the plots
    plt.close()
    plt.tight_layout()
    st.pyplot(fig)



# Display random imgs
def displayRandomImages(df:pd.DataFrame, n:int=3, bad_images = False) -> None:
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

    if bad_images == True:
        for i, (index,row) in enumerate(random_images.iterrows()):
            img = plt.imread(row['image_path'])
            cols[i].image(img, caption="", use_column_width=True)

    else:
        for i, (index, row) in enumerate(random_images.iterrows()):
            img = plt.imread(row['image_path'])
            # Display image in the corresponding column
            cols[i].image(img, caption=row['label'], use_column_width=True)



def displayRandomImagesFromDatasetTF(n: int = 3) -> None:
    """
    Display random images from a TensorFlow dataset.

    Parameters:
        n (int): The number of random images to display. Default is 3.

    Returns:
        None
    """
    train_dataset = tf.data.experimental.load(train_dataset_path)
    train_dataset = train_dataset.shuffle(buffer_size=1000)
    datasets_elements = train_dataset.take(n)
    for images, labels in datasets_elements:
        for i in range(n):
            image = images[i].numpy()
            image = np.clip(image, 0.0, 1.0)  # Clip values to the range [0.0, 1.0]
            st.image(image, caption=f"Label: {labels[i].numpy()}", use_column_width=True)



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
    fig, ax = plt.subplots(figsize=(14,10))
    sns.boxenplot(data=df, ax=ax)
    ax.grid(False)
    ax.set_title('Features Box plot')
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
    plt.subplots_adjust(wspace=0.8, hspace=0.5)

    attributes = ['red mean', 'green mean', 'blue mean', 'color mean']
    colors = ['red', 'green', 'blue', 'purple']
    titles = ['Red mean distribution', 'Green mean distribution', 'Blue mean distribution', 'Color mean distribution']

    for ax, attr, color, title in zip(axes.flatten(), attributes, colors, titles):
        plotHist(ax, attr, df, color, title)
    st.pyplot(fig)



def plot_training_history(history) -> None:
    """
    Plots the training history of a model.

    Parameters:
    - history (dict): A dictionary containing the training history of the model.

    Returns:
    - None

    """
    # Accuracy & Val_accuracy plot (training model)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Training loss & Val loss plot (training model)
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Display the plot in Streamlit
    st.pyplot(plt)



import numpy as np

def get_y_test(test_datas) -> np.ndarray:
    """
    Get the ground truth labels from the test data.

    Args:
        test_datas (list): A list of tuples containing the test data and labels.

    Returns:
        np.ndarray: An array of ground truth labels.

    """
    y_test = []
    for _, label in test_datas:
        y_test.extend(np.argmax(label.numpy(), axis=-1)) 
    y_test = np.array(y_test)
    return y_test



def display_classifciation_report(y_test, y_pred) -> None:
    """
    Display the classification report for a given set of predictions.

    Parameters:
    - y_test (array-like): The true labels of the test data.
    - y_pred (array-like): The predicted labels for the test data.
    - test_datas (array-like): The test data used for prediction.

    Returns:
    None
    """
    # Get y_pred classes (argmax)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Generate classification report
    report = classification_report(y_test, y_pred_classes)
    report_df = pd.DataFrame(report).transpose()
    st.table(report_df)



def display_confusion_matrix(y_test, y_pred) -> None:
    """
    Display the confusion matrix and normalized confusion matrix for a classification model.

    Parameters:
    y_test (array-like): The true labels of the test data.
    y_pred (array-like): The predicted labels of the test data.
    test_datas (array-like): The test data used for prediction.

    Returns:
    None
    """
    y_pred_classes = np.argmax(y_pred, axis=1)
    # Confusion Matrix
    confusion_mtx = confusion_matrix(y_test, y_pred_classes)
    confusion_mtx_normalized = confusion_mtx.astype('float') / confusion_mtx.sum(axis=1)[:, np.newaxis]

    # Plot
    plt.figure(figsize=(8, 8))
    sns.heatmap(confusion_mtx, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predictions')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    st.pyplot(plt)  # Affiche le graphique sur Streamlit

    # normalized CM plot
    plt.figure(figsize=(8, 8))
    sns.heatmap(confusion_mtx_normalized, annot=True, cmap='Greens')
    plt.xlabel('Predictions')
    plt.ylabel('True labels')
    plt.title('Normalized Confusion Matrix')
    st.pyplot(plt)



def display_test_images_and_get_predictions(test_datas, encoded_labels, y_pred_classes) -> None:
    """
    Display test images and their corresponding predictions.

    Args:
        test_datas (Iterable): The test dataset containing images and labels.

    Returns:
        None
    """
    # Convert dataset to list
    test_data_list = list(test_datas)

    # Choose 9 randoms indexes
    random_indices = random.sample(range(len(test_data_list)), 9)

    # Display images & labels
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))

    for i, ax in enumerate(axes.flat):
        img, label = test_data_list[random_indices[i]]
        img = img.numpy()

        random_index = random.randint(0, len(img) - 1)
        img = img[random_index]

        true_label = encoded_labels[np.argmax(label)]
        predicted_label = encoded_labels[y_pred_classes[random_indices[i]]]

        ax.imshow(img)
        ax.set_title(f"True: {true_label}\nPred: {predicted_label}")

    plt.tight_layout()
    st.pyplot(fig)  # Display the figure in Streamlit



def get_img_prediction(uploaded_file, model, encoded_labels) -> None:
    """
    Get the image prediction using the provided model and display the result in Streamlit.

    Parameters:
    - uploaded_file: The uploaded image file.
    - model: The pre-trained model used for prediction.
    - encoded_labels: The encoded labels used for mapping the predicted class.

    Returns:
    None
    """
    # load img, pre-process
    image_data = uploaded_file
    img_web = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    img_web = cv2.cvtColor(img_web, cv2.COLOR_BGR2RGB)
    img_web = cv2.resize(img_web, (224, 224))

    w = image.img_to_array(img_web)
    w = np.expand_dims(w, axis=0)
    #w = preprocess_input(w)
        
    # Predict
    web_prediction = model.predict(w)
    index_classe_web = np.argmax(web_prediction)
    nom_classe_predite = encoded_labels[index_classe_web]

    # Show img with predicted class
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img_web)
    ax.set_title(f'Predicted class : {nom_classe_predite}')
    ax.axis('off')
    st.pyplot(fig)  # Display the figure in Streamlit