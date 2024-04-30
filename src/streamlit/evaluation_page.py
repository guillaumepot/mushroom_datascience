import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
import json


# Variables
from streamlit_vars import available_trained_models, trained_model_url_dict, test_dataset_path, encoded_labels_json_path, history_model_url_dict
# Functions
from streamlit_vars import plot_training_history, get_y_test, display_classification_report, display_confusion_matrix, display_test_images_and_get_predictions, get_img_prediction



def load_evaluation_page():
    st.title("Model Evaluation page")
    st.write("This page shows model evaluation for a given trained model.")


    # Select model
    s = st.selectbox('Model', available_trained_models)
    model_url = trained_model_url_dict[s]

    # Load model
    model = tf.keras.models.load_model(model_url)


    # Load history
    history_path = history_model_url_dict[s]
    with open(history_path, 'rb') as file:
        history = pickle.load(file)

    # Load test Dataset
    test_datas = tf.data.Dataset.load(test_dataset_path)


    # Get predictions
    y_test = get_y_test(test_datas)
    y_pred = model.predict(test_datas)
    y_pred_classes = np.argmax(y_pred, axis=1)


    # Get encoded labels dictionary
    with open(encoded_labels_json_path, 'r') as file:
        encoded_labels = json.load(file)
    encoded_labels = {v: k for k, v in encoded_labels.items()}

    # Display images and predictions
    if st.button('Display random images from test dataset and show predictions \n (This can take a few seconds)'):
        display_test_images_and_get_predictions(test_datas, encoded_labels, y_pred_classes)

    # Display history plot
    plot_training_history(history)

    # Display classification report
    st.header("Classification Report")
    display_classification_report(y_test, y_pred)

    # Display confusion matrix
    display_confusion_matrix(y_test, y_pred)