import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
import json


# Variables
from streamlit_vars import available_models, trained_model_url_dict, test_dataset_path, encoded_labels_json_path, history_path_dict
# Functions
from streamlit_vars import plot_training_history, get_y_test, display_classifciation_report, display_confusion_matrix, display_test_images_and_get_predictions, get_img_prediction



def load_evaluation_page():
    st.title("Model Evaluation page")
    st.write("This page shows model evaluation for a given trained model.")


    # Select model
    s = st.selectbox('Model', available_models)
    model_url = trained_model_url_dict[s]

    # Load model
    model = tf.keras.models.load_model(model_url)


    # Load history
    history_path = history_path_dict[s]
    with open(history_path, 'rb') as file:
        history = pickle.load(file)

    # Load test Dataset
    test_datas = tf.data.Dataset.load(test_dataset_path)

    # Display history plot
    plot_training_history(history)

    # Get predictions
    y_test = get_y_test()
    y_pred = model.predict(test_datas)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Display classification report
    display_classifciation_report(y_test, y_pred)

    # Display confusion matrix
    display_confusion_matrix(y_test, y_pred)

    # Get encoded labels dictionary
    with open(encoded_labels_json_path, 'r') as file:
        encoded_labels = json.load(file)
    encoded_labels = {v: k for k, v in encoded_labels.items()}

    # Display images and predictions
    display_test_images_and_get_predictions(test_datas, encoded_labels, y_pred_classes)


    # Get image prediction
    st.write("Get image prediction")
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    get_img_prediction(uploaded_file, model, encoded_labels)