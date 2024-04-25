import streamlit as st
import tensorflow as tf
import io
import sys


# Variables
from streamlit_vars import available_models, trained_model_url_dict
# Functions
from streamlit_vars import displayRandomImages, displayRandomImagesFromDatasetTF








def load_evaluation_page():
    st.title("Model Evaluation page")
    st.write("This page shows model evaluation for a given trained model.")


    # Select model
    s = st.selectbox('Model', available_models)
    model_url = trained_model_url_dict[s]

    # Load model
    model = tf.keras.models.load_model(model_url)
    
    # 