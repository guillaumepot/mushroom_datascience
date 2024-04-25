import streamlit as st
import pandas as pd
import tensorflow as tf



# Variables
from streamlit_vars import available_models, model_url_dict
# Functions
from streamlit_vars import displayRandomImages, displayRandomImagesFromDatasetTF



def load_modelization_page():
        st.title("Model pages")
        st.write("This page shows some generated models ready to train on the dataset")

        # Select model
        s = st.selectbox('Model', available_models)
        model_url = model_url_dict[s]


        # Load model
        model = tf.keras.models.load_model(model_url)
        st.write(model.summary())


