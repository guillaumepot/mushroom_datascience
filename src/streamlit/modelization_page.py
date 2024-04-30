import streamlit as st
import tensorflow as tf
import io
import sys


# Variables
from streamlit_vars import available_builded_models, builded_model_url_dict



def load_modelization_page():
    st.title("Model page")
    st.write("This page shows some generated models ready to train on the dataset")

    # Select model
    s = st.selectbox('Model', available_builded_models)
    model_url = builded_model_url_dict[s]

    # Load model
    model = tf.keras.models.load_model(model_url)

    # Create a buffer to capture the output (model.summary())
    stream = io.StringIO()
    sys.stdout = stream

    # Call the summary function
    model.summary()

    # Reset the standard output
    sys.stdout = sys.__stdout__

    # Get the summary from the buffer
    model_summary = stream.getvalue()

    # Display the model summary
    st.text(model_summary)