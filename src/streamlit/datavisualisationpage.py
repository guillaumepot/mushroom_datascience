import streamlit as st
import pandas as pd
import os

pd.set_option('display.max_columns', None)


def load_data_visualization_page():
    st.write("### Data Visualization Page")
    st.write("This page shows somes graphs after original dataset treatments.")

    