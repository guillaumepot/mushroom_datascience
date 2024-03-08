import streamlit as st


def load_home_page():
    st.write("### Home Page")
    with st.container():
        st.write("Welcome to the Streamlit application for the mushroom classification project.")
        st.write("This application aims to present the different steps of the project, including data exploration, visualization, modeling, and evaluation.")
        st.write("To navigate between the different steps, use the left menu.")
        st.write("For more information about the project, please refer to the [README]")