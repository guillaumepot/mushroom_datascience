import streamlit as st

from home_page import load_home_page
from datasetcsv_page import load_dataset_csv_page
from dataanalysis_page import load_data_visualization_page
from imageviz_page import load_image_viz_page
from modelization_page import load_modelization_page
from evaluation_page import load_evaluation_page

from streamlit_vars import addSidebarFooter

## Cache ##
@st.cache_data
def generate_cache():
    pass


## Sidebar ##

# Title
st.sidebar.title("Mushroom Observations and Classification project")

# Pages
pages=["Home","Dataset csv file","Data Analysis","Image Visualization","Modelization","Evaluation"]
page=st.sidebar.radio("", pages)



## Page call ##

# Home page
if page == pages[0]: 
    load_home_page()


# Dataset csv file
if page == pages[1]: 
    load_dataset_csv_page()


# Data Analysis
if page == pages[2]:
    load_data_visualization_page()


# Image Visualization
if page == pages[3]:
    load_image_viz_page()


# Modelization
if page == pages[4]:
    load_modelization_page()


if page == pages[5]:
    load_evaluation_page()


## Sidebar Footer ##
addSidebarFooter()