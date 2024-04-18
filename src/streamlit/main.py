import streamlit as st

from homepage import load_home_page
from datasetcsvpage import load_dataset_csv_page
from dataanalysispage import load_data_visualization_page
from imagevizpage import load_image_viz_page

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


# Original dataset
if page == pages[1]: 
    load_dataset_csv_page()


# Data visualization
if page == pages[2]:
    load_data_visualization_page()


# Image visualization
if page == pages[3]:
    load_image_viz_page()



## Sidebar Footer ##
addSidebarFooter()