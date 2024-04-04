import streamlit as st

from homepage import load_home_page
from originaldatasetpage import load_original_dataset_page
from dataanalysispage import load_data_visualization_page
from classificationchoicepage import load_classification_choice

from streamlit_vars import addSidebarFooter

## Cache ##
@st.cache_data
def generate_cache():
    pass


## Sidebar ##

# Title
st.sidebar.title("Mushroom Observations and Classification project")

# Pages
pages=["Home","Original Dataset","Data Analysis","Classification choice","Modelization","Evaluation"]
page=st.sidebar.radio("", pages)



## Page call ##

# Home page
if page == pages[0]: 
    load_home_page()


# Original dataset
if page == pages[1]: 
    load_original_dataset_page()


# Data visualization
if page == pages[2]:
    load_data_visualization_page()


# Data preprocessing
if page == pages[3]:
    load_classification_choice()



## Sidebar Footer ##
addSidebarFooter()