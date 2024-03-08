import streamlit as st

from homepage import load_home_page
from originaldatasetpage import load_original_dataset_page



## Cache ##

@st.cache_data
def generate_cache():
    pass


## Sidebar ##

# Title
st.sidebar.title("Mushroom Observations and Classification project")

# Pages
pages=["Home","Original Dataset","Data Vizualization","Modelization","Evaluation"]
page=st.sidebar.radio("", pages)



# Sidebar Footer
st.markdown("""
    <style>
    .reportview-container .main footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

footer="""
<footer>
    <p>Author: Guillaume Pot</p>
    <p>Email: <a href="mailto:guillaumepot.pro@outlook.com">guillaumepot.pro@outlook.com</a></p>
    <p>LinkedIn: <a href="https://www.linkedin.com/in/062guillaumepot/" target="_blank">Click Here</a></p>
</footer>
"""
st.sidebar.markdown(footer, unsafe_allow_html=True)



## Page call ##

# Home page
if page == pages[0]: 
    load_home_page()


# Original dataset
if page == pages[1]: 
    load_original_dataset_page()






