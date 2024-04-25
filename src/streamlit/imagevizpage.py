import streamlit as st
import pandas as pd

# Variables
from streamlit_vars import cleaned_dataset_top10_species_url, top10_species_img_url, bad_images_df
# Functions
from streamlit_vars import displayRandomImages, displayRandomImagesFromDatasetTF


pd.set_option('display.max_columns', None)

# Load DF
cleaned_dataset_top10_species = pd.read_csv(cleaned_dataset_top10_species_url)
bad_imgs = pd.read_csv(bad_images_df)

cleaned_dataset_top10_species['image_path'] = cleaned_dataset_top10_species['image_lien'].apply(lambda x: top10_species_img_url + x)


def load_image_viz_page():
    st.title("Image Visualization")
    st.write("This page shows some images depending the set you choose.")
    

    choice =  st.radio('Select Dataset', ['Top 10 species', 'Bad images removed from dataset', 'tf_datasets'])
    n = st.selectbox('Number of images to display', list(range(1, 4)))

    # Display random images from df
    if choice == 'Top 10 species':
        unique_species = cleaned_dataset_top10_species['species'].unique()
        s = st.selectbox('Specy', unique_species)
        df_img_to_show = cleaned_dataset_top10_species[cleaned_dataset_top10_species['species'] == s]

        if st.button('Show random images from Dataset'):
             displayRandomImages(df_img_to_show, n=n)

    # Display random images from sorted bad images
    elif choice == 'Bad images removed from dataset':
        st.markdown("Displayed images have been sorted manually and are considered as bad images.")
        st.markdown("A image is considered as bad following these criterias:")
        st.markdown("1. The image is not a mushroom.")
        st.markdown("2. The mushroom is too tiny in a big environment picture")
        st.markdown("3. There are humans on the picture.")
        st.markdown("4. The context is not clear (cooking, ...)")
        st.markdown("5. The image is a microscopic analysis of celluls.")
        if st.button('Show random bad images'):
            displayRandomImages(bad_imgs, n=n, bad_images = True)


    # Display random images from tf_datasets
    else:
        st.markdown("This dataset is composed of 10 (+1) classes of mushrooms.")
        if st.button('Show random images from train tf_dataset'):
            displayRandomImagesFromDatasetTF(n=n)
