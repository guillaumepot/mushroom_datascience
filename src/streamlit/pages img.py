    


from streamlit_vars import displayRandomImages, displayFeaturesCharts



    # Part 3 - Imgs Visualization
    elif page == "Imgs Visualization":
        st.subheader("Images Visualization")
        n = st.selectbox('Number of images to display', list(range(1, 6)))
        if st.button('Show random images from Dataset'):
            displayRandomImages(cleaned_dataset_with_features_dimensions, n=n)

        st.subheader("Image features charts")
        displayFeaturesCharts(cleaned_dataset_with_features_dimensions)