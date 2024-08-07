# A Deep Learning classifier for Datascience Projet 



<img src="./media/mushroom_img.jpeg" width="350" height="350">





**OVERVIEW**

This project was initially realized as a "red thread" project while I was following a Datascientist diploma. After a few weeks, I decided to add a soft update to clean the code.


>**1.** Initially, the project contains a report (French) and some notebooks (Pipeline ETL, model training, evaluation)

--

>**2.** With the new iteration, I added a Streamlit interface to navigate and discover the project, refactorized some code, added notebooks, optimized functions.

--

This project is a prototype, not designed for production.




## Table of content
- [1. Repository Architecture](#1-repository-architecture)
- [2. Prerequisites](#2-prerequisites)
    - [2.1. Venv](#21-venv)
    - [2.2. Files](#22-files)
    - [2.3. Generate Objects](#22-generate-objects)
- [3. Prerequisites](#3-prerequisites)

## 1. Repository Architecture
```
    /   
    │
    ├── old  < Contains old report from first project iteration (French)
    |
    │
    ├── src
    |       │  
    |       ├── logs < Contains logs generated by some functions (utils)
    |       |            
    |       ├── notebooks                            
    |       │   │
    |       │   ├── 1-dataset_clean.ipynb           < Used to clean the baase dataset
    |       │   │                                   
    |       │   ├── 2-data_analysis.ipynb           < Used to analyse datas
    |       │   │
    |       │   ├── 3-classification_choice.ipynb   < Used to find & justify classifiction choice among the libing classification
    |       │   │
    |       │   ├── 4-file_sort.ipynb               < Used to sort the .jpg files according to the csv files
    |       │   │
    |       │   ├── 5-imgs_analysis.ipynb           < Used to analyse images (colors, dimensions)
    |       │   │ 
    |       │   ├── 6-data_preprocessing.ipynb      < Used to pre-process datas & generate tf datasets
    |       │   │
    |       │   ├── 8-model_train.ipynb             < Used to train the model(s)
    |       │   │
    |       │   └── 9-model_evaluate.ipynb          < Used to evaluate the model(s)
    |
    |
    ├── streamlit
    |       ├── (deprecated)classificationchoicepage.py < Deprecated page about classification choice
    |       |
    |       ├── dataanalysispage.py < Streamlit page for data analysis
    |       |
    |       ├── datasetcsvpage.py < Streamlit page for CSV display
    |       |
    |       ├── evaluationpage.py < Streamlit page for model evaluation
    |       |
    |       ├── homepage.py < Streamlit home page
    |       |
    |       ├── imagevizpage.py < Streamlit page for image visualization
    |       |
    |       ├── main.py < Streamlit main (call pages)
    |       |
    |       ├── modelizationpage.py < Streamlit page for display model(s)
    |       |
    |       ├── streamlit_vars.py < Streamlit used in pages variables python file
    |       |
    |       └── streamlit.env < .env file for a Streamlit container (not done)
    |
    ├── utils
    |       │  
    |       ├── __init__.py < Python file to use utils as a package
    |       |            
    |       ├── file_processing.py < Functions to sort & move files
    |       |                           
    |       ├── img_preprocessing.py < Functions to pre-process imgs files & generate tf datasets
    |       |                           
    |       ├── import_df.py < Functions to import pandas DF
    |       |                           
    |       ├── model_generation.py < Functions to generate a keras model
    │
    ├── storage
    |       │  
    |       ├── datas            < Contains logs generated by some functions (utils)
    |       │   │
    |       │   ├── csv          < Used to clean the baase dataset
    |       │   │                                   
    |       │   ├── img          < Used to analyse datas
    |       │   │                                   
    |       │   ├── json         < Used to analyse datas
    |       │   │                                   
    |       │   └── tf_datasets  < Used to analyse datas
    |       |                
    |       ├── mlruns < Contains mlruns (autologged model fitting)                            
    |       │  
    |       ├── models < Contains logs generated by some functions (utils)
    |       │   │
    |       │   ├── builded          < Builded model(s) from 7-modelization.ipynb
    |       │   │                                                          
    |       │   ├── histories        < Training histories
    |       │   │                                   
    |       │   ├── model_checkpoint < Model weights (callback)
    |       │   │
    |       │   └── trained          < trained model(s)
    │
    ├── poetry.lock < Environment file for poetry
    │
    ├── pyproject.toml <  Conf file for poetry
    │
    ├── README.md < This current file
    │
    └── requirementstxt < Venv dependencies
```


## 2. Prerequisites

### 2.1. Venv
- Create a venv according to requirements.txt or pyproject.toml (poetry)

The list below contains all the dependencies you need :

    
    ├── python 3.10
    │
    ├── pandas = "^2.2.1"
    │
    ├── seaborn = "^0.13.2"
    │
    ├── ipykernel = "^6.29.3"
    │
    ├── opencv-python = "^4.9.0.80"
    │
    ├── streamlit = "^1.32.2"
    │
    ├── bokeh = "^3.4.0"
    │
    ├── scikit-learn = "^1.4.1.post1"
    │    
    ├── tf-keras = "^2.16.0"
    │
    ├── mlflow = "^2.12.1"
    │
    ├── tensorflow = "^2.16.1"



### 2.2 Files
- Get some file from my shared link (*link no longer available, please pm me on LinkedIn or mail to get a new one*)

    This link contains some files you have to unzip here if you want to execute all notebooks :
    - Unzip cleaned_dataset here : ../../storage/datas/imgs/**cleaned_dataset/**
    - Unzip bad_images.zip here : ../../storage/datas/imgs/**bad_images/**
    - Unzip models.zip : ../../storage/**models/**
    - Copy observations_mushroom.csv here : ../../storage/datas/csv/raw/**observations_mushroom.csv**


### 2.3. Generate Objects 
**Required**
- Generate tf datasets using notebook 6-data_preprocessing.ipynb

**Optional**
- Generate a model using 7-modelization.ipynb


## 3. Start Streamlit
- Navigate to ./src/streamlit
- Run streamlit with the command : streamlit run main.py