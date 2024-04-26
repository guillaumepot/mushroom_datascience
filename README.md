# A Datascience Projet 

**OVERVIEW**

This project was initially realized as a "red thread" project while I was following a Datascientist diploma. After a few weeks, I decided to add a soft update to clean the code.


>**1.** Initially, the project contains a report (French) and some notebooks (Pipeline ETL, model training, evaluation)

--

>**2.** With the new iteration, I added a Streamlit interface to navigate and discover the project, refactorized some code, added notebooks, optimized functions.

--

This project is a prototype, not designed for production.




## Table of content
- [1. Repository](#1-repository)
- [2. Prerequisites](#1-prerequisites)
    - [2.1. Venv](#11-venv)
    - [2.2. Configuration Defaults](#42-configuration-defaults)


## 1. Repository Architecture

    /   
    │
    ├── data_ml_functions
    │       │  
    │       ├── common_variables.py             <-
    │       │
    │       ├── archive_datas_source.py         <- 
    │       │
    │       ├── scrap_match_history.py          <-
    │       ├── scrap_bookmakers_odds.py        <-
    │       ├── geckodriver                     <-
    │       │
    │       ├── data_preprocessing_matches.py   <-
    │       │
    │       ├── train_model.py                  <-
    │       │
    │       ├── model_predictions.py            <-
    │       │
    │       └── unit_test.py                    <-
    │
    └── storage
            │  
            ├── data                                        <-
            │   │
            │   ├── archives                                <-
            │   │   │
            │   │   ├── {yyyy}-{mm}-{dd}                    <-
            │   │   │       ├── {championship}{1}_odds.csv  <-
            │   │   │       ├── {championship}{2}_odds.csv  <-
            │   │   │       └── main_leagues_data.zip       <-
            │   │   │            
            │   │   └── {yyyy}-{mm}-{dd}                    <-
            │   │           ├── {championship}{1}_odds.csv  <-
            │   │           ├── {championship}{2}_odds.csv  <-
            │   │           └── main_leagues_data.zip       <-
            │   │
            │   ├── clean                                   <-
            │   │   │
            │   │   ├── {championship}{1}                   <-
            │   │   │       ├── match_data_for_modeling.csv <-
            │   │   │       ├── odds.csv                    <-
            │   │   │       ├── statsAwayTeam.csv           <-
            │   │   │       └── statsHomeTeam.csv           <-
            │   │   │
            │   │   └── {championship}{2}                   <-
            │   │           ├── match_data_for_modeling.csv <-
            │   │           ├── odds.csv                    <-
            │   │           ├── statsAwayTeam.csv           <-
            │   │           └── statsHomeTeam.csv           <-
            │   │
            │   ├── raw                                     <-
            │   │   ├── {championship}{1}.csv               <-
            │   │   ├── {championship}{1}_odds.csv          <-
            │   │   │       
            │   │   ├── {championship}{2}.csv               <-
            │   │   ├── {championship}{2}_odds.csv          <-
            │   │   │       
            │   │   ├── {championship}{3}.csv               <-
            │   │   │       
            │   │   ├── {championship}{4}.csv               <-
            │   │   │       
            │   │   ├── {championship}{5}.csv               <-
            │   │
            │   └── source                                  <-
            │       ├── {championship}{1}_odds.csv          <-
            │       ├── {championship}{2}_odds.csv          <-
            │       │       
            │       └── main_leagues_data.zip               <-
            │
            ├── database                                    <-
            │   └── userDB.json                             <-
            │
            ├── models                                      <-
            │   ├── {championship}{1}_{model}.pkl           <-
            │   └── {championship}{2}_{model}.pkl           <-
            │
            └── mlflow                                      <-
                └── mlruns                                  <-
                    │
                    ├── {num}                               <-
                    │   │
                    │   └── {ref}                           <-
                    │       │
                    │       ├── artifacts
                    │       │   ├── confusion_matrix.png                <-
                    │       │   └── artifacts_{championship}_{model}    <-            
                    │       │       ├── conda.yaml                      <-
                    │       │       ├── input_example.json              <-
                    │       │       ├── MLmodel                         <-
                    │       │       ├── model.pkl                       <-
                    │       │       ├── python_env.yaml                 <-      
                    │       │       └── requirements.txt                <-
                    │       │
                    │       ├── metrics                     <-
                    │       │   ├── accuracy                <-
                    │       │   ├── f1                      <-
                    │       │   └── recall                  <-
                    │       │
                    │       ├── params                      <-
                    │       │   ├── C                       <-
                    │       │   ├── gamma                   <-
                    │       │   ├── kernel                  <-
                    │       │   └── probability             <-
                    │       │
                    │       ├── tags
                    │       │   ├── mlflow.log-model.history    <-
                    │       │   ├── mlflow.runName              <-
                    │       │   ├── mlflow.source.name          <-
                    │       │   ├── mlflow.source.type          <-
                    │       │   ├── mlflow.user                 <- 
                    │       │   └── quality                     <-
                    │       │
                    │       └── meta.yaml                   <-
                    │
                    └── models                              <-
                        │
                        ├── {championship}{1}_{model}       <-
                        │   │
                        │   ├── aliases                     <-
                        │   │   └── production              <-
                        │   │
                        │   ├── version-{1}                 <-
                        │   │   ├── tags                    <-
                        │   │   │   └── quality             <-
                        │   │   │
                        │   │   └── meta.yaml               <-
                        │   │           
                        │   └── meta.yaml                   <-
                        │
                        └── {championship}{2}_{model}       <-
                            │
                            ├── aliases                     <-
                            │   └── production              <-
                            │
                            ├── version-{1}                 <-
                            │   ├── tags                    <-
                            │   │   └── quality             <-
                            │   │
                            │   └── meta.yaml               <-
                            │           
                            └── meta.yaml                   <-







## 2. Prerequisites

### 2.1. Venv
- Create a venv according to requirements.txt or pyproject.toml (poetry)

The list below contains all the dependencies you need :
    Requirements  
    │
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



### 2.2 Copy & unzip files
- Get some file from my shared link :

    This link contains some files you have to unzip here if you want to execute all notebooks :
    - Unzip cleaned_dataset here : ../../storage/datas/imgs/**cleaned_dataset/**
    - Unzip bad_images.zip here : ../../storage/datas/imgs/**bad_images/**
    - Unzip models.zip : ../../storage/**models/**
    - Copy observations_mushroom.csv here : ../../storage/datas/csv/raw/**observations_mushroom.csv**


### 3. Genereate objects 
**Required**
- Generate tf datasets using notebook 6-data_preprocessing.ipynb
- Generate a model using 7-modelization.ipynb











# Start Streamlit
Execute script start_streamlit.sh (ajouter le dezip de datas csv source vers datas csv raw et generer les TF datasets)