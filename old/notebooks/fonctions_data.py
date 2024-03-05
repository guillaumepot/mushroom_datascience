# Version 1.5 finale

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2







# -------------------------------------------------------------------------------------------------
# Fonction de controle de la présence des fichiers images
# -------------------------------------------------------------------------------------------------
def controle_presence_fichiers(df, chemin_images):

    '''
    Controle que les fichiers images soient bien présents sur le disque.
        - df : DataFrame contenant les url des fichiers images
        - chemin_images : Variable du DF contenant les url
    '''

    image_directory = chemin_images
    missing_files = []

    # Parcourir chaque ligne du DataFrame
    for index, row in df.iterrows():
        image_path = os.path.join(image_directory, row['image_lien'])

        if not os.path.exists(image_path):
            missing_files.append(image_path)

    # Afficher les fichiers non trouvés
    if missing_files:
        print("\nFichiers non trouvés :")
        for file_path in missing_files:
            print(file_path)

    # Ou préciser que tous les fichiers sont présents
    else:
        print("\nTous les fichiers sont présents.")





# -------------------------------------------------------------------------------------------------
# Fonction d'undersampling des observations
# -------------------------------------------------------------------------------------------------
def undersampling_df(df, col):

    '''
    Undersample le df donné pour équilibrer le nombre d'observations par classe.
        - df : df à undersampler
        - col : colonne concernée par le GroupBy pour générer l'undersampling
    '''

    compte = df.groupby(col).count()
    min_samples = compte['image_url'].min()
    min_samples = int(min_samples)

    df_undersample = pd.DataFrame()

    for label, group in df.groupby('label'):
        df_undersample = pd.concat([df_undersample, group.sample(min_samples, replace=True)])
        df_undersample = df_undersample.reset_index(drop=True)

    return df_undersample


# -------------------------------------------------------------------------------------------------
# Fonction de tirage aléatoires d'images selon un DataFrame
# -------------------------------------------------------------------------------------------------

def tirage_aleatoire(df, dataframe_col_label, dataframe_col_url, nb_tirages=5):

    '''
    Créé une figure composée d'images tirées aléatoirement d'un dataframe.
        - df : DataFrame source
        - dataframe_col_label : Colonne contenant les labels des images
        - dataframe_col_url : Colonne contenant les URL des images
        - nb_tirages : Nombre de tirages à effectuer
    '''

    plt.figure(figsize=(12,12))
    indice_aleatoire = np.random.choice(len(df), size=nb_tirages, replace=False)
    subplot_colonnes = nb_tirages
    subplot_lignes = (nb_tirages + subplot_colonnes - 1) // subplot_colonnes

    for i, j in enumerate(indice_aleatoire):
        plt.subplot(subplot_lignes,subplot_colonnes,i+1)
        plt.subplots_adjust(wspace=0.8, hspace=0.2)               # Eviter que les subplots se retrouvent trop proches
        img = plt.imread(dataframe_col_url[j])                    # Lecture de l'image
        height, width, _ = img.shape                              # Lecture des dimensions de l'image
        plt.axis('off')                                           # Retrait des axes
        plt.imshow(img)                                           # Affichage de l'image
        plt.title(f"{dataframe_col_label[j]}\n{width}x{height}")  # Titre de l'image -> Nom d'espèce et dimensions


# -------------------------------------------------------------------------------------------------
# Fonction de tirage aléatoires d'images selon un DataFrame
# -------------------------------------------------------------------------------------------------

def extract_features(url_img):

    '''
    Extrait les features des images, les renvoient sous forme d'un DataFrame contenant les largeurs, hauteurs, et moyennes RGB
        - url_img : Chemin des images
    '''

    img = cv2.imread(url_img)
    hauteur, largeur, canal = img.shape
    features = {
        'largeur': largeur,
        'hauteur': hauteur,
        'moyenne_rouge': np.mean(img[:,:,2]),
        'moyenne_vert': np.mean(img[:,:,1]),
        'moyenne_bleu': np.mean(img[:,:,0])}
    
    return features