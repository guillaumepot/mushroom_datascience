import os
import shutil
import logging
import pandas as pd


# "/home/guillaume/Téléchargements/mushroom-dataset/images"

# Get URL
img_url=input("Enter the path to the folder containing the images:")
destination_folder=input("Enter the path to the folder where you want to move the images:")
csv_url="../storage/datas/clean/full_cleaned_dataset.csv"

# Set logging file
logging.basicConfig(filename='sort_imgs.log', level=logging.INFO)


# load DF
df = pd.read_csv(csv_url)


# Create folders if they don't exist
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)
if not os.path.exists("sort_imgs.log"):
    with open("sort_imgs.log", 'w') as f:
        pass


# Sort files
files_not_found = []
for file in df['image_lien'].values:
    if os.path.isfile(img_url + "/" + file):
        shutil.move(img_url + "/" + file, destination_folder + "/" + file)
    else:
        logging.info(f"File not found: {file}")
        files_not_found.append(file)



# Remove files not found from DF and save
old_df_shape = df.shape[0]
df = df[~df['image_lien'].isin(files_not_found)]
new_df_shape = df.shape[0]
logging.info(f"\n\n Number of files not found: {len(files_not_found)} \n"\
             f"Old dataframe shape: {old_df_shape} \n"\
                f"New dataframe shape: {new_df_shape}")
df.to_csv(csv_url, index=False)