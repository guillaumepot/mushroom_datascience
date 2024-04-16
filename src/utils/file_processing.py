"""
This file contains somes functions used to process files.
Available functions :
- make_file_list_as_csv: List all files in a folder and save the list to a CSV.
- move_files_to_folder: Move files from a folder to another according to a CSV file.
- check_if_file_exists: Check if the files specified in a DataFrame exist in a given directory.
"""



# Lib
import os
import pandas as pd
import logging
import shutil



# Vars




# Functions
def make_file_list_as_csv(file_name="file_list"):
    """
    Main function to list files in a folder and save the file list to a CSV.

    Args:
        file_name (str): The name of the CSV file to be saved. Default is "file_list".

    Returns:
        None
    """
    path = input("Enter the path to the folder containing the files to list:")
    file_name = file_name + ".csv"
    csv_url = f"../../storage/datas/csv/clean/{file_name}.csv"

    # Set logging file
    logging.basicConfig(filename='../logs/file_list.log', level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
    logging.info("Starting to list files...")
    time_start = pd.Timestamp.now()


    # Start listing files
    files = os.listdir(path)

    for file in files:
        logging.info(f"Found: {file}")

    # Create a DataFrame
    df = pd.DataFrame(files, columns=['image_found'])


    # Sort by filename & reset index
    df['file_number'] = df['image_found'].apply(lambda x: int(x.split('.')[0]))
    df.sort_values(by='file_number', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.drop('file_number', axis=1, inplace=True)

    logging.info("\n**************\n")
    logging.info("Files listed successfully")
    time_end = pd.Timestamp.now()
    logging.info(f"Took {time_end - time_start} (H:m:s) to list {len(files)} files")

    # Save to csv
    logging.info("\n**************\n")
    logging.info(f"Saving CSV to: {csv_url}")
    df.to_csv(csv_url, index=False)
    logging.info("CSV saved successfully")




def move_files_to_folder(csv_url:str):
    """
    Move files from a specified folder to a destination folder based on a CSV file.

    Parameters:
    csv_url (str): The URL or file path of the CSV file containing the list of files to be moved.

    Returns:
    None
    """
    # Get URL
    img_url=input("Enter the path to the folder containing the images:")
    destination_folder=input("Enter the path to the folder where you want to move the images:")


    # Set logging file
    logging.basicConfig(filename='../logs/move_files_to_folder.log', level=logging.INFO)


    # load DF
    df = pd.read_csv(csv_url)


    # Create folders if they don't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    if not os.path.exists("move_files_to_folder.log"):
        with open("../logs/move_files_to_folder.log", 'w') as f:
            pass


    # Sort files
    files_not_found = []
    count=0
    for file in df['image_lien']:
        if os.path.isfile(img_url + "/" + file):
            # Move existing files
            shutil.move(img_url + "/" + file, destination_folder + "/" + file)
            count+=1
        else:
            # Log files not found
            logging.info(f"File not found: {file}")
            files_not_found.append(file)



    # Remove files not found from DF and save
    old_df_shape = df.shape[0]
    df = df[~df['image_lien'].isin(files_not_found)]
    new_df_shape = df.shape[0]
    logging.info(f"\n\n Number of files not found: {len(files_not_found)} \n"\
                f"Old dataframe shape: {old_df_shape} \n"\
                    f"New dataframe shape: {new_df_shape} \n"\
                        f"Files moved: {count}")
    df.to_csv(csv_url, index=False)



def check_if_file_exists(img_directory:str, df:pd.DataFrame) -> None:
    """
    Check if the files specified in the DataFrame exist in the given image directory.

    Args:
        img_directory (str): The directory where the image files are located.
        df (pd.DataFrame): The DataFrame containing the image file information.

    Returns:
        None

    """
    missing_files = []

    # For each row in the DataFrame
    for index, row in df.iterrows():
        img_path = os.path.join(img_directory, row['image_lien'])
        missing_file = row['image_lien']

        if not os.path.exists(img_path):
            missing_files.append(missing_file)

    # Print the missing files
    if missing_files:
        print(f"{len(missing_files)} files not found")
        print(f"Missing files: {missing_files}")

    # Or print that all files are found
    else:
        print("All files found!")


if __name__ == "__main__":
    print("Available functions: \n - make_file_list_as_csv \n - move_files_to_folder \n - check_if_file_exists")
    function_to_run = input("Enter the function to run: ")
    if function_to_run == "make_file_list_as_csv":
        make_file_list_as_csv()
    elif function_to_run == "move_files_to_folder":
        csv_url = input("Enter the URL of the CSV file: ")
        move_files_to_folder(csv_url)
    elif function_to_run == "check_if_file_exists":
        img_directory = input("Enter the path to the image directory: ")
        csv_url = input("Enter the URL of the CSV file: ")
        df = pd.read_csv(csv_url)
        check_if_file_exists(img_directory, df)
    else:
        print("Function not found")