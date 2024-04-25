"""
This file contains somes functions used to process files.
Available functions :
- unzip_files: Unzips the files from the given source path and extracts them to the specified destination path.
- make_file_list_as_csv: List all files in a folder and save the list to a CSV.
- move_files_to_folder: Move files from a folder to another according to a CSV file.
- check_if_file_exists: Check if the files specified in a DataFrame exist in a given directory.
"""



# Lib
import os
import pandas as pd
import logging
import shutil
import time
import zipfile


# Vars




# Functions
def unzip_files(source = None, destination = None) -> None:
    """
    Unzips the files from the given source path and extracts them to the specified destination path.

    Args:
        source (str): The path of the zip file to be extracted.
        destination (str): The path where the extracted files will be saved.

    Returns:
        None

    """
    if source is None or destination is None:
        source = input("Enter the path of the zip file to be extracted:")
        destination = input("Enter the path where the extracted files will be saved:")


    start_time = time.time()

    with zipfile.ZipFile(source, 'r') as zip_ref:
        zip_ref.extractall(destination)

    end_time = time.time()

    execution_time = (end_time - start_time)/60

    print(f"execution time: {execution_time} minutes")



def check_if_file_exists(img_directory: str, csv_url: str, column: str, auto_clean_csv=False) -> None:
    """
    Check if files specified in a CSV exist in the given image directory.

    Args:
        img_directory (str): The directory where the image files are located.
        csv_url (str): The URL or file path of the CSV file containing the file names.
        column (str): The name of the column in the CSV file that contains the file names.
        auto_clean_csv (bool, optional): Whether to automatically clean the CSV file by removing rows with missing files.
            Defaults to False.

    Returns:
        None

    """
    # Set logging file
    logging.basicConfig(filename='../logs/check_if_file_exists.log', level=logging.INFO)

    # Read CSV
    df = pd.read_csv(csv_url)

    # Create a list to store missing files
    missing_files = []
    start_time = time.time()

    # For each row in the DataFrame
    for index, row in df.iterrows():
        img_path = os.path.join(img_directory, row[column])
        file = row[column]

        if not os.path.exists(img_path):
            missing_files.append(file)
            logging.info(f"File not found: {file}")

    end_time = time.time()
    execution_time = (end_time - start_time) / 60

    # Log the missing files total
    if missing_files:
        logging.info(f"\n {len(missing_files)} files not found")
    else:
        logging.info("All files found!")

    logging.info(f"Execution time: {execution_time} minutes")

    # Clean CSV
    if auto_clean_csv:
        df = df[~df[column].isin(missing_files)]
        df.to_csv(csv_url, index=False)
        logging.info("CSV cleaned successfully")



def copy_files_to_folder(csv_url:str, column:str, source:str = None, destination:str = None, auto_clean_csv = False) -> None:
    """
    Copy files from a source folder to a destination folder based on a CSV file.

    Args:
        csv_url (str): The URL or file path of the CSV file containing the list of files to be copied.
        column (str): The name of the column in the CSV file that contains the file names.
        source (str, optional): The path to the folder containing the source files. If not provided, the user will be prompted to enter it.
        destination (str, optional): The path to the folder where the files will be copied. If not provided, the user will be prompted to enter it.
        auto_clean_csv (bool, optional): Whether to automatically clean the CSV file by removing entries for files that were not found in the source folder. Default is False.

    Returns:
        None

    Raises:
        None

    """
    # Get URL
    if source is None or destination is None:
        source=input("Enter the path to the folder containing the images:")
        destination=input("Enter the path to the folder where you want to move the images:")


    # Set logging file
    logging.basicConfig(filename='../logs/move_files_to_folder.log', level=logging.INFO)


    # load DF
    df = pd.read_csv(csv_url)



    start_time = time.time()


    # Create folders if they don't exist
    if not os.path.exists(destination):
        os.makedirs(destination)
    if not os.path.exists("move_files_to_folder.log"):
        with open("../logs/move_files_to_folder.log", 'w') as f:
            pass


    # Sort files
    files_not_found = []
    count=0
    for file in df[column]:
        if os.path.isfile(source + "/" + file):
            # Move existing files
            shutil.copy(source + "/" + file, destination + "/" + file)
            count+=1
        else:
            # Log files not found
            logging.info(f"File not found: {file}")
            files_not_found.append(file)


    end_time = time.time()
    execution_time = (end_time - start_time) / 60
    logging.info(f"Execution time: {execution_time} minutes")


    # Clean CSV
    if auto_clean_csv:
        # Remove files not found from DF and save
        old_df_shape = df.shape[0]
        df = df[~df['image_lien'].isin(files_not_found)]
        new_df_shape = df.shape[0]
        logging.info(f"\n\n Number of files not found: {len(files_not_found)} \n"\
                    f"Old dataframe shape: {old_df_shape} \n"\
                        f"New dataframe shape: {new_df_shape} \n"\
                            f"Files moved: {count}")
        df.to_csv(csv_url, index=False)



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
    csv_url = f"../../storage/datas/csv/clean/{file_name}"

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



if __name__ == "__main__":
    print("Available functions: \n - unzip_files \n - make_file_list_as_csv \n - move_files_to_folder \n - check_if_file_exists")
    function_to_run = input("Enter the function to run: ")
    if function_to_run == "unzip_files":
        unzip_files()
    elif function_to_run == "make_file_list_as_csv":
        make_file_list_as_csv()
    elif function_to_run == "move_files_to_folder":
        csv_url = input("Enter the URL of the CSV file: ")
        copy_files_to_folder(csv_url)
    elif function_to_run == "check_if_file_exists":
        img_directory = input("Enter the path to the image directory: ")
        csv_url = input("Enter the URL of the CSV file: ")
        df = pd.read_csv(csv_url)
        check_if_file_exists(img_directory, df)
    else:
        print("Function not found")