"""
This file is used to create a DF from a list of files in a directory.
"""

import os
import pandas as pd
import logging



# List all files
def main(file_name="file_list"):
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

    files = os.listdir(path)

    logging.info("Starting to list files...")
    time_start = pd.Timestamp.now()

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


if __name__ == '__main__':
    main()