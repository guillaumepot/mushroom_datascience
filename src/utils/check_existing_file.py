import os
import pandas as pd

def check_file(img_directory:str, df:pd.DataFrame) -> None:
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