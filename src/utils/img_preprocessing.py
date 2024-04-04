import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def get_features_target(df: pd.DataFrame, target_column_name: str):
    """
    Extracts the features and target from a pandas DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data.
        target_column_name (str): The name of the target column in the DataFrame.

    Returns:
        features & target series
    """
    features = df.drop(target_column_name, axis=1)
    target = df[target_column_name]

    return features, target


def img_preprocessing():
    """
    
    """
    def load_img():
        pass


    def resize_img():
        pass


    def normalize_img():
        pass


    def apply_augmentation():
        pass





def generate_dataset():
    """
    
    """





