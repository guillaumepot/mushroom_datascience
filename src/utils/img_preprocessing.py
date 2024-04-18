"""
This file contains functions for image preprocessing and dataset generation
Available functions:
- get_features_target
- img_preprocessing
- generate_dataset
"""

# LIB
import pandas as pd
import json
import os
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# FUNCTIONS
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




def img_preprocessing(img_url:str = None, dimensions:tuple = (224,224), resize:bool = False, augment:bool = False, normalize:bool = False,
                      flip_left_right:bool = False, flip_up_down:bool = False, brightness:bool = False, contrast:bool = False, saturation:bool = False) -> tf.Tensor:
                      
    """
    Preprocesses an image based on the specified parameters.

    Args:
        img_url (str): The URL or file path of the image.
        dimensions (tuple, optional): The desired dimensions of the image after resizing. Defaults to (224, 224).
        resize (bool, optional): Whether to resize the image. Defaults to False.
        augment (bool, optional): Whether to apply data augmentation to the image. Defaults to False.
        normalize (bool, optional): Whether to normalize the image. Defaults to False.
        flip_left_right (bool, optional): Whether to randomly flip the image horizontally. Defaults to False.
        flip_up_down (bool, optional): Whether to randomly flip the image vertically. Defaults to False.
        brightness (bool, optional): Whether to randomly adjust the brightness of the image. Defaults to False.
        contrast (bool, optional): Whether to randomly adjust the contrast of the image. Defaults to False.
        saturation (bool, optional): Whether to randomly adjust the saturation of the image. Defaults to False.

    Returns:
        tf.Tensor: The preprocessed image as a TensorFlow tensor.
    """

    # Functions
    def load_img(img_url:str):
        img = tf.io.read_file(img_url)
        img = tf.image.decode_png(img, channels=3)
        return img


    def resize_img(img, dimensions:tuple):
        img = tf.image.resize(img, dimensions)
        return img


    def apply_augmentation(img, flip_left_right:bool = False, flip_up_down:bool = False, brightness:bool = False, contrast:bool = False, saturation:bool = False):
        if flip_left_right:
            img = tf.image.random_flip_left_right(img)
        if flip_up_down:
            img = tf.image.random_flip_up_down(img)
        if brightness:
            img = tf.image.random_brightness(img, max_delta=0.2)
        if contrast:
            img = tf.image.random_contrast(img, lower=0.7, upper=1.3)
        if saturation:
            img = tf.image.random_saturation(img, lower = 2, upper = 10)
        return img


    def normalize_img(img):
        img = img / 255.0
        return img


    # Apply functions
    img = load_img(img_url)
    img = tf.clip_by_value(img, 0, 255)


    if resize:
        img = resize_img(img, dimensions)

    if augment:
        img = apply_augmentation(img, flip_left_right, flip_up_down, brightness, contrast, saturation)

    if normalize:
        img = normalize_img(img)


    return img




def generate_dataset(target_to_encode: pd.Series, features_to_split: pd.DataFrame,
                     train_size: float = 0.8, test_size = 0.5, random_state: int = 10,
                     img_dimensions:tuple = (224,224), resize = False,
                     augment = False, normalize = False,
                     flip_left_right = False,
                     flip_up_down = False,
                     brightness = False,
                     contrast = False,
                     saturation = False,
                     batch_size = 32):
    """
    Generate train, validation, and test datasets for image preprocessing.

    Parameters:
    - target_to_encode (pd.Series): The target variable to encode.
    - features_to_split (pd.DataFrame): The features to split into train, validation, and test sets.
    - train_size (float): The proportion of the dataset to include in the train split (default: 0.8).
    - test_size (float): The proportion of the dataset to include in the test split (default: 0.5).
    - random_state (int): The random seed for reproducibility (default: 10).
    - img_dimensions (tuple): The desired dimensions of the images (default: (224, 224)).
    - resize (bool): Whether to resize the images (default: False).
    - augment (bool): Whether to apply data augmentation to the images (default: False).
    - normalize (bool): Whether to normalize the images (default: False).
    - flip_left_right (bool): Whether to flip the images horizontally (default: False).
    - flip_up_down (bool): Whether to flip the images vertically (default: False).
    - brightness (bool): Whether to adjust the brightness of the images (default: False).
    - contrast (bool): Whether to adjust the contrast of the images (default: False).
    - saturation (bool): Whether to adjust the saturation of the images (default: False).
    - batch_size (int): The batch size for the generated datasets (default: 32).

    Returns:
    - train_dataset (tf.data.Dataset): The generated train dataset.
    - val_dataset (tf.data.Dataset): The generated validation dataset.
    - test_dataset (tf.data.Dataset): The generated test dataset.
    """

    # Functions
    def encode_target(target_to_encode: pd.Series):
        label_encoder = LabelEncoder()
        encoded_label_set = label_encoder.fit_transform(target_to_encode)
        print(f"Encoded target: {encoded_label_set}")
        encoded_label_set = to_categorical(encoded_label_set)
        
        label_mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
        print(f"Label mapping: {label_mapping}")


        # Save label_mapping
        json_path = '../../storage/datas/json/encoded_labels.json'
        if os.path.exists(json_path) and os.path.getsize(json_path) > 0:
            with open(json_path, 'r') as f:
                existing_data = json.load(f)
        else:
            existing_data = {}

        existing_data.update(label_mapping)

        with open(json_path, 'w') as f:
            json.dump(existing_data, f)

        return encoded_label_set




    def split_features_target(features_to_split: pd.DataFrame, encoded_label_set: pd.Series):
        X_train, X_temp, y_train, y_temp = train_test_split(features_to_split, encoded_label_set, train_size = train_size, random_state = random_state)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size = test_size, random_state = random_state)
        print("Train, validation & test splits created successfully.")
        print(f"\n Current shapes: \n X_train: {X_train.shape} \n X_val: {X_val.shape} \n X_test: {X_test.shape} ")
        return X_train, X_val, X_test, y_train, y_val, y_test



    def generate_tf_dataset(set):
        if set not in ['train', 'validation', 'test']:
            raise ValueError("Invalid value for 'set'. Please use 'train', 'validation' or 'test'.")

        if set == 'train':
            # Generate Train Dataset
            train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
                # Apply image preprocessing function
            train_dataset = train_dataset.map(lambda x,y: [img_preprocessing(img_url = x,
                                                                            dimensions = img_dimensions,
                                                                            resize = resize,
                                                                            augment = augment,
                                                                            normalize = normalize,
                                                                            flip_left_right = flip_left_right,
                                                                            flip_up_down = flip_up_down,
                                                                            brightness = brightness,
                                                                            contrast = contrast,
                                                                            saturation = saturation),y], # End of function img_preprocessing
                                                                            num_parallel_calls=-1)

            train_dataset = train_dataset.cache()
            train_dataset = train_dataset.shuffle(buffer_size = train_dataset.cardinality())        
            train_dataset = train_dataset.batch(batch_size)

            print(f"Train dataset generated successfully. Number of batches: {train_dataset.cardinality()}")

            return train_dataset
        
        
        
        if set == 'validation':
            val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
                # Apply image preprocessing function
            val_dataset = val_dataset.map(lambda x,y: [img_preprocessing(img_url = x,
                                                                            dimensions = img_dimensions,
                                                                            resize = resize,
                                                                            augment = False,
                                                                            normalize = normalize,
                                                                            flip_left_right = False,
                                                                            flip_up_down = False,
                                                                            brightness = False,
                                                                            contrast = False,
                                                                            saturation = False),y], # End of function img_preprocessing
                                                                            num_parallel_calls=-1)
            
            val_dataset = val_dataset.batch(batch_size)

            print(f"Validation dataset generated successfully. Number of batches: {val_dataset.cardinality()}")

            return val_dataset
        

        if set == 'test':
            # Generate Test Dataset
            test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
                # Apply image preprocessing function
            test_dataset = test_dataset.map(lambda x,y: [img_preprocessing(img_url = x,
                                                                            dimensions = img_dimensions,
                                                                            resize = resize,
                                                                            augment = False,
                                                                            normalize = normalize,
                                                                            flip_left_right = False,
                                                                            flip_up_down = False,
                                                                            brightness = False,
                                                                            contrast = False,
                                                                            saturation = False),y], # End of function img_preprocessing
                                                                            num_parallel_calls=-1)

            test_dataset = test_dataset.batch(batch_size)

            print(f"Test dataset generated successfully. Number of batches: {test_dataset.cardinality()}")

            return test_dataset
        
    # End of function generate_tf_dataset
        
        
    ##############################################
        

    # Encode target
    encoded_label_set = encode_target(target_to_encode)
    

    # Get train, test & validation datasets
    X_train, X_val, X_test, y_train, y_val, y_test = split_features_target(features_to_split, encoded_label_set)



    # Generate Train Dataset
    train_dataset = generate_tf_dataset('train')

    # Generate Validation Dataset
    val_dataset = generate_tf_dataset('validation')

    # Generate Test Dataset
    test_dataset = generate_tf_dataset('test')

    return train_dataset, val_dataset, test_dataset