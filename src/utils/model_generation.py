"""
This module contains functions to generate models for image classification using transfer learning.
"""


# LIB
import tensorflow as tf



# VARS



# FUNCTIONS
def generate_model(num_classes:int, input_shape:tuple=(224, 224, 3)) -> tf.keras.Model:
    """
    Generate a model for image classification using transfer learning.

    Args:
        pre_trained_model_url (str): The URL or path to the pre-trained model.
        num_classes (int): The number of classes in the classification task.
        input_shape (tuple, optional): The input shape of the images. Defaults to (224, 224, 3).

    Returns:
        tf.keras.Model: The generated model.

    """
    # Load base model
    base_model = tf.keras.applications.EfficientNetV2B0(
        include_top=False,
        weights='imagenet',
        input_shape=(224,224,3),
        include_preprocessing=False)

    base_model.trainable = False


    # Define model
    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1024)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(512)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(256)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    
    print("model summary: \n", model.summary())

    return model