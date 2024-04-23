"""
...
"""


# LIB
import tensorflow as tf
import tensorflow_hub as hub


# Solve Keras conflict version
version_fn = getattr(tf.keras, "version", None)
if version_fn and version_fn().startswith("3."):
  import tf_keras as keras
else:
  keras = tf.keras

# VARS



# FUNCTIONS
def generate_model(pre_trained_model_url:str, num_classes:int, input_shape:tuple=(224, 224, 3)) -> tf.keras.Model:
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
    base_model = hub.KerasLayer(pre_trained_model_url,
                                trainable=False,
                                name="base_model_efficientnet",
                                input_shape=input_shape)


    # Define model
    model = keras.Sequential([
        base_model,
        keras.layers.Flatten(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation='softmax')
    ])

    print("model summary: \n", model.summary())

    return model