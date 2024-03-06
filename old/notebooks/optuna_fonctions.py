# v1.0


import optuna
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Sequential
from tensorflow.keras import layers
import tensorflow as tf


def create_model(trial):

    # Paramètres suggérés pour l'optimisation
    num_hidden_layers = trial.suggest_int('num_hidden_layers', 1, 10)
    num_hidden_units = trial.suggest_int('num_hidden_units', 64, 1024)
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3)




    # import du modèle efficientNetv2 pré-entrainé, les couches de convolution sont gelées
    base_model = tf.keras.applications.EfficientNetV2M(input_shape = (256,256,3),
                                                        include_top = False,
                                                        weights = 'imagenet')

    base_model.trainable = False



    # Construction du modèle optuna

    model = Sequential()
    model.add(base_model)
    model.add(layers.Flatten())

    for _ in range(num_hidden_layers):
        model.add(layers.Dropout(dropout_rate))
        model.add(layers.Dense(num_hidden_units, activation='relu'))

    # Couche de sortie avec 10 classes
    model.add(layers.Dense(10, activation='softmax'))

    # Compilation du modèle
    optimizer = Adam(learning_rate = learning_rate)
    model.compile(optimizer = optimizer, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

    return model




def objective(trial, dataset_train, dataset_val):


    # Créer le modèle
    model = create_model(trial)

    # Entraîner le modèle


    history = model.fit(dataset_train,
                    validation_data = dataset_val,
                    epochs = 3)
    



    # Récupérer la précision sur la validation finale
    final_val_accuracy = history.history['val_accuracy'][-1]

    # Mettre à jour l'étude Optuna avec la métrique de précision
    trial.report(final_val_accuracy, step=0)

    # Indiquer à Optuna si l'objectif a été atteint (maximisation de la précision)
    if trial.should_prune():
        raise optuna.TrialPruned()

    return final_val_accuracy