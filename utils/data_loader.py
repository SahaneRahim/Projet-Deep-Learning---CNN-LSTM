import os
import zipfile
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# CIFAR-10

def charger_cifar10():
    # Keras télécharge CIFAR-10 automatiquement
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Normalisation : pixels de [0, 255] → [0.0, 1.0]
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32")  / 255.0

    print("Données CIFAR-10 chargées !")
    print(f"  Entraînement : {x_train.shape}")
    print(f"  Test         : {x_test.shape}")

    return x_train, y_train, x_test, y_test


def creer_datasets_cifar(x_train, y_train, x_test, y_test, taille_batch=64):
    # On crée des tf.data.Dataset pour l'entraînement
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(10000).batch(taille_batch).prefetch(tf.data.AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_ds = test_ds.batch(taille_batch).prefetch(tf.data.AUTOTUNE)

    return train_ds, test_ds


# MÉTÉO (LSTM)
 
def charger_meteo():
    # 1. Télécharger le zip (sans extraction automatique)
    zip_path = tf.keras.utils.get_file(
        origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
        fname='jena_climate_2009_2016.csv.zip',
        extract=False
    )
 
    # 2. Extraire manuellement le CSV
    dossier  = os.path.dirname(zip_path)
    csv_path = os.path.join(dossier, 'jena_climate_2009_2016.csv')
 
    if not os.path.exists(csv_path):
        print("Extraction du fichier zip...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dossier)
        print("Extraction terminée !")
 
    # 3. Lire le CSV
    df = pd.read_csv(csv_path)
 
    # On garde la température, 1 mesure par heure (1 ligne sur 6)
    temperature = df['T (degC)'].values[::6].reshape(-1, 1)
 
    print(f"Températures chargées : {len(temperature)} points")
 
    # 4. Normalisation entre 0 et 1
    scaler = MinMaxScaler()
    temperature_normalisee = scaler.fit_transform(temperature)
 
    return temperature_normalisee, scaler
 
 
# LSTM DATASET 
 
def creer_datasets_lstm(data, longueur_sequence=48, taille_batch=32, ratio_train=0.8):
    dataset = tf.keras.utils.timeseries_dataset_from_array(
        data=data[:-1],
        targets=data[longueur_sequence:],
        sequence_length=longueur_sequence,
        batch_size=taille_batch,
        shuffle=False
    )
 
    total            = len(data) - longueur_sequence
    taille_train     = int(total * ratio_train)
    nb_batches_train = taille_train // taille_batch
 
    train_ds = dataset.take(nb_batches_train)
    test_ds  = dataset.skip(nb_batches_train)
 
    print(f"Batches entraînement : {nb_batches_train}")
 
    return train_ds, test_ds