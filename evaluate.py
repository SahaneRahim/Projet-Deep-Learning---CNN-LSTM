"""
Usage :
    python evaluate.py cnn
    python evaluate.py lstm
"""

import sys
import numpy as np
import tensorflow as tf
import pickle

from models.cnn_model import CustomCNN
from models.rnn_model import CustomLSTM
from utils.data_loader import (charger_cifar10, creer_datasets_cifar,
                                charger_meteo,   creer_datasets_lstm)
from utils.visualize   import afficher_matrice_confusion, afficher_predictions_lstm

CLASSES = ['avion', 'auto', 'oiseau', 'chat', 'cerf',
           'chien', 'grenouille', 'cheval', 'bateau', 'camion']

if len(sys.argv) < 2:
    print("Usage : python evaluate.py cnn   OU   python evaluate.py lstm")
    sys.exit(1)

mission = sys.argv[1]


#  MISSION 1 : CNN

if mission == 'cnn':
    print("=== Évaluation CNN ===")

    # 1. Données
    x_train, y_train, x_test, y_test = charger_cifar10()
    _, test_ds = creer_datasets_cifar(x_train, y_train, x_test, y_test)

    # 2. Charger le modèle sauvegardé
    modele = tf.keras.models.load_model('saved_models/meilleur_cnn.keras', custom_objects={'CustomCNN': CustomCNN})
    modele.summary()

    # 3. Accuracy globale
    loss, accuracy = modele.evaluate(test_ds)
    print(f"\nAccuracy : {accuracy * 100:.2f}%")

    # 4. Prédictions pour la matrice de confusion
    y_predit = []
    y_reel   = []

    for images, labels in test_ds:
        predictions = modele.predict(images, verbose=0)
        y_predit.extend(np.argmax(predictions, axis=1))
        y_reel.extend(labels.numpy().flatten())

    y_predit = np.array(y_predit)
    y_reel   = np.array(y_reel)

    # 5. Matrice de confusion
    afficher_matrice_confusion(y_reel, y_predit, CLASSES)


#  MISSION 2 : LSTM

elif mission == 'lstm':
    print("=== Évaluation LSTM ===")

    # 1. Données
    temperature, _ = charger_meteo()

    # 2. Scaler sauvegardé
    with open('saved_models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    # 3. Modèle sauvegardé
    modele = tf.keras.models.load_model('saved_models/meilleur_lstm.keras', custom_objects={'CustomLSTM': CustomLSTM})
    modele.summary()

    # 4. Dataset de test
    _, test_ds = creer_datasets_lstm(temperature)

    # 5. Prédictions
    y_predit = []
    y_reel   = []

    for sequences, labels in test_ds:
        predictions = modele.predict(sequences, verbose=0)
        y_predit.extend(predictions.flatten())
        y_reel.extend(labels.numpy().flatten())

    y_predit = np.array(y_predit)
    y_reel   = np.array(y_reel)

    # 6. Graphique
    afficher_predictions_lstm(y_reel, y_predit, scaler)


else:
    print(f"Mission inconnue : '{mission}'. Utilise 'cnn' ou 'lstm'.")