"""
Usage :
    python train.py cnn
    python train.py lstm
"""

import sys
import os
import pickle
import tensorflow as tf

from models.cnn_model  import CustomCNN
from models.rnn_model  import CustomLSTM
from utils.data_loader import (charger_cifar10, creer_datasets_cifar,
                                charger_meteo,   creer_datasets_lstm)
from utils.visualize   import afficher_historique

os.makedirs('saved_models', exist_ok=True)

if len(sys.argv) < 2:
    print("Usage : python train.py cnn   OU   python train.py lstm")
    sys.exit(1)

mission = sys.argv[1]


#  MISSION 1 : CNN

if mission == 'cnn':
    print("=== Entraînement CNN — CIFAR-10 ===")

    # 1. Données
    x_train, y_train, x_test, y_test = charger_cifar10()
    train_ds, test_ds = creer_datasets_cifar(x_train, y_train, x_test, y_test)

    # 2. Modèle
    modele = CustomCNN()

    # 3. Compilation
    modele.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # 4. Callbacks
    arret_anticipe = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=8,
        restore_best_weights=True
    )

    sauvegarde = tf.keras.callbacks.ModelCheckpoint(
        filepath='saved_models/meilleur_cnn.keras',
        monitor='val_accuracy',
        save_best_only=True
    )

    reduire_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        verbose=1
    )

    # 5. Entraînement
    historique = modele.fit(
        train_ds,
        epochs=80,
        validation_data=test_ds,
        callbacks=[arret_anticipe, sauvegarde, reduire_lr]
    )

    # 6. Résultat
    loss, accuracy = modele.evaluate(test_ds, verbose=0)
    print(f"\nAccuracy finale : {accuracy * 100:.2f}%")

    # 7. Graphiques
    afficher_historique(historique, "cnn_historique.png")

    # 8. Sauvegarde
    modele.save('saved_models/cnn_final.keras')
    print("Modèle CNN sauvegardé !")


#  MISSION 2 : LSTM

elif mission == 'lstm':
    print("=== Entraînement LSTM — Météo ===")

    # 1. Données
    temperature, scaler = charger_meteo()
    train_ds, test_ds = creer_datasets_lstm(temperature)

    # 2. Modèle
    modele = CustomLSTM()

    # 3. Compilation
    modele.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )

    # 4. Callbacks
    arret_anticipe = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    sauvegarde = tf.keras.callbacks.ModelCheckpoint(
        filepath='saved_models/meilleur_lstm.keras',
        monitor='val_loss',
        save_best_only=True
    )

    # 5. Entraînement
    historique = modele.fit(
        train_ds,
        epochs=30,
        validation_data=test_ds,
        callbacks=[arret_anticipe, sauvegarde]
    )

    # 6. Graphiques
    afficher_historique(historique, "lstm_historique.png")

    # 7. Sauvegarde
    modele.save('saved_models/lstm_final.keras')
    with open('saved_models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    print("Modèle LSTM sauvegardé !")


else:
    print(f"Mission inconnue : '{mission}'. Utilise 'cnn' ou 'lstm'.")