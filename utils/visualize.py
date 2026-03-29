import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns


def afficher_historique(history, nom_fichier="historique.png"):
    # Affiche les courbes Loss et Accuracy pendant l'entraînement
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Courbe de perte
    axes[0].plot(history.history['loss'],     label='Train')
    axes[0].plot(history.history['val_loss'], label='Validation')
    axes[0].set_title('Loss (Perte)')
    axes[0].set_xlabel('Époque')
    axes[0].legend()
    axes[0].grid(True)

    # Courbe de précision (seulement pour la classification)
    if 'accuracy' in history.history:
        axes[1].plot(history.history['accuracy'],     label='Train')
        axes[1].plot(history.history['val_accuracy'], label='Validation')
        axes[1].set_title('Accuracy (Précision)')
        axes[1].set_xlabel('Époque')
        axes[1].legend()
        axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(nom_fichier, dpi=150)
    plt.show()
    print(f"Graphique sauvegardé → {nom_fichier}")


def afficher_matrice_confusion(y_reel, y_predit, classes, nom_fichier="confusion.png"):
    cm = confusion_matrix(y_reel, y_predit)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes,
                yticklabels=classes)
    plt.title('Matrice de Confusion — CNN')
    plt.ylabel('Vraie classe')
    plt.xlabel('Classe prédite')
    plt.tight_layout()
    plt.savefig(nom_fichier, dpi=150)
    plt.show()
    print(f"Matrice sauvegardée → {nom_fichier}")


def afficher_predictions_lstm(y_reel, y_predit, scaler, nom_fichier="predictions.png"):
    # On repasse aux vraies températures (annuler la normalisation)
    y_reel_reel   = scaler.inverse_transform(y_reel.reshape(-1, 1)).flatten()
    y_predit_reel = scaler.inverse_transform(y_predit.reshape(-1, 1)).flatten()

    plt.figure(figsize=(14, 6))
    plt.plot(y_reel_reel,   label='Températures réelles', color='blue')
    plt.plot(y_predit_reel, label='Prédictions LSTM',     color='red', alpha=0.8)
    plt.title('Prédiction de Température — LSTM')
    plt.xlabel('Heures')
    plt.ylabel('Température (°C)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(nom_fichier, dpi=150)
    plt.show()
    print(f"Graphique sauvegardé → {nom_fichier}")
