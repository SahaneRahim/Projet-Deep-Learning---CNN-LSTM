import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns


def afficher_historique(history, nom_fichier="historique.png"):
    # history est l'objet retourné par model.fit()
    # Il contient les valeurs de loss et accuracy à chaque époque

    # On vérifie si 'accuracy' existe dans l'historique
    # → True pour le CNN (classification), False pour le LSTM (régression)
    a_accuracy = 'accuracy' in history.history

    # Si CNN → 2 graphiques côte à côte (loss + accuracy)
    # Si LSTM → 1 seul graphique (loss seulement, pas d'accuracy en régression)
    if a_accuracy:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(8, 5))
        axes = [axes]  # on met dans une liste pour pouvoir faire axes[0] dans les deux cas

    # ── Graphique 1 : courbe de perte (Loss) ──────────────────
    axes[0].plot(history.history['loss'],     label='Train')       # perte sur les données d'entraînement
    axes[0].plot(history.history['val_loss'], label='Validation')  # perte sur les données de validation
    axes[0].set_title('Loss (Perte)')
    axes[0].set_xlabel('Époque')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    # ── Graphique 2 : courbe de précision (Accuracy) ──────────
    # Uniquement pour le CNN (pas pour le LSTM)
    if a_accuracy:
        axes[1].plot(history.history['accuracy'],     label='Train')
        axes[1].plot(history.history['val_accuracy'], label='Validation')
        axes[1].set_title('Accuracy (Précision)')
        axes[1].set_xlabel('Époque')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(nom_fichier, dpi=150)  # sauvegarde l'image en haute résolution
    plt.close()                         # ferme la fenêtre sans bloquer l'exécution
    print(f"Graphique sauvegardé → {nom_fichier}")


def afficher_matrice_confusion(y_reel, y_predit, classes, nom_fichier="confusion.png"):
    # La matrice de confusion montre combien de fois chaque classe
    # a été correctement ou incorrectement prédite
    # Les valeurs sur la diagonale = prédictions correctes
    cm = confusion_matrix(y_reel, y_predit)

    plt.figure(figsize=(10, 8))
    # heatmap = tableau coloré où les couleurs représentent les valeurs
    # annot=True  : affiche les chiffres dans les cases
    # fmt='d'     : format entier (pas de décimales)
    # cmap='Blues': palette de couleurs bleues
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes,
                yticklabels=classes)
    plt.title('Matrice de Confusion — CNN')
    plt.ylabel('Vraie classe')
    plt.xlabel('Classe prédite')
    plt.tight_layout()
    plt.savefig(nom_fichier, dpi=150)
    plt.close()
    print(f"Matrice sauvegardée → {nom_fichier}")


def afficher_predictions_lstm(y_reel, y_predit, scaler, nom_fichier="predictions.png"):
    # Les valeurs sont normalisées (entre 0 et 1)
    # On utilise inverse_transform pour repasser aux vraies températures en °C
    y_reel_reel   = scaler.inverse_transform(y_reel.reshape(-1, 1)).flatten()
    y_predit_reel = scaler.inverse_transform(y_predit.reshape(-1, 1)).flatten()

    # On superpose les deux courbes pour comparer visuellement
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
    plt.close()
    print(f"Graphique sauvegardé → {nom_fichier}")