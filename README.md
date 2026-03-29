# Projet Deep Learning — CNN & LSTM (ENSP Douala)

Ce dépôt contient l'implémentation de deux modèles avancés de Deep Learning. Le projet a pour but de résoudre deux types de problèmes fondamentaux d'Intelligence Artificielle de manière claire, structurée et suivant les meilleures pratiques de programmation orientée objet sous TensorFlow/Keras.

## 🎯 Les deux missions du projet

1. **Mission CNN (Image Classification)** : Un modèle Convolutional Neural Network (CNN) spécialisé dans la classification de l'ensemble de données d'images **CIFAR-10**. Il inclut une architecture par blocs avec Normalisation par Lots (Batch Normalization), Dropout pour réduire l'overfitting, et augmentation de données (Data Augmentation).
2. **Mission LSTM (Time-Series Forecasting)** : Un modèle Récurrent (RNN/LSTM) pour prédire une valeur continue dans les séries temporelles. Il analyse les 48 dernières heures du dataset de la météo (températures) de **Jena Climate** pour prédire l'heure suivante.

---

## 📁 Structure du Projet

```text
📂 PROJECT_ML_2_ENSPD/
├── 📂 models/
│   ├── cnn_model.py       # Architecture du modèle CNN (Custom subclassed Model)
│   └── rnn_model.py       # Architecture du modèle LSTM
├── 📂 utils/
│   ├── data_loader.py     # Script de téléchargement, extraction et formatage des datasets
│   └── visualize.py       # Fonctions générant les graphiques et visualisations (Loss, Accuracy, Matrice...)
├── 📂 saved_models/       # Modèles entraînés (.keras) & filtres/scalers (.pkl) sauvegardés
├── train.py               # Script principal d'entraînement (pour CNN ou LSTM)
├── evaluate.py            # Script principal d'évaluation (pour CNN ou LSTM)
└── requirements.txt       # Liste des dépendances pip
```

---

## ⚙️ Prérequis & Installation

Avant de manipuler le projet, assurez-vous de posséder `Python 3.x` et d'initialiser les modules nécessaires :

Dans votre terminal, exécutez simplement :
```bash
pip install -r requirements.txt
```
*(Ceci installera TensorFlow, Numpy, Pandas, Scikit-learn, Matplotlib et Seaborn).*

---

## 🚀 Comment exécuter le projet ?

Ce projet a été architecturé pour être utilisé depuis deux points d'entrée uniques : `train.py` (pour apprendre) et `evaluate.py` (pour tester).

Ici se trouve l'ordre exact et les commandes à taper pour lancer vos scripts.

### 1) Entraîner les modèles (Training)
Toujours procéder à l'entraînement **AVANT** d'essayer d'évaluer le projet afin de générer un modèle dans le dossier `saved_models/`. 
Le processus intègre déjà des fonctions d'*Early Stopping* (pour éviter le surapprentissage) et de sauvegarde automatique du meilleur modèle.

**Pour entraîner le CNN (Classification d'Image) :**
```bash
python train.py cnn
```
**Pour entraîner le LSTM (Série temporelle - Météo) :**
```bash
python train.py lstm
```

---

### 2) Évaluer les modèles (Evaluation)
Une fois vos modèles entièrement entraînés et sauvegardés, vous pouvez visualiser leurs performances sur un jeu de données "jamais vus" (Test Set).

**Pour tester le CNN :**
```bash
python evaluate.py cnn
```
> *Génère une Matrice de Confusion montrant les réussites et échecs sur chaque classe (Avion, Chat, Camion...).*

**Pour tester le LSTM :**
```bash
python evaluate.py lstm
```
> *Génère un graphique superposant les prédictions du modèle sur la vraie courbe de Températures (Thermomètre virtuel).*
