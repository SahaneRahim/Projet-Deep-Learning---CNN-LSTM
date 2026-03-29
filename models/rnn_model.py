import tensorflow as tf

# Modèle LSTM pour prédire la prochaine valeur d'une série temporelle
# Entrée : une fenêtre de 48 heures de températures
# Sortie : la température à l'heure suivante (T+1)

class CustomLSTM(tf.keras.Model):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # 1ère couche LSTM : return_sequences=True car une 2ème LSTM suit
        self.lstm1 = tf.keras.layers.LSTM(64, return_sequences=True)

        # 2ème couche LSTM : return_sequences=False → on prend juste le dernier état
        self.lstm2 = tf.keras.layers.LSTM(32, return_sequences=False)

        self.dropout = tf.keras.layers.Dropout(0.2)

        # Couche de sortie : prédit 1 seule valeur (la température T+1)
        self.sortie = tf.keras.layers.Dense(1)

    def call(self, x, training=False):
        x = self.lstm1(x)
        x = self.lstm2(x)
        x = self.dropout(x, training=training)
        x = self.sortie(x)
        return x
