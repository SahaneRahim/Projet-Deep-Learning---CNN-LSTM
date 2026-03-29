import tensorflow as tf

# On crée notre CNN en héritant de tf.keras.Model
# C'est ce qu'on appelle l'API Subclassing

class CustomCNN(tf.keras.Model):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Augmentation des données (actif seulement pendant l'entraînement)
        self.augment = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.15),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
        ])

        # Bloc 1 : détecte les formes simples (bords, coins...)
        self.conv1a = tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu')
        self.conv1b = tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu')
        self.bn1    = tf.keras.layers.BatchNormalization()
        self.pool1  = tf.keras.layers.MaxPooling2D((2,2))

        # Bloc 2 : détecte des formes plus complexes
        self.conv2a = tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu')
        self.conv2b = tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu')
        self.bn2    = tf.keras.layers.BatchNormalization()
        self.pool2  = tf.keras.layers.MaxPooling2D((2,2))

        # Bloc 3 : détecte des formes encore plus complexes
        self.conv3a = tf.keras.layers.Conv2D(256, (3,3), padding='same', activation='relu')
        self.conv3b = tf.keras.layers.Conv2D(256, (3,3), padding='same', activation='relu')
        self.bn3    = tf.keras.layers.BatchNormalization()
        self.pool3  = tf.keras.layers.MaxPooling2D((2,2))
        
        # Classificateur final
        self.flatten  = tf.keras.layers.Flatten()
        self.dense1   = tf.keras.layers.Dense(512, activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(0.5)
        self.dense2   = tf.keras.layers.Dense(256, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(0.3)
        self.sortie   = tf.keras.layers.Dense(10, activation='softmax')

    # Cette fonction décrit comment les données traversent le réseau
    def call(self, x, training=False):
        x = self.augment(x, training=training)

        # Bloc 1 : 2 convolution + BatchNorm
        x = self.conv1a(x)
        x = self.conv1b(x)
        x = self.bn1(x, training=training)
        x = self.pool1(x)

        # Bloc 2 
        x = self.conv2a(x)
        x = self.conv2b(x)
        x = self.bn2(x, training=training)
        x = self.pool2(x)

        # Bloc 3 
        x = self.conv3a(x)
        x = self.conv3b(x)
        x = self.bn3(x, training=training)
        x = self.pool3(x)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        x = self.sortie(x)

        return x
