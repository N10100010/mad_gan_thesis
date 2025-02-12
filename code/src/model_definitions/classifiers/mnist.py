import tensorflow as tf
from model_definitions.classifiers import BaseClassifier


class MNISTClassifier(BaseClassifier):
    dataset = BaseClassifier.MNIST
    
    input_shape = (28, 28, 1)
    
    def __init__(self, num_classes=10):
        super().__init__()
        # Define convolutional layers
        self.conv1 = tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=self.input_shape
        )
        self.pool1 = tf.keras.layers.MaxPooling2D((2, 2))
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu")
        self.pool2 = tf.keras.layers.MaxPooling2D((2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation="relu")
        self.dense2 = tf.keras.layers.Dense(num_classes, activation="softmax")

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)
