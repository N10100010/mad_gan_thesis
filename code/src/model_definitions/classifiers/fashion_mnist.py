import tensorflow as tf
from model_definitions.classifiers import BaseClassifier


class FashionMNISTClassifier(BaseClassifier):
    dataset = BaseClassifier.FASHION_MNIST
    input_shape = (28, 28, 1)

    def __init__(self, num_classes=10):
        super().__init__()

        # Define model using Functional API
        inputs = tf.keras.Input(shape=self.input_shape, name="classifier_input")

        x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu")(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)

        x = tf.keras.layers.Conv2D(64, (3, 3), activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)

        x = tf.keras.layers.Conv2D(128, (3, 3), activation="relu")(x)
        x = tf.keras.layers.Flatten()(x)

        x = tf.keras.layers.Dense(256, activation="relu", name="feature_extractor")(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

        # Create the model
        self.model = tf.keras.Model(
            inputs=inputs, outputs=outputs, name="FashionMNISTClassifier"
        )

    def call(self, inputs, training=False):
        return self.model(inputs, training=training)
