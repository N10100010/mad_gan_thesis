import tensorflow as tf
from model_definitions.classifiers import BaseClassifier


class CIFAR10Classifier(BaseClassifier):
    dataset = BaseClassifier.CIFAR10
    input_shape = (32, 32, 3)

    def __init__(self, num_classes=10):
        super().__init__()

        # Define the model using Functional API
        inputs = tf.keras.Input(shape=self.input_shape, name="classifier_input")

        x = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(
            inputs
        )
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)

        x = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)

        x = tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same")(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)

        x = tf.keras.layers.Dense(512, activation="relu", name="feature_extractor")(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

        # Create the model
        self.model = tf.keras.Model(
            inputs=inputs, outputs=outputs, name="CIFAR10Classifier"
        )

    def call(self, inputs, training=False):
        return self.model(inputs, training=training)
