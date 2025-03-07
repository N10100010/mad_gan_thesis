import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import load_model


class CIFAR10Classifier(tf.keras.Model):
    input_shape = (32, 32, 3)
    
    def __init__(self, num_classes=10):
        super().__init__()

        # Define the model using Functional API
        inputs = tf.keras.Input(shape=self.input_shape, name="classifier_input")

        x = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(inputs)
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
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs, name="CIFAR10Classifier")

    def call(self, inputs, training=False):
        return self.model(inputs, training=training)


# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize images
y_train, y_test = tf.keras.utils.to_categorical(y_train, 10), tf.keras.utils.to_categorical(y_test, 10)  # One-hot encode labels

# Instantiate and compile model
classifier = CIFAR10Classifier()
classifier.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train model for 2 epochs
classifier.model.fit(x_train, y_train, epochs=2, batch_size=64, validation_data=(x_test, y_test))

# Save model
model_path = "cifar10_classifier.h5"
classifier.model.save(model_path)
print(f"Model saved to {model_path}")

# Load model
loaded_model = load_model(model_path)
print("Model loaded successfully.")

# Pick a random image for classification
random_idx = np.random.randint(len(x_test))
random_image = x_test[random_idx]
random_label = np.argmax(y_test[random_idx])

# Predict
predictions = loaded_model.predict(np.expand_dims(random_image, axis=0))
predicted_label = np.argmax(predictions)

# Display the image with the prediction
plt.imshow(random_image)
plt.title(f"Predicted: {predicted_label}, Actual: {random_label}")
plt.axis("off")
plt.show()
