import numpy as np
import tensorflow as tf


def dataset_func(random_state=None):
    (train_images, train_labels), (_, _) = tf.keras.datasets.cifar10.load_data()

    train_images = train_images.reshape(train_images.shape[0], 32, 32, 3).astype(
        "int32"
    )
    train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

    return train_images, train_labels


def conditional_dataset_func(random_state=None):
    (train_images, train_labels), (_, _) = tf.keras.datasets.cifar10.load_data()

    train_images = train_images.reshape(train_images.shape[0], 32, 32, 3).astype(
        "int32"
    )
    train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
    return train_images, train_labels


def dataset_func_black_and_white(random_state=None):
    (train_images, train_labels), (_, _) = tf.keras.datasets.cifar10.load_data()

    # Convert RGB to Grayscale manually using NumPy
    train_images = np.dot(
        train_images[..., :3], [0.2989, 0.5870, 0.1140]
    )  # Standard grayscale conversion
    train_images = np.expand_dims(train_images, axis=-1)  # Ensure shape is (32, 32, 1)

    train_images = (train_images - 127.5) / 127.5

    # Convert back to a TensorFlow tensor
    train_images = tf.convert_to_tensor(train_images, dtype=tf.float32)

    return train_images, train_labels
