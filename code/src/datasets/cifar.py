import numpy as np
import tensorflow as tf


def cmadgan_dataset_func(random_state=None, conddition_dim=10, batch_size=128):
    (x_train, y_train), (_, _) = tf.keras.datasets.cifar10.load_data()

    # Combine datasets if desired (more training data) -> using only train set here for simplicity
    all_digits = x_train.astype("float32")
    all_labels = y_train.astype("int32")

    # Add channel dimension and Normalize images to [-1, 1]
    all_digits = np.expand_dims(all_digits, axis=-1)
    all_digits = (all_digits - 127.5) / 127.5

    # One-hot encode labels
    all_labels_one_hot = tf.one_hot(all_labels, depth=conddition_dim)
    all_labels_one_hot = tf.squeeze(all_labels_one_hot, axis=1)

    # Create tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices((all_digits, all_labels_one_hot))
    dataset = (
        dataset.shuffle(buffer_size=len(all_digits))
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    return dataset


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
