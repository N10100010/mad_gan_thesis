from typing import Tuple

import numpy as np
import tensorflow as tf


def cmadgan_dataset_func(random_state=None, conddition_dim=10, batch_size=128):
    (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()

    # Combine datasets if desired (more training data) -> using only train set here for simplicity
    all_digits = x_train.astype("float32")
    all_labels = y_train.astype("int32")

    # Add channel dimension and Normalize images to [-1, 1]
    all_digits = np.expand_dims(all_digits, axis=-1)
    all_digits = (all_digits - 127.5) / 127.5

    # One-hot encode labels
    all_labels_one_hot = tf.one_hot(all_labels, depth=conddition_dim)

    # Create tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices((all_digits, all_labels_one_hot))
    dataset = (
        dataset.shuffle(buffer_size=len(all_digits))
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    return dataset


def dataset_func(random_state=None):
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype(
        "float32"
    )
    train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

    # Convert to stacked-mnist(rgb images)
    # t1 = tf.random.shuffle(train_images, seed = 10)
    # t2 = tf.random.shuffle(train_images, seed = 20)
    # train_images = tf.concat([train_images, t1, t2], axis=-1)

    return train_images, np.unique(train_labels)


def conditional_dataset_func(random_state=None):
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype(
        "float32"
    )
    # Normalize images to [-1, 1]
    train_images = (train_images.astype("float64") - 127.5) / 127.5

    return train_images, train_labels


def get_dataset(
    size_dataset: int, n_gen: int, batch_size: int
) -> Tuple[tf.data.Dataset, np.ndarray, np.ndarray]:
    data, labels = dataset_func()
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = (
        dataset.repeat()
        .shuffle(10 * size_dataset, reshuffle_each_iteration=True)
        .batch(n_gen * batch_size, drop_remainder=True)
    )

    return dataset, labels, data
