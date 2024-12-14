from typing import Tuple

import numpy as np
import tensorflow as tf


def dataset_func(random_state=None):
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype(
        "float32"
    )
    # train_images = tf.image.resize(train_images, [32,32])
    train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

    # Convert to stacked-mnist(rgb images)
    # t1 = tf.random.shuffle(train_images, seed = 10)
    # t2 = tf.random.shuffle(train_images, seed = 20)
    # train_images = tf.concat([train_images, t1, t2], axis=-1)

    return train_images, np.unique(train_labels)


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
