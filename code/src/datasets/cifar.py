import tensorflow as tf


def dataset_func(random_state=None):
    (train_images, train_labels), (_, _) = tf.keras.datasets.cifar10.load_data()

    train_images = train_images.reshape(train_images.shape[0], 32, 32, 3).astype(
        "int32"
    )
    train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

    return train_images, train_labels
