from typing import Tuple 

import tensorflow as tf
import numpy as np

def dataset_func(random_state: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load MNIST dataset and return it as a numpy array.

    Args:
        random_state: An optional seed for shuffling the dataset.

    Returns:
        A tuple of two numpy arrays. The first array contains the images,
        and the second array contains the unique labels.
    """
    print("Downloading MNIST dataset")
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    print("Downloaded MNIST dataset")
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    # train_images = tf.image.resize(train_images, [32,32])
    train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

    # Convert to stacked-mnist(rgb images)
    #t1 = tf.random.shuffle(train_images, seed = 10)
    #t2 = tf.random.shuffle(train_images, seed = 20)
    #train_images = tf.concat([train_images, t1, t2], axis=-1)
    
    return train_images, train_labels

def get_dataset(size_dataset: int, n_gen: int, batch_size: int) -> (tf.data.Dataset, np.ndarray):
    """
    Create a TensorFlow dataset for training, with shuffled data and labels.

    Parameters
    ----------
    size_dataset : int
        The size of the dataset.
    n_gen : int
        The number of generators.
    batch_size : int
        The batch size for training.

    Returns
    -------
    tf.data.Dataset
        A TensorFlow dataset object for training.
    np.ndarray
        The labels associated with the dataset, shuffled in the same way.
    """
    # Load and preprocess the MNIST dataset
    data, labels = dataset_func()  # Assume this returns (data, labels) as numpy arrays
    
    # Combine data and labels into a single dataset to ensure consistent shuffling
    combined_dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    combined_dataset = combined_dataset.repeat().shuffle(10 * size_dataset, reshuffle_each_iteration=True).batch(n_gen * batch_size, drop_remainder=True)
    
    print("Shuffeling MNIST dataset")
    # Extract shuffled data and labels
    shuffled_data, shuffled_labels = [], []
    for batch_data, batch_labels in combined_dataset.take(size_dataset):
        shuffled_data.append(batch_data.numpy())
        shuffled_labels.append(batch_labels.numpy())
    
    print("Shuffeled MNIST dataset")
    shuffled_data = np.concatenate(shuffled_data, axis=0)
    shuffled_labels = np.concatenate(shuffled_labels, axis=0)
    
    # Create a new dataset with just the shuffled data
    dataset = tf.data.Dataset.from_tensor_slices(shuffled_data).batch(n_gen * batch_size, drop_remainder=True)
    
    return dataset, shuffled_labels

    
