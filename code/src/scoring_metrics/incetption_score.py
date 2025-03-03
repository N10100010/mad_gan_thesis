from typing import Tuple

import numpy as np
import tensorflow as tf
from scipy.stats import entropy


def calculate_inception_score(
    generated_images, classifier, batch_size=32, splits=10
) -> Tuple[float, float]:
    """
    Computes the Inception Score (IS) for generated generated_images using a given classifier.

    Parameters:
        generated_images (numpy.ndarray):
            Generated generated_images with shape (N, H, W, C).
            For example, MNIST, FASION: (N, 28, 28, 1), CIFAR: (N, 32, 32, 3).
        classifier (tf.keras.Model):
            A pretrained classifier (e.g., Inception-V3 for CIFAR or a CNN for MNIST/Fashion-MNIST).
        batch_size (int):
            Batch size for model predictions.
        splits (int):
            Number of splits for IS computation.

    Returns:
        tuple: (mean_inception_score, std_inception_score)
    """
    # Determine the expected number of channels and spatial dimensions from the classifier's input shape.
    # classifier.input_shape is generally like (None, H_expected, W_expected, C_expected)
    if len(classifier.input_shape) == 4:
        expected_height, expected_width, expected_channels = classifier.input_shape[1:4]
    elif len(classifier.input_shape) == 3:
        expected_height, expected_width, expected_channels = classifier.input_shape
    else:
        raise ValueError("Invalid input shape for classifier.")

    # Adjust the channel dimension if needed:
    if generated_images.shape[-1] != expected_channels:
        if generated_images.shape[-1] == 1 and expected_channels == 3:
            generated_images = np.repeat(generated_images, 3, axis=-1)
        elif generated_images.shape[-1] == 3 and expected_channels == 1:
            generated_images = np.mean(generated_images, axis=-1, keepdims=True)
        else:
            raise ValueError(
                f"Incompatible channel dimensions: images have {generated_images.shape[-1]} channels, but classifier expects {expected_channels}."
            )

    # Normalize generated_images to [0, 1] if they are not already
    if generated_images.max() > 1:
        generated_images = generated_images.astype(np.float32) / 255.0

    if (
        generated_images.shape[1] == 1
    ):  # we have a not needed batch dimension, so we can remove it.
        generated_images = np.squeeze(generated_images, axis=1)

    # Resize generated_images if their spatial dimensions don't match the classifier's expected dimensions.
    if (
        generated_images.shape[1] != expected_height
        or generated_images.shape[2] != expected_width
    ):
        # Convert generated_images to a TensorFlow tensor, resize, then convert back to numpy
        generated_images = tf.convert_to_tensor(generated_images)

        # Create the size tensor explicitly with int32 type
        target_size = tf.constant([expected_height, expected_width], dtype=tf.int32)

        # Resize the images using the target size
        generated_images = tf.image.resize(generated_images, target_size).numpy()


    num_images = generated_images.shape[0]
    preds = []

    # Process generated_images in batches.
    for i in range(0, num_images, batch_size):
        batch = generated_images[i : i + batch_size]
        # Get classifier predictions (logits)
        logits = classifier.predict(batch, verbose=0)
        # Convert logits to probabilities via softmax
        prob = tf.nn.softmax(logits).numpy()
        preds.append(prob)

    # free memory of the feature extractor
    classifier = None

    preds = np.concatenate(preds, axis=0)  # Shape: (N, num_classes)

    # Compute the Inception Score using KL divergence
    split_scores = []
    split_size = num_images // splits
    for i in range(splits):
        part = preds[i * split_size : (i + 1) * split_size]
        p_y = np.mean(part, axis=0)  # Marginal distribution over classes
        kl_divs = [entropy(p, p_y) for p in part]
        split_score = np.exp(np.mean(kl_divs))
        split_scores.append(split_score)

    return float(np.mean(split_scores)), float(np.std(split_scores))
