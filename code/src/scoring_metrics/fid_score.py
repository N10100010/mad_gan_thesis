import numpy as np
import tensorflow as tf
from scipy.linalg import sqrtm
from tensorflow.keras.applications.inception_v3 import preprocess_input


def _load_real_images(dataset, num_samples=10000):
    """Loads real images from MNIST, Fashion-MNIST, or CIFAR-10."""
    if dataset == "mnist":
        (x_train, _), _ = tf.keras.datasets.mnist.load_data()
        x_train = np.expand_dims(x_train, axis=-1)  # Add channel dimension
    elif dataset == "fashion_mnist":
        (x_train, _), _ = tf.keras.datasets.fashion_mnist.load_data()
        x_train = np.expand_dims(x_train, axis=-1)  # Add channel dimension
    elif dataset == "cifar10":
        (x_train, _), _ = tf.keras.datasets.cifar10.load_data()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    x_train = x_train.astype(np.float32) / 255.0
    idx = np.random.choice(len(x_train), num_samples, replace=False)
    return x_train[idx]


def calculate_fid_score(generated_images, dataset, classifier, batch_size=32):
    """
    Computes the Fréchet Inception Distance (FID) between generated and real images.

    Parameters:
        generated_images (numpy.ndarray): Generated images with shape (N, H, W, C).
        dataset (str): One of ["mnist", "fashion-mnist", "cifar10"].
        classifier (tf.keras.Model): Model for feature extraction.
                                   For CIFAR-10, use InceptionV3 (include_top=False, pooling="avg").
        batch_size (int): Batch size for inference.

    Returns:
        float: FID score (lower is better).
    """
    real_images = _load_real_images(dataset)
    input_shape = (
        classifier.input_shape
    )  # e.g., (299, 299, 3) or (28, 28, 1) or (32, 32, 3)
    expected_height, expected_width, expected_channels = (
        input_shape[0],
        input_shape[1],
        input_shape[2],
    )

    if expected_height is None or expected_width is None or expected_channels is None:
        if dataset == "cifar10":
            expected_height, expected_width, expected_channels = 299, 299, 3
        elif dataset in ["mnist", "fashion-mnist"]:
            expected_height, expected_width, expected_channels = 28, 28, 1

    if generated_images.shape[-1] != expected_channels:
        if generated_images.shape[-1] == 1 and expected_channels == 3:
            generated_images = np.repeat(generated_images, 3, axis=-1)
        elif generated_images.shape[-1] == 3 and expected_channels == 1:
            generated_images = np.mean(generated_images, axis=-1, keepdims=True)
        else:
            raise ValueError(
                f"Incompatible channel dimensions: images have {generated_images.shape[-1]} channels, but classifier expects {expected_channels}."
            )

    if generated_images.max() > 1:
        generated_images = generated_images.astype(np.float32) / 255.0

    # Resize images on CPU to avoid GPU memory overflow
    if (
        generated_images.shape[1] == 1
    ):  # we have a not needed batch dimension, so we can remove it.
        generated_images = np.squeeze(generated_images, axis=1)

    if (generated_images.shape[1] != expected_height) or (
        generated_images.shape[2] != expected_width
    ):
        resize_shape = [int(expected_height), int(expected_width)]
        with tf.device("/CPU:0"):
            generated_images = tf.image.resize(generated_images, resize_shape).numpy()
            real_images = tf.image.resize(real_images, resize_shape).numpy()

    if dataset == "cifar10":
        generated_images = preprocess_input(generated_images * 255.0)
        real_images = preprocess_input(real_images * 255.0)

    # Extract features from an intermediate layer
    if dataset != "cifar10":
        # we assume that the passed classifier has a layer named "feature_extractor". This is assumed to be the last dense-layer in the deep net.
        if "feature_extractor" not in [_.name for _ in classifier.layers]:
            raise ValueError(
                "Classifier is not a model with a layer named `feature_extractor`."
            )
        feature_extractor = tf.keras.Model(
            classifier.model.input,
            classifier.model.get_layer("feature_extractor").output,
        )
    else:
        # we assume the following definition for the classifier of the inception model:
        # classifier = tf.keras.applications.InceptionV3(weights="imagenet", include_top=False, pooling="avg")
        feature_extractor = classifier

    classifier = None

    # Define a helper function to extract features in batches
    def get_features(images):
        features = []
        for i in range(0, images.shape[0], batch_size):
            batch = images[i : i + batch_size]
            feat = feature_extractor.predict(batch, verbose=0)
            features.append(feat)
        return np.concatenate(features, axis=0)

    real_features = get_features(real_images)
    generated_features = get_features(generated_images)

    # free memory of the feature extractor
    feature_extractor = None

    # Compute mean and covariance statistics
    mu_real, sigma_real = (
        real_features.mean(axis=0),
        np.cov(real_features, rowvar=False),
    )
    mu_gen, sigma_gen = (
        generated_features.mean(axis=0),
        np.cov(generated_features, rowvar=False),
    )

    # Compute squared difference between means
    diff = mu_real - mu_gen
    diff_squared = diff.dot(diff)

    # Compute sqrt of the product of covariance matrices
    covmean, _ = sqrtm(sigma_real.dot(sigma_gen), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real  # Numerical stability fix

    # Compute final FID score
    fid = diff_squared + np.trace(sigma_real + sigma_gen - 2 * covmean)
    return fid
