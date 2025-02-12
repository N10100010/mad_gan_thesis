from typing import Tuple

import numpy as np
import tensorflow as tf
from scipy.linalg import sqrtm


def calculate_fid(
    real_images: np.ndarray,
    generated_images: np.ndarray,
    classifier: tf.keras.Model,
    n_splits: int = 10,
) -> Tuple[float, float]:
    """
    Calculate the Frechet Inception Distance (FID) between two sets of images,
    using a provided classifier as a feature extractor.

    **Important:**
    For meaningful FID computation, the classifier should output features
    (i.e. it should be a feature extractor). If your classifier outputs classification
    probabilities, consider building a new model that outputs an intermediate layer.

    Args:
        real_images (np.ndarray): Array of real images with shape (N, H, W, 3).
        generated_images (np.ndarray): Array of generated images with shape (N, H, W, 3).
        classifier (tf.keras.Model): A model (typically from the Functional API) that outputs features.
            The input images will be resized to match the classifier's input shape.
        n_splits (int): Number of splits to partition the data for computing statistics.

    Returns:
        tuple: (mean_fid, std_fid)
    """
    fid_scores = []
    n_samples = real_images.shape[0]
    n_part = n_samples // n_splits

    # Determine the expected input size from the classifier.
    target_size = classifier.input_shape[1:3]  # (height, width)

    for i in range(n_splits):
        start = i * n_part
        end = (i + 1) * n_part

        # Select corresponding subsets.
        subset_real = real_images[start:end]
        subset_generated = generated_images[start:end]

        # Resize images to the classifier's input size.
        real_resized = tf.image.resize(subset_real, target_size)
        gen_resized = tf.image.resize(subset_generated, target_size)
        # (Optional) Add any preprocessing steps required by your classifier here.

        # Extract features using the provided classifier.
        features_real = classifier.predict(real_resized)
        features_gen = classifier.predict(gen_resized)

        # Compute statistics: mean and covariance of the features.
        mu_real, sigma_real = (
            np.mean(features_real, axis=0),
            np.cov(features_real, rowvar=False),
        )
        mu_gen, sigma_gen = (
            np.mean(features_gen, axis=0),
            np.cov(features_gen, rowvar=False),
        )

        # Compute squared difference between means.
        ssd = np.sum((mu_real - mu_gen) ** 2)
        # Compute sqrt of the product of covariances.
        covmean, _ = sqrtm(sigma_real.dot(sigma_gen), disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = ssd + np.trace(sigma_real + sigma_gen - 2 * covmean)
        fid_scores.append(fid)

    mean_fid = np.mean(fid_scores)
    std_fid = np.std(fid_scores)
    return mean_fid, std_fid


# Example usage:
if __name__ == "__main__":
    # Here we use MNIST as an example. In practice, you might train a domain-specific classifier.
    import numpy as np

    # Load MNIST data.
    (mnist_train, _), (mnist_test, _) = tf.keras.datasets.mnist.load_data()
    # Normalize and add a channel dimension.
    mnist_train = mnist_train.astype("float32") / 255.0
    mnist_test = mnist_test.astype("float32") / 255.0
    mnist_train = np.expand_dims(mnist_train, axis=-1)
    mnist_test = np.expand_dims(mnist_test, axis=-1)
    # Convert grayscale to RGB.
    mnist_train_tensor = tf.convert_to_tensor(mnist_train)
    mnist_test_tensor = tf.convert_to_tensor(mnist_test)
    mnist_train_rgb = tf.image.grayscale_to_rgb(mnist_train_tensor).numpy()
    mnist_test_rgb = tf.image.grayscale_to_rgb(mnist_test_tensor).numpy()

    # For demonstration, let's assume we have a simple classifier.
    # (In practice, you would use a well-trained model on MNIST or your dataset.)
    # Here we build a simple model that accepts 28x28x3 images.
    input_img = tf.keras.Input(shape=(28, 28, 3))
    x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu")(input_img)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    output = tf.keras.layers.Dense(10, activation="softmax")(x)
    simple_classifier = tf.keras.Model(input_img, output)
    # Note: This model is untrained and is used here for demonstration only.

    # For FID, it is recommended to use a feature extractor.
    # One option is to remove the final softmax layer.
    # For example, we can build a feature extractor from the simple_classifier.
    feature_extractor = tf.keras.Model(
        inputs=simple_classifier.input, outputs=simple_classifier.layers[-2].output
    )
    fid_mean, fid_std = calculate_fid(
        mnist_test_rgb, mnist_train_rgb, feature_extractor, n_splits=10
    )
    print(f"FID: {fid_mean:.4f} ± {fid_std:.4f}")
