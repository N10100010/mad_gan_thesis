from typing import Tuple
import tensorflow as tf
import numpy as np

def calculate_inception_score(
    images: np.ndarray, 
    classifier: tf.keras.Model,
    n_splits: int = 10
) -> Tuple[float, float]:
    """
    Calculate the Inception Score (IS) using a provided classifier.

    Args:
        images (np.ndarray): Array of images with shape (N, H, W, 3). 
            These images can be of any size; they will be resized to match the classifier's input.
        classifier (tf.keras.Model): A model (typically from the Functional API) that outputs a probability distribution.
            For example, a classifier trained on your dataset with a softmax output.
        n_splits (int): Number of splits to partition the data for computing statistics.

    Returns:
        tuple: (mean_inception_score, std_inception_score)
    """
    # Determine the expected input size from the classifier.
    target_size = classifier.input_shape[1:3]  # (height, width)
    
    scores = []
    n_samples = images.shape[0]
    n_part = n_samples // n_splits

    for i in range(n_splits):
        # Select a subset of images
        subset = images[i * n_part:(i + 1) * n_part]
        # Resize to classifier's expected input size.
        subset_resized = tf.image.resize(subset, target_size)
        # (Optional) Add any preprocessing steps here if required by your classifier.
        # For example, if your classifier expects images in the range [-1,1], normalize accordingly.
        #
        # Compute predictions using the provided classifier.
        preds = classifier.predict(subset_resized)
        # Compute the marginal distribution (average probability over images)
        py = np.mean(preds, axis=0)
        # Compute the KL divergence for each image and then the exponential of its mean.
        kl_divs = preds * (np.log(preds + 1e-16) - np.log(py + 1e-16))
        kl_div_sum = np.sum(kl_divs, axis=1)
        split_score = np.exp(np.mean(kl_div_sum))
        scores.append(split_score)
    
    inception_score = np.mean(scores)
    inception_score_std = np.std(scores)
    return inception_score, inception_score_std


# Example usage:
if __name__ == '__main__':
    # Here we use MNIST as an example. In practice, you might train a domain-specific classifier.
    import numpy as np

    # Load MNIST data.
    (mnist_train, _), (mnist_test, _) = tf.keras.datasets.mnist.load_data()
    # Normalize and add a channel dimension.
    mnist_train = mnist_train.astype('float32') / 255.0
    mnist_test = mnist_test.astype('float32') / 255.0
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
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(input_img)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    output = tf.keras.layers.Dense(10, activation='softmax')(x)
    simple_classifier = tf.keras.Model(input_img, output)
    # Note: This model is untrained and is used here for demonstration only.
    
    # Compute the Inception Score using the classifier.
    is_score, is_std = calculate_inception_score(mnist_train_rgb, simple_classifier, n_splits=10)
    print(f"Inception Score: {is_score:.4f} ± {is_std:.4f}")
