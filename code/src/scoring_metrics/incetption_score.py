import tensorflow as tf


def calculate_inception_score(
    generated_images: tf.Tensor, real_images: tf.Tensor, n_splits: int = 10
) -> tuple:
    """
    Calculate the Inception score for a set of generated images.

    Args:
        generated_images (tf.Tensor): Generated images as a tensor.
        real_images (tf.Tensor): Real images as a tensor.

    Returns:
        tuple: Inception score and standard deviation.
    """
    # Use the model with the top classification layer
    model = tf.keras.applications.InceptionV3(include_top=True, weights="imagenet")

    scores = []
    n_part = generated_images.shape[0] // n_splits

    for split in range(n_splits):
        ix_start, ix_end = split * n_part, (split + 1) * n_part
        subset_generated = generated_images[ix_start:ix_end]
        subset_real = real_images[ix_start:ix_end]

        if subset_generated.shape[-1] == 1:
            subset_generated = tf.image.grayscale_to_rgb(subset_generated)
        if subset_real.shape[-1] == 1:
            subset_real = tf.image.grayscale_to_rgb(subset_real)

        subset_generated = tf.image.resize(subset_generated, (299, 299))
        subset_real = tf.image.resize(subset_real, (299, 299))
        subset_generated = tf.keras.applications.inception_v3.preprocess_input(
            subset_generated
        )
        subset_real = tf.keras.applications.inception_v3.preprocess_input(subset_real)

        # Get predictions and apply softmax to obtain probabilities
        preds_generated = tf.nn.softmax(model.predict(subset_generated))
        preds_real = tf.nn.softmax(model.predict(subset_real))

        for j in range(len(preds_generated)):
            p_yx = preds_generated[j]
            p_y = tf.reduce_mean(preds_real, axis=0)
            # To avoid division by zero, add a small epsilon if necessary
            kl_div = tf.reduce_sum(p_yx * tf.math.log(p_yx / (p_y + 1e-10)))
            scores.append(kl_div)

    scores = tf.convert_to_tensor(scores)
    inception_score = tf.exp(tf.reduce_mean(scores))
    inception_score_std = tf.exp(tf.math.reduce_std(scores))

    return inception_score.numpy(), inception_score_std.numpy()


# Example usage:

# Load MNIST dataset
(mnist_train_images, _), (mnist_test_images, _) = tf.keras.datasets.mnist.load_data()
mnist_train_images = mnist_train_images.astype("float32") / 255.0
mnist_test_images = mnist_test_images.astype("float32") / 255.0

# Ensure MNIST images have 3 channels
mnist_train_images = tf.image.grayscale_to_rgb(
    tf.expand_dims(mnist_train_images, axis=-1)
)
mnist_test_images = tf.image.grayscale_to_rgb(
    tf.expand_dims(mnist_test_images, axis=-1)
)

mnist_train_images = mnist_train_images[:1000]
mnist_test_images = mnist_test_images[:1000]

# Calculate Inception Score
inception_score, inception_score_std = calculate_inception_score(
    mnist_train_images, mnist_test_images
)

print(f"Inception Score: {inception_score}")
print(f"Inception Score Standard Deviation: {inception_score_std}")
