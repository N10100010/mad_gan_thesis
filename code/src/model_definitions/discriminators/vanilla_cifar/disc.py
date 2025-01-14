import tensorflow as tf


def define_discriminator(in_shape=(32, 32, 3)):
    """
    Define a discriminator model for CIFAR-10 images.

    Parameters:
    - n_gen: Number of generator models + 1 (for real data class).
    - in_shape: Shape of the input images (default is CIFAR-10 shape).

    Returns:
    - A compiled discriminator model.
    """
    kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
    alpha = 0.2  # LeakyReLU alpha parameter

    inp = tf.keras.layers.Input(shape=in_shape)

    # Initial Convolutional Layer
    x = tf.keras.layers.Conv2D(
        64, (3, 3), padding="same", kernel_initializer=kernel_initializer
    )(inp)
    x = tf.keras.layers.BatchNormalization()(x)  # Normalize after convolutions
    x = tf.keras.layers.LeakyReLU(alpha=alpha)(x)

    # Downsampling Layers
    x = tf.keras.layers.Conv2D(
        128,
        (3, 3),
        strides=(2, 2),
        padding="same",
        kernel_initializer=kernel_initializer,
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)  # Normalize after convolutions
    x = tf.keras.layers.LeakyReLU(alpha=alpha)(x)

    x = tf.keras.layers.Conv2D(
        128,
        (3, 3),
        strides=(2, 2),
        padding="same",
        kernel_initializer=kernel_initializer,
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=alpha)(x)

    x = tf.keras.layers.Conv2D(
        256,
        (3, 3),
        strides=(2, 2),
        padding="same",
        kernel_initializer=kernel_initializer,
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=alpha)(x)

    # Flatten and Dense Layers
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.4)(x)  # Regularization

    # Output Layer with Softmax
    out = tf.keras.layers.Dense(
        1, activation="sigmoid", kernel_initializer=kernel_initializer
    )(x)

    # Define and compile model
    model = tf.keras.models.Model(inp, out, name="Discriminator")

    return model
