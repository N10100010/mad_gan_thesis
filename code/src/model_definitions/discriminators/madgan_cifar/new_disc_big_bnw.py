import tensorflow as tf


def define_discriminator(n_gen, in_shape=(32, 32, 1)):
    """
    Define a 'big' discriminator model for CIFAR-10 images with extra layers and wider filters.

    Parameters:
    - n_gen: Number of generator models + 1 (for real data class).
    - in_shape: Shape of the input images (default is CIFAR-10 shape).

    Returns:
    - A discriminator model.
    """
    kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
    alpha = 0.2  # LeakyReLU alpha

    inp = tf.keras.layers.Input(shape=in_shape)

    # Initial Convolution
    x = tf.keras.layers.Conv2D(
        64, (3, 3), padding="same", kernel_initializer=kernel_initializer
    )(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=alpha)(x)

    # Downsampling layers with increasing filters
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

    x = tf.keras.layers.Conv2D(
        512,
        (3, 3),
        strides=(2, 2),
        padding="same",
        kernel_initializer=kernel_initializer,
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=alpha)(x)

    # Flatten and Dense Layers
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    # Output Layer: n_gen+1 classes (n generators and 1 for real)
    out = tf.keras.layers.Dense(
        n_gen + 1, activation="softmax", kernel_initializer=kernel_initializer
    )(x)

    model = tf.keras.models.Model(inp, out, name="Discriminator_Big")
    return model


# if __name__ == '__main__':
#     # Example usage:
#     model = define_discriminator(n_gen=3)
#     model.summary()
