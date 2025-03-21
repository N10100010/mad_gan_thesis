import tensorflow as tf


def define_generators(n_gen, latent_dim):
    """
    Define multiple 'big' generator models for generating CIFAR-10 images with more layers and wider filters.

    Parameters:
    - n_gen: Number of generator models to create.
    - latent_dim: Dimensionality of the latent space.

    Returns:
    - List of generator models.
    """
    kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
    alpha = 0.2  # LeakyReLU alpha

    models = []

    for label in range(n_gen):
        inp = tf.keras.layers.Input(shape=(latent_dim,), name=f"input_{label}")

        # Dense layer to form a 4x4x512 tensor
        x = tf.keras.layers.Dense(
            4 * 4 * 512, use_bias=False, kernel_initializer=kernel_initializer
        )(inp)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=alpha)(x)
        x = tf.keras.layers.Reshape((4, 4, 512))(x)

        # Upsample to 8x8
        x = tf.keras.layers.Conv2DTranspose(
            256,
            (5, 5),
            strides=(2, 2),
            padding="same",
            use_bias=False,
            kernel_initializer=kernel_initializer,
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=alpha)(x)

        # Upsample to 16x16
        x = tf.keras.layers.Conv2DTranspose(
            128,
            (5, 5),
            strides=(2, 2),
            padding="same",
            use_bias=False,
            kernel_initializer=kernel_initializer,
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=alpha)(x)

        # Upsample to 32x32
        x = tf.keras.layers.Conv2DTranspose(
            64,
            (5, 5),
            strides=(2, 2),
            padding="same",
            use_bias=False,
            kernel_initializer=kernel_initializer,
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=alpha)(x)

        # Output layer: produce 32x32x3 image with tanh activation
        out = tf.keras.layers.Conv2DTranspose(
            1,
            (5, 5),
            strides=(1, 1),
            padding="same",
            use_bias=False,
            activation="tanh",
            kernel_initializer=kernel_initializer,
        )(x)

        model = tf.keras.models.Model(inp, out, name=f"Generator_Big_{label}")
        models.append(model)
    return models


# if __name__ == '__main__':
#     # Example usage:
#     models = define_generators(n_gen=3, latent_dim=100)
#     for model in models:
#         model.summary()
