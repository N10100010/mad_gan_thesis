import tensorflow as tf


def define_generators(n_gen, latent_dim):
    """
    Define multiple generator models for generating CIFAR-10 images.

    Parameters:
    - n_gen: Number of generator models to create.
    - latent_dim: Dimensionality of the latent space.

    Returns:
    - List of generator models.
    """
    # Dense layer to project the latent space into an initial 8x8x256 tensor
    dens = tf.keras.layers.Dense(
        units=8 * 8 * 256, use_bias=False, input_shape=(latent_dim,)
    )
    batchnorm0 = tf.keras.layers.BatchNormalization()
    rel0 = tf.keras.layers.LeakyReLU()
    reshape0 = tf.keras.layers.Reshape([8, 8, 256])

    # First upsampling block
    con2dt1 = tf.keras.layers.Conv2DTranspose(
        128, (5, 5), strides=(2, 2), padding="same", use_bias=False
    )
    batchnorm1 = tf.keras.layers.BatchNormalization()
    rel1 = tf.keras.layers.LeakyReLU()

    # Second upsampling block
    con2dt2 = tf.keras.layers.Conv2DTranspose(
        64, (5, 5), strides=(2, 2), padding="same", use_bias=False
    )
    batchnorm2 = tf.keras.layers.BatchNormalization()
    rel2 = tf.keras.layers.LeakyReLU()

    # Final convolution to output a 32x32x3 image
    models = []
    for label in range(n_gen):
        input = tf.keras.layers.Input(
            shape=(latent_dim,), dtype=tf.float32, name=f"input_{label}"
        )
        x = dens(input)
        x = batchnorm0(x)
        x = rel0(x)
        x = reshape0(x)

        x = con2dt1(x)
        x = batchnorm1(x)
        x = rel1(x)

        x = con2dt2(x)
        x = batchnorm2(x)
        x = rel2(x)

        # Adjust output layer for 3 channels (RGB) and tanh activation
        x = tf.keras.layers.Conv2DTranspose(
            3, (5, 5), strides=(1, 1), padding="same", use_bias=False, activation="tanh"
        )(x)

        models.append(tf.keras.models.Model(input, x, name=f"generator{label}"))
    return models
