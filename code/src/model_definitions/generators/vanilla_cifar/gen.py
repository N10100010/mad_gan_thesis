import tensorflow as tf


def define_generator(latent_dim):
    """ """
    kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    inp = tf.keras.layers.Input(shape=(latent_dim,), dtype=tf.float32)
    x = tf.keras.layers.Dense(
        units=8 * 8 * 256,
        use_bias=False,
        input_shape=(latent_dim,),
        kernel_initializer=kernel_initializer,
    )(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Reshape([8, 8, 256])(x)

    x = tf.keras.layers.Conv2DTranspose(
        128,
        (5, 5),
        strides=(2, 2),
        padding="same",
        use_bias=False,
        kernel_initializer=kernel_initializer,
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2DTranspose(
        64,
        (5, 5),
        strides=(2, 2),
        padding="same",
        use_bias=False,
        kernel_initializer=kernel_initializer,
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # Adjust output layer for 3 channels (RGB) with tanh activation
    x = tf.keras.layers.Conv2DTranspose(
        3,
        (5, 5),
        strides=(1, 1),
        padding="same",
        use_bias=False,
        activation="tanh",
        kernel_initializer=kernel_initializer,
    )(x)

    return tf.keras.models.Model(inp, x, name="Generator")
