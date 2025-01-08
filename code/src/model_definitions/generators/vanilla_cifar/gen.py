import tensorflow as tf


def define_generator(latent_dim: int = 256) -> tf.keras.Model:
    # ki = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(latent_dim,))
    )
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Reshape((7, 7, 256)))

    model.add(
        tf.keras.layers.Conv2DTranspose(
            128, (5, 5), strides=(1, 1), padding="same", use_bias=False
        )
    )
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(
        tf.keras.layers.Conv2DTranspose(
            64, (5, 5), strides=(2, 2), padding="same", use_bias=False
        )
    )
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(
        tf.keras.layers.Conv2DTranspose(
            1, (5, 5), strides=(2, 2), padding="same", use_bias=False, activation="tanh"
        )
    )

    return model
