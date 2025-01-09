import tensorflow as tf


def define_generator(latent_dim: int = 256) -> tf.keras.Model:
    # ki = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    inp = tf.keras.layers.Input(shape=(latent_dim,))
    x = tf.keras.layers.Dense(7 * 7 * 256)(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Reshape((7, 7, 256))(x)

    x = tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    out = tf.keras.layers.Conv2DTranspose(
        1, (5, 5), strides=(2, 2), padding="same", activation="tanh"
    )(x)

    return tf.keras.Model(inp, out, name="Generator")
