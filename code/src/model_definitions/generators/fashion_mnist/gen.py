import tensorflow as tf


def define_generators(n_gen, latent_dim):
    dens = tf.keras.layers.Dense(
        units=7 * 7 * 256, use_bias=False, input_shape=(latent_dim,)
    )
    batchnorm0 = tf.keras.layers.BatchNormalization()
    rel0 = tf.keras.layers.LeakyReLU()
    reshape0 = tf.keras.layers.Reshape([7, 7, latent_dim])

    con2dt1 = tf.keras.layers.Conv2DTranspose(
        128, (5, 5), strides=(1, 1), padding="same", use_bias=False
    )
    batchnorm1 = tf.keras.layers.BatchNormalization()
    rel1 = tf.keras.layers.LeakyReLU()

    con2dt2 = tf.keras.layers.Conv2DTranspose(
        64, (5, 5), strides=(2, 2), padding="same", use_bias=False
    )
    batchnorm2 = tf.keras.layers.BatchNormalization()
    rel2 = tf.keras.layers.LeakyReLU()

    models = []
    for label in range(n_gen):
        input = tf.keras.layers.Input(
            shape=(latent_dim,), dtype=tf.float64, name=f"input_{label}"
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

        x = tf.keras.layers.Conv2DTranspose(
            1, (5, 5), strides=(2, 2), padding="same", use_bias=False, activation="tanh"
        )(x)

        models.append(tf.keras.models.Model(input, x, name=f"generator{label}"))
    return models
