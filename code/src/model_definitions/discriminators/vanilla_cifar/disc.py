import tensorflow as tf


def define_discriminator(in_shape=(32, 32, 3)):
    kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
    alpha = 0.2

    inp = tf.keras.layers.Input(shape=in_shape)

    # Start with stronger features
    x = tf.keras.layers.Conv2D(
        64,
        (5, 5),
        strides=(2, 2),
        padding="same",
        kernel_initializer=kernel_initializer,
    )(inp)
    x = tf.keras.layers.LeakyReLU(alpha)(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Conv2D(
        128,
        (5, 5),
        strides=(2, 2),
        padding="same",
        kernel_initializer=kernel_initializer,
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha)(x)

    x = tf.keras.layers.Conv2D(
        256,
        (5, 5),
        strides=(2, 2),
        padding="same",
        kernel_initializer=kernel_initializer,
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    return tf.keras.Model(inp, x, name="Discriminator")
