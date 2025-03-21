import tensorflow as tf


# define the standalone discriminator model
def define_discriminator(n_gen):
    inp = tf.keras.layers.Input(shape=(28, 28, 1))

    x = tf.keras.layers.Conv2D(
        64, (5, 5), strides=(2, 2), padding="same", input_shape=[28, 28, 1]
    )(inp)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same")(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Flatten()(x)
    out = tf.keras.layers.Dense(n_gen + 1, activation="softmax")(x)

    model = tf.keras.models.Model(inp, out, name="Discriminator")
    return model
