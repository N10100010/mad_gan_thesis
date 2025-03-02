import tensorflow as tf


def define_generator(latent_dim):
    n_nodes = 256 * 4 * 4

    inp = tf.keras.layers.Input(shape=(latent_dim,))
    x = tf.keras.layers.Dense(n_nodes, input_dim=latent_dim)(inp)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Reshape((4, 4, 256))(x)
    x = tf.keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same")(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same")(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same")(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    out = tf.keras.layers.Conv2D(3, (3, 3), activation="tanh", padding="same")(x)

    return tf.keras.Model(inp, out, name="Generator")
