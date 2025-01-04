import tensorflow as tf


# define the standalone discriminator model
def define_discriminator(n_gen, in_shape=(32, 32, 3)):
    inp = tf.keras.layers.Input(shape=in_shape)

    # Initial Convolutional Layer
    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", input_shape=in_shape)(inp)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    # Downsampling Layers
    x = tf.keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same")(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    x = tf.keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same")(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    x = tf.keras.layers.Conv2D(256, (3, 3), strides=(2, 2), padding="same")(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    # Flatten and Dense Layers
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    out = tf.keras.layers.Dense(n_gen + 1, activation="softmax")(x)

    model = tf.keras.models.Model(inp, out, name="Discriminator")
    return model
