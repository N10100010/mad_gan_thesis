import tensorflow as tf


def define_generator(latent_dim, n_classes):
    # Label input
    label_input = tf.keras.layers.Input(shape=(1,), dtype=tf.int32)
    li = tf.keras.layers.Embedding(n_classes, 50)(label_input)
    li = tf.keras.layers.Flatten()(li)

    li = tf.keras.layers.Dense(8 * 8 * 64)(li)
    li = tf.keras.layers.LeakyReLU(alpha=0.2)(li)
    li = tf.keras.layers.Reshape((8, 8, 64))(li)

    # Noise input
    noise_input = tf.keras.layers.Input(shape=(latent_dim,))
    n_nodes = 128 * 8 * 8
    gen = tf.keras.layers.Dense(n_nodes)(noise_input)
    gen = tf.keras.layers.LeakyReLU(alpha=0.2)(gen)
    gen = tf.keras.layers.Reshape((8, 8, 128))(gen)

    li = tf.keras.layers.Reshape((8, 8, 64))(li)

    merge = tf.keras.layers.Concatenate(axis=-1)([gen, li])

    x = tf.keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same")(
        merge
    )
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    x = tf.keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same")(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    x = tf.keras.layers.Conv2D(3, (8, 8), activation="tanh", padding="same")(x)

    return tf.keras.Model([noise_input, label_input], x, name="Conditional_Generator")
