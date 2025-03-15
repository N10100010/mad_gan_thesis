import tensorflow as tf


def define_generator(latent_dim: int = 2048, n_classes: int = 10):
    noise = tf.keras.layers.Input(shape=(latent_dim,))
    label = tf.keras.layers.Input(shape=(1,))

    label_embedding = tf.keras.layers.Flatten()(
        tf.keras.layers.Embedding(n_classes, latent_dim)(label)
    )

    model_input = tf.keras.layers.multiply([noise, label_embedding])

    x = tf.keras.layers.Dense(2048)(model_input)

    x = tf.keras.layers.Reshape((2, 2, 512))(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    x = tf.keras.layers.LeakyReLU(0.1)(x)

    x = tf.keras.layers.Conv2DTranspose(256, (5, 5), padding="same", strides=2)(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    x = tf.keras.layers.LeakyReLU(0.1)(x)

    x = tf.keras.layers.Conv2DTranspose(128, (5, 5), padding="same", strides=2)(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    x = tf.keras.layers.LeakyReLU(0.1)(x)

    x = tf.keras.layers.Conv2DTranspose(64, (5, 5), padding="same", strides=2)(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    x = tf.keras.layers.LeakyReLU(0.1)(x)

    x = tf.keras.layers.Conv2DTranspose(3, (5, 5), padding="same", strides=2)(x)
    img = tf.keras.layers.Activation("tanh")(x)

    return tf.keras.Model([noise, label], img)
