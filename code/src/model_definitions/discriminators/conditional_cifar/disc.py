import tensorflow as tf


def define_discriminator(
    latent_dim: int = 2048, n_classes: int = 10, image_shape=(32, 32, 3)
):
    """Creates a conditional discriminator model using projection-based conditioning."""

    img = tf.keras.layers.Input(shape=image_shape)

    x = tf.keras.layers.GaussianNoise(0.1)(img)

    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", strides=2)(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = tf.keras.layers.Conv2D(128, (3, 3), padding="same", strides=2)(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = tf.keras.layers.Conv2D(256, (3, 3), padding="same", strides=2)(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = tf.keras.layers.Conv2D(512, (3, 3), padding="same", strides=2)(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    label = tf.keras.layers.Input(shape=(1,))
    label_embedding = tf.keras.layers.Flatten()(
        tf.keras.layers.Embedding(n_classes, latent_dim)(label)
    )

    flat_img = tf.keras.layers.Flatten()(x)

    model_input = tf.keras.layers.multiply([flat_img, label_embedding])

    nn = tf.keras.layers.Dropout(0.3)(model_input)

    validity = tf.keras.layers.Dense(1, activation="sigmoid")(nn)

    return tf.keras.Model([img, label], validity)
