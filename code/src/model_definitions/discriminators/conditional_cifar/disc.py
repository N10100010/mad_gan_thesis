import tensorflow as tf


def define_discriminator(n_classes, image_shape=(32, 32, 3)):
    """Creates a conditional discriminator model using projection-based conditioning."""

    # Embed and expand labels
    label_input = tf.keras.layers.Input(shape=(1,), dtype=tf.int32)
    li = tf.keras.layers.Embedding(n_classes, 50)(label_input)  # 1, 50
    li = tf.keras.layers.Dense(image_shape[0] * image_shape[1])(li)  # 1, 1024
    li = tf.keras.layers.Reshape((image_shape[0], image_shape[1], 1))(li)
    li = tf.keras.layers.Reshape((image_shape[0], image_shape[1], 1))(li)  # 32, 32, 1

    # Image input
    image_input = tf.keras.layers.Input(shape=image_shape)

    merged = tf.keras.layers.Concatenate()([image_input, li])  # 32, 32, 4

    x = tf.keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same")(merged)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)  # 16, 16, 128

    x = tf.keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same")(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)  # 8, 8, 128

    x = tf.keras.layers.Flatten()(x)  # 8192
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(1, activation="sigmoid")(x)  # 1

    return tf.keras.Model(
        [image_input, label_input], x, name="Conditional_Discriminator"
    )
