import tensorflow as tf


def define_generator(latent_dim):
    kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    inp = tf.keras.layers.Input(shape=(latent_dim,))
    x = tf.keras.layers.Dense(
        8 * 8 * 512, use_bias=False, kernel_initializer=kernel_initializer
    )(inp)
    x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)  # Increased momentum
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Reshape((8, 8, 512))(x)

    # Add dropout for regularization
    x = tf.keras.layers.Dropout(0.3)(x)

    # Upsample with larger kernels
    x = tf.keras.layers.Conv2DTranspose(
        256,
        (5, 5),
        strides=(2, 2),
        padding="same",
        use_bias=False,  # Changed from 4x4
    )(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2DTranspose(
        128, (5, 5), strides=(2, 2), padding="same", use_bias=False
    )(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
    x = tf.keras.layers.ReLU()(x)

    # Additional convolutional layer
    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
    x = tf.keras.layers.ReLU()(x)

    out = tf.keras.layers.Conv2D(3, (3, 3), padding="same", activation="tanh")(
        x
    )  # Direct conv instead of transpose

    return tf.keras.Model(inp, out, name="Generator")
