import tensorflow as tf
from model_definitions.custom_layers.minibatchnorm import MinibatchDiscrimination


def define_discriminator(in_shape=(32, 32, 3)):
    # kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
    # alpha = 0.2

    # testing new discriminator architecture
    inp = tf.keras.layers.Input(shape=in_shape)
    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", input_shape=in_shape)(inp)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same")(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same")(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), strides=(2, 2), padding="same")(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = MinibatchDiscrimination()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(1)(x)

    return tf.keras.Model(inp, x, name="Discriminator")
