import tensorflow as tf


def define_discriminator(in_shape=(32, 32, 3), ngf=64):
    inp = tf.keras.Input(shape=in_shape)
    x = tf.keras.layers.Conv2D(ngf, (3, 3), padding="same", input_shape=in_shape)(inp)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    x = tf.keras.layers.Conv2D(ngf * 2, (3, 3), strides=(2, 2), padding="same")(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(ngf * 4, (3, 3), strides=(2, 2), padding="same")(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(ngf * 8, (3, 3), strides=(2, 2), padding="same")(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    return tf.keras.Model(inputs=inp, outputs=x, name="Discriminator")


# Create the model
# discriminator = make_discriminator_model()
# discriminator.summary()
