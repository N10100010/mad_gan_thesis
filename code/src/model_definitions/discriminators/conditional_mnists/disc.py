import tensorflow as tf
from tensorflow.keras.layers import (
    Concatenate,
    Conv2D,
    Dense,
    Dropout,
    Embedding,
    Flatten,
    Input,
    LeakyReLU,
    Reshape,
)
from tensorflow.keras.models import Model


def define_discriminator(image_shape=(28, 28, 1), n_classes: int = 10):
    img = Input(shape=image_shape)
    label = Input(shape=(1,), dtype="int32")

    label_embedding = Embedding(n_classes, tf.math.reduce_prod(image_shape))(label)
    label_embedding = Dense(tf.math.reduce_prod(image_shape))(label_embedding)
    label_embedding = Reshape(image_shape)(label_embedding)

    merged = Concatenate(axis=-1)([img, label_embedding])

    x = Conv2D(64, kernel_size=3, strides=2, padding="same")(merged)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.3)(x)

    x = Conv2D(128, kernel_size=3, strides=2, padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.3)(x)

    x = Flatten()(x)
    x = Dense(1, activation="sigmoid")(x)

    return Model([img, label], x)
