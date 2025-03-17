import tensorflow as tf
from tensorflow.keras.layers import (
    BatchNormalization,
    Concatenate,
    Conv2D,
    Conv2DTranspose,
    Dense,
    Dropout,
    Embedding,
    Flatten,
    Input,
    LeakyReLU,
    Multiply,
    Reshape,
)
from tensorflow.keras.models import Model


def define_generator(latent_dim: int = 100, n_classes: int = 10):
    noise = Input(shape=(latent_dim,))
    label = Input(shape=(1,), dtype="int32")

    label_embedding = Embedding(n_classes, latent_dim)(label)
    label_embedding = Flatten()(label_embedding)

    merged = Multiply()([noise, label_embedding])

    x = Dense(128 * 7 * 7, activation="relu")(merged)
    x = Reshape((7, 7, 128))(x)
    x = BatchNormalization(momentum=0.8)(x)

    x = Conv2DTranspose(
        128, kernel_size=3, strides=2, padding="same", activation="relu"
    )(x)
    x = BatchNormalization(momentum=0.8)(x)

    x = Conv2DTranspose(
        64, kernel_size=3, strides=2, padding="same", activation="relu"
    )(x)
    x = BatchNormalization(momentum=0.8)(x)

    x = Conv2DTranspose(1, kernel_size=3, strides=1, padding="same", activation="tanh")(
        x
    )

    return Model([noise, label], x)


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


# Example usage
generator = define_generator()
discriminator = define_discriminator()

generator.summary()
discriminator.summary()

# generator = define_generator()
# discriminator = define_discriminator()
# gan: tf.keras.Model = ConditionalGAN(
#     generator=generator,
#     discriminator=discriminator,
#     latent_dim=100,
#     n_classes=10,
# )
