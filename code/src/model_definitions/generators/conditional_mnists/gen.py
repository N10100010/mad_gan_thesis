from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2DTranspose,
    Dense,
    Embedding,
    Flatten,
    Input,
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
