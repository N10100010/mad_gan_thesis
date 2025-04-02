from tensorflow import keras
from tensorflow.keras import layers


def define_generators(latent_dim, condition_dim, data_shape, name="Generator_CNN"):
    """Builds a Conditional Generator Model using Conv2DTranspose."""
    noise_input = keras.Input(shape=(latent_dim,), name="noise_input")
    condition_input = keras.Input(shape=(condition_dim,), name="condition_input")

    # Combine noise and condition, project using Dense
    merged_input = layers.Concatenate()([noise_input, condition_input])

    x = layers.Dense(7 * 7 * 128)(merged_input)  # Project to 7x7x128 features
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Reshape((7, 7, 128))(x)  # Reshape to start convolutional transpose

    # Upsample to 14x14
    x = layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding="same")(
        x
    )  # Output: 14x14x64
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    # Upsample to 28x28
    x = layers.Conv2DTranspose(32, kernel_size=4, strides=2, padding="same")(
        x
    )  # Output: 28x28x32
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    output = layers.Conv2D(
        data_shape[-1], kernel_size=5, strides=1, padding="same", activation="tanh"
    )(x)

    model = keras.Model([noise_input, condition_input], output, name=name)
    return model
