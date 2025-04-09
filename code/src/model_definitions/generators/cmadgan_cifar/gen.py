from tensorflow import keras
from tensorflow.keras import layers


def build_generator_cnn_cifar(
    latent_dim, condition_dim, data_shape, name="Generator_CNN_CIFAR"
):
    """Builds a Conditional Generator Model for CIFAR-10 using Conv2DTranspose."""
    noise_input = keras.Input(shape=(latent_dim,), name="noise_input")
    condition_input = keras.Input(shape=(condition_dim,), name="condition_input")

    merged_input = layers.Concatenate()([noise_input, condition_input])

    # Project and reshape to start convolution at 4x4
    start_filters = 256  # Can be adjusted (e.g., 512)
    x = layers.Dense(4 * 4 * start_filters)(merged_input)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Reshape((4, 4, start_filters))(x)  # Shape: (4, 4, start_filters)

    # Upsample to 8x8
    x = layers.Conv2DTranspose(
        start_filters // 2, kernel_size=4, strides=2, padding="same"
    )(x)  # (8, 8, 128)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    # Upsample to 16x16
    x = layers.Conv2DTranspose(
        start_filters // 4, kernel_size=4, strides=2, padding="same"
    )(x)  # (16, 16, 64)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    # Upsample to 32x32
    x = layers.Conv2DTranspose(
        start_filters // 8, kernel_size=4, strides=2, padding="same"
    )(x)  # (32, 32, 32)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    # Final Conv layer to get 3 channel image (RGB)
    # Use tanh activation for pixel values in [-1, 1]
    output = layers.Conv2D(
        data_shape[-1], kernel_size=5, strides=1, padding="same", activation="tanh"
    )(x)  # (32, 32, 3)

    model = keras.Model([noise_input, condition_input], output, name=name)
    return model
