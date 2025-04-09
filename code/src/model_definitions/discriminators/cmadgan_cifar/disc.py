from tensorflow import keras
from tensorflow.keras import layers


def build_discriminator_cnn_cifar(
    data_shape, condition_dim, name="Discriminator_CNN_CIFAR"
):
    """Builds a Conditional Discriminator Model for CIFAR-10 using Conv2D."""
    data_input = keras.Input(shape=data_shape, name="data_input")  # (32, 32, 3)
    condition_input = keras.Input(
        shape=(condition_dim,), name="condition_input"
    )  # (10,)

    # Process condition: Embed and reshape for concatenation
    # Project condition to match spatial dims * channels of an early layer or just spatially
    # Let's try reshaping to 32x32x1 for simple concatenation
    cond_embedding_size = data_shape[0] * data_shape[1]  # 32*32 = 1024
    c = layers.Dense(cond_embedding_size)(condition_input)
    c = layers.Reshape((data_shape[0], data_shape[1], 1))(c)  # Shape: (32, 32, 1)

    # Concatenate processed condition with image data along the channel axis
    merged_input = layers.Concatenate(axis=-1)(
        [data_input, c]
    )  # Shape: (32, 32, 3+1=4)

    start_filters = 64  # Can be adjusted

    # Downsample 32x32 -> 16x16
    x = layers.Conv2D(start_filters, kernel_size=4, strides=2, padding="same")(
        merged_input
    )  # (16, 16, 64)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.3)(x)

    # Downsample 16x16 -> 8x8
    x = layers.Conv2D(start_filters * 2, kernel_size=4, strides=2, padding="same")(
        x
    )  # (8, 8, 128)
    # Using LayerNormalization instead of BatchNormalization can sometimes help in Discriminators
    x = layers.LayerNormalization()(x)  # Optional: Experiment with LayerNorm/BatchNorm
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.3)(x)

    # Downsample 8x8 -> 4x4
    x = layers.Conv2D(start_filters * 4, kernel_size=4, strides=2, padding="same")(
        x
    )  # (4, 4, 256)
    x = layers.LayerNormalization()(x)  # Optional
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.3)(x)

    # Flatten and add Dense layers
    x = layers.Flatten()(x)
    # x = layers.Dense(128)(x) # Optional intermediate dense layer
    # x = layers.LeakyReLU(alpha=0.2)(x)

    # Output layer: Single logit
    output = layers.Dense(1)(x)

    model = keras.Model([data_input, condition_input], output, name=name)
    return model
