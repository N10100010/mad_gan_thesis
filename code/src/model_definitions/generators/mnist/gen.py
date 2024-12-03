import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Reshape, Conv2DTranspose, BatchNormalization, LeakyReLU
def define_generators(n_gen: int, latent_dim: int, class_labels: list) -> list[Model]:
    """
    Defines a number of generator models for a GAN.

    Parameters
    ----------
    n_gen : int
        The number of generator models to define.
    latent_dim : int
        The dimensionality of the latent space.
    class_labels : list
        A list of class labels for conditional generation.

    Returns
    -------
    list
        A list of generator models.
    """
    # Define the initial dense layer
    dens = Dense(units=7*7*256, use_bias=False, input_shape=(latent_dim,))
    batchnorm0 = BatchNormalization()
    rel0 = LeakyReLU()
    reshape0 = Reshape([7, 7, latent_dim])

    # Define the first Conv2DTranspose layer
    con2dt1 = Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)
    batchnorm1 = BatchNormalization()
    rel1 = LeakyReLU()

    # Define the second Conv2DTranspose layer
    con2dt2 = Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)
    batchnorm2 = BatchNormalization()
    rel2 = LeakyReLU()

    models = []
    for label in range(n_gen):
        # Input layer for the generator
        input = Input(shape=(latent_dim,), dtype=tf.float64, name=f"input_{label}")
        
        # Apply dense layer and reshape
        x = dens(input)
        x = batchnorm0(x)
        x = rel0(x)
        x = reshape0(x)

        # Apply first Conv2DTranspose layer
        x = con2dt1(x)
        x = batchnorm1(x)
        x = rel1(x)
        
        # Apply second Conv2DTranspose layer
        x = con2dt2(x)
        x = batchnorm2(x)
        x = rel2(x)
        
        # Final Conv2DTranspose layer to output image
        x = Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')(x)
        
        # Create and store model
        models.append(Model(input, x, name=f"generator{label}"))
        
    return models
