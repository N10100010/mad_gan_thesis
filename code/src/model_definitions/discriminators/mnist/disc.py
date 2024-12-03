from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, Flatten, Dense, Dropout
# define the standalone discriminator model
def define_discriminator(n_gen: int):
    """
    Defines the standalone discriminator model.

    Parameters
    ----------
    n_gen : int
        The number of generators in the GAN.

    Returns
    -------
    model : Model
        The discriminator model.
    """
    inp = Input(shape=(28, 28, 1))

    # Conv 1
    x = Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1])(inp)
    x = LeakyReLU()(x)
    x = Dropout(0.3)(x)

    # Conv 2
    x = Conv2D(128, (5, 5), strides=(2, 2), padding='same')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.3)(x)
    
    # Flatten
    x = Flatten()(x)
    
    # Output layer
    out = Dense(n_gen + 1, activation = 'softmax')(x)

    model = Model(inp, out, name="Discriminator")
    return model
