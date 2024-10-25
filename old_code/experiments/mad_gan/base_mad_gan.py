
from tensorflow.keras.layers import (Input, Dense, Dropout, LeakyReLU, 
                                     ReLU, Conv2D,Conv2DTranspose, Flatten,
                                     Reshape, BatchNormalization)
from tensorflow.keras import Model

import tensorflow as tf 


# define the standalone discriminator model
def define_discriminator(n_gen, num_classes: int):
    inp = Input(shape=(28, 28, 1))

    x = Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1])(inp)
    x = LeakyReLU()(x)
    x = Dropout(0.3)(x)

    x = Conv2D(128, (5, 5), strides=(2, 2), padding='same')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.3)(x)
    
    x = Flatten()(x)
    out = Dense(n_gen + 1, activation = 'softmax')(x)

    model = Model(inp, out, name="Discriminator")
    return model

def define_generators(n_gen, latent_dim, num_classes: int):
    dens = Dense(units=7*7*256, use_bias=False, input_shape=(latent_dim,))
    batchnorm0 = BatchNormalization()
    rel0 = LeakyReLU()
    reshape0 = Reshape([7,7,latent_dim])

    con2dt1 = Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)
    batchnorm1 = BatchNormalization()
    rel1 = LeakyReLU()

    con2dt2 = Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)
    batchnorm2 = BatchNormalization()
    rel2 = LeakyReLU()

    models = []
    for label in range(n_gen):
        input = Input(shape=(latent_dim,), dtype = tf.float64, name=f"input_{label}")
        x = dens(input)
        x = batchnorm0(x)
        x = rel0(x)
        x = reshape0(x)

        x = con2dt1(x)
        x = batchnorm1(x)
        x = rel1(x)
        
        x = con2dt2(x)
        x = batchnorm2(x)
        x = rel2(x)
        
        x = Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')(x)
        
        models.append(Model(input, x, name = f"generator{label}"))
    return models