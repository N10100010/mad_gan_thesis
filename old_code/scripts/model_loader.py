import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Input, Dense, Dropout, LeakyReLU, 
                                 ReLU, Conv2D,Conv2DTranspose, Flatten,
                                 Reshape, BatchNormalization)
from tensorflow.keras import Model


from experiments.pydantic_models.config import DiscriminatorConfig, GeneratorsConfig, LayerConfig

def create_layer(layer: LayerConfig) -> tf.keras.layers.Layer:
    if layer.type == "Dense":
        return tf.keras.layers.Dense(
            units=layer.units,
            use_bias=layer.bias,
            input_shape=(layer.input_shape,)
        )
    elif layer.type == "BatchNormalization":
        return tf.keras.layers.BatchNormalization()
    elif layer.type == "LeakyReLU":
        return tf.keras.layers.LeakyReLU()
    elif layer.type == "Reshape":
        return tf.keras.layers.Reshape(target_shape=layer.values)
    elif layer.type == "Conv2DTranspose":
        return tf.keras.layers.Conv2DTranspose(
            filters=layer.units,
            kernel_size=layer.filter,
            strides=layer.stride,
            padding=layer.padding,
            use_bias=layer.bias,
            activation=layer.activation
        )
    elif layer.type == "Conv2D":
        return tf.keras.layers.Conv2D(
            layer.units,
            kernel_size=layer.filter,
            strides=layer.stride,
            padding=layer.padding
        )
    elif layer.type == "Dropout":
        return tf.keras.layers.Dropout(rate=layer.rate)
    elif layer.type == "Flatten":
        return tf.keras.layers.Flatten()
    elif layer.type == "Input":
        return tf.keras.layers.Input(shape=layer.shape)
    else:
        raise ValueError(f"Unsupported layer type: {layer.type}")

def build_generators(config: GeneratorsConfig) -> tf.keras.Model:
    latent_dim = config.latent_dim
    shared_layers = []
    
    # shared layers precreation
    for layer in config.shared_layers:
        shared_layers.append(create_layer(layer))
    
    models = []
    
    for geneartor_idx in range(config.num_generators): 
        
        # Define input
        input = tf.keras.layers.Input(shape=(latent_dim,))
        x = input
    
        # Apply separate layers before
        for layer in config.separate_layers["before"]:
            x = create_layer(layer)(x)
        
        # Apply shared layers
        for shared_layer in shared_layers: 
            x = shared_layer(x)
    
        # Apply separate layers after
        for layer in config.separate_layers["after"]:
            x = create_layer(layer)(x)
            
        models.append(tf.keras.Model(input, x, name=f"generator_{geneartor_idx}"))
    
    return models
    
def build_discriminator(config: DiscriminatorConfig) -> tf.keras.Model:
    
    # inputs = tf.keras.layers.Input(shape=(28, 28, 1))
    # x = inputs
    # 
    # for layer in config.layers:
    #     x = create_layer(layer)(x)  # Only call create_layer(layer)(x) once
    # 
    # cool_disc = tf.keras.Model(inputs, x, name="Discriminator")
    # return cool_disc

    inp = Input(shape=(28, 28, 1))
    x = inp

    x = Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1])(inp)
    x = LeakyReLU()(x)
    x = Dropout(0.3)(x)

    x = Conv2D(128, (5, 5), strides=(2, 2), padding='same')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.3)(x)
    
    x = Flatten()(x)
    out=Dense(3 + 1, activation = 'softmax')(x)   

    model = Model(inp, out, name="Discriminator")
    return model
