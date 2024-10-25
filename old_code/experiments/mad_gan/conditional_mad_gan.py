from tensorflow.keras.layers import (Input, Dense, Dropout, LeakyReLU, 
                                     ReLU, Conv2D, Conv2DTranspose, Flatten,
                                     Reshape, BatchNormalization, Embedding, Multiply, Concatenate)
from tensorflow.keras import Model
import tensorflow as tf


def define_discriminator(n_gen: int, num_classes: int):
    # Input layers
    img_input = Input(shape=(28, 28, 1))
    label_input = Input(shape=(1,), dtype=tf.int32, name="label_input")

    # Embed and reshape the label to concatenate with the image
    label_embedding = Embedding(input_dim=num_classes, output_dim=28*28)(label_input)
    label_embedding = Reshape((28, 28, 1))(label_embedding)

    # Concatenate the image and label embedding
    combined_input = Concatenate()([img_input, label_embedding])

    # Convolutional layers
    x = Conv2D(64, (5, 5), strides=(2, 2), padding='same')(combined_input)
    x = LeakyReLU()(x)
    x = Dropout(0.3)(x)

    x = Conv2D(128, (5, 5), strides=(2, 2), padding='same')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.3)(x)

    # Flatten and Dense layers for final decision
    x = Flatten()(x)
    output = Dense(n_gen + 1, activation='softmax')(x)  # n_gen for fake classes, 1 for real class

    model = Model([img_input, label_input], output, name="Discriminator")
    return model

def define_generators(n_gen: int = 3, IN_SHAPE = (7, 7, 1)):
    NUM_CLASSES = 10
    LATENT_DIM = 128
    
    # Shared image foundation: 
    
    # Image Foundation 
    n_nodes = IN_SHAPE[0] * IN_SHAPE[1] * LATENT_DIM
    dens0 = Dense(n_nodes, name='SHARED-Foundation-Layer')
    batchnorm1 = BatchNormalization(name='SHARED-BatchNormalization')    
    relu1 = ReLU(name='SHARED-Foundation-Layer-Activation')
    resh1 = Reshape((IN_SHAPE[0], IN_SHAPE[1], 128), name='SHARED-Foundation-Layer-Reshape')
    concat1 = Concatenate(name='SHARED-Combine-Layer')
    conv1 = Conv2DTranspose(filters=128, kernel_size=(4,4), strides=(2,2), padding='same', name="SHARED-Conv2dT")
    
    batchnorm2 = BatchNormalization()    
    reulu2 = ReLU()
    
    conv2 = Conv2DTranspose(filters=128, kernel_size=(4,4), strides=(2,2), padding='same')
    batchnorm3 = BatchNormalization()    
    reulu3 = ReLU()
    

    models = []
    for gen_index in range(n_gen): 
        # Label Input:
        label_in = Input(shape=(1,), name='Label-Input') # Shape=(1,) because of class label
        lbls = Embedding(NUM_CLASSES, 50, name='Label-Embedding-Layer')(label_in) # Embed label to vector

        n_nodes = IN_SHAPE[0] * IN_SHAPE[1] 
        lbls = Dense(n_nodes, name='Label-Dense-Layer')(lbls)
        lbls = Reshape((IN_SHAPE[0], IN_SHAPE[1], 1), name='Label-Reshape-Layer')(lbls)

        # Normal Generator Input:
        latent_in = Input(shape=(LATENT_DIM ,), name='Latent-Input-Layer')
    
        gen = dens0(latent_in)
        gen = batchnorm1(gen)
        gen = relu1(gen)
        gen = resh1(gen)
        concat = concat1([gen, lbls])
        
        gen = conv1(concat) 
        gen = batchnorm2(gen)
        gen = reulu2(gen)
        
        gen = conv2(gen)
        gen = batchnorm3(gen)
        gen = reulu3(gen)
        
        out = Conv2D(filters=1, kernel_size=(7,7), activation='tanh', padding='same', name='Output-Layer')(gen)
        
        models.append(Model([latent_in, label_in], out, name=f"Generator_{gen_index}"))
    
    return models
        