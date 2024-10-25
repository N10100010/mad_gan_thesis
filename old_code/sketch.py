
from tensorflow.keras.layers import *
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Model
from utils.utils import gpu_detection


from tensorflow.keras.layers import (Input, Dense, Dropout, LeakyReLU, 
                                     ReLU, Conv2D,Conv2DTranspose, Flatten,
                                     Reshape, BatchNormalization, Concatenate)

def build_generators(n_gen: int = 3, IN_SHAPE = (7, 7, 1)):
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
        
        out = Conv2DTranspose(filters=128, kernel_size=(4,4), strides=(2,2), padding='same')(gen)
        
        models.append(Model([latent_in, label_in], out, name=f"Generator_{gen_index}"))
    
    return models
        

if __name__ == "__main__":
    devices = gpu_detection()
    import os 
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(current_dir, 'code')) 
    
    generators = build_generators()
    generators[0].summary()
