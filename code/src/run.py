import datetime 
import os
import imageio
import glob

import numpy as np 



import tensorflow as tf


from monitors.generator import GANMonitor
from model_definitions.mad_gan.mnist import MADGAN 
from model_definitions.generators.mnist.gen import define_generators
from model_definitions.discriminators.mnist.disc import define_discriminator

from loss_functions.generator import generators_loss_function
from latent_points.mnist import generate_latent_points
from datasets.mnist import dataset_func, get_dataset

def Generators_loss_function(y_true, y_pred): 
    logarithm = -tf.math.log(y_pred[:,-1] + 1e-15)
    return tf.reduce_mean(logarithm, axis=-1)


def func():
    
    n_gen = 2 #number of generators
    latent_dim = 256 #dimention of input noise
    batch_size = 256 #number of batches
    size_dataset = 60_000 #size MNIST dataset - 60_000
    epochs = 3
    steps_per_epoch = (size_dataset//batch_size)//n_gen
    type = 'no-stack'
    
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")

    dir_name = f"experiments/MNIST_{n_gen}-gen_{epochs}-ep_{type}-type_{current_date}"  
    # Check if the folder exists, if not, create it
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print(f"Folder created at: {dir_name}")
    else:
        print(f"Folder already exists at: {dir_name}")
        
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    if tf.test.gpu_device_name() == '/device:GPU:0':
        print("Using a GPU")
        print(f"GPUs available: {tf.config.list_physical_devices('GPU')}")
    else:
        print("Using a CPU")
        
    data, unique_labels = dataset_func() 
    dataset = tf.data.Dataset.from_tensor_slices(data) 
    dataset = dataset.repeat().shuffle(10 * size_dataset, reshuffle_each_iteration=True).batch(n_gen * batch_size, drop_remainder=True)

    discriminator = define_discriminator(n_gen)
    print(discriminator.summary())
    generators = define_generators(n_gen, latent_dim, class_labels=unique_labels)
    print(generators[0].summary())
    
    # creating MADGAN
    madgan = MADGAN(discriminator = discriminator, generators = generators, 
                latent_dim = latent_dim, n_gen = n_gen)
    
    madgan.compile(
        d_optimizer = tf.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
        g_optimizer = [tf.optimizers.Adam(learning_rate=1e-4, beta_1=0.5) for g in range(n_gen)],
        d_loss_fn = tf.keras.losses.CategoricalCrossentropy(),
        g_loss_fn = Generators_loss_function
    )   
    
    
    checkpoint_filepath = f'{dir_name}\checkpoint.weights.h5'
    random_latent_vectors = generate_latent_points(latent_dim = latent_dim, batch_size = 11, n_gen = n_gen)
    my_callbacks = [
        GANMonitor(
            random_latent_vectors=random_latent_vectors, 
            data = data, 
            n_classes=len(unique_labels), 
            latent_dim = latent_dim, 
            dir_name = dir_name, 
        ),    
        # This callback is for Saving the model
        tf.keras.callbacks.ModelCheckpoint(filepath = checkpoint_filepath , save_freq = 10 , save_weights_only = True),
    ]
    history = madgan.fit(dataset, epochs = 2, steps_per_epoch = steps_per_epoch, verbose = 1, callbacks = my_callbacks)    
    
    

if __name__ == '__main__':
    
    func()