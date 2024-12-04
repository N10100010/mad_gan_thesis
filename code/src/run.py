import datetime 
import os
import imageio
import glob

import tensorflow as tf

from datasets.mnist import get_dataset

from model_definitions.mad_gan.mnist import MADGAN 
from model_definitions.discriminators.mnist.disc import define_discriminator 
from model_definitions.generators.mnist.gen import define_generators

from loss_functions.generator import generators_loss_function


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
        
    print("Loading dataset")
    dataset, labels = get_dataset(n_gen, batch_size, size_dataset)
    unique_labels = np.unique(labels)
    print("Dataset loaded")
    
    print("Defining generator loss")
    generator_loss = generators_loss_function()
    print("Generator loss defined")
    
    print("Defining discriminator")
    discriminator = define_discriminators(n_gen)
    print("Discriminator defined")
    
    print("Defining generators")
    generators = define_generators(n_gen,latent_dim, class_labels = unique_labels)
    print("Generators defined")
    
    print("Defining MADGAN")
    madgan = MADGAN(discriminator, generators, latent_dim, n_gen)
    print("MADGAN defined")
    
    print("Compiling MADGAN")
    madgan.compile(
        d_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
        g_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
        d_loss_fn=tf.keras.losses.BinaryCrossentropy(),
        g_loss_fn=generator_loss
    )
    print("MADGAN compiled")
    
    print("Defining checkpoint filepath")
    checkpoint_filepath = f'{dir_name}\checkpoint.weights.h5'
    print("Checkpoint filepath defined")
    
    print("Generating random latent vectors")
    random_latent_vectors = generate_latent_points(latent_dim = latent_dim, batch_size = 11, n_gen = n_gen)
    print("Random latent vectors generated")
    
    print("Defining callbacks")
    my_callbacks = [
        GANMonitor(
            random_latent_vectors=random_latent_vectors, 
            data = data, 
            n_classes=len(unique_labels), 
            latent_dim = latent_dim, 
            dir_name = dir_name, 
        ),    
        # This callback is for Saving the model
        tf.keras.callbacks.ModelCheckpoint(filepath = checkpoint_filepath , save_freq = 1 , save_weights_only = True),
    ]
    print("Callbacks defined")
    
    print("Training MADGAN")
    history = madgan.fit(dataset, epochs = 2, steps_per_epoch = steps_per_epoch, verbose = 1, callbacks = my_callbacks)
    print("MADGAN trained")

if __name__ == '__main__':
    
    func()