import sys
import os


from experiments.pydantic_models.config import Config
from scripts.config_loader import load_pydantic_object
from experiments.losses.mad_gan_generators_loss import generators_loss_function

from models.mad_gan import MADGAN
from models.callbacks.mad_gan_monitor import GANMonitor

from utils.utils import gpu_detection

from scripts.config_loader import load_pydantic_object
from scripts.latent_data import generate_latent_points

import tensorflow as tf 
import pathlib
import numpy as np 



def dataset_func(random_state = None):
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    # train_images = tf.image.resize(train_images, [32,32])
    train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

    # Convert to stacked-mnist(rgb images)
    #t1 = tf.random.shuffle(train_images, seed = 10)
    #t2 = tf.random.shuffle(train_images, seed = 20)
    #train_images = tf.concat([train_images, t1, t2], axis=-1)
    
    return train_images, train_labels

def plot_training_history(history, dir_name, save: bool = True):
    from matplotlib import pyplot as plt 
    # Extract losses from history
    history_dict = history.history
    generator_losses = []
    discriminator_loss = None
    
    # Separate generator losses and discriminator loss
    for key in history_dict.keys():
        if 'g_loss' in key:
            generator_losses.append((key, history_dict[key]))
        elif key == 'd_loss':
            discriminator_loss = history_dict[key]
    
    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Plot generator losses
    for gen_loss_key, gen_loss_values in generator_losses:
        plt.plot(gen_loss_values, label=gen_loss_key)
    
    # Plot discriminator loss
    if discriminator_loss is not None:
        plt.plot(discriminator_loss, label='d_loss', linewidth=2, linestyle='--', color='black')
    
    plt.title('Training Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    if save: 
        plt.savefig(f'{dir_name}/training_summary.png', dpi=200, format="png")






def run_normal_mad_gan(): 
    
    config = load_pydantic_object(
        path=r'C:\Users\NiXoN\Desktop\_thesis\_thesis_master\code\experiments\mad_gan\gen_3.yaml',
        pydantic_class=Config
        )
    
    
    #############################
    #           TODO:           #
    # - test the model loader 
    # - 
    #############################
    
    
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.losses import CategoricalCrossentropy
    from models.callbacks.mad_gan_monitor import GANMonitor
    
    from experiments.mad_gan.base_mad_gan import define_discriminator, define_generators
    
    madgan = MADGAN(config, define_generators, define_discriminator)
    
    madgan.compile(
        d_optimizer = Adam(learning_rate=2e-4, beta_1=0.5),
        g_optimizer = [Adam(learning_rate=1e-4, beta_1=0.5) for g in range(config.model.num_generators)],
        d_loss_fn = CategoricalCrossentropy(),
        g_loss_fn = generators_loss_function
    )
    
    madgan.discriminator.summary()
    madgan.generators[0].summary()
    
    n_gen = config.model.num_generators #number of generators
    latent_dim = config.model.latent_dim #dimention of input noise
    batch_size = latent_dim #number of batches
    size_dataset = 60_000 #size MNIST dataset - 60_000
    epochs = 3
    steps_per_epoch = (size_dataset//batch_size)//n_gen

    
    data, unique_labels = dataset_func()
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.repeat().shuffle(10 * size_dataset, reshuffle_each_iteration=True).batch(n_gen * batch_size, drop_remainder=True)

    dir_name = pathlib.Path(__file__).parent.resolve() / pathlib.Path(f'experiments\_experiments\{config.general.experiment_name}')
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
        tf.keras.callbacks.ModelCheckpoint(filepath = checkpoint_filepath , save_freq = 50 , save_weights_only = True),
    ]
    
    history = madgan.fit(dataset, epochs = 3, steps_per_epoch = steps_per_epoch, verbose = 1, callbacks = my_callbacks)
    
    plot_training_history(history, dir_name = dir_name)
    

import pathlib
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from models.cond_mad_gan import CondMADGAN

from experiments.mad_gan.conditional_mad_gan import define_discriminator, define_generators
from experiments.losses.mad_gan_generators_loss import generators_loss_function
from models.callbacks.mad_gan_monitor import GANMonitor
from scripts.latent_data import generate_latent_points_cond
from experiments.pydantic_models.config import Config
from scripts.config_loader import load_pydantic_object

def dataset_func_cond():
    # Load MNIST dataset
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    
    # Normalize the images to the range [-1, 1] and expand dimensions to include the channel dimension
    train_images = (train_images.astype('float32') - 127.5) / 127.5
    train_images = tf.expand_dims(train_images, axis=-1)
    
    # Convert labels to one-hot encoding
    unique_labels = tf.constant(tf.range(10))
    train_labels = tf.one_hot(train_labels, depth=10)
    
    return train_images, train_labels, unique_labels

import numpy as np
import tensorflow as tf

def generate_latent_points_cond(latent_dim, batch_size, n_gen, num_classes):
    """
    Generates random latent points and corresponding labels for each generator.
    
    :param latent_dim: Dimension of the latent space.
    :param batch_size: Number of latent points to generate per generator.
    :param n_gen: Number of generators.
    :param num_classes: Number of classes for conditional generation.
    
    :return: Tuple containing a list of latent points (one per generator) and corresponding labels.
    """
    latent_points_list = []
    labels_list = []
    
    for _ in range(n_gen):
        # Generate latent points
        latent_points = np.random.randn(batch_size, latent_dim)
        
        # Generate labels
        labels = np.random.randint(0, num_classes, batch_size)
        labels = tf.one_hot(labels, num_classes)
        
        latent_points_list.append(latent_points)
        labels_list.append(labels)
    
    return latent_points_list, labels_list


def run_conditional_mad_gan(): 
    config = load_pydantic_object(
        path=r'C:\Users\NiXoN\Desktop\_thesis\_thesis_master\code\experiments\mad_gan\gen_3.yaml',
        pydantic_class=Config
    )
        
    latent_dim = config.model.latent_dim  # Dimension of input noise
    batch_size = latent_dim  # Number of batches

    cond_mad_gan = CondMADGAN(config, define_generators, define_discriminator, batch_size=batch_size)
    
    cond_mad_gan.compile(
        d_optimizer=Adam(learning_rate=2e-4, beta_1=0.5),
        g_optimizer=[Adam(learning_rate=1e-4, beta_1=0.5) for _ in range(config.model.num_generators)],
        d_loss_fn=CategoricalCrossentropy(),
        g_loss_fn=generators_loss_function
    )
    
    cond_mad_gan.discriminator.summary()
    cond_mad_gan.generators[0].summary()
    
    n_gen = config.model.num_generators  # Number of generators
    latent_dim = config.model.latent_dim  # Dimension of input noise
    size_dataset = 60_000  # Size of MNIST dataset - 60,000
    epochs = 3
    steps_per_epoch = (size_dataset // batch_size) // n_gen

    data, labels, unique_labels = dataset_func_cond()  # Load dataset and labels
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.repeat().shuffle(10 * size_dataset, reshuffle_each_iteration=True).batch(n_gen * batch_size, drop_remainder=True)

    dir_name = pathlib.Path(__file__).parent.resolve() / pathlib.Path(f'experiments\_experiments\{config.general.experiment_name}')
    checkpoint_filepath = f'{dir_name}\checkpoint.weights.h5'
    random_latent_vectors = generate_latent_points_cond(latent_dim=latent_dim, batch_size=11, n_gen=n_gen, num_classes=10)

    my_callbacks = [
        GANMonitor(
            random_latent_vectors=random_latent_vectors, 
            data=data, 
            n_classes=len(unique_labels), 
            latent_dim=latent_dim, 
            dir_name=dir_name, 
        ),    
        # This callback is for saving the model
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, save_freq=50, save_weights_only=True),
    ]
    
    
    from matplotlib import pyplot as plt 
    LATENT_DIM = 128

    
    latent_input = np.random.randn(LATENT_DIM)
    latent_input = latent_input.reshape(1, LATENT_DIM)
    
    pred = cond_mad_gan.generators[0].predict([latent_input, np.array([1])])
    plt.figure(figsize=(5, 5))
    plt.title("Example image from Generator")
    plt.imshow(pred[0], cmap='gray')
    plt.xlabel("Pixel")
    plt.ylabel("Pixel")
    plt.colorbar()
    # plt.show()
    
    history = cond_mad_gan.fit(dataset, epochs=epochs, steps_per_epoch=steps_per_epoch, verbose=1, callbacks=my_callbacks)
    
    plot_training_history(history)




if __name__ == "__main__":
    devices = gpu_detection()
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(current_dir, 'code'))  # Platform-independent
    
    #run_normal_mad_gan()
    run_conditional_mad_gan()

    

print("finished")