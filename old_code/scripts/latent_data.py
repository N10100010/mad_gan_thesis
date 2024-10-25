from tensorflow_probability import distributions as tfd
import tensorflow_probability as tfp


# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, batch_size, n_gen):
    # Multivariate normal diagonal distribution
    mvn = tfd.MultivariateNormalDiag(
        loc=[0]*latent_dim,
        scale_diag=[1.0]*latent_dim)

    noise = []
    for i in range(n_gen):
        # Some samples from MVN
        x_input = mvn.sample(batch_size)
        noise.append(x_input)
    return noise


# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, batch_size, n_gen):
    # Multivariate normal diagonal distribution
    mvn = tfd.MultivariateNormalDiag(
        loc=[0]*latent_dim,
        scale_diag=[1.0]*latent_dim)

    noise = []
    for i in range(n_gen):
        # Some samples from MVN
        x_input = mvn.sample(batch_size)
        noise.append(x_input)
        
    
    return noise

import tensorflow as tf
import tensorflow_probability as tfp

import tensorflow as tf
import tensorflow_probability as tfp

import numpy as np 

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
        latent_points = np.random.randn(latent_dim * batch_size).reshape(batch_size,latent_dim)
        
        # Generate labels
        labels = np.random.randint(0, num_classes, batch_size)
        labels = tf.one_hot(labels, num_classes)
        
        latent_points_list.append(latent_points)
        labels_list.append(labels)
    
    return latent_points_list, labels_list
