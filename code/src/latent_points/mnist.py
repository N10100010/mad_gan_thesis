from typing import List 

from tensorflow_probability import distributions as tfd
import tensorflow_probability as tfp


# generate points in latent space as input for the generator
def generate_latent_points(
    latent_dim: int,
    batch_size: int,
    n_gen: int
) -> List[tf.Tensor]:
    """
    Generate points in latent space as input for the generator.

    Parameters
    ----------
    latent_dim : int
        The dimensionality of the latent space.
    batch_size : int
        The batch size to generate.
    n_gen : int
        The number of generators to generate points for.

    Returns
    -------
    List[tf.Tensor]
        A list of `n_gen` tensors of shape `(batch_size, latent_dim)`.
    """
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
