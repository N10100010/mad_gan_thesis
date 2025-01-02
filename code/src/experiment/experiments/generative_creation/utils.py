from pathlib import Path
from typing import List, Type

import numpy as np
import tensorflow as tf
from experiment.base_experiments.base_mad_gan_experiment import BaseMADGANExperiment
from model_definitions.mad_gan import MADGAN


def load_madgan_model(path_to_weights: str) -> tf.keras.Model:
    return tf.keras.models.load_model(
        path_to_weights, custom_objects={"MADGAN": MADGAN}
    )


def load_madgan_generators(path_to_weights: str) -> List[tf.keras.Model]:
    madgan = load_madgan_model(path_to_weights)
    return madgan.generators


def generate_madgan_images(
    experiment_class: Type[BaseMADGANExperiment],
    model_path: str,
    latent_point_generator: callable,
    n_images: int,
    use_generator: int = None,
) -> List[np.ndarray]:
    experiment = experiment_class.load_from_path(Path(model_path))
    experiment.load_model_weights()

    latent_points = latent_point_generator(
        experiment.latent_dim, experiment.batch_size, experiment.n_gen
    )
    return experiment.generate_images(n_images, latent_points, use_generator)
