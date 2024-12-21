from typing import List

import tensorflow as tf
from model_definitions.mad_gan import MADGAN


def load_madgan_model(path_to_weights: str) -> tf.keras.Model:
    return tf.keras.models.load_model(
        path_to_weights, custom_objects={"MADGAN": MADGAN}
    )


def load_madgan_generators(path_to_weights: str) -> List[tf.keras.Model]:
    madgan = load_madgan_model(path_to_weights)
    return madgan.generators
