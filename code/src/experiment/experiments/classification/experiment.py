import json
import os
from pathlib import Path
from typing import Dict

import tensorflow as tf
from experiment.base_experiments import BaseExperiment
from experiment.experiments.classification.utils import preprocess_image

"""
CLASSIFICATION LABELS OF CURRENT USED DATASETS:

    MNIST images classification dataset.
    | Label | Description |
    |:-----:|-------------|
    |   0   | 0           |
    |   1   | 1           |
    |   2   | 2           |
    |   3   | 3           |
    |   4   | 4           |
    |   5   | 5           |
    |   6   | 6           |
    |   7   | 7           |
    |   8   | 8           |
    |   9   | 9           |

    FASHION MNIST images classification dataset.
    | Label | Description |
    |:-----:|-------------|
    |   0   | T-shirt/top |
    |   1   | Trouser     |
    |   2   | Pullover    |
    |   3   | Dress       |
    |   4   | Coat        |
    |   5   | Sandal      |
    |   6   | Shirt       |
    |   7   | Sneaker     |
    |   8   | Bag         |
    |   9   | Ankle boot  |

    CIFAR10 small images classification dataset.
    | Label | Description |
    |:-----:|-------------|
    |   0   | airplane    |
    |   1   | automobile  |
    |   2   | bird        |
    |   3   | cat         |
    |   4   | deer        |
    |   5   | dog         |
    |   6   | frog        |
    |   7   | horse       |
    |   8   | ship        |
    |   9   | truck       |
"""


class CLASSIFICATION_Experiment(BaseExperiment):
    """
    Experiment for a bunch of images in a folder, given a classifier model definition and a path to the weights.

    Eventually, we gotta do this in batches to avoid memory issues.
    --> However, this depends on if we do this on the local machine or on the server.
    """

    labels_json_file_name: str = "labels.json"
    certainties_json_file_name: str = "certainties.json"

    created_images_folder_path: Path
    model_path: Path

    classifications: Dict
    classifier_class = None
    classifier = None

    def __init__(self, *args, **kwargs):
        #  Todo: make the init explicit? Then we can have propper type hints, when calling the constructor.

        super().__init__(*args, **kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)

        self.labels_json_file_path = (
            self.created_images_folder_path / ".." / self.labels_json_file_name
        )
        self.certainties_json_file_path = (
            self.created_images_folder_path / ".." / self.certainties_json_file_name
        )

    def _setup(self):
        self.logger.info("################# Setup")
        pass

    def _load_data(self):
        image_file_names = os.listdir(self.created_images_folder_path)
        image_file_names = [fn for fn in image_file_names if fn.endswith(".png")]
        self.logger.info(f"Found {len(image_file_names)} images.")

        image_file_names = [
            self.created_images_folder_path / fn for fn in image_file_names
        ]

        self.logger.info(f"Loading {len(image_file_names)} images.")
        self.images = {
            fn.name: preprocess_image(fn, target_size=self.classifier_class.input_shape)
            for fn in image_file_names
        }

        # apparently most memory efficient way to get the first element in a dict.
        self.image_data_shape = next(iter(self.images.values())).shape

    def _initialize_models(self):
        self.classifier = self.classifier_class()

        _ = self.classifier(tf.random.normal(shape=self.image_data_shape))
        self.classifier.load_weights(self.model_path)

    def _run(self):
        self.classifications = {}
        self.certainties = {}

        for fn, img in self.images.items():
            classification = self.classifier(img)
            breakpoint()
            index = tf.argmax(classification, axis=-1).numpy().item()
            maximum = tf.reduce_max(classification, axis=-1).numpy().item()
            # --> to get to labels
            # classification = dataset_labels[self.classifier.dataset][index]
            self.classifications[fn] = index
            self.certainties[fn] = maximum

    def _save_results(self):
        classifications_json = json.dumps(self.classifications)
        certainties_json = json.dumps(self.certainties)
        with open(self.labels_json_file_path, "w") as f:
            f.write(classifications_json)
        with open(self.certainties_json_file_path, "w") as f:
            f.write(certainties_json)
