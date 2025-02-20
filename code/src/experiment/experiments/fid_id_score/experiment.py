import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from experiment.base_experiments import BaseExperiment
from experiment.experiments.classification.utils import preprocess_image
from model_definitions.classifiers.base import BaseClassifier
from scoring_metrics import calculate_fid_score


class ScoringExperiment(BaseExperiment):
    """
    Creates metrics in form of json and plots, given a set of images.
    """

    model_path: str = None
    classifier_class = (
        BaseClassifier  # a reference to the class defining the classifier model
    )
    classifier: tf.keras.Model = None

    # The experiment that generated images - assumed to contain a folder named generated_images
    generation_experiment_path: str = None

    def __init__(self, *args, **kwargs):
        #  Todo: make the init explicit? Then we can have propper type hints, when calling the constructor.

        super().__init__(*args, **kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)

        self.dataset = self.classifier_class.dataset

    def _load_data(self):
        """
        Loads all images from a specified folder into a NumPy array.

        Parameters:
            folder_path (str): Path to the folder containing images.
            target_size (tuple): Desired image size (height, width).

        Returns:
            numpy.ndarray: Array of images with shape (N, H, W, C).
        """

        image_target_size = self.classifier_class.input_shape[0:2]
        folder_path = os.path.join(self.generation_experiment_path, "generated_images")

        images = []
        for filename in os.listdir(folder_path)[:1000]:
            img_path = Path(folder_path) / filename
            try:
                img = preprocess_image(
                    img_path, target_size=self.classifier_class.input_shape
                )
                images.append(img)
            except Exception as e:
                print(f"Could not load image {img_path}: {e}")

        self.image_data_shape = next(iter(images)).shape
        self.generated_images = np.array(images)

    def _initialize_models(self):
        self.classifier = self.classifier_class()
        _ = self.classifier(tf.random.normal(shape=self.image_data_shape))
        self.classifier.load_weights(self.model_path)

    def _run(self):
        """
        Eventually, this function has to become more elaborate.
        Currently, we just calculate the scores over all the generated images. BUT, we could also calculate the scores per class.
        --> however, therefor we'd need to also load the labels for the generated images and e.g. only calc the score on those the have a certainty higher than X.

        """
        self.logger.info("Calculating scores...")

        fid_score = calculate_fid_score(
            generated_images=self.generated_images,
            dataset=self.dataset,
            classifier=self.classifier,
        )
        self.logger.info(f"FID score: {fid_score}")
        # is_score = calculate_inception_score(generated_images=self.generated_images, classifier=self.classifier)
        # self.logger.info("Inception score: ", is_score)
