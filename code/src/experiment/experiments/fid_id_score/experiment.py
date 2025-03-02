from pathlib import Path

import numpy as np
import tensorflow as tf
from experiment.base_experiments import BaseExperiment
from experiment.experiments.classification.utils import preprocess_image
from model_definitions.classifiers.base import BaseClassifier
from scoring_metrics import calculate_fid_score, calculate_inception_score


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
    n_generated_images: int = None

    def __init__(self, *args, **kwargs):
        #  Todo: make the init explicit? Then we can have propper type hints, when calling the constructor.

        super().__init__(*args, **kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)

        if self.model_path is None and self.classifier:
            # in this case, we assume the inceptionV3
            self.dataset = BaseClassifier.CIFAR10
        else:
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

        folder_path = Path(self.generation_experiment_path) / "generated_images"

        all_image_paths = [
            p for p in folder_path.iterdir() if p.suffix in [".jpg", ".jpeg", ".png"]
        ]

        if self.n_generated_images:
            idx = np.random.choice(
                len(all_image_paths), self.n_generated_images, replace=False
            )
            selected_image_paths = np.array(all_image_paths)[idx]
        else:
            selected_image_paths = all_image_paths

        images = []
        for filename in selected_image_paths[:100]:
            img_path = Path(folder_path) / filename
            try:
                if self.dataset == BaseClassifier.CIFAR10:
                    img = preprocess_image(img_path, target_size=(32, 32, 3))
                else:
                    img = preprocess_image(
                        img_path, target_size=self.classifier_class.input_shape
                    )
                images.append(img)
            except Exception as e:
                self.logger.warning(f"Could not load image {img_path}: {e}")

        if len(images) == 0:
            raise Exception(f"Could not load any images from folder {folder_path}.")

        self.image_data_shape = next(iter(images)).shape
        self.generated_images = np.array(images)

    def _initialize_models(self):
        if self.dataset != BaseClassifier.CIFAR10:
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
        if self.dataset == BaseClassifier.CIFAR10:
            fid_score = calculate_fid_score(
                generated_images=self.generated_images,
                dataset=self.dataset,
                classifier=tf.keras.applications.InceptionV3(
                    weights="imagenet", include_top=False, pooling="avg"
                ),
            )
            is_score = calculate_inception_score(
                generated_images=self.generated_images,
                classifier=tf.keras.applications.InceptionV3(
                    weights="imagenet", include_top=True
                ),
            )
        else:
            fid_score = calculate_fid_score(
                generated_images=self.generated_images,
                dataset=self.dataset,
                classifier=self.classifier,
            )
            is_score = calculate_inception_score(
                generated_images=self.generated_images,
                classifier=self.classifier,
            )

        self.logger.info(f"FID score: {fid_score}")
        self.logger.info(f"Inception score: {is_score}")
