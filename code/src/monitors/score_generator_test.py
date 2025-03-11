import json
from pathlib import Path
from typing import Callable

import numpy as np
import tensorflow as tf
from model_definitions.classifiers.base import BaseClassifier
from scoring_metrics import calculate_fid_score, calculate_inception_score
from utils.logging import setup_logger


class ScoreGANMonitor(tf.keras.callbacks.Callback):
    logger = setup_logger(name="ScoreGANMonitor", prefix="\n MONITOR: ")

    def __init__(
        self,
        dir_name: str,
        latent_dim: int,
        latent_point_generator: Callable,
        dataset: str,
        classifier_class: tf.keras.Model,
        classifier: tf.keras.Model = None,
        model_path: str = None,
        score_calculation_freq: int = 1,
    ):
        super().__init__()

        self.dir_name = Path(dir_name)
        self.latent_dim = latent_dim
        self.latent_point_generator = latent_point_generator
        self.dataset = dataset
        self.classifier_class = classifier_class
        self.classifier = classifier
        self.model_path = Path(model_path) if model_path else None
        self.score_calculation_freq = score_calculation_freq

        if dataset == BaseClassifier.CIFAR10:
            self.image_data_shape = (1, 32, 32, 3)
        else:
            self.image_data_shape = (1, 28, 28, 1)

        (self.dir_name / "scores").mkdir(parents=True, exist_ok=True)
        self.scores_file = self.dir_name / "scores" / "metrics.json"

    def on_epoch_end(self, epoch: int, logs: dict = None) -> None:
        self.logger.info(f"Calculating scores for epoch {epoch}...")
        if epoch % self.score_calculation_freq == 0:
            random_latent_vectors = tf.random.normal(
                shape=(1000, 1, 1, self.latent_dim)
            )
            generated_samples = self.model.generator(random_latent_vectors)
            generated_samples = generated_samples.numpy()
            generated_samples = np.clip(generated_samples, 0, 1)
            ## TODO: ensure that the generated samples are in the correct scale
            generated_samples = generated_samples.reshape(
                -1, *self.image_data_shape[1:]
            )

            # we want to load the classifier ad-hoc.
            # otherwise, it would be loaded throughout the entire training and we don't have the memory for that
            # --> We also unload the model at the end of scoring
            if self.classifier is None:
                self.classifier: tf.keras.Model = self.classifier_class()
                _ = self.classifier(tf.random.normal(shape=self.image_data_shape))
                self.classifier.load_weights(self.model_path)

            if self.dataset == BaseClassifier.CIFAR10:
                fid_score = calculate_fid_score(
                    generated_images=generated_samples,
                    dataset=self.dataset,
                    classifier=tf.keras.applications.InceptionV3(
                        weights="imagenet", include_top=False, pooling="avg"
                    ),
                )
                is_score, is_std = calculate_inception_score(
                    generated_images=generated_samples,
                    classifier=tf.keras.applications.InceptionV3(
                        weights="imagenet", include_top=True
                    ),
                )
            else:
                fid_score = calculate_fid_score(
                    generated_images=generated_samples,
                    dataset=self.dataset,
                    classifier=self.classifier,
                )
                is_score, is_std = calculate_inception_score(
                    generated_images=generated_samples,
                    classifier=self.classifier,
                )

            self.logger.info(
                f"Epoch {epoch}: FID: {fid_score}, IS: {is_score} +/- {is_std}"
            )

            scores = {
                "epoch": epoch,
                "FID": fid_score,
                "IS": is_score,
                "IS_std": is_std,
            }
            if self.scores_file.exists():
                with open(self.scores_file, "r") as f:
                    data = json.load(f)
            else:
                data = []

            data.append(scores)

            with open(self.scores_file, "w") as f:
                json.dump(data, f, indent=4)

            self.classifier = None
