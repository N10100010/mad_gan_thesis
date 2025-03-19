import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from latent_points.utils import generate_latent_points
from model_definitions.classifiers.base import BaseClassifier
from scoring_metrics import calculate_fid_score, calculate_inception_score
from utils.logging import setup_logger


class ScoreMADGANMonitor(tf.keras.callbacks.Callback):
    logger = setup_logger(name="ScoreGANMonitor", prefix="\n MONITOR: ")

    def __init__(
        self,
        dir_name: str,
        latent_dim: int,
        dataset: str,
        total_epochs: int,
        score_calculation_freq: int = 1,
    ):
        super().__init__()

        self.dir_name = Path(dir_name)
        self.latent_dim = latent_dim
        self.dataset = dataset
        self.total_epochs = total_epochs
        self.score_calculation_freq = score_calculation_freq
        if dataset == BaseClassifier.CIFAR10:
            self.image_data_shape = (1, 32, 32, 3)
        else:
            self.image_data_shape = (1, 28, 28, 1)

        (self.dir_name / "scores").mkdir(parents=True, exist_ok=True)
        self.scores_file = self.dir_name / "scores" / "metrics.json"

        # Delete the file for restarts
        if self.scores_file.exists():
            self.scores_file.unlink()

    def on_epoch_end(self, epoch: int, logs: dict = None) -> None:
        self.logger.info(f"Calculating scores for epoch {epoch}...")

        if epoch % self.score_calculation_freq == 0:
            random_latent_vectors = np.array(generate_latent_points(self.latent_dim, 1, 1000))

            fid_classifier = tf.keras.applications.InceptionV3(
                weights="imagenet", include_top=False, pooling="avg"
            )
            is_classifier = tf.keras.applications.InceptionV3(weights="imagenet", include_top=True)

            scores_by_gen = {}
            for gen_nr, generator in enumerate(self.model.generators):
                generated_samples = []
                for random_vector in random_latent_vectors:
                    img = generator(random_vector).numpy()
                    img = np.clip(img, -1, 1)
                    img = img.reshape(-1, *self.image_data_shape[1:])
                    generated_samples.append(img)
                generated_samples = np.array(generated_samples)

                fid_score = calculate_fid_score(
                    generated_images=generated_samples, dataset=self.dataset, classifier=fid_classifier
                )
                is_score, is_std = calculate_inception_score(
                    generated_images=generated_samples, classifier=is_classifier
                )

                self.logger.info(
                    f"Epoch {epoch} - Generator {gen_nr}: FID: {fid_score}, IS: {is_score} +/- {is_std}"
                )
                scores_by_gen[gen_nr] = {
                    "epoch": epoch,
                    "FID": fid_score,
                    "IS": is_score,
                    "IS_std": is_std,
                }

            if self.scores_file.exists():
                with open(self.scores_file, "r") as f:
                    data = json.load(f)
            else:
                data = {}

            data[epoch] = scores_by_gen

            with open(self.scores_file, "w") as f:
                json.dump(data, f, indent=4)

            # Plot results if final epoch
            if epoch + 1 == self.total_epochs:
                self.plot_scores(data)

    def plot_scores(self, data):
        scores_dir = self.dir_name / "scores"

        epochs = sorted(data.keys())
        num_generators = len(next(iter(data.values())))

        fid_scores = {gen: [] for gen in range(num_generators)}
        is_scores = {gen: [] for gen in range(num_generators)}
        is_stds = {gen: [] for gen in range(num_generators)}

        for epoch in epochs:
            for gen in range(num_generators):
                fid_scores[gen].append(data[epoch][gen]["FID"])
                is_scores[gen].append(data[epoch][gen]["IS"])
                is_stds[gen].append(data[epoch][gen]["IS_std"])

        # Plot FID scores
        plt.figure(figsize=(10, 6))
        for gen in range(num_generators):
            plt.plot(epochs, fid_scores[gen], label=f"Generator {gen}")
        plt.xlabel("Epochs")
        plt.ylabel("FID Score")
        plt.title("FID Score Over Epochs")
        plt.legend()
        plt.savefig(scores_dir / "fid_plot.png")
        plt.close()

        # Plot IS scores with uncertainty bands
        fig, axes = plt.subplots(1, num_generators, figsize=(18, 6))
        gens_per_plot = (num_generators + 2) // num_generators

        for i, ax in enumerate(axes):
            for gen in range(i * gens_per_plot, min((i + 1) * gens_per_plot, num_generators)):
                ax.plot(epochs, is_scores[gen], label=f"Generator {gen}")
                ax.fill_between(
                    epochs,
                    np.array(is_scores[gen]) - np.array(is_stds[gen]),
                    np.array(is_scores[gen]) + np.array(is_stds[gen]),
                    alpha=0.2,
                )
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Inception Score")
            ax.set_title(f"Inception Score (Generators {i * gens_per_plot}-{min((i + 1) * gens_per_plot - 1, num_generators - 1)})")
            ax.legend()
        plt.tight_layout()
        plt.savefig(scores_dir / "is_plot.png")
        plt.close()
