import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
        self.scores_folder = self.dir_name / "scores"
        self.scores_file = self.scores_folder / "metrics.json"

        # Delete the file for restarts
        if self.scores_file.exists():
            self.scores_file.unlink()

    def on_epoch_end(self, epoch: int, logs: dict = None) -> None:
        self.logger.info(f"Calculating scores for epoch {epoch}...")

        if epoch % self.score_calculation_freq == 0:
            random_latent_vectors = np.array(
                generate_latent_points(self.latent_dim, 1, 1000)
            )

            scores_by_gen = {}
            for gen_nr, generator in enumerate(self.model.generators):
                generated_samples = []
                for random_vector in random_latent_vectors:
                    img = generator(random_vector).numpy()
                    img = np.clip(img, -1, 1)
                    img = img.reshape(-1, *self.image_data_shape[1:])
                    generated_samples.append(img)
                generated_samples = np.array(generated_samples)
                generated_samples = np.squeeze(generated_samples, axis=1)

                fid_score = calculate_fid_score(
                    generated_images=generated_samples, dataset=self.dataset
                )
                is_score, is_std = calculate_inception_score(
                    generated_images=generated_samples
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
    # Convert string keys to integers if necessary
    data = {
        int(epoch): {int(gen): values for gen, values in gen_data.items()}
        for epoch, gen_data in data.items()
    }

    epochs = sorted(data.keys())
    num_generators = len(
        next(iter(data.values()))
    )  # Assumes all epochs have the same number of generators

    # FID Scores Plot (Single Figure)
    plt.figure(figsize=(10, 5))
    for gen in range(num_generators):
        fid_scores = [data[epoch][gen]["FID"] for epoch in epochs]
        plt.plot(epochs, fid_scores, marker="o", label=f"Generator {gen}")

    plt.xlabel("Epochs")
    plt.ylabel("FID Score")
    plt.title("FID Scores Over Epochs")
    plt.legend()
    plt.grid(True)

    plt.savefig(self.scores_folder / "FID.png")

    plt.close()

    # IS Scores Plot (Subplots for each Generator)
    fig, axes = plt.subplots(
        num_generators, 1, figsize=(10, 5 * num_generators), sharex=True
    )

    if num_generators == 1:
        axes = [axes]  # Ensure axes is iterable even for one generator

    for gen in range(num_generators):
        is_scores = [data[epoch][gen]["IS"] for epoch in epochs]
        is_stds = [data[epoch][gen]["IS_std"] for epoch in epochs]

        axes[gen].plot(
            epochs, is_scores, marker="o", label=f"Generator {gen}", color=f"C{gen}"
        )
        axes[gen].fill_between(
            epochs,
            np.array(is_scores) - np.array(is_stds),
            np.array(is_scores) + np.array(is_stds),
            alpha=0.2,
            color=f"C{gen}",
        )

        axes[gen].set_ylabel("IS Score")
        axes[gen].set_title(f"IS Score - Generator {gen}")
        axes[gen].legend()
        axes[gen].grid(True)

    axes[-1].set_xlabel("Epochs")
    plt.tight_layout()
    plt.savefig(self.scores_folder / "IS.png")
