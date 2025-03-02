from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class VanillaGANMonitor(tf.keras.callbacks.Callback):
    def __init__(
        self,
        random_latent_vectors: list,
        data: np.ndarray,
        n_classes: int,
        latent_dim: int = 128,
        latent_point_generator: Callable = None,
        dir_name: Path = "Model",
        samples_subfolder: str = "generators_examples",
        generator_example_freq: int = 10,
        save: bool = True,
    ):
        super().__init__()

        self.data = data
        self.random_latent_vectors = random_latent_vectors
        self.latent_dim = latent_dim
        self.latent_point_generator = latent_point_generator
        self.n_classes = n_classes
        self.dir_name = dir_name
        self.samples_subfolder = samples_subfolder
        self.generator_example_freq = generator_example_freq
        self.save = save

    def create_generator_images(self, epoch: int) -> None:
        n_cols = 2
        fig, axes = plt.subplots(nrows=1, ncols=n_cols, figsize=(12, 8))
        fig.suptitle(f"Epoch: {epoch}", fontsize=20)
        axes = axes.flatten()
        generated_sample = self.model.generator(self.random_latent_vectors)

        for ax_index, ax in enumerate(axes):
            if (ax_index + 1) % n_cols == 0:
                ax.imshow(
                    (
                        self.data[np.random.randint(self.data.shape[0]), :, :] * 127.5
                        + 127.5
                    )
                    / 255,
                    cmap="gray",
                )
                ax.set_title("Real (random)")
            else:
                ax.imshow(
                    (
                        generated_sample[
                            np.random.randint(generated_sample.shape[0]), :, :
                        ]
                        * 127.5
                        + 127.5
                    )
                    / 255,
                    cmap="gray",
                )
                ax.set_title("Generated Image")
                ax.axis("off")

        fig.tight_layout()

        if self.save:
            Path(self.dir_name / self.samples_subfolder).mkdir(
                exist_ok=True, parents=True
            )
            plt.savefig(
                self.dir_name
                / self.samples_subfolder
                / f"image_at_epoch_{(epoch + 1):04}.png",
                dpi=200,
                format="png",
            )
            plt.close()

    def on_epoch_end(self, epoch: int, logs: dict = None) -> None:
        if epoch % self.generator_example_freq == 0:
            self.create_generator_images(epoch)
