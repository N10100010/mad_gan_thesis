from pathlib import Path

import numpy as np
import tensorflow as tf
from utils.plotting import plot_generators_examples


class MADGANMonitor(tf.keras.callbacks.Callback):
    def __init__(
        self,
        random_latent_vectors: list,
        data: np.ndarray,
        n_classes: int,
        latent_dim: int = 128,
        dir_name: Path = "Model",
        generate_after_epochs: int = 10,
    ) -> None:
        """
        Parameters
        ----------
        random_latent_vectors : list
            A list of fixed random latent vectors for generating images with.
        data : np.ndarray
            The training data.
        n_classes : int
            The number of classes in the dataset.
        latent_dim : int, optional
            The dimensionality of the latent space. Defaults to 128.
        dir_name : Path, optional
            The directory name to save the generated images in. Defaults to 'Model'.
        generate_after_epochs : int, optional
            The interval (in epochs) after which to generate images. Defaults to 10.
        """
        self.data = data[0:10]
        self.random_latent_vectors = random_latent_vectors
        self.latent_dim = latent_dim
        self.dir_name = dir_name
        self.n_classes = n_classes
        self.generate_after_epochs = generate_after_epochs

    def on_epoch_end(self, epoch: int, logs: dict = None) -> None:
        """
        Saves generated images after each epoch if the epoch number is divisible by `generate_after_epochs`.

        Parameters
        ----------
        epoch : int
            The epoch number.
        logs : dict, optional
            A dictionary of logs. Defaults to None.

        Returns
        -------
        None
        """
        if epoch % self.generate_after_epochs == 0:
            plot_generators_examples(
                n_rows=len(self.model.generators),
                n_cols=len(self.model.generators) + 1,
                dir_name=self.dir_name,
                random_latent_vectors=self.random_latent_vectors,
                data=self.data,
                generators=self.model.generators,
                epoch=epoch,
                save=True,
            )
