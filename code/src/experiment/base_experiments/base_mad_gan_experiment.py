import json
from pathlib import Path
from typing import Dict, List
from matplotlib import pyplot as plt 
import numpy as np


from experiment.base_experiments.base_experiment import BaseExperiment
from model_definitions.mad_gan.mad_gan import MADGAN
from utils.plotting import generate_gan_training_gif, plot_training_history


class BaseMADGANExperiment(BaseExperiment):
    n_gen: int = None
    latent_dim: int = None
    size_dataset: int = None
    batch_size: int = None
    epochs: int = None
    steps_per_epoch: int = None
    generate_after_epochs: int = None
    generator_training_samples_subfolder: str = None
    madgan: MADGAN = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for k, v in kwargs.items():
            setattr(self, k, v)

    def _save_results(self):
        model_weights_path = Path(self.dir_path, "final_model.weights.h5")
        self.madgan.save_weights(model_weights_path)
        self.logger.info(f"Model saved to: {model_weights_path}")

        history_path = Path(self.dir_path, "training_history.npy")
        np.save(history_path, self.history.history)
        self.logger.info(f"Training history saved to: {history_path}")

        plot_training_history(
            history=self.history,
            path=self.dir_path,
        )

    def _final(self):
        generate_gan_training_gif(
            image_folder=self.dir_path / self.generator_training_samples_subfolder,
            output_gif=Path(self.dir_path, "generator_training.gif"),
            duration=300,
        )

    @classmethod
    def load_from_path(cls, path: Path):
        try:
            json_path = Path(path, "metadata.json")
            with open(json_path, "r") as f:
                metadata = json.load(f)
        except Exception as e:
            raise Exception(f"Error loading from metadata file {json_path}: {e}")
        return cls(**metadata)

    def load_model_weights(self, checkpoint: int = None):
        self._initialize_models()
        self.madgan.built = True
        if isinstance(self.dir_path, str):
            self.dir_path = Path(self.dir_path)

        if checkpoint is not None:
            file_path = (
                self.dir_path / "checkpoints" / f"backup_epoch_{checkpoint}.weights.h5"
            )
        else:
            file_path = self.dir_path / "final_model.weights.h5"
        if not file_path.exists():
            raise Exception(f"Model weights not found at {file_path}")

        self.madgan.load_weights(file_path)

    def generate_images(
        self, n_images: int, latent_vectors: List[np.ndarray], use_generator: int, save: bool = True
    ) -> Dict[int, np.ndarray]:
        """
        Generates images from a list of latent vectors. If use_generator is set, that specific generator will be used.
        Otherwise, the images will be generated from all generators.

        Args:
            n_images (int): _number of images to generate_
            latent_vectors (List[np.ndarray]): _list of latent vectors_
            use_generator (int): _index of generator to use_

        Raises:
            Exception: Out-of-Bounds-Error - if a generator index is out of bounds

        Returns:
            Dict[int: np.ndarray]: Dictionary with generator index as key and generated image as value
        """
        if self.madgan is None:
            raise Exception("MADGAN is not initialized")

        if use_generator:
            if 0 > use_generator or use_generator >= self.madgan.n_gen:
                raise Exception(
                    f"Generator index {use_generator} is out of bounds. Generator indizes: {range(self.madgan.n_gen)}"
                )
            image_data = {use_generator: []}
            for i in range(n_images):
                image_data[use_generator].append(
                    self.madgan.generators[use_generator](latent_vectors[i])
                )

        else:
            image_data = {i: [] for i in range(self.madgan.n_gen)}
            for i in range(n_images):
                for j in range(self.madgan.n_gen):
                    image = self.madgan.generators[j](latent_vectors[j])
                    plt.imshow(image[0])
                    image_data[j].append(image)

        return image_data
