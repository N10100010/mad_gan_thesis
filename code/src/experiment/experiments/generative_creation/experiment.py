from pathlib import Path
from typing import Dict, Type, TypeVar

import numpy as np
from experiment.base_experiments.base_experiment import BaseExperiment
from experiment.base_experiments.base_mad_gan_experiment import BaseMADGANExperiment
from matplotlib import pyplot as plt
from model_definitions.mad_gan.mad_gan import MADGAN
from utils.logging import setup_logger

X = TypeVar("X", bound=BaseExperiment)


class GenerativeCreationExperiment(BaseExperiment):
    experiment_class: Type[X]
    experiment_path: str = None

    latent_point_generator: callable
    n_images: int

    use_generator: int = None
    save: bool = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.logger = setup_logger(name=self.name)

    def _setup(self):
        self.experiment: BaseMADGANExperiment = self.experiment_class.load_from_path(
            Path(self.experiment_path)
        )
        pass

    def _load_data(self):
        self.latent_vectors = self.latent_point_generator(
            self.experiment.latent_dim, self.n_images, self.experiment.n_gen
        )

    def _initialize_models(self):
        self.experiment.load_model_weights()

        self.madgan: MADGAN = self.experiment.madgan

    def _run(self):
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

        if self.use_generator:
            if 0 > self.use_generator or self.use_generator >= self.madgan.n_gen:
                raise Exception(
                    f"Generator index {self.use_generator} is out of bounds. Generator indizes: {range(self.madgan.n_gen)}"
                )
            image_data = {self.use_generator: []}
            for i in range(self.n_images):
                image_data[self.use_generator].append(
                    self.madgan.generators[self.use_generator](self.latent_vectors[i])
                )

        else:
            if len(self.latent_vectors) < self.madgan.n_gen:
                raise Exception(
                    f"Number of latent vectors {len(self.latent_vectors)} is less than number of generators {self.madgan.n_gen}. Use setting with specific generator."
                )
            image_data = {i: [] for i in range(self.madgan.n_gen)}
            for i in range(self.n_images):
                for j in range(self.madgan.n_gen):
                    image = self.madgan.generators[j](self.latent_vectors[j])
                    plt.imshow(image[0])
                    image_data[j].append(image)

        self.image_data: Dict[int, np.ndarray] = image_data

    def _save_results(self):
        # Placeholder for saving results logic

        saving_path = Path(self.dir_path, "generated_images")
        saving_path.mkdir(parents=True, exist_ok=True)

        for gen_idx, batch in self.image_data.items():
            for batch_number, _images in enumerate(batch):
                for image_number, image in enumerate(_images):
                    plt.imshow(image / 127.5 * 127.5)
                    plt.title(f"Generator {gen_idx}")
                    plt.savefig(
                        saving_path
                        / f"gen_{gen_idx}_{batch_number}__{image_number}.png"
                    )
