from pathlib import Path
from typing import Type, TypeVar

import numpy as np
from experiment.base_experiments.base_experiment import BaseExperiment
from experiment.base_experiments.base_gan_experiment import BaseGANExperiment
from matplotlib import pyplot as plt
from model_definitions.vanilla_gan.gan import VanillaGAN
from utils.logging import setup_logger

X = TypeVar("X", bound=BaseExperiment)


class GAN_GenerativeCreationExperiment(BaseExperiment):
    experiment_class: Type[X]
    experiment_path: str = None

    latent_point_generator: callable
    n_images: int

    save: bool = True
    save_raw_image: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.logger = setup_logger(name=self.name)

    def _setup(self):
        self.experiment: BaseGANExperiment = self.experiment_class.load_from_path(
            Path(self.experiment_path)
        )

    def _load_data(self):
        self.latent_vectors = self.latent_point_generator(
            [self.n_images, self.experiment.latent_dim]
        )

    def _initialize_models(self):
        self.experiment.load_model_weights()

        self.gan: VanillaGAN = self.experiment.gan

    def _run(self):
        if self.gan is None:
            raise Exception("MADGAN is not initialized")

        image_data = []
        for i in range(self.n_images):
            image_data.append(self.gan.generator(self.latent_vectors))

        self.image_data: np.ndarray = image_data

    def _save_results(self):
        saving_path = Path(self.dir_path, "generated_images")
        saving_path.mkdir(parents=True, exist_ok=True)

        for i, batch in enumerate(self.image_data):
            for j, image in enumerate(batch):
                if self.save_raw_image:
                    plt.imsave(fname=saving_path / f"image_{j}.png", arr=image)
                else:
                    plt.imshow(image, cmap="gray")
                    plt.title(f"Image {j}")
                    plt.savefig(saving_path / f"image_{j}.png")
                    plt.close()
