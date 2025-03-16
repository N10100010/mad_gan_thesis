from pathlib import Path
from typing import Type, TypeVar

import numpy as np
import tensorflow as tf
from experiment.base_experiments.base_experiment import BaseExperiment
from experiment.base_experiments.base_gan_experiment import BaseGANExperiment
from matplotlib import pyplot as plt
from model_definitions.conditional_gan.gan import ConditionalGAN
from utils.logging import setup_logger

X = TypeVar("X", bound=BaseExperiment)


class CondGAN_GenerativeCreationExperiment(BaseExperiment):
    experiment_class: Type[X]
    experiment_path: str = None

    latent_point_generator: callable
    n_images_per_class: int

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

        self.experiment.generator_func = self.generator_func
        self.experiment.discriminator_func = self.discriminator_func

    def _load_data(self):
        # self.latent_vectors = self.latent_point_generator(
        #    [self.n_images, self.experiment.latent_dim]
        # )
        ...

    def _initialize_models(self):
        self.experiment.load_model_weights()

        self.gan: ConditionalGAN = self.experiment.gan

    def _run(self):
        if self.gan is None:
            raise Exception("GAN is not initialized")

        image_data = {}
        for i in range(self.experiment.n_classes):
            labels = tf.reshape(
                tf.repeat([i], repeats=self.n_images_per_class), (-1, 1)
            )
            image_data[i] = self.gan.generator(
                [
                    self.latent_point_generator(
                        [self.n_images_per_class, self.experiment.latent_dim]
                    ),
                    labels,
                ],
                training=False,
            )

        self.image_data = image_data
        image_data = None

    def _save_results(self):
        saving_path = Path(self.dir_path, "generated_images")
        saving_path.mkdir(parents=True, exist_ok=True)

        for cls, images in self.image_data.items():
            for i, image in enumerate(images):
                image = image.numpy() if hasattr(image, "numpy") else np.array(image)
                image = (image + 1) / 2

                # If image has an extra channel (grayscale image with shape HxWx1), remove it.
                if image.ndim == 3 and image.shape[-1] == 1:
                    image = image.squeeze(axis=-1)

                # Determine the colormap: if image is 2D, we assume grayscale.
                cmap = "gray" if image.ndim == 2 else None

                if self.save_raw_image:
                    plt.imsave(
                        fname=saving_path / f"image_cls_{cls}_{i}.png",
                        arr=image,
                        cmap=cmap,
                    )  # sclae images from -1 ... 1 -> 0 ... 1
                else:
                    plt.imshow(image, cmap="gray")
                    plt.title(f"Image class {cls} - {i}")
                    plt.axis("off")
                    plt.savefig(saving_path / f"image_cls_{cls}_{i}.png")
                    plt.close()
