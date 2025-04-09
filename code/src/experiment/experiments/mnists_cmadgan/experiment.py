import os

import tensorflow as tf
from datasets.fasion_mnist import cmadgan_dataset_func as fashion_mnist_dataset_func
from datasets.mnist import cmadgan_dataset_func as mnist_dataset_func
from experiment.base_experiments import BaseMADGANExperiment
from matplotlib import pyplot as plt
from model_definitions.cmadgan.cmadgan import CMADGAN
from monitors.cmadgan_generator import CMADGANMonitor


def save_generated_images(
    epoch, generated_sample_sets, output_dir, examples=10, dim=(3, 10), figsize=(10, 3)
):
    """Saves a grid of generated digits for each generator."""
    fig, axes = plt.subplots(dim[0], dim[1], figsize=figsize)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    for i in range(dim[0]):  # Loop through generators
        for j in range(dim[1]):  # Loop through examples (digits 0-9)
            img = generated_sample_sets[i][
                j
            ]  # Get the j-th sample from the i-th generator
            img = tf.reshape(img, (28, 28, 1))  # Reshape if necessary
            # De-normalize from [-1, 1] to [0, 1] for display
            img = (img + 1.0) / 2.0
            axes[i, j].imshow(img[:, :, 0], cmap="gray")  # Display grayscale channel
            axes[i, j].axis("off")
            if j == 0:  # Add generator label
                axes[i, j].text(
                    -0.1,
                    0.5,
                    f"Gen {i}",
                    horizontalalignment="right",
                    verticalalignment="center",
                    transform=axes[i, j].transAxes,
                    fontsize=10,
                )
            if i == 0:  # Add digit label
                axes[i, j].set_title(f"{j}", fontsize=10)

    plt.suptitle(f"CMAD-GAN MNIST Generated Digits - Epoch {epoch+1}", fontsize=14)
    save_path = os.path.join(output_dir, f"mnist_epoch_{epoch+1:04d}.png")
    plt.savefig(save_path)
    print(f"Saved generated image grid to {save_path}")
    plt.close(fig)  # Close the figure to free memory


class MNISTS_CMADGAN_Experiment(BaseMADGANExperiment):
    """Test implementation of the BaseExperiments class

    Args:
        BaseExperiment (_type_): _description_
    """

    dataset_name: str
    n_gen: int = 1
    latent_dim: int = 256
    size_dataset: int = 60_000
    batch_size: int = 256
    epochs: int = 20
    steps_per_epoch: int = (size_dataset // batch_size) // n_gen
    generator_training_samples_subfolder: str = "generators_examples"
    generate_after_epochs = 1
    diversity_weight: float = 0.3
    conditional_dim: int = 10  # Number of classes in the dataset

    # the functions defining the discriminator and generator models
    # are passed as arguments to the MADGAN class, for rapid prototyping
    define_discriminator = None
    define_generators = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for k, v in kwargs.items():
            setattr(self, k, v)

        if self.dataset_name == "mnist":
            self.dataset_func = mnist_dataset_func
        elif self.dataset_name == "fashion_mnist":
            self.dataset_func = fashion_mnist_dataset_func
        else:
            raise ValueError(f"Unknown dataset name: {self.dataset_name}")

    def _setup(self):
        pass

    def _load_data(self):
        self.data = self.dataset_func()
        self.logger.info(f"Data loaded: {self.data}")

    def _initialize_models(self):
        self.madgan: CMADGAN = CMADGAN(
            latent_dim=self.latent_dim,
            condition_dim=self.conditional_dim,
            data_shape=(28, 28, 1),
            num_generators=self.n_gen,
            diversity_weight=self.diversity_weight,
            define_generator=self.define_generators,
            define_discriminator=self.define_discriminator,
        )

        d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        g_optimizers = [
            tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
            for _ in range(self.n_gen)
        ]

        self.madgan.compile(d_optimizer=d_optimizer, g_optimizers=g_optimizers)

    def _run(self):
        checkpoint_filepath = self.dir_path / "checkpoints" / "backup"
        checkpoint_filepath.parent.mkdir(parents=True, exist_ok=True)
        self.callbacks = [
            CMADGANMonitor(
                latent_dim=self.latent_dim,
                n_classes=self.conditional_dim,
                dir_name=self.dir_path,
                samples_subfolder=self.generator_training_samples_subfolder,
                generate_after_epochs=self.generate_after_epochs,
                num_examples_per_class=1,
                class_labels=[
                    "0",
                    "1",
                    "2",
                    "3",
                    "4",
                    "5",
                    "6",
                    "7",
                    "8",
                    "9",
                ],
                real_dataset=self.data,
            ),
            # ScoreCMADGANMonitor(
            #     dir_name=self.dir_path,
            #     latent_dim=self.latent_dim,
            #     dataset=self.dataset_name,
            #     total_epochs=self.epochs,
            #     num_samples_for_scoring=1000,
            #     score_calculation_freq=self.score_calculation_freq,
            # ),
            # the epoch variable in the f-string is available in the callback
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_filepath.__str__() + "_epoch_{epoch}.weights.h5",
                save_freq=469 * 25,
                save_weights_only=True,
            ),
        ]

        self.history = self.madgan.fit(
            self.data,
            epochs=self.epochs,
            verbose=1,
            callbacks=self.callbacks,
        )
