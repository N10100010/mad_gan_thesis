from pathlib import Path

import numpy as np
import tensorflow as tf
from datasets.fasion_mnist import dataset_func
from experiment import BaseExperiment
from latent_points.mnist import generate_latent_points
from loss_functions.generator import generators_loss_function
from model_definitions.discriminators.mnist.disc import define_discriminator
from model_definitions.generators.mnist.gen import define_generators
from model_definitions.mad_gan.mnist import MADGAN
from monitors.generators import MADGANMonitor
from utils.plotting import generate_gan_training_gif, plot_training_history


class FASHION_MNIST_MADGAN_Experiment(BaseExperiment):
    """Test implementation of the BaseExperiments class

    Args:
        BaseExperiment (_type_): _description_
    """

    n_gen: int = 1
    latent_dim: int = 256
    size_dataset: int = 60_000
    batch_size: int = 256
    epochs: int = 2
    steps_per_epoch: int = (size_dataset // batch_size) // n_gen  # 78
    generator_training_samples_subfolder: str = "generators_examples"
    generate_after_epochs = 1

    def __init__(self, *args, **kwargs):
        pop_keys = []
        # Update class attributes if provided in kwargs
        for k, v in kwargs.items():
            if hasattr(self, k):  # Only set attributes that already exist
                setattr(self, k, v)
                pop_keys.append(k)

        # Remove keys from kwargs that have been used
        for k in pop_keys:
            kwargs.pop(k)

        super().__init__(*args, **kwargs)

    def _setup(self):
        pass

    def _load_data(self):
        self.data, self.unique_labels = dataset_func()
        self.dataset = tf.data.Dataset.from_tensor_slices(self.data)
        self.dataset = (
            self.dataset.repeat()
            .shuffle(10 * self.size_dataset, reshuffle_each_iteration=True)
            .batch(self.n_gen * self.batch_size, drop_remainder=True)
        )
        self.logger.info(f"Data loaded with shape: {self.data.shape}")

    def _initialize_models(self):
        self.discriminator = define_discriminator(self.n_gen)
        self.generators = define_generators(
            self.n_gen, self.latent_dim, class_labels=self.unique_labels
        )

        self.madgan = MADGAN(
            discriminator=self.discriminator,
            generators=self.generators,
            latent_dim=self.latent_dim,
            n_gen=self.n_gen,
        )

        self.madgan.compile(
            d_optimizer=tf.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
            g_optimizer=[
                tf.optimizers.Adam(learning_rate=1e-4, beta_1=0.5)
                for _ in range(self.n_gen)
            ],
            d_loss_fn=tf.keras.losses.CategoricalCrossentropy(),
            g_loss_fn=generators_loss_function,
        )

    def _run(self):
        checkpoint_filepath = self.dir_path / "last_checkpoint.weights.h5"
        random_latent_vectors = generate_latent_points(
            latent_dim=self.latent_dim, batch_size=11, n_gen=self.n_gen
        )
        self.callbacks = [
            MADGANMonitor(
                random_latent_vectors=random_latent_vectors,
                n_gen=self.n_gen,
                data=self.data,
                n_classes=len(self.unique_labels),
                latent_dim=self.latent_dim,
                dir_name=self.dir_path,
                sub_folder=self.generator_training_samples_subfolder,
                generate_after_epochs=self.generate_after_epochs,
            ),
            # This callback is for Saving the model
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_filepath, save_freq=10, save_weights_only=True
            ),
        ]

        self.history = self.madgan.fit(
            self.dataset,
            epochs=self.epochs,
            steps_per_epoch=self.steps_per_epoch,
            verbose=1,
            callbacks=self.callbacks,
        )

    def _save_results(self):
        model_path = Path(self.dir_path, "final_model.weights.h5")
        self.madgan.save_weights(model_path)
        self.logger.info(f"Model saved to: {model_path}")

        # Save history
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
