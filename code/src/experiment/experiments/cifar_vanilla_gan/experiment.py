from pathlib import Path

import numpy as np
import tensorflow as tf
from datasets.cifar import dataset_func
from experiment.base_experiments import BaseGANExperiment
from latent_points.utils import generate_latent_points
from model_definitions.discriminators.vanilla_cifar.disc import define_discriminator
from model_definitions.generators.vanilla_cifar.gen import define_generator
from model_definitions.vanilla_gan.gan import VanillaGAN
from monitors.vanilla_gan_geneator import VanillaGANMonitor
from utils.plotting import plot_gan_training_history


class CIFAR_VanillaGAN_Experiment(BaseGANExperiment):
    latent_dim: int = 100
    generator_training_samples_subfolder: str = "generators_examples"
    generator_example_freq: int = 1
    save_freq: int = 50
    size_dataset: int = 50_000
    batch_size: int = 256
    epochs: int = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for k, v in kwargs.items():
            setattr(self, k, v)

    def _setup(self):
        pass

    def _load_data(self):
        self.data, self.unique_labels = dataset_func()
        self.dataset = tf.data.Dataset.from_tensor_slices(self.data)
        self.dataset = (
            self.dataset.repeat()
            .shuffle(10 * self.size_dataset, reshuffle_each_iteration=True)
            .batch(self.batch_size, drop_remainder=True)
        )
        # The cifar dataset is loaded as float64, but the model expects float32
        self.dataset = self.dataset.map(lambda x: tf.cast(x, tf.float32))
        self.logger.info(f"Data loaded with shape: {self.data.shape}")

    def _initialize_models(self):
        self.generator = define_generator(self.latent_dim)
        self.discriminator = define_discriminator()

        # Use learning rate decay
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            0.0002, decay_steps=1000, decay_rate=0.95
        )

        self.gan: tf.keras.Model = VanillaGAN(
            self.generator, self.discriminator, self.latent_dim
        )
        self.gan.compile(
            generator_optimizer=tf.keras.optimizers.Adam(lr_schedule, beta_1=0.5),
            discriminator_optimizer=tf.keras.optimizers.Adam(lr_schedule, beta_1=0.5),
        )

    def _save_results(self):
        model_weights_path = Path(self.dir_path, "final_model.weights.h5")
        self.gan.save_weights(model_weights_path)
        self.logger.info(f"Model saved to: {model_weights_path}")

        history_path = Path(self.dir_path, "training_history.npy")
        np.save(history_path, self.history.history)
        self.logger.info(f"Training history saved to: {history_path}")

        plot_gan_training_history(
            history=self.history,
            path=self.dir_path,
        )

    def _run(self):
        checkpoint_filepath = self.dir_path / "checkpoints" / "backup"
        checkpoint_filepath.parent.mkdir(parents=True, exist_ok=True)
        random_latent_vectors = generate_latent_points(
            latent_dim=self.latent_dim, batch_size=self.batch_size, n_gen=1
        )

        self.callbacks = [
            VanillaGANMonitor(
                random_latent_vectors=random_latent_vectors,
                data=self.data,
                n_classes=len(self.unique_labels),
                latent_dim=self.latent_dim,
                dir_name=self.dir_path,
                samples_subfolder=self.generator_training_samples_subfolder,
                generator_example_freq=self.generator_example_freq,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_filepath.__str__() + "_epoch_{epoch}.weights.h5",
                save_freq=234 * self.save_freq,
                save_weights_only=True,
            ),
        ]

        self.history = self.gan.fit(
            self.dataset,
            epochs=self.epochs,
            steps_per_epoch=self.size_dataset // self.batch_size,
            callbacks=self.callbacks,
            verbose=1,
        )
