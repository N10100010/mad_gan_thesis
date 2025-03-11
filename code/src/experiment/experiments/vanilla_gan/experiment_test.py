from pathlib import Path
from typing import Callable

import numpy as np
import tensorflow as tf
from datasets.cifar import dataset_func as cifar_dataset_func
from datasets.fasion_mnist import dataset_func as fashion_dataset_func
from datasets.mnist import dataset_func as mnist_dataset_func
from experiment.base_experiments import BaseGANExperiment
from latent_points.utils import generate_latent_points
from model_definitions.vanilla_gan.gan_test import VanillaGAN as VanillaGAN

# from model_definitions.vanilla_gan.gan import VanillaGAN as VanillaGAN
from monitors.vanilla_gan_geneator_test import VanillaGANMonitor
from monitors.score_generator_test import ScoreGANMonitor
from utils.plotting import plot_gan_training_history


class VanillaGAN_Experiment(BaseGANExperiment):
    latent_dim: int = 100
    generator_training_samples_subfolder: str = "generators_examples"
    generator_example_freq: int = 1
    score_calculation_freq: int = 1
    save_freq: int = 50
    size_dataset: int = 50_000
    batch_size: int = 64
    epochs: int = 2
    learning_rate: float = 0.0002
    discriminator_func: Callable
    generator_func: Callable

    classifier: tf.keras.Model = None
    classifier_class: tf.keras.Model
    dataset_name: str
    classifier_model_path: str = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for k, v in kwargs.items():
            setattr(self, k, v)

    def _setup(self):
        pass

    def _load_data(self):
        if self.dataset_name == "mnist":
            self.data, self.unique_labels = mnist_dataset_func()
        elif self.dataset_name == "cifar10":
            self.data, self.unique_labels = cifar_dataset_func()
        elif self.dataset_name == "fashion_mnist":
            self.data, self.unique_labels = fashion_dataset_func()

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
        self.generator = self.generator_func(latent_dim=self.latent_dim)
        self.discriminator = self.discriminator_func()

        # Use learning rate decay
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            self.learning_rate, decay_steps=1000, decay_rate=0.95
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

        batch_z = tf.random.normal(shape=(self.batch_size, 1, 1, self.latent_dim))

        self.callbacks = [
            VanillaGANMonitor(
                random_latent_vectors=batch_z,
                data=self.data,
                n_classes=len(self.unique_labels),
                latent_dim=self.latent_dim,
                latent_point_generator=generate_latent_points,
                dir_name=self.dir_path,
                samples_subfolder=self.generator_training_samples_subfolder,
                generator_example_freq=self.generator_example_freq,
            ),
            # ScoreGANMonitor(
            #     dir_name=self.dir_path,
            #     latent_dim=self.latent_dim,
            #     latent_point_generator=generate_latent_points,
            #     dataset=self.dataset_name,
            #     classifier_class=self.classifier_class,
            #     classifier=self.classifier,
            #     model_path=self.classifier_model_path,
            #     score_calculation_freq=self.score_calculation_freq,
            # ),
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
            verbose=0,
        )
