from typing import Callable

import tensorflow as tf
from datasets.cifar import conditional_dataset_func as cifar_dataset_func
from datasets.fasion_mnist import conditional_dataset_func as fashion_dataset_func
from datasets.mnist import conditional_dataset_func as mnist_dataset_func
from experiment.base_experiments.base_gan_experiment import BaseGANExperiment
from model_definitions.conditional_gan.gan import ConditionalGAN
from monitors.conditional_gan_generator import ConditionalGANMonitor


class ConditionalGAN_Experiment(BaseGANExperiment):
    dataset_name: str

    n_classes: int = 10
    latent_dim: int = 100
    batch_size: int = 64
    save_freq: int = 50
    size_dataset: int = 50_000
    batch_size: int = 64
    epochs: int = 2
    learning_rate: float = 0.0002
    discriminator_func: Callable
    generator_func: Callable
    generator_example_freq: int = 1
    generator_training_samples_subfolder: str = "generators_examples"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for k, v in kwargs.items():
            setattr(self, k, v)

    def _setup(self):
        pass

    def _load_data(self):
        if self.dataset_name == "mnist":
            train_images, train_labels = mnist_dataset_func()
        elif self.dataset_name == "cifar10":
            train_images, train_labels = cifar_dataset_func()
        elif self.dataset_name == "fashion_mnist":
            train_images, train_labels = fashion_dataset_func()

        # Create a tf.data.Dataset
        dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        dataset = (
            dataset.shuffle(10000)
            .batch(self.batch_size, drop_remainder=True)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )

        self.dataset = dataset

    def _initialize_models(self):
        generator = self.generator_func(
            latent_dim=self.latent_dim, n_classes=self.n_classes
        )
        discriminator = self.discriminator_func(n_classes=self.n_classes)

        self.gan: tf.keras.Model = ConditionalGAN(
            generator=generator,
            discriminator=discriminator,
            latent_dim=self.latent_dim,
            n_classes=self.n_classes,
        )

        self.gan.compile(
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            discriminator_optimizer=tf.keras.optimizers.Adam(self.learning_rate),
            generator_optimizer=tf.keras.optimizers.Adam(self.learning_rate),
        )

    def _run(self):
        checkpoint_filepath = self.dir_path / "checkpoints" / "backup"
        checkpoint_filepath.parent.mkdir(parents=True, exist_ok=True)

        callbacks = [
            ConditionalGANMonitor(
                data=self.dataset,
                n_classes=self.n_classes,
                latent_dim=self.latent_dim,
                random_latent_vectors=tf.random.normal(
                    shape=(self.n_classes * 9, self.latent_dim)
                ),
                dir_name=self.dir_path,
                samples_subfolder="generators_examples",
                generator_example_freq=self.generator_example_freq,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_filepath.__str__() + "_epoch_{epoch}.h5",
                save_freq=234 * self.save_freq,
                # save_weights_only=True,
            ),
        ]

        self.history = self.gan.fit(
            self.dataset,
            epochs=self.epochs,
            callbacks=callbacks,
        )
