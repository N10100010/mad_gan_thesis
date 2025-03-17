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
    save_freq: int = 1
    size_dataset: int = 50_000
    batch_size: int = 64
    epochs: int = 2
    learning_rate_disc: float = 0.0002
    learning_rate_gen: float = 0.001
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

        def augment_image(image, label):
            if self.dataset_name in ["fashion_mnist"]:
                image = tf.image.grayscale_to_rgb(image)
            image = tf.image.random_flip_left_right(image)  # Horizontal flip
            if self.dataset_name in ["fashion_mnist"]:
                image = tf.image.rgb_to_grayscale(image)
            image = tf.image.random_brightness(image, 0.1)  # Small brightness change
            image = tf.image.random_contrast(
                image, 0.9, 1.1
            )  # Small contrast variation
            noise = tf.random.normal(
                shape=tf.shape(image), mean=0.0, stddev=0.05, dtype=tf.float64
            )  # Match dtype
            image = image + noise
            image = tf.clip_by_value(image, -1.0, 1.0)  # Keep pixel values valid
            return image, label

        # Apply augmentations when loading the dataset
        dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        dataset = (
            dataset.shuffle(10000)
            .map(augment_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .batch(self.batch_size, drop_remainder=True)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )

        self.dataset = dataset

    def _initialize_models(self):
        generator = self.generator_func(latent_dim=self.latent_dim)
        discriminator = self.discriminator_func()

        self.gan: tf.keras.Model = ConditionalGAN(
            generator=generator,
            discriminator=discriminator,
            latent_dim=self.latent_dim,
            n_classes=self.n_classes,
        )

        self.gan.compile(
            generator_optimizer=tf.keras.optimizers.Adam(
                learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
                    self.learning_rate_gen, decay_steps=1000, decay_rate=0.95
                ),
                beta_1=0.5,
            ),
            discriminator_optimizer=tf.keras.optimizers.Adam(
                learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
                    self.learning_rate_disc, decay_steps=1000, decay_rate=0.95
                ),
                beta_1=0.5,
            ),
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
                save_weights_only=True,
            ),
        ]

        self.history = self.gan.fit(
            self.dataset,
            epochs=self.epochs,
            callbacks=callbacks,
        )
