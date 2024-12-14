import tensorflow as tf
from datasets.mnist import dataset_func
from experiment import BaseExperiment
from loss_functions.generator import generators_loss_function
from model_definitions.discriminators.mnist.disc import define_discriminator
from model_definitions.generators.mnist.gen import define_generators
from model_definitions.mad_gan.mnist import MADGAN


class MNIST_MADGAN_Experiment(BaseExperiment):
    """Test implementation of the BaseExperiments class

    Args:
        BaseExperiment (_type_): _description_
    """

    n_gen: int = 3
    latent_dim: int = 256
    size_dataset: int = 60_000
    batch_size: int = 256
    epochs: int = 3

    # @call_super
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # @call_super
    def _load_data(self):
        self.data, self.unique_labels = dataset_func()
        self.dataset = tf.data.Dataset.from_tensor_slices(self.data)
        self.dataset = (
            self.dataset.repeat()
            .shuffle(10 * self.size_dataset, reshuffle_each_iteration=True)
            .batch(self.n_gen * self.batch_size, drop_remainder=True)
        )
        self.logger.info(f"Data loaded with shape: {self.data.shape}")

    # @call_super
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

    # @call_super
    def _run(self):
        pass

    # @call_super
    def _setup(self):
        pass

    # @call_super
    def _save_results(self):
        pass
