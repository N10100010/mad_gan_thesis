import tensorflow as tf

# from datasets.cifar import dataset_func
from datasets.cifar import dataset_func_black_and_white
from experiment.base_experiments import BaseMADGANExperiment
from latent_points.utils import generate_latent_points
from loss_functions.generator import generators_loss_function

# from model_definitions.discriminators.cifar.disc import define_discriminator
# from model_definitions.generators.cifar.gen import define_generators
from model_definitions.mad_gan import MADGAN
from monitors.madgan_generator import MADGANMonitor


class CIFAR_MADGAN_Experiment(BaseMADGANExperiment):
    """Test implementation of the BaseExperiments class

    Args:
        BaseExperiment (_type_): _description_
    """

    n_gen: int = 1
    latent_dim: int = 256
    size_dataset: int = 60_000
    batch_size: int = 256
    epochs: int = 2
    steps_per_epoch: int = (size_dataset // batch_size) // n_gen
    generator_training_samples_subfolder: str = "generators_examples"
    generate_after_epochs = 1

    # the functions defining the discriminator and generator models
    # are passed as arguments to the MADGAN class, for rapid prototyping
    define_discriminator = None
    define_generators = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for k, v in kwargs.items():
            setattr(self, k, v)

    def _setup(self):
        pass

    def _load_data(self):
        self.data, self.unique_labels = dataset_func_black_and_white()

        self.dataset = tf.data.Dataset.from_tensor_slices(self.data)
        self.dataset = (
            self.dataset.repeat()
            .shuffle(10 * self.size_dataset, reshuffle_each_iteration=True)
            .batch(self.n_gen * self.batch_size, drop_remainder=True)
        )
        # The cifar dataset is loaded as float64, but the model expects float32
        self.dataset = self.dataset.map(lambda x: tf.cast(x, tf.float32))
        self.logger.info(f"Data loaded with shape: {self.data.shape}")

    def _initialize_models(self):
        self.discriminator = self.define_discriminator(self.n_gen)
        self.generators = self.define_generators(self.n_gen, self.latent_dim)

        self.madgan: MADGAN = MADGAN(
            discriminator=self.discriminator,
            generators=self.generators,
            latent_dim=self.latent_dim,
            n_gen=self.n_gen,
        )

        # Apply learning rate decay
        d_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-4,
            decay_steps=10000,
            decay_rate=0.95,
            staircase=True,
        )
        g_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=2e-4,
            decay_steps=10000,
            decay_rate=0.95,
            staircase=True,
        )

        self.madgan.compile(
            d_optimizer=tf.optimizers.Adam(learning_rate=d_lr_schedule, beta_1=0.5),
            g_optimizer=[
                tf.optimizers.Adam(learning_rate=g_lr_schedule, beta_1=0.5)
                for _ in range(self.n_gen)
            ],
            d_loss_fn=tf.keras.losses.CategoricalCrossentropy(),
            g_loss_fn=generators_loss_function,
        )

    def _run(self):
        checkpoint_filepath = self.dir_path / "checkpoints" / "backup"
        checkpoint_filepath.parent.mkdir(parents=True, exist_ok=True)
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
            # the epoch variable in the f-string is available in the callback
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_filepath.__str__() + "_epoch_{epoch}.weights.h5",
                save_freq=234 * 25,
                save_weights_only=True,
            ),
        ]

        self.history = self.madgan.fit(
            self.dataset,
            epochs=self.epochs,
            steps_per_epoch=self.steps_per_epoch,
            verbose=1,
            callbacks=self.callbacks,
        )
