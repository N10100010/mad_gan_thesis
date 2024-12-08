import os
import datetime
import tensorflow as tf
import numpy as np

class ExperimentManager:
    def __init__(self, n_gen, latent_dim, batch_size, size_dataset, epochs, experiment_type, dataset_func, define_discriminator, define_generators, MADGAN, Generators_loss_function, GANMonitor, generate_latent_points):
        self.n_gen = n_gen
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.size_dataset = size_dataset
        self.epochs = epochs
        self.steps_per_epoch = (size_dataset // batch_size) // n_gen
        self.type = experiment_type
        self.dataset_func = dataset_func
        self.define_discriminator = define_discriminator
        self.define_generators = define_generators
        self.MADGAN = MADGAN
        self.Generators_loss_function = Generators_loss_function
        self.GANMonitor = GANMonitor
        self.generate_latent_points = generate_latent_points
        self.dir_name = self._create_experiment_directory()
        self.discriminator = None
        self.generators = None
        self.madgan = None
        self.dataset = None
        self.data = None
        self.unique_labels = None

    def _create_experiment_directory(self):
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        dir_name = f"experiments/MNIST_{self.n_gen}-gen_{self.epochs}-ep_{self.type}-type_{current_date}"
        os.makedirs(dir_name, exist_ok=True)
        print(f"Experiment directory: {dir_name}")
        return dir_name

    def _check_gpu(self):
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        if tf.test.gpu_device_name() == '/device:GPU:0':
            print("Using a GPU")
        else:
            print("Using a CPU")

    def load_data(self):
        self.data, self.unique_labels = self.dataset_func()
        self.dataset = tf.data.Dataset.from_tensor_slices(self.data)
        self.dataset = (
            self.dataset
            .repeat()
            .shuffle(10 * self.size_dataset, reshuffle_each_iteration=True)
            .batch(self.n_gen * self.batch_size, drop_remainder=True)
        )
        print(f"Data loaded with shape: {self.data.shape}")

    def initialize_models(self):
        self.discriminator = self.define_discriminator(self.n_gen)
        print("Discriminator Summary:")
        print(self.discriminator.summary())

        self.generators = self.define_generators(self.n_gen, self.latent_dim, class_labels=self.unique_labels)
        print("Generator[0] Summary:")
        print(self.generators[0].summary())

        self.madgan = self.MADGAN(
            discriminator=self.discriminator,
            generators=self.generators,
            latent_dim=self.latent_dim,
            n_gen=self.n_gen,
        )
        self.madgan.compile(
            d_optimizer=tf.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
            g_optimizer=[
                tf.optimizers.Adam(learning_rate=1e-4, beta_1=0.5) for _ in range(self.n_gen)
            ],
            d_loss_fn=tf.keras.losses.CategoricalCrossentropy(),
            g_loss_fn=self.Generators_loss_function,
        )

    def run_experiment(self):
        checkpoint_filepath = os.path.join(self.dir_name, "checkpoint.weights.h5")
        random_latent_vectors = self.generate_latent_points(
            latent_dim=self.latent_dim, batch_size=11, n_gen=self.n_gen
        )
        callbacks = [
            self.GANMonitor(
                random_latent_vectors=random_latent_vectors,
                data=self.data,
                n_classes=len(self.unique_labels),
                latent_dim=self.latent_dim,
                dir_name=self.dir_name,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_filepath, save_freq=10, save_weights_only=True
            ),
        ]

        print("Starting training...")
        history = self.madgan.fit(
            self.dataset,
            epochs=self.epochs,
            steps_per_epoch=self.steps_per_epoch,
            verbose=1,
            callbacks=callbacks,
        )
        self._save_results(history)

    def _save_results(self, history):
        # Save model weights and training history
        model_path = os.path.join(self.dir_name, "final_model.weights.h5")
        self.madgan.save_weights(model_path)
        print(f"Model saved to: {model_path}")

        # Save history
        history_path = os.path.join(self.dir_name, "training_history.npy")
        np.save(history_path, history.history)
        print(f"Training history saved to: {history_path}")
