import tensorflow as tf
from experiment.experiment_queue import ExperimentQueue
from experiment.experiments.cifar_vanilla_gan.experiment import (
    CIFAR_VanillaGAN_Experiment,
)
from experiment.experiments.fashion_mnist_vanilla_gan.experiment import (
    FASHION_MNIST_VanillaGAN_Experiment,
)
from experiment.experiments.generative_creation.gan.experiment import (
    GAN_GenerativeCreationExperiment,
)
from experiment.experiments.mnist_vanilla_gan.experiment import (
    MNIST_VanillaGAN_Experiment,
)

experiments = [
    GAN_GenerativeCreationExperiment(
        name="generative_creation_test",
        experiment_class=MNIST_VanillaGAN_Experiment,
        experiment_path="experiments/2025-01-14_MNIST_VanillaGAN_Experiment___epochs_10",
        latent_point_generator=tf.random.normal,
        n_images=50,
    ),
    GAN_GenerativeCreationExperiment(
        name="generative_creation_test_fashion",
        experiment_class=FASHION_MNIST_VanillaGAN_Experiment,
        experiment_path="experiments/2025-01-14_FASHION_MNIST_VanillaGAN_Experiment___epochs_10",
        latent_point_generator=tf.random.normal,
        n_images=50,
    ),
    GAN_GenerativeCreationExperiment(
        name="generative_creation_test_cifar",
        experiment_class=CIFAR_VanillaGAN_Experiment,
        experiment_path="experiments/2025-01-14_CIFAR_VanillaGAN_Experiment___latent_200_epochs_200",
        latent_point_generator=tf.random.normal,
        n_images=50,
    ),
    GAN_GenerativeCreationExperiment(
        name="generative_creation_test_cifar_1",
        experiment_class=CIFAR_VanillaGAN_Experiment,
        experiment_path="experiments/2025-02-02_CIFAR_VanillaGAN_Experiment___latent_200_epochs_1000",
        latent_point_generator=tf.random.normal,
        n_images=5,
        save_raw_image=True,
    ),
    GAN_GenerativeCreationExperiment(
        name="CIFAR_GENERATIVE_VanillaGAN_Experiment",
        experiment_class=CIFAR_VanillaGAN_Experiment,
        experiment_path="experiments/2025-01-14_CIFAR_VanillaGAN_Experiment___latent_200_epochs_200",
        latent_point_generator=tf.random.normal,
        experiment_suffix="__latent_200_epochs_200",
        save_raw_image=True,
        n_images=20,
    ),
    GAN_GenerativeCreationExperiment(
        name="CIFAR_GENERATIVE_VanillaGAN_Experiment",
        experiment_class=CIFAR_VanillaGAN_Experiment,
        experiment_path="experiments/2025-01-14_CIFAR_VanillaGAN_Experiment___latent_100_epochs_200",
        latent_point_generator=tf.random.normal,
        experiment_suffix="__latent_100_epochs_200",
        n_images=20,
    ),
    GAN_GenerativeCreationExperiment(
        name="CIFAR_GENERATIVE_VanillaGAN_Experiment",
        experiment_class=CIFAR_VanillaGAN_Experiment,
        experiment_path="experiments/2025-01-14_CIFAR_VanillaGAN_Experiment___latent_150_epochs_200",
        latent_point_generator=tf.random.normal,
        experiment_suffix="__latent_150_epochs_200",
        n_images=20,
    ),
    GAN_GenerativeCreationExperiment(
        name="CIFAR_GENERATIVE_VanillaGAN_Experiment",
        experiment_class=CIFAR_VanillaGAN_Experiment,
        experiment_path="experiments/2025-01-14_CIFAR_VanillaGAN_Experiment___latent_200_epochs_200",
        latent_point_generator=tf.random.normal,
        experiment_suffix="__latent_200_epochs_200",
        save_raw_image=True,
        n_images=20,
    ),
]

queue = ExperimentQueue()
for exp in experiments:
    queue.add_experiment(exp)
queue.run_all()
