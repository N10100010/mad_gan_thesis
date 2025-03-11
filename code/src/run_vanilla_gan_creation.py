import tensorflow as tf
from experiment.experiment_queue import ExperimentQueue
from experiment.experiments.generative_creation.gan.experiment import (
    GAN_GenerativeCreationExperiment,
)
from experiment.experiments.vanilla_gan.experiment_test import (
    VanillaGAN_Experiment as VanillaGAN_Experiment_Test,
)
from model_definitions.discriminators.vanilla_cifar.new_disc import (
    define_discriminator as define_discriminator_cifar_new,
)
from model_definitions.generators.vanilla_cifar.new_gen import (
    define_generator as define_generator_cifar_new,
)

experiments = [
    # GAN_GenerativeCreationExperiment(
    #     name="generative_creation_test",
    #     experiment_class=MNIST_VanillaGAN_Experiment,
    #     experiment_path="experiments/2025-01-14_MNIST_VanillaGAN_Experiment___epochs_10",
    #     latent_point_generator=tf.random.normal,
    #     n_images=50,
    # ),
    # GAN_GenerativeCreationExperiment(
    #     name="generative_creation_test_fashion",
    #     experiment_class=FASHION_MNIST_VanillaGAN_Experiment,
    #     experiment_path="experiments/2025-01-14_FASHION_MNIST_VanillaGAN_Experiment___epochs_10",
    #     latent_point_generator=tf.random.normal,
    #     n_images=50,
    # ),
    # GAN_GenerativeCreationExperiment(
    #     name="generative_creation_test_cifar",
    #     experiment_class=CIFAR_VanillaGAN_Experiment,
    #     experiment_path="experiments/2025-01-14_CIFAR_VanillaGAN_Experiment___latent_200_epochs_200",
    #     latent_point_generator=tf.random.normal,
    #     n_images=50,
    # ),
    # GAN_GenerativeCreationExperiment(
    #     name="generative_creation_test_cifar_1",
    #     experiment_class=CIFAR_VanillaGAN_Experiment,
    #     experiment_path="experiments/2025-02-02_CIFAR_VanillaGAN_Experiment___latent_200_epochs_1000",
    #     latent_point_generator=tf.random.normal,
    #     n_images=5,
    #     save_raw_image=True,
    # ),
    # GAN_GenerativeCreationExperiment(
    #     name="CIFAR_GENERATIVE_VanillaGAN_Experiment",
    #     experiment_class=CIFAR_VanillaGAN_Experiment,
    #     experiment_path="experiments/2025-01-14_CIFAR_VanillaGAN_Experiment___latent_200_epochs_200",
    #     latent_point_generator=tf.random.normal,
    #     experiment_suffix="__latent_200_epochs_200",
    #     save_raw_image=True,
    #     n_images=20,
    # ),
    # GAN_GenerativeCreationExperiment(
    #     name="CIFAR_GENERATIVE_VanillaGAN_Experiment",
    #     experiment_class=CIFAR_VanillaGAN_Experiment,
    #     experiment_path="experiments/2025-01-14_CIFAR_VanillaGAN_Experiment___latent_100_epochs_200",
    #     latent_point_generator=tf.random.normal,
    #     experiment_suffix="__latent_100_epochs_200",
    #     n_images=20,
    # ),
    # GAN_GenerativeCreationExperiment(
    #     name="CIFAR_GENERATIVE_VanillaGAN_Experiment",
    #     experiment_class=CIFAR_VanillaGAN_Experiment,
    #     experiment_path="experiments/2025-01-14_CIFAR_VanillaGAN_Experiment___latent_150_epochs_200",
    #     latent_point_generator=tf.random.normal,
    #     experiment_suffix="__latent_150_epochs_200",
    #     n_images=20,
    # ),
    # GAN_GenerativeCreationExperiment(
    #     experiments_base_path="./experiments/VANILLA_GAN_DATACREATION",
    #     name="__CIFAR_GENERATIVE_VanillaGAN_Experiment",
    #     experiment_class=CIFAR_VanillaGAN_Experiment,
    #     experiment_path="experiments/VANILLA_GAN_MODELS/2025-03-01_CIFAR_VanillaGAN_Experiment___adjusted",
    #     latent_point_generator=tf.random.normal,
    #     experiment_suffix="",
    #     save_raw_image=True,
    #     n_images=90_000,
    # ),
    #
    # # MNIST data creation
    # GAN_GenerativeCreationExperiment(
    #     experiments_base_path="./experiments/VANILLA_GAN_DATACREATION",
    #     name="__MNIST_GENERATIVE_VanillaGAN_Experiment",
    #     experiment_class=MNIST_VanillaGAN_Experiment,
    #      experiment_path="experiments/VANILLA_GAN_MODELS/2025-03-01_MNIST_VanillaGAN_Experiment___adjusted",
    #     latent_point_generator=tf.random.normal,
    #     experiment_suffix="",
    #     save_raw_image=True,
    #     n_images=90_000,
    # ),
    #
    # # FASHION-MNIST data creation
    #  GAN_GenerativeCreationExperiment(
    #     experiments_base_path="./experiments/VANILLA_GAN_DATACREATION",
    #      name="__FASHION_MNIST_GENERATIVE_VanillaGAN_Experiment",
    #      experiment_class=FASHION_MNIST_VanillaGAN_Experiment,
    #     experiment_path="experiments/VANILLA_GAN_MODELS/2025-03-01_FASHION_MNIST_VanillaGAN_Experiment___adjusted",
    #      latent_point_generator=tf.random.normal,
    #      experiment_suffix="",
    #      save_raw_image=True,
    #      n_images=90_000,
    #  ),
    GAN_GenerativeCreationExperiment(
        experiments_base_path="./experiments/VANILLA_GAN_DATACREATION",
        name="__CIFAR_GENERATIVE_VanillaGAN_Experiment_dc_paper_like",
        experiment_class=VanillaGAN_Experiment_Test,
        experiment_path="experiments/VANILLA_GAN_MODELS/2025-03-11_CIFAR_VanillaGAN_Experiment_dcgan_paper_like_no_score",
        latent_point_generator=tf.random.normal,
        experiment_suffix="",
        save_raw_image=True,
        n_images=10,
        discriminator_func=define_discriminator_cifar_new,
        generator_func=define_generator_cifar_new,
    ),
]

queue = ExperimentQueue()
for exp in experiments:
    queue.add_experiment(exp)
queue.run_all()
