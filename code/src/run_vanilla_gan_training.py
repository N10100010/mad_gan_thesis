from experiment.experiment_queue import ExperimentQueue
from experiment.experiments.cifar_vanilla_gan.experiment import (
    CIFAR_VanillaGAN_Experiment,
)
from experiment.experiments.fashion_mnist_vanilla_gan.experiment import FASHION_MNIST_VanillaGAN_Experiment
from experiment.experiments.mnist_vanilla_gan.experiment import MNIST_VanillaGAN_Experiment

experiments = [
    # MNIST_VanillaGAN_Experiment(
    #     name="MNIST_VanillaGAN_Experiment__",
    #     latent_dim=100,
    #     epochs=10,
    #     experiment_suffix="epochs_10",
    # ),
    # FASHION_MNIST_VanillaGAN_Experiment(
    #     name="FASHION_MNIST_VanillaGAN_Experiment__",
    #     latent_dim=100,
    #     epochs=10,
    #     experiment_suffix="epochs_10",
    # ),
    # CIFAR_VanillaGAN_Experiment(
    #     name="CIFAR_VanillaGAN_Experiment__",
    #     latent_dim=128,
    #     epochs=100,
    #     experiment_suffix="latent_128_epochs_200",
    # ),
    # CIFAR_VanillaGAN_Experiment(
    #     name="CIFAR_VanillaGAN_Experiment__",
    #     latent_dim=256,
    #     epochs=200,
    #     experiment_suffix="latent_256_epochs_200",
    # ),
    # # Best Cifar vanilla GAN thus far
    CIFAR_VanillaGAN_Experiment(
        name="CIFAR_VanillaGAN_Experiment__",
        latent_dim=256,
        epochs=300,
        experiment_suffix="adjusted",
        experiments_base_path="./experiments/VANILLA_GAN_MODELS"
    ),
    FASHION_MNIST_VanillaGAN_Experiment(
        name="FASHION_MNIST_VanillaGAN_Experiment__",
        latent_dim=128,
        epochs=300,
        experiment_suffix="adjusted",
        experiments_base_path="./experiments/VANILLA_GAN_MODELS"
    ),
    MNIST_VanillaGAN_Experiment(
        name="MNIST_VanillaGAN_Experiment__",
        latent_dim=128,
        epochs=250,
        experiment_suffix="adjusted",
        experiments_base_path="./experiments/VANILLA_GAN_MODELS"
    )
]

queue = ExperimentQueue()
for exp in experiments:
    queue.add_experiment(exp)
queue.run_all()
