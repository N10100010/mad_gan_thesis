from experiment.experiment_queue import ExperimentQueue
from experiment.experiments.vanilla_gan.experiment import VanillaGAN_Experiment
from model_definitions.classifiers import MNISTClassifier
from model_definitions.discriminators.vanilla_mnist.disc import (
    define_discriminator as define_discriminator_mnist,
)
from model_definitions.generators.vanilla_mnist.gen import (
    define_generator as define_generator_mnist,
)

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
    # CIFAR_VanillaGAN_Experiment(
    #     name="CIFAR_VanillaGAN_Experiment__",
    #     latent_dim=256,
    #     epochs=2,
    #     experiment_suffix="adjusted",
    #     experiments_base_path="./experiments/VANILLA_GAN_MODELS"
    # ),
    # FASHION_MNIST_VanillaGAN_Experiment(
    #     name="FASHION_MNIST_VanillaGAN_Experiment__",
    #     latent_dim=128,
    #     epochs=300,
    #     experiment_suffix="adjusted",
    #     experiments_base_path="./experiments/VANILLA_GAN_MODELS"
    # ),
    # MNIST_VanillaGAN_Experiment(
    #     name="MNIST_VanillaGAN_Experiment__",
    #     latent_dim=128,
    #     epochs=250,
    #     experiment_suffix="adjusted",
    #     experiments_base_path="./experiments/VANILLA_GAN_MODELS"
    # )
    VanillaGAN_Experiment(
        name="VanillaGAN_Experiment__MNIST",
        latent_dim=128,
        epochs=50,
        experiment_suffix="adjusted",
        experiments_base_path="./experiments/VANILLA_GAN_MODELS",
        dataset_name="mnist",
        discriminator_func=define_discriminator_mnist,
        generator_func=define_generator_mnist,
        classifier_class=MNISTClassifier,
        classifier_model_path="C:\\Users\\NiXoN\\Desktop\\_thesis\\mad_gan_thesis\\code\\experiments\\2025-03-01_CLASSFIER_MNIST\\checkpoints\\best_weights.h5",
        score_calculation_freq=1,
    )
]

queue = ExperimentQueue()
for exp in experiments:
    queue.add_experiment(exp)
queue.run_all()
