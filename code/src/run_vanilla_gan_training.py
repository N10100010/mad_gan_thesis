import tensorflow as tf
from experiment.experiment_queue import ExperimentQueue
from experiment.experiments.vanilla_gan.experiment_gan_hacks import (
    VanillaGAN_Experiment as VanillaGAN_Experiment_GAN_Hacks,
)
from model_definitions.discriminators.vanilla_cifar.new_disc import (
    define_discriminator as define_discriminator_cifar_new,
)
from model_definitions.generators.vanilla_cifar.new_gen import (
    define_generator as define_generator_cifar_new,
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
    # GENERALIZED
    # VanillaGAN_Experiment(
    #     name="MNIST_VanillaGAN_Experiment",
    #     latent_dim=128,
    #     epochs=100,
    #     experiment_suffix="_mnibatchnorm",
    #     experiments_base_path="./experiments/VANILLA_GAN_MODELS",
    #     dataset_name="mnist",
    #     discriminator_func=define_discriminator_mnist,
    #     generator_func=define_generator_mnist,
    #     classifier_class=MNISTClassifier,
    #     classifier_model_path="/home/stud/n/nr063/mounted_home/mad_gan_thesis/code/experiments/2025-03-01_CLASSFIER_MNIST/checkpoints/best_weights.h5",
    #     score_calculation_freq=5,
    # ),
    # VanillaGAN_Experiment(
    #     name="FASHION_VanillaGAN_Experiment",
    #     latent_dim=128,
    #     epochs=125,
    #     experiment_suffix="_mnibatchnorm",
    #     experiments_base_path="./experiments/VANILLA_GAN_MODELS",
    #     dataset_name="fashion_mnist",
    #     discriminator_func=define_discriminator_fashion,
    #     generator_func=define_generator_fashion,
    #     classifier_class=FashionMNISTClassifier,
    #     classifier_model_path="/home/stud/n/nr063/mounted_home/mad_gan_thesis/code/experiments/2025-03-01_CLASSFIER_FashionMNIST/checkpoints/best_weights.h5",
    #     score_calculation_freq=5,
    # ),
    # VanillaGAN_Experiment(
    #     name="CIFAR_VanillaGAN_Experiment",
    #     latent_dim=256,
    #     epochs=150,
    #     experiment_suffix="_mnibatchnorm",
    #     experiments_base_path="./experiments/VANILLA_GAN_MODELS",
    #     dataset_name="cifar10",
    #     discriminator_func=define_discriminator_cifar,
    #     generator_func=define_generator_cifar,
    #     classifier=tf.keras.applications.InceptionV3,
    #     classifier_class=None,
    #     score_calculation_freq=5,
    # ),
    # VanillaGAN_Experiment_Test(  # github.com/carpedm20/DCGAN-tensorflow
    #     name="CIFAR_VanillaGAN_Experiment",
    #     latent_dim=100,
    #     batch_size=64,
    #     epochs=1,
    #     experiment_suffix="dcgan_paper_like",
    #     experiments_base_path="./experiments/VANILLA_GAN_MODELS",
    #     dataset_name="cifar10",
    #     discriminator_func=define_discriminator_cifar_new,
    #     generator_func=define_generator_cifar_new,
    #     classifier=tf.keras.applications.InceptionV3,
    #     classifier_class=None,
    #     score_calculation_freq=0.5,
    # ),
    VanillaGAN_Experiment_GAN_Hacks(  # github.com/carpedm20/DCGAN-tensorflow
        name="CIFAR_VanillaGAN_Experiment",
        latent_dim=100,
        batch_size=64,
        epochs=1,
        experiment_suffix="dcgan_paper_like_with_gan_hacks",
        experiments_base_path="./experiments/VANILLA_GAN_MODELS",
        dataset_name="cifar10",
        discriminator_func=define_discriminator_cifar_new,
        generator_func=define_generator_cifar_new,
        classifier=tf.keras.applications.InceptionV3,
        classifier_class=None,
        score_calculation_freq=0.5,
    ),
]

queue = ExperimentQueue()
for exp in experiments:
    queue.add_experiment(exp)
queue.run_all()
