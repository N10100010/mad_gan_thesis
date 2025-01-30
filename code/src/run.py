from experiment.experiment_queue import ExperimentQueue
from experiment.experiments.classification.cifar.experiment import (
    CLASS_CIFAR10_Experiment,
)
from experiment.experiments.classification.fashion_mnist.experiment import (
    CLASS_FashionMNIST_Experiment,
)
from experiment.experiments.classification.mnist.experiment import (
    CLASS_MNIST_Experiment,
)

# from experiment.experiments.cifar_vanilla_gan.experiment import (
#     CIFAR_VanillaGAN_Experiment,
# )

if __name__ == "__main__":
    experiments = [
        # GenerativeCreationExperiment(
        #     name="Fashion_MNIST_DataCreation",
        #     experiment_class=FASHION_MNIST_MADGAN_Experiment,
        #     experiment_path="experiments\\2025-01-02_FASHION_MNIST_MADGAN_Experiment__6_n_gen_6",
        #     latent_point_generator=generate_latent_points,
        #     n_images=1,
        # ),
        # CIFAR_MADGAN_Experiment(
        #     name="TEST_better_discriminator_CIFAR_MADGAN_Experiment_2",
        #     n_gen=2,
        #     latent_dim=256,
        #     epochs=2,
        #     experiment_suffix="n_gen_2",
        # ),
        # MNIST_VanillaGAN_Experiment(
        #     name="MNIST_VanillaGAN_Experiment__150",
        #     latent_dim=100,
        #     epochs=150,
        #     experiment_suffix="",
        # ),
        # MADGAN_GenerativeCreationExperiment(
        #     name="Fashion_MNIST_DataCreation",
        #     experiment_class=FASHION_MNIST_MADGAN_Experiment,
        #     experiment_path="experiments\\2025-01-02_FASHION_MNIST_MADGAN_Experiment__6_n_gen_6",
        #     latent_point_generator=generate_latent_points,
        #     n_images=1,
        # ),
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
        # CIFAR_VanillaGAN_Experiment(
        #     name="CIFAR_VanillaGAN_Experiment__",
        #     latent_dim=128,
        #     epochs=200,
        #     experiment_suffix="latent_128_epochs_200",
        # ),
        # CIFAR_VanillaGAN_Experiment(
        #     name="CIFAR_VanillaGAN_Experiment__",
        #     latent_dim=256,
        #     epochs=200,
        #     experiment_suffix="latent_256_epochs_200",
        # ),
        # CIFAR_VanillaGAN_Experiment(
        #     name="CIFAR_VanillaGAN_Experiment__",
        #     latent_dim=200,
        #     epochs=200,
        #     experiment_suffix="latent_200_epochs_200",
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
        #     name="CIFAR_GENERATIVE_VanillaGAN_Experiment",
        #     experiment_class=CIFAR_VanillaGAN_Experiment,
        #     experiment_path="experiments/2025-01-14_CIFAR_VanillaGAN_Experiment___latent_200_epochs_200",
        #     latent_point_generator=tf.random.normal,
        #     experiment_suffix="__latent_200_epochs_200",
        #     n_images=20,
        # )
        CLASS_MNIST_Experiment(name="TEST--CLASS_MNIST_Experiment__", epochs=2),
        CLASS_FashionMNIST_Experiment(
            name="TEST--CLASS_FashionMNIST_Experiment__", epochs=2
        ),
        CLASS_CIFAR10_Experiment(name="TEST--CLASS_CIFAR10_Experiment__", epochs=2),
    ]

    queue = ExperimentQueue()
    for exp in experiments:
        queue.add_experiment(exp)
    queue.run_all()
