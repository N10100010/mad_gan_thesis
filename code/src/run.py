# import tensorflow as tf
from experiment.experiment_queue import ExperimentQueue

# from experiment.experiments.generative_creation.gan import (
#     GAN_GenerativeCreationExperiment,
# )
# from experiment.experiments.mnist_vanilla_gan.experiment import (
#     MNIST_VanillaGAN_Experiment,
# )
# from experiment.experiments.fashion_mnist_vanilla_gan.experiment import (
#     FASHION_MNIST_VanillaGAN_Experiment,
# )
from experiment.experiments.cifar_vanilla_gan.experiment import (
    CIFAR_VanillaGAN_Experiment,
)

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
        #     name="MNIST_VanillaGAN_Experiment__5",
        #     latent_dim=100,
        #     epochs=100,
        #     experiment_suffix="",
        # ),
        # FASHION_MNIST_VanillaGAN_Experiment(
        #     name="MNIST_VanillaGAN_Experiment__5",
        #     latent_dim=100,
        #     epochs=100,
        #     experiment_suffix="",
        # ),
        # CIFAR_WassersteinGAN_Experiment(
        #     name="CIFAR_WassersteinGAN_Experiment__",
        #     latent_dim=128,
        #     epochs=100,
        #     experiment_suffix="applied_wg_loss_latent_128_epochs_50",
        # ),
        CIFAR_VanillaGAN_Experiment(
            name="NEW_CIFAR_VanillaGAN_Experiment__",
            latent_dim=128,
            epochs=100,
            experiment_suffix="latent_128_epochs_100",
        ),
        CIFAR_VanillaGAN_Experiment(
            name="NEW_CIFAR_VanillaGAN_Experiment__",
            latent_dim=256,
            epochs=100,
            experiment_suffix="latent_256_epochs_100",
        ),
        # GAN_GenerativeCreationExperiment(
        #     name="MNIST_GENERATIVE_VanillaGAN_Experiment",
        #     experiment_class=MNIST_VanillaGAN_Experiment,
        #     experiment_path="experiments\\2025-01-10_MNIST_VanillaGAN_Experiment__5",
        #     latent_point_generator=tf.random.normal,
        #     n_images=1,
        # )
    ]

    queue = ExperimentQueue()
    for exp in experiments:
        queue.add_experiment(exp)
    queue.run_all()
