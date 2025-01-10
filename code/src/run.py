import tensorflow as tf
from experiment.experiment_queue import ExperimentQueue
from experiment.experiments.generative_creation.gan import (
    GAN_GenerativeCreationExperiment,
)
from experiment.experiments.mnist_vanilla_gan.experiment import (
    MNIST_VanillaGAN_Experiment,
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
        #     epochs=5,
        #     experiment_suffix="",
        # ),
        GAN_GenerativeCreationExperiment(
            name="MNIST_GENERATIVE_VanillaGAN_Experiment",
            experiment_class=MNIST_VanillaGAN_Experiment,
            experiment_path="experiments\\2025-01-10_MNIST_VanillaGAN_Experiment__5",
            latent_point_generator=tf.random.normal,
            n_images=1,
        )
    ]

    queue = ExperimentQueue()
    for exp in experiments:
        queue.add_experiment(exp)
    queue.run_all()
