from experiment.experiment_queue import ExperimentQueue
from experiment.experiments.cifar_madgan import CIFAR_MADGAN_Experiment

if __name__ == "__main__":
    experiments = [
        # GenerativeCreationExperiment(
        #     name="Fashion_MNIST_DataCreation",
        #     experiment_class=FASHION_MNIST_MADGAN_Experiment,
        #     experiment_path="experiments\\2025-01-02_FASHION_MNIST_MADGAN_Experiment__6_n_gen_6",
        #     latent_point_generator=generate_latent_points,
        #     n_images=1,
        # ),
        CIFAR_MADGAN_Experiment(
            name="TEST_better_discriminator_CIFAR_MADGAN_Experiment_2",
            n_gen=2,
            latent_dim=256,
            epochs=2,
            experiment_suffix="n_gen_2",
        ),
    ]

    queue = ExperimentQueue()
    for exp in experiments:
        queue.add_experiment(exp)
    queue.run_all()
