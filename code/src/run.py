from experiment.experiment_queue import ExperimentQueue
from experiment.experiments.fashion_mnist_madgan import FASHION_MNIST_MADGAN_Experiment
from experiment.experiments.generative_creation.experiment import (
    GenerativeCreationExperiment,
)
from latent_points.utils import generate_latent_points

if __name__ == "__main__":
    experiments = [
        GenerativeCreationExperiment(
            name="Fashion_MNIST_DataCreation",
            experiment_class=FASHION_MNIST_MADGAN_Experiment,
            experiment_path="experiments\\2025-01-02_FASHION_MNIST_MADGAN_Experiment__6_n_gen_6",
            latent_point_generator=generate_latent_points,
            n_images=1,
        ),
    ]

    queue = ExperimentQueue()
    for exp in experiments:
        queue.add_experiment(exp)
    queue.run_all()
