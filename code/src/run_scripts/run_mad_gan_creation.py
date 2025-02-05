from experiment.experiment_queue import ExperimentQueue
from src.experiment.experiments.fashion_mnist_madgan.experiment import (
    FASHION_MNIST_MADGAN_Experiment,
)
from src.experiment.experiments.generative_creation.madgan.experiment import (
    MADGAN_GenerativeCreationExperiment,
)
from src.latent_points.utils import generate_latent_points

experiments = [
    MADGAN_GenerativeCreationExperiment(
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
