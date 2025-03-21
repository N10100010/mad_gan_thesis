from experiment.experiment_queue import ExperimentQueue
from experiment.experiments.mnists_madgan.experiments import MNISTS_MADGAN_Experiment
from model_definitions.discriminators.madgan_mnists.disc import (
    define_discriminator as define_mnists_discriminator,
)
from model_definitions.generators.madgan_mnists.gen import (
    define_generators as define_mnists_generators,
)

experiments = [
    # CIFAR_MADGAN_Experiment(
    #     name="CIFAR_MADGAN_Experiment__",
    #     n_gen=3,
    #     latent_dim=2048,
    #     epochs=500,
    #     experiment_suffix="big__latent_2048_3_gen_500_epochs",
    #     experiments_base_path="./experiments/CIFAR_MADGAN_MODELS_PROTOTYPES",
    #     define_discriminator=define_cifar_discriminator_big,
    #     define_generators=define_cifar_generators_big,
    # ),
    MNISTS_MADGAN_Experiment(
        name="MNIST_MADGAN_Experiment__",
        n_gen=3,
        latent_dim=1024,
        epochs=2,
        experiment_suffix="latent_1024_3_gen_500_epochs",
        experiments_base_path="./experiments/CIFAR_MADGAN_MODELS_PROTOTYPES",
        define_discriminator=define_mnists_discriminator,
        define_generators=define_mnists_generators,
        dataset_name="mnist",
    ),
    # MNISTS_MADGAN_Experiment(
    #     name="FASHION_MADGAN_Experiment__",
    #     n_gen=3,
    #     latent_dim=1024,
    #     epochs=2,
    #     experiment_suffix="latent_1024_3_gen_500_epochs",
    #     experiments_base_path="./experiments/CIFAR_MADGAN_MODELS_PROTOTYPES",
    #     define_discriminator=define_mnists_discriminator,
    #     define_generators=define_mnists_generators,
    #     dataset_name="fashion_mnist",
    # ),
]

queue = ExperimentQueue()
for exp in experiments:
    queue.add_experiment(exp)
queue.run_all()
