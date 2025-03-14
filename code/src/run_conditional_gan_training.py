from experiment.experiment_queue import ExperimentQueue
from experiment.experiments.conditional_gan import ConditionalGAN_Experiment
from model_definitions.discriminators.conditional_cifar.disc import (
    define_discriminator as define_discriminator_cifar,
)
from model_definitions.generators.conditional_cifar.gen import (
    define_generator as define_generator_cifar,
)

experiments = [
    ConditionalGAN_Experiment(
        name="CIFAR_ConditionalGAN_Experiment",
        latent_dim=100,
        epochs=150,
        batch_size=64,
        discriminator_func=define_discriminator_cifar,
        generator_func=define_generator_cifar,
        dataset_name="cifar10",
        experiments_base_path="./experiments/CONDITIONAL_GAN_MODELS",
        experiment_suffix="latent_100",
    ),
]

queue = ExperimentQueue()
for exp in experiments:
    queue.add_experiment(exp)
queue.run_all()
