from experiment.experiment_queue import ExperimentQueue
from experiment.experiments.cifar_cmadgan.experiment import CIFAR_CMADGAN_Experiment
from model_definitions.discriminators.cmadgan_cifar.disc import (
    build_discriminator_cnn_cifar as define_discriminator,
)
from model_definitions.generators.cmadgan_cifar.gen import (
    build_generator_cnn_cifar as define_generators,
)

experiments = [
    # MNISTS_CMADGAN_Experiment(
    #     name="MNIST_CMADGAN_Experiment__",
    #     n_gen=2,
    #     latent_dim=100,
    #     epochs=1,
    #     experiment_suffix=f"latent__{2}_gen_500_epochs",
    #     experiments_base_path="./experiments/CMADGAN_MODELS_PROTOTYPES/MNIST",
    #     dataset_name="mnist",
    #     define_discriminator=define_discriminator_mnist,
    #     define_generators=define_generators_mnist,
    #     conditional_dim=10,
    #     score_calculation_freq=5,
    # ),
]

for i in range(1, 2):
    experiments.append(
        CIFAR_CMADGAN_Experiment(
            name="CIFAR_CMADGAN_Experiment__",
            n_gen=2,
            latent_dim=256,
            epochs=1,
            experiment_suffix=f"latent_256_{2}_gen_500_epochs",
            experiments_base_path="./experiments/CMADGAN_MODELS_PROTOTYPES/CIFAR",
            dataset_name="cifar",
            define_discriminator=define_discriminator,
            define_generators=define_generators,
            conditional_dim=10,
            score_calculation_freq=5,
        ),
    )


queue = ExperimentQueue()
for exp in experiments:
    queue.add_experiment(exp)
queue.run_all()
