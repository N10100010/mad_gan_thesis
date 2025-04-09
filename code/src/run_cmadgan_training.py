from experiment.experiment_queue import ExperimentQueue

# from experiment.experiments.mnists_cmadgan.experiment import MNISTS_CMADGAN_Experiment
from experiment.experiments.cifar_cmadgan.experiment import CIFAR_CMADGAN_Experiment

# from model_definitions.discriminators.cmadgan_mnists.disc import define_discriminator
# from model_definitions.generators.cmadgan_mnists.gen import define_generators
from model_definitions.discriminators.cmadgan_cifar.disc import (
    build_discriminator_cnn_cifar as define_discriminator,
)
from model_definitions.generators.cmadgan_cifar.gen import (
    build_generator_cnn_cifar as define_generators,
)

experiments = []

for i in range(1, 11):
    experiments.append(
        CIFAR_CMADGAN_Experiment(
            name="FASHION_CMADGAN_Experiment__",
            n_gen=i,
            latent_dim=256,
            epochs=2,
            experiment_suffix=f"latent_256_{i}_gen_500_epochs",
            experiments_base_path="./experiments/CMADGAN_MODELS_PROTOTYPES/FASHION",
            dataset_name="fashion_mnist",
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
