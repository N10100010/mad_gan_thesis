from experiment.experiment_queue import ExperimentQueue
from experiment.experiments.mnists_cmadgan.experiment import MNISTS_CMADGAN_Experiment
from model_definitions.discriminators.cmadgan_mnists.disc import define_discriminator
from model_definitions.generators.cmadgan_mnists.gen import define_generators

experiments = [
    MNISTS_CMADGAN_Experiment(
        name="MNIST1_CMADGAN_Experiment__",
        n_gen=3,
        latent_dim=100,
        epochs=30,
        experiment_suffix="latent_100_3_gen_30_epochs",
        experiments_base_path="./experiments/CMADGAN_MODELS_PROTOTYPES/MNIST",
        dataset_name="mnist",
        define_discriminator=define_discriminator,
        define_generators=define_generators,
        conditional_dim=10,
        batch_size=256,
    ),
]


queue = ExperimentQueue()
for exp in experiments:
    queue.add_experiment(exp)
queue.run_all()
