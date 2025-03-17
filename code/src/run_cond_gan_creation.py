import tensorflow as tf
from experiment.experiment_queue import ExperimentQueue
from experiment.experiments.conditional_gan.experiment import ConditionalGAN_Experiment
from experiment.experiments.generative_creation.condgan.experiment import (
    CondGAN_GenerativeCreationExperiment,
)
from model_definitions.discriminators.conditional_cifar.disc import (
    define_discriminator as define_discriminator_cifar,
)
from model_definitions.discriminators.conditional_mnists.disc import (
    define_discriminator as define_discriminator_mnists,
)
from model_definitions.generators.conditional_cifar.gen import (
    define_generator as define_generator_cifar,
)
from model_definitions.generators.conditional_mnists.gen import (
    define_generator as define_generator_mnists,
)

experiments = [
    CondGAN_GenerativeCreationExperiment(
        name="CIFAR_CONDITIONAL_DataCreation",
        experiment_class=ConditionalGAN_Experiment,
        experiment_path="experiments/CONDITIONAL_GAN_MODELS/2025-03-17_CIFAR_ConditionalGAN_Experiment_latent_2048",
        latent_point_generator=tf.random.normal,
        n_images_per_class=7000,
        save_raw_image=True,
        discriminator_func=define_discriminator_cifar,
        generator_func=define_generator_cifar,
        experiments_base_path="./experiments/CONDITIONAL_GAN_DATACREATION",
    ),
    CondGAN_GenerativeCreationExperiment(
        name="MNIST_CONDITIONAL_DataCreation",
        experiment_class=ConditionalGAN_Experiment,
        experiment_path="experiments/CONDITIONAL_GAN_MODELS/2025-03-17_FASHION_ConditionalGAN_Experiment_latent_100",
        latent_point_generator=tf.random.normal,
        n_images_per_class=7000,
        save_raw_image=True,
        discriminator_func=define_discriminator_mnists,
        generator_func=define_generator_mnists,
        experiments_base_path="./experiments/CONDITIONAL_GAN_DATACREATION",
    ),
    CondGAN_GenerativeCreationExperiment(
        name="FASHION_CONDITIONAL_DataCreation",
        experiment_class=ConditionalGAN_Experiment,
        experiment_path="experiments/CONDITIONAL_GAN_MODELS/2025-03-17_FASHION_ConditionalGAN_Experiment_latent_100",
        latent_point_generator=tf.random.normal,
        n_images_per_class=7000,
        save_raw_image=True,
        discriminator_func=define_discriminator_mnists,
        generator_func=define_generator_mnists,
        experiments_base_path="./experiments/CONDITIONAL_GAN_DATACREATION",
    ),
]

queue = ExperimentQueue()
for experiment in experiments:
    queue.add_experiment(experiment)

queue.run_all()
