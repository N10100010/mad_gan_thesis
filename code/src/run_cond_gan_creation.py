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
        name="test_cifar",
        experiment_class=ConditionalGAN_Experiment,
        experiment_path="C:\\Users\\NiXoN\\Desktop\\_thesis\\mad_gan_thesis\\code\\experiments\\CONDITIONAL_GAN_MODELS\\2025-03-17_CIFAR_ConditionalGAN_Experiment_latent_100",
        latent_point_generator=tf.random.normal,
        n_images_per_class=10,
        save_raw_image=True,
        discriminator_func=define_discriminator_cifar,
        generator_func=define_generator_cifar,
    ),
    CondGAN_GenerativeCreationExperiment(
        name="test_mnist",
        experiment_class=ConditionalGAN_Experiment,
        experiment_path="C:\\Users\\NiXoN\\Desktop\\_thesis\\mad_gan_thesis\\code\\experiments\\CONDITIONAL_GAN_MODELS\\2025-03-17_MNIST_ConditionalGAN_Experiment_latent_100",
        latent_point_generator=tf.random.normal,
        n_images_per_class=10,
        save_raw_image=True,
        discriminator_func=define_discriminator_mnists,
        generator_func=define_generator_mnists,
    ),
    CondGAN_GenerativeCreationExperiment(
        name="test_fashion",
        experiment_class=ConditionalGAN_Experiment,
        experiment_path="C:\\Users\\NiXoN\\Desktop\\_thesis\\mad_gan_thesis\\code\\experiments\\CONDITIONAL_GAN_MODELS\\2025-03-17_FASHION_ConditionalGAN_Experiment_latent_100",
        latent_point_generator=tf.random.normal,
        n_images_per_class=10,
        save_raw_image=True,
        discriminator_func=define_discriminator_mnists,
        generator_func=define_generator_mnists,
    ),
]

queue = ExperimentQueue()
for experiment in experiments:
    queue.add_experiment(experiment)

queue.run_all()
