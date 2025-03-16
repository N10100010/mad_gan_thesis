import tensorflow as tf
from experiment.experiment_queue import ExperimentQueue
from experiment.experiments.conditional_gan.experiment import ConditionalGAN_Experiment
from experiment.experiments.generative_creation.condgan.experiment import (
    CondGAN_GenerativeCreationExperiment,
)
from model_definitions.discriminators.conditional_cifar.disc import (
    define_discriminator as define_discriminator_cifar,
)
from model_definitions.generators.conditional_cifar.gen import (
    define_generator as define_generator_cifar,
)

s = CondGAN_GenerativeCreationExperiment(
    name="test",
    experiment_class=ConditionalGAN_Experiment,
    experiment_path="C:\\Users\\NiXoN\\Desktop\\_thesis\\mad_gan_thesis\\code\\experiments\\CONDITIONAL_GAN_MODELS\\2025-03-15_CIFAR_ConditionalGAN_Experiment_latent_100",
    latent_point_generator=tf.random.normal,
    n_images_per_class=10,
    save_raw_image=True,
    discriminator_func=define_discriminator_cifar,
    generator_func=define_generator_cifar,
)


queue = ExperimentQueue()
queue.add_experiment(s)
queue.run_all()
