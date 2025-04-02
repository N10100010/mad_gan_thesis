import tensorflow as tf 
import numpy as np 
import random 

SEED = 420

tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


from experiment.experiment_queue import ExperimentQueue
from experiment.experiments.cifar_madgan.experiment import CIFAR_MADGAN_Experiment
from experiment.experiments.mnists_madgan.experiments import MNISTS_MADGAN_Experiment
from model_definitions.discriminators.madgan_cifar.new_disc_big import (
    define_discriminator as define_cifar_discriminator_big,
)
from model_definitions.generators.madgan_cifar.new_gen_big import (
    define_generators as define_cifar_generators_big,
)
from model_definitions.discriminators.madgan_mnists.disc import (
    define_discriminator as define_mnists_discriminator,
)
from model_definitions.generators.madgan_mnists.gen import (
    define_generators as define_mnists_generators,
)

n_generators = [5,6,7,8,9,10]

cifar_exps = []
mnist_exps = []
fashion_exps = []

for n in n_generators: 
    # cifar_exps.append(
    #     CIFAR_MADGAN_Experiment(
    #         name="CIFAR_MADGAN_Experiment__",
    #         n_gen=n,
    #         latent_dim=2048,
    #         epochs=500,
    #         experiment_suffix=f"big__latent_2048_{n}_gen_500_epochs",
    #         experiments_base_path="./experiments/MADGAN_MODELS_PROTOTYPES/CIFAR",
    #         define_discriminator=define_cifar_discriminator_big,
    #         define_generators=define_cifar_generators_big,
    #     )
    # )

    mnist_exps.append(
        MNISTS_MADGAN_Experiment(
            name="MNIST_MADGAN_Experiment__",
            n_gen=n,
            latent_dim=784,
            epochs=250,
            experiment_suffix=f"latent_784_{n}_gen_500_epochs",
            experiments_base_path="./experiments/MADGAN_MODELS_PROTOTYPES/MNIST",
            define_discriminator=define_mnists_discriminator,
            define_generators=define_mnists_generators,
            dataset_name="mnist",
        ),
    )

experiments = mnist_exps


# experiments = [
#     # CIFAR_MADGAN_Experiment(
#     #     name="CIFAR_MADGAN_Experiment__",
#     #     n_gen=3,
#     #     latent_dim=2048,
#     #     epochs=2,
#     #     experiment_suffix="big__latent_2048_3_gen_500_epochs",
#     #     experiments_base_path="./experiments/TEST_MADGAN_MODELS_PROTOTYPES",
#     #     define_discriminator=define_cifar_discriminator_big,
#     #     define_generators=define_cifar_generators_big,
#     # ),
#     MNISTS_MADGAN_Experiment(
#         name="TEST__MNIST_MADGAN_Experiment__",
#         n_gen=3,
#         latent_dim=1024,
#         epochs=30,
#         experiment_suffix="latent_1024_3_gen_500_epochs",
#         experiments_base_path="./experiments/TEST_MADGAN_MODELS_PROTOTYPES",
#         define_discriminator=define_mnists_discriminator,
#         define_generators=define_mnists_generators,
#         dataset_name="mnist",
#     ),
#     # MNISTS_MADGAN_Experiment(
#     #     name="FASHION_MADGAN_Experiment__",
#     #     n_gen=3,
#     #     latent_dim=1024,
#     #     epochs=500,
#     #     experiment_suffix="latent_1024_3_gen_500_epochs",
#     #     experiments_base_path="./experiments/TEST_MADGAN_MODELS_PROTOTYPES",
#     #     define_discriminator=define_mnists_discriminator,
#     #     define_generators=define_mnists_generators,
#     #     dataset_name="fashion_mnist",
#     # ),
# ]

queue = ExperimentQueue()
for exp in experiments:
    queue.add_experiment(exp)
queue.run_all()
