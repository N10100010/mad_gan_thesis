from experiment.experiment_queue import ExperimentQueue
from experiment.experiments.cifar_madgan.experiment import CIFAR_MADGAN_Experiment
from model_definitions.discriminators.cifar.disc_bnw import (
    define_discriminator as define_discriminator_base,
)
from model_definitions.discriminators.cifar.new_disc_big_bnw import (
    define_discriminator as define_discriminator_big,
)
from model_definitions.discriminators.cifar.new_disc_small_bnw import (
    define_discriminator as define_discriminator_small,
)
from model_definitions.generators.cifar.gen_bnw import (
    define_generators as define_generators_base,
)
from model_definitions.generators.cifar.new_gen_big_bnw import (
    define_generators as define_generators_big,
)
from model_definitions.generators.cifar.new_gen_small_bnw import (
    define_generators as define_generators_small,
)

# from model_definitions.discriminators.cifar.disc import (
#     define_discriminator as define_discriminator_base,
# )
# from model_definitions.discriminators.cifar.new_disc_big import (
#     define_discriminator as define_discriminator_big,
# )
# from model_definitions.discriminators.cifar.new_disc_small import (
#     define_discriminator as define_discriminator_small,
# )
# from model_definitions.generators.cifar.gen import (
#     define_generators as define_generators_base,
# )
# from model_definitions.generators.cifar.new_gen_big import (
#     define_generators as define_generators_big,
# )
# from model_definitions.generators.cifar.new_gen_small import (
#     define_generators as define_generators_small,
# )


experiments = [
    CIFAR_MADGAN_Experiment(
        name="CIFAR_MADGAN_Experiment__",
        n_gen=3,
        latent_dim=256,
        epochs=100,
        experiment_suffix="base__latent_256_bnw",
        experiments_base_path="./experiments/CIFAR_MADGAN_MODELS_PROTOTYPES",
        define_discriminator=define_discriminator_base,
        define_generators=define_generators_base,
    ),
    CIFAR_MADGAN_Experiment(
        name="CIFAR_MADGAN_Experiment__",
        n_gen=3,
        latent_dim=256,
        epochs=100,
        experiment_suffix="big__latent_256_bnw",
        experiments_base_path="./experiments/CIFAR_MADGAN_MODELS_PROTOTYPES",
        define_discriminator=define_discriminator_big,
        define_generators=define_generators_big,
    ),
    CIFAR_MADGAN_Experiment(
        name="CIFAR_MADGAN_Experiment__",
        n_gen=3,
        latent_dim=256,
        epochs=100,
        experiment_suffix="small__latent_256_bnw",
        experiments_base_path="./experiments/CIFAR_MADGAN_MODELS_PROTOTYPES",
        define_discriminator=define_discriminator_small,
        define_generators=define_generators_small,
    ),
]

queue = ExperimentQueue()
for exp in experiments:
    queue.add_experiment(exp)
queue.run_all()
