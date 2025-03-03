from experiment.experiment_queue import ExperimentQueue
from experiment.experiments.cifar_madgan.experiment import CIFAR_MADGAN_Experiment
from experiment.experiments.fashion_mnist_madgan.experiment import FASHION_MNIST_MADGAN_Experiment
# from experiment.experiments.fashion_mnist_vanilla_gan.experiment import (
#     FASHION_MNIST_VanillaGAN_Experiment,
# )
# from experiment.experiments.mnist_vanilla_gan.experiment import (
#     MNIST_VanillaGAN_Experiment,
# )

experiments = [
    # CIFAR_MADGAN_Experiment(
    #     name="TEST_better_discriminator_CIFAR_MADGAN_Experiment_2",
    #     n_gen=2,
    #     latent_dim=256,
    #     epochs=2,
    #     experiment_suffix="n_gen_2",
    # ),
    # MNIST_VanillaGAN_Experiment(
    #     name="MNIST_VanillaGAN_Experiment__150",
    #     latent_dim=100,
    #     epochs=150,
    #     experiment_suffix="",
    # ),
    # FASHION_MNIST_MADGAN_Experiment(
    #     name="FASHION_MNIST_MADGAN_Experiment__7_n_gen_7",
    #     experiments_base_path="./experiments/FASHIONMNIST_MADGAN_MODELS",
    #     latent_dim=256,
    #     epochs=250,
    #     experiment_suffix="",
    #     n_gen=7
    # ),

    
    CIFAR_MADGAN_Experiment(
        name="TEST_better_discriminator_CIFAR_MADGAN_Experiment_2",
        n_gen=2,
        latent_dim=256,
        epochs=2,
        experiment_suffix="n_gen_2",
    ),

]

queue = ExperimentQueue()
for exp in experiments:
    queue.add_experiment(exp)
queue.run_all()
