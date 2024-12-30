from experiment.experiment_queue import ExperimentQueue
from experiment.experiments.fashion_mnist_madgan import FASHION_MNIST_MADGAN_Experiment
from experiment.experiments.mnist_madgan import MNIST_MADGAN_Experiment

if __name__ == "__main__":
    experiments = [
        MNIST_MADGAN_Experiment(
            name="MNIST_MADGAN_Experiment__1",
            experiment_suffix="n_gen_1",
            epochs=150,
            n_gen=1,
        ),
        MNIST_MADGAN_Experiment(
            name="MNIST_MADGAN_Experiment__2",
            experiment_suffix="n_gen_2",
            epochs=150,
            n_gen=2,
        ),
        MNIST_MADGAN_Experiment(
            name="MNIST_MADGAN_Experiment__3",
            experiment_suffix="n_gen_3",
            epochs=200,
            n_gen=3,
        ),
        FASHION_MNIST_MADGAN_Experiment(
            name="FASHION_MNIST_MADGAN_Experiment__1",
            experiment_suffix="n_gen_1",
            epochs=150,
            n_gen=1,
        ),
        FASHION_MNIST_MADGAN_Experiment(
            name="FASHION_MNIST_MADGAN_Experiment__2",
            experiment_suffix="n_gen_2",
            epochs=150,
            n_gen=2,
        ),
        FASHION_MNIST_MADGAN_Experiment(
            name="FASHION_MNIST_MADGAN_Experiment__3",
            experiment_suffix="n_gen_3",
            epochs=200,
            n_gen=3,
        ),
    ]

    queue = ExperimentQueue()
    for exp in experiments:
        queue.add_experiment(exp)
    queue.run_all()

    # if __name__ == "__main__":
    # to load an exwperiment, with the models initialized
    # experiment = FASHION_MNIST_MADGAN_Experiment.load_from_path(
    #     Path("experiments\\2024-12-22_FASHION_MNIST_MADGAN_Experiment__1_n_gen_1")
    # )
    #
    # experiment.load_model_weights()
    # # to load an exwperiment, with the models initialized
    #
    # from latent_points.utils import generate_latent_points
    # from matplotlib import pyplot as plt
    #
    # latent_points = generate_latent_points(
    #     latent_dim=experiment.latent_dim,
    #     batch_size=experiment.batch_size,
    #     n_gen=experiment.n_gen,
    # )
    #
    # generators = experiment.madgan.generators
    # generated_images = []
    # for g in range(experiment.n_gen):
    #     generated_images.append(generators[g](latent_points[g]))
    # for image in generated_images:
    #     print(type(image))
    #     plt.imshow(image[0])
    #     plt.show()
    #
    # print("Hi! It works")
