from pathlib import Path

from experiment.experiments.fashion_mnist_madgan import FASHION_MNIST_MADGAN_Experiment
from experiment.experiments.generative_creation.utils import generate_madgan_images
from latent_points.utils import generate_latent_points
from matplotlib import pyplot as plt

if __name__ == "__main__":
    experiments = [
        FASHION_MNIST_MADGAN_Experiment(
            name="FASHION_MNIST_MADGAN_Experiment__6",
            experiment_suffix="n_gen_6",
            epochs=2,
            n_gen=6,
        ),
    ]

    # queue = ExperimentQueue()
    # for exp in experiments:
    #    queue.add_experiment(exp)
    # queue.run_all()

    # if __name__ == "__main__":
    # to load an exwperiment, with the models initialized
    # experiment = FASHION_MNIST_MADGAN_Experiment.load_from_path(
    #    Path("experiments\\2025-01-02_FASHION_MNIST_MADGAN_Experiment__6_n_gen_6")
    # )

    # experiment.load_model_weights()
    # to load an exwperiment, with the models initialized
    # from latent_points.utils import generate_latent_points
    # from matplotlib import pyplot as plt

    # latent_points = generate_latent_points(
    #    latent_dim=experiment.latent_dim,
    #    batch_size=experiment.batch_size,
    #    n_gen=experiment.n_gen,
    # )

    # generators = experiment.madgan.generators
    # generated_images = []
    # for g in range(experiment.n_gen):
    #    generated_images.append(generators[g](latent_points[g]))
    # for image in generated_images:
    #    print(type(image))
    #    plt.imshow(image[0])
    #    plt.show()

    generated_images = generate_madgan_images(
        experiment_class=FASHION_MNIST_MADGAN_Experiment,
        model_path=Path(
            "experiments\\2025-01-02_FASHION_MNIST_MADGAN_Experiment__5_n_gen_5"
        ),
        latent_point_generator=generate_latent_points,
        n_images=100,
    )

    for gen_idx, images in generated_images.items():
        for _, image in enumerate(images):
            plt.imshow(image[0])
            plt.title(f"Generator {gen_idx}")
            plt.show()

            if _ == 2:
                break

    print("Hi! It works")
