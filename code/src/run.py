from pathlib import Path

from experiment.experiments.fashion_mnist_madgan import FASHION_MNIST_MADGAN_Experiment
from experiment.experiments.generative_creation.utils import generate_madgan_images
from latent_points.utils import generate_latent_points
from matplotlib import pyplot as plt

if __name__ == "__main__":
    experiments = [
        # MNIST_MADGAN_Experiment(
        #     name="MNIST_MADGAN_Experiment__1",
        #     experiment_suffix="n_gen_1",
        #     epochs=150,
        #     n_gen=1,
        # ),
        # MNIST_MADGAN_Experiment(
        #     name="MNIST_MADGAN_Experiment__2",
        #     experiment_suffix="n_gen_2",
        #     epochs=150,
        #     n_gen=2,
        # ),
        # MNIST_MADGAN_Experiment(
        #     name="MNIST_MADGAN_Experiment__3",
        #     experiment_suffix="n_gen_3",
        #     epochs=200,
        #     n_gen=3,
        # ),
        # MNIST_MADGAN_Experiment(
        #     name="MNIST_MADGAN_Experiment__4",
        #     experiment_suffix="n_gen_4",
        #     epochs=200,
        #     n_gen=4,
        # ),
        # MNIST_MADGAN_Experiment(
        #     name="MNIST_MADGAN_Experiment__5",
        #     experiment_suffix="n_gen_5",
        #     epochs=200,
        #     n_gen=5,
        # ),
        # MNIST_MADGAN_Experiment(
        #     name="MNIST_MADGAN_Experiment__6",
        #     experiment_suffix="n_gen_6",
        #     epochs=200,
        #     n_gen=6,
        # ),
        # MNIST_MADGAN_Experiment(
        #     name="MNIST_MADGAN_Experiment__7",
        #     experiment_suffix="n_gen_7",
        #     epochs=250,
        #     n_gen=7,
        # ),
        # MNIST_MADGAN_Experiment(
        #     name="MNIST_MADGAN_Experiment__8",
        #     experiment_suffix="n_gen_8",
        #     epochs=250,
        #     n_gen=8,
        # ),
        # MNIST_MADGAN_Experiment(
        #     name="MNIST_MADGAN_Experiment__9",
        #     experiment_suffix="n_gen_9",
        #     epochs=250,
        #     n_gen=9,
        # ),
        # MNIST_MADGAN_Experiment(
        #    name="MNIST_MADGAN_Experiment__10",
        #     experiment_suffix="n_gen_10",
        #     epochs=250,
        #     n_gen=10,
        # ),
        # FASHION_MNIST_MADGAN_Experiment(
        #     name="FASHION_MNIST_MADGAN_Experiment__1",
        #     experiment_suffix="n_gen_1",
        #     epochs=150,
        #     n_gen=1,
        # ),
        # FASHION_MNIST_MADGAN_Experiment(
        #     name="FASHION_MNIST_MADGAN_Experiment__2",
        #     experiment_suffix="n_gen_2",
        #     epochs=150,
        #     n_gen=2,
        # ),
        # FASHION_MNIST_MADGAN_Experiment(
        #     name="FASHION_MNIST_MADGAN_Experiment__3",
        #     experiment_suffix="n_gen_3",
        #     epochs=200,
        #     n_gen=3,
        # ),
        # FASHION_MNIST_MADGAN_Experiment(
        #     name="FASHION_MNIST_MADGAN_Experiment__4",
        #     experiment_suffix="n_gen_4",
        #     epochs=200,
        #     n_gen=4,
        # ),
        # FASHION_MNIST_MADGAN_Experiment(
        #     name="FASHION_MNIST_MADGAN_Experiment__5",
        #     experiment_suffix="n_gen_5",
        #     epochs=200,
        #     n_gen=5,
        # ),
        FASHION_MNIST_MADGAN_Experiment(
            name="FASHION_MNIST_MADGAN_Experiment__6",
            experiment_suffix="n_gen_6",
            epochs=200,
            n_gen=6,
        ),
        # MNIST_MADGAN_Experiment(
        #     name="MNIST_MADGAN_Experiment__7",
        #     experiment_suffix="n_gen_7",
        #     epochs=250,
        #     n_gen=7,
        # ),
        # MNIST_MADGAN_Experiment(
        #     name="MNIST_MADGAN_Experiment__8",
        #     experiment_suffix="n_gen_8",
        #     epochs=250,
        #     n_gen=8,
        # ),
        # MNIST_MADGAN_Experiment(
        #     name="MNIST_MADGAN_Experiment__9",
        #     experiment_suffix="n_gen_9",
        #     epochs=250,
        #     n_gen=9,
        # ),
        # MNIST_MADGAN_Experiment(
        #    name="MNIST_MADGAN_Experiment__10",
        #     experiment_suffix="n_gen_10",
        #     epochs=250,
        #     n_gen=10,
        # ),
        # MNIST_MADGAN_Experiment(
        #     name="MNIST_MADGAN_Experiment__1",
        #     experiment_suffix="n_gen_1",
        #     epochs=150,
        #     n_gen=1,
        # ),
        # MNIST_MADGAN_Experiment(
        #     name="MNIST_MADGAN_Experiment__2",
        #     experiment_suffix="n_gen_2",
        #     epochs=150,
        #     n_gen=2,
        # ),
        # MNIST_MADGAN_Experiment(
        #     name="MNIST_MADGAN_Experiment__3",
        #     experiment_suffix="n_gen_3",
        #     epochs=200,
        #     n_gen=3,
        # ),
        # MNIST_MADGAN_Experiment(
        #     name="MNIST_MADGAN_Experiment__4",
        #     experiment_suffix="n_gen_4",
        #     epochs=200,
        #     n_gen=4,
        # ),
        # MNIST_MADGAN_Experiment(
        #     name="MNIST_MADGAN_Experiment__5",
        #     experiment_suffix="n_gen_5",
        #     epochs=200,
        #     n_gen=5,
        # ),
        # MNIST_MADGAN_Experiment(
        #     name="MNIST_MADGAN_Experiment__6",
        #     experiment_suffix="n_gen_6",
        #     epochs=200,
        #     n_gen=6,
        # ),
        # MNIST_MADGAN_Experiment(
        #     name="MNIST_MADGAN_Experiment__7",
        #     experiment_suffix="n_gen_7",
        #     epochs=250,
        #     n_gen=7,
        # ),
        # MNIST_MADGAN_Experiment(
        #     name="MNIST_MADGAN_Experiment__8",
        #     experiment_suffix="n_gen_8",
        #     epochs=250,
        #     n_gen=8,
        # ),
        # MNIST_MADGAN_Experiment(
        #     name="MNIST_MADGAN_Experiment__9",
        #     experiment_suffix="n_gen_9",
        #     epochs=250,
        #     n_gen=9,
        # ),
        # MNIST_MADGAN_Experiment(
        #    name="MNIST_MADGAN_Experiment__10",
        #     experiment_suffix="n_gen_10",
        #     epochs=250,
        #     n_gen=10,
        # ),
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
            "experiments/2025-01-02_FASHION_MNIST_MADGAN_Experiment__5_n_gen_5"
        ),
        latent_point_generator=generate_latent_points,
        n_images=100,
    )

    saving_path = Path(f"experiments/2025-01-02_FASHION_MNIST_MADGAN_Experiment__5_n_gen_5/generated_images")
    saving_path.mkdir(parents=True, exist_ok=True)
    
    for gen_idx, images in generated_images.items():
        for _, _images in enumerate(images):
            for __, image in enumerate(_images): 
                plt.imshow(image / 127.5 * 127.5)
                plt.title(f"Generator {gen_idx}")
                plt.savefig(saving_path / f"gen_{gen_idx}_{_}__{__}.png")

            if _ == 2:
                break

    print("Hi! It works")
