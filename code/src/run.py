from pathlib import Path

from experiment.experiments.fashion_mnist_madgan import FASHION_MNIST_MADGAN_Experiment

if __name__ == "__main__":
    experiments = [
        # FASHION_MNIST_MADGAN_Experiment(
        #     name="FASHION_MNIST_MADGAN_Experiment__1",
        #     experiment_suffix="n_gen_1",
        #     epochs=10,
        #     n_gen=1,
        # ),
        # MNIST_MADGAN_Experiment(
        #     name="MNIST_MADGAN_Experiment__1",
        #     experiment_suffix="n_gen_1",
        #     epochs=10,
        #     n_gen=1,
        # ),
        # MNIST_MADGAN_Experiment(
        #     name="MNIST_MADGAN_Experiment__2",
        #     experiment_suffix="n_gen_2",
        #     epochs=2,
        #     n_gen=2,
        # ),
        # MNIST_MADGAN_Experiment(
        #     name="MNIST_MADGAN_Experiment__3",
        #     experiment_suffix="n_gen_3",
        #     epochs=2,
        #     n_gen=3,
        # ),
        # MNIST_MADGAN_Experiment(name="MNIST_MADGAN_Experiment__4", experiment_suffix="n_gen_4", epochs=250, n_gen=4),
        # MNIST_MADGAN_Experiment(name="MNIST_MADGAN_Experiment__5", experiment_suffix="n_gen_5", epochs=250, n_gen=5),
        # MNIST_MADGAN_Experiment(name="MNIST_MADGAN_Experiment__6", experiment_suffix="n_gen_6", epochs=250, n_gen=6),
        # MNIST_MADGAN_Experiment(name="MNIST_MADGAN_Experiment__7", experiment_suffix="n_gen_7", epochs=250, n_gen=7),
        # MNIST_MADGAN_Experiment(name="MNIST_MADGAN_Experiment__8", experiment_suffix="n_gen_8", epochs=250, n_gen=8),
        # MNIST_MADGAN_Experiment(name="MNIST_MADGAN_Experiment__9", experiment_suffix="n_gen_9", epochs=250, n_gen=9),
        # MNIST_MADGAN_Experiment(name="MNIST_MADGAN_Experiment__10",experiment_suffix="n_gen_10",epochs=250, n_gen=10),
        # MNIST_MADGAN_Experiment(name="MNIST_MADGAN_Experiment__3", experiment_suffix="n_gen_3", epochs=200, n_gen=3),
        # MNIST_MADGAN_Experiment(name="MNIST_MADGAN_Experiment__4", experiment_suffix="n_gen_4", epochs=200, n_gen=4),
        # MNIST_MADGAN_Experiment(name="MNIST_MADGAN_Experiment__5", experiment_suffix="n_gen_5", epochs=200, n_gen=5),
        # MNIST_MADGAN_Experiment(name="MNIST_MADGAN_Experiment__6", experiment_suffix="n_gen_6", epochs=200, n_gen=6),
        # MNIST_MADGAN_Experiment(name="MNIST_MADGAN_Experiment__7", experiment_suffix="n_gen_7", epochs=300, n_gen=7),
        # MNIST_MADGAN_Experiment(
        #     name="MNIST_MADGAN_Experiment__10",
        #     experiment_suffix="n_gen_10",
        #     epochs=300,
        #     n_gen=10,
        # ),
    ]

    # queue = ExperimentQueue()
    # for exp in experiments:
    #     queue.add_experiment(exp)
    #
    # queue.run_all()

    if __name__ == "__main__":
        experiment = FASHION_MNIST_MADGAN_Experiment(
            name="MNIST_MADGAN_Experiment__1",
            experiment_suffix="n_gen_1",
            epochs=10,
            n_gen=1,
        )
        madgan = experiment.load_model_from_path(
            Path(
                "experiments\\2024-12-19_FASHION_MNIST_MADGAN_Experiment__1_n_gen_1\\final_model.weights.h5"
            )
        )

        madgan.fit(
            experiment.dataset,
            epochs=1,
            steps_per_epoch=(experiment.size_dataset // experiment.batch_size)
            // experiment.n_gen,  # 78
            verbose=1,
        )

        print("Hi! It works")
