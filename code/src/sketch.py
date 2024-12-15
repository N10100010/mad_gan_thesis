from experiment.experiment_queue import ExperimentQueue
from experiment.experiments.mnist_madgan import (
    MNIST_MADGAN_Experiment,
    MNIST_MADGAN_Experiment1,
)

if __name__ == "__main__":
    exp1 = MNIST_MADGAN_Experiment(
        name="MNIST_MADGAN_Experiment1", experiment_suffix="test_1"
    )
    exp2 = MNIST_MADGAN_Experiment1(
        name="MNIST_MADGAN_Experiment2", experiment_suffix="test_2"
    )

    queue = ExperimentQueue()
    queue.add_experiment(exp1)
    queue.add_experiment(exp2)
    queue.run_all()
