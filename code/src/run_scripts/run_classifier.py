from experiment.experiment_queue import ExperimentQueue
from experiment.experiments.classifier.cifar.experiment import CLASS_CIFAR10_Experiment
from experiment.experiments.classifier.fashion_mnist.experiment import (
    CLASS_FashionMNIST_Experiment,
)
from experiment.experiments.classifier.mnist.experiment import CLASS_MNIST_Experiment

experiments = [
    CLASS_MNIST_Experiment(name="TEST--CLASS_MNIST_Experiment__", epochs=20),
    CLASS_FashionMNIST_Experiment(
        name="TEST--CLASS_FashionMNIST_Experiment__", epochs=20
    ),
    CLASS_CIFAR10_Experiment(name="TEST--CLASS_CIFAR10_Experiment__", epochs=50),
]

queue = ExperimentQueue()
for exp in experiments:
    queue.add_experiment(exp)
queue.run_all()
