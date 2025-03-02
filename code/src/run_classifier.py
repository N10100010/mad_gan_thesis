from experiment.experiment_queue import ExperimentQueue
from experiment.experiments.classifier import (
    CLASS_CIFAR10_Experiment,
    CLASS_FashionMNIST_Experiment,
    CLASS_MNIST_Experiment,
)

experiments = [
    CLASS_MNIST_Experiment(
        name="CLASSFIER_MNIST", epochs=35, traditional_data_augmentation=True
    ),
    CLASS_FashionMNIST_Experiment(
        name="CLASSFIER_FashionMNIST", epochs=40, traditional_data_augmentation=True
    ),
    CLASS_CIFAR10_Experiment(
        name="CLASSFIER_CIFAR10", epochs=50, traditional_data_augmentation=True
    ),
]

queue = ExperimentQueue()
for exp in experiments:
    queue.add_experiment(exp)
queue.run_all()
