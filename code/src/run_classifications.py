from pathlib import Path

from experiment.experiment_queue import ExperimentQueue
from experiment.experiments.classification.experiment import (
    CLASSIFICATION_Experiment,
)
from model_definitions.classifiers.cifar10 import CIFAR10Classifier
from model_definitions.classifiers.mnist import MNISTClassifier


experiments = [
    ########## CLASSIFY CIFAR10 IMAGES ##########
    # CLASSIFICATION_Experiment(
    #     name="TEST--CLASSIFICATION_CIFAR_Experiment__",
    #     created_images_folder_path=Path(
    #         "C:\\Users\\NiXoN\\Desktop\\_thesis\\mad_gan_thesis\\code\\experiments\\2025-02-03_generative_creation_test_cifar10\\generated_images"
    #     ),
    #     model_path=Path(
    #         "C:\\Users\\NiXoN\\Desktop\\_thesis\\mad_gan_thesis\\code\\experiments\\2025-01-30_TEST--CLASS_CIFAR10_Experiment__\\final_model.weights.h5"
    #     ),
    #     classifier_class=CIFAR10Classifier,
    # ),
    
    
    ########## CLASSIFY MNIST IMAGES ##########
    CLASSIFICATION_Experiment(
        name="TEST--CLASSIFICATION_MNIST_Experiment__",
        created_images_folder_path=Path(
            "C:\\Users\\NiXoN\\Desktop\\_thesis\\mad_gan_thesis\\code\\experiments\\2025-02-08_MADGAN_3_GEN_MNIST_DataCreation_SPEC_GEN_0\\generated_images"
        ),
        model_path=Path(
            "C:\\Users\\NiXoN\\Desktop\\_thesis\\mad_gan_thesis\\code\\experiments\\2025-01-30_TEST--CLASS_MNIST_Experiment__\\final_model.weights.h5"
        ),
        classifier_class=MNISTClassifier,
    ),
]

queue = ExperimentQueue()
for exp in experiments:
    queue.add_experiment(exp)
queue.run_all()
