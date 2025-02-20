from pathlib import Path
import os
import re
 
from experiment.experiment_queue import ExperimentQueue
from experiment.experiments.classification.experiment import CLASSIFICATION_Experiment
from model_definitions.classifiers import FashionMNISTClassifier
from model_definitions.classifiers import MNISTClassifier
from model_definitions.classifiers import CIFAR10Classifier


experiments_path = Path("/home/stud/n/nr063/mounted_home/mad_gan_thesis/code/experiments")

## classify madgan mnist experiments. 
# classifier_path = "experiments/2025-02-12_CLASSFIER_MNIST/checkpoints/best_weights.h5"
# dataset_identifier = "_MNIST_"
# classification_experiment_name = "ClassificationExperiment_MNIST_MADGAN"
# classifier_class = MNISTClassifier

## classify madgan fashion-mnist experiments. 
# classifier_path = "experiments/2025-02-12_CLASSFIER_FashionMNIST/checkpoints/best_weights.h5"
# dataset_identifier = "_FASHIONMNIST_"
# classification_experiment_name = "ClassificationExperiment_FASHIONMNIST_MADGAN"
# classifier_class = FashionMNISTClassifier

# all_experiments = os.listdir(experiments_path)
# creation_experiments = [exp for exp in all_experiments if (f'{dataset_identifier}' in exp) and ('DataCreation_SPEC' in exp)]
# creation_experiments = [exp for exp in creation_experiments if not (Path(experiments_path) / exp / "labels.json").exists()]
# experiments_to_run = []

# for ex in creation_experiments: 
#     # Extract number of trained generators
#     match1 = re.search(r'(\d+)_GEN', ex)
#     generators_trained = int(match1.group(1)) if match1 else None
# 
#     # Extract the generator used
#     match2 = re.search(r'_GEN_(\d+)', ex)
#     generator_used = int(match2.group(1)) if match2 else None
# 
#     experiments_to_run.append(
#         CLASSIFICATION_Experiment(
#             name=f"{classification_experiment_name}_{generators_trained}_used_generator_{generator_used}",
#             created_images_folder_path=experiments_path / ex / "generated_images",
#             model_path=Path(classifier_path),
#             classifier_class=classifier_class
#         )
#     )

experiments = [
    CLASSIFICATION_Experiment(
        name=f"ClassificationExperiment_MNIST_VanillaGAN",
        created_images_folder_path=experiments_path / "2025-02-18___MNIST_GENERATIVE_VanillaGAN_Experiment" / "generated_images",
        model_path=Path("experiments/2025-02-12_CLASSFIER_MNIST/checkpoints/best_weights.h5"),
        classifier_class=MNISTClassifier
    ),
    CLASSIFICATION_Experiment(
        name=f"ClassificationExperiment_FASHIONMNIST_VanillaGAN",
        created_images_folder_path=experiments_path / "2025-02-18___FASHION_MNIST_GENERATIVE_VanillaGAN_Experiment" / "generated_images",
        model_path=Path("experiments/2025-02-12_CLASSFIER_FashionMNIST/checkpoints/best_weights.h5"),
        classifier_class=FashionMNISTClassifier
    ),
    CLASSIFICATION_Experiment(
        name=f"ClassificationExperiment_CIFAR_VanillaGAN",
        created_images_folder_path=experiments_path / "2025-02-18___CIFAR_GENERATIVE_VanillaGAN_Experiment" / "generated_images",
        model_path=Path("experiments/2025-02-12_CLASSFIER_CIFAR10/checkpoints/best_weights.h5"),
        classifier_class=CIFAR10Classifier
    )
]

queue = ExperimentQueue()
for exp in experiments:
    queue.add_experiment(exp)
queue.run_all()
