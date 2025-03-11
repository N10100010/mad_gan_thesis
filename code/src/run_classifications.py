from pathlib import Path
import os
import re
 
from experiment.experiment_queue import ExperimentQueue
from experiment.experiments.classification.experiment import CLASSIFICATION_Experiment
from model_definitions.classifiers import FashionMNISTClassifier
from model_definitions.classifiers import MNISTClassifier
from model_definitions.classifiers import CIFAR10Classifier


experiments_path = Path("/home/stud/n/nr063/mounted_home/mad_gan_thesis/code/experiments/CIFAR_MADGAN_DATACREATION_PROTOTYPES")
experiments_path = Path("/home/stud/n/nr063/mounted_home/mad_gan_thesis/code/experiments/VANILLA_GAN_DATACREATION")

## classify madgan mnist experiments. 
classifier_path = "experiments/2025-03-05_CLASSFIER_CIFAR10/checkpoints/best_weights.h5"
classification_experiment_name = "ClassificationExperiment_CIFAR10_MADGAN"
classifier_class = CIFAR10Classifier
all_experiments = os.listdir(experiments_path)


all_experiments = [exp for exp in all_experiments if '15' in exp]

experiments_to_run = []
for ex in all_experiments: 
    # Extract number of trained generators
    # match1 = re.search(r'(\d+)_GEN', ex)
    # generators_trained = int(match1.group(1)) if match1 else None
# 
    # # Extract the generator used
    # match2 = re.search(r'_GEN_(\d+)', ex)
    # generator_used = int(match2.group(1)) if match2 else None

    t = ex.split('_')
    t = t[-1]

    experiments_to_run.append(
        CLASSIFICATION_Experiment(
            name=f"{classification_experiment_name}_{t}",
            created_images_folder_path=experiments_path / ex / "generated_images",
            model_path=Path(classifier_path),
            classifier_class=classifier_class
        )
    )


## classify madgan mnist experiments. 
# classifier_path = "experiments/2025-02-12_CLASSFIER_MNIST/checkpoints/best_weights.h5"
# dataset_identifier = "_MNIST_"
# classification_experiment_name = "ClassificationExperiment_MNIST_MADGAN"
# classifier_class = MNISTClassifier

## classify madgan fashion-mnist experiments. 
# classifier_path = "experiments/2025-02-12_CLASSFIER_FashionMNIST/checkpoints/best_weights.h5"
# dataset_identifier = "_FASHIONMNIST_7"
# classification_experiment_name = "ClassificationExperiment_FASHIONMNIST_MADGAN"
# classifier_class = FashionMNISTClassifier

# all_experiments = os.listdir(experiments_path)
# creation_experiments = [exp for exp in all_experiments if (f'{dataset_identifier}' in exp) and ('DataCreation_SPEC' in exp) and ('02-27' in exp)]
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

# all_experiments = os.listdir(experiments_path)
# 
# 
# experiments = [exp for exp in all_experiments if "2025-03-01" in exp]
# 
# 
# mnist_classifier = "experiments/2025-03-01_CLASSFIER_MNIST/checkpoints/best_weights.h5"
# fashion_classifier = "experiments/2025-03-01_CLASSFIER_FashionMNIST/checkpoints/best_weights.h5"
# cifar_classifier = "experiments/2025-03-01_CLASSFIER_CIFAR10/checkpoints/best_weights.h5"
# 
# classifier_path = ""
# 
# experiments_to_run = []
# 
# for exp in experiments: 
#     
#     ds_match = re.search(r"___([^_]+)_", exp)
#     print(exp)
#     ds = ds_match.group(1)
#     print(ds)
# 
# 
# 
#     if ds == "MNIST": 
#         classifier_path = mnist_classifier
#         classifier_class = MNISTClassifier
#     elif ds == "FASHION": 
#         classifier_path = fashion_classifier
#         classifier_class = FashionMNISTClassifier
#     elif ds == "CIFAR": 
#         classifier_path = cifar_classifier
#         classifier_class = CIFAR10Classifier
# 
#     experiments_to_run.append(
#         CLASSIFICATION_Experiment(
#             name=f"{ds}ClassificationExperiment_VanillaGAN",
#             created_images_folder_path=experiments_path / exp / "generated_images",
#             model_path=Path(classifier_path),
#             classifier_class=classifier_class
#         )
#     )




queue = ExperimentQueue()
for exp in experiments_to_run:
    queue.add_experiment(exp)
queue.run_all()
