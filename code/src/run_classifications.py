from pathlib import Path
 
from experiment.experiment_queue import ExperimentQueue
from experiment.experiments.classification.experiment import CLASSIFICATION_Experiment
# from model_definitions.classifiers.fashion_mnist import FashionMNISTClassifier
from model_definitions.classifiers import MNISTClassifier

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
    # CLASSIFICATION_Experiment(
    #     name="TEST--CLASSIFICATION_MNIST_Experiment__",
    #     created_images_folder_path=Path(
    #         "C:\\Users\\NiXoN\\Desktop\\_thesis\\mad_gan_thesis\\code\\experiments\\2025-02-08_MADGAN_3_GEN_MNIST_DataCreation_SPEC_GEN_0\\generated_images"
    #     ),
    #     model_path=Path(
    #         "C:\\Users\\NiXoN\\Desktop\\_thesis\\mad_gan_thesis\\code\\experiments\\2025-01-30_TEST--CLASS_MNIST_Experiment__\\final_model.weights.h5"
    #     ),
    #     classifier_class=MNISTClassifier,
    # ),
    ########## CLASSIFY FASHION MNIST IMAGES ##########
    # CLASSIFICATION_Experiment(
    #     name="TEST--CLASSIFICATION_FASHION_MNIST_Experiment__",
    #     created_images_folder_path=Path(
    #         "C:\\Users\\NiXoN\\Desktop\\_thesis\\mad_gan_thesis\\code\\experiments\\2025-02-12_MADGAN_FASHION_MNIST_5_GEN_DataCreation_SPEC_GEN_4\\generated_images"
    #     ),
    #     model_path=Path(
    #         "code/experiments/2025-02-12_CLASSFIER_MNIST/checkpoints/best_weights.h5"
    #     ),
    #     classifier_class=FashionMNISTClassifier,
    # ),
]

# classify madgan mnist experiments. 
import os
import re
from pathlib import Path

mnist_classifier_path = "experiments/2025-02-12_CLASSFIER_MNIST/checkpoints/best_weights.h5"
experiments_path = Path("/home/stud/n/nr063/mounted_home/mad_gan_thesis/code/experiments")

all_experiments = os.listdir(experiments_path)
mnist_creation_experiments = [exp for exp in all_experiments if ('_MNIST_' in exp) and ('DataCreation_SPEC' in exp)]

experiments_to_run = []

for ex in mnist_creation_experiments: 
    # Extract number of trained generators
    match1 = re.search(r'(\d+)_GEN', ex)
    generators_trained = int(match1.group(1)) if match1 else None

    # Extract the generator used
    match2 = re.search(r'_GEN_(\d+)', ex)
    generator_used = int(match2.group(1)) if match2 else None

    experiments_to_run.append(
        CLASSIFICATION_Experiment(
            name=f"ClassificationExperiment_MNIST_MADGAN_{generators_trained}_used_generator_{generator_used}",
            created_images_folder_path=experiments_path / ex / "generated_images",
            model_path=Path(mnist_classifier_path),
            classifier_class=MNISTClassifier
        )
    )


queue = ExperimentQueue()
for exp in experiments_to_run:
    queue.add_experiment(exp)
queue.run_all()
