import os
import re
from pathlib import Path

import tensorflow as tf
from experiment.experiment_queue import ExperimentQueue
from experiment.experiments.classifier import StratifiedClassifierExperiment

experiments_path = Path(
    "/home/stud/n/nr063/mounted_home/mad_gan_thesis/code/experiments"
)

dataset_identifier = "MADGAN_MNIST_"
experiment_name = "Stratified_classifierExperiment_MNIST_"

all_experiments = os.listdir(experiments_path)
creation_experiments = [
    exp
    for exp in all_experiments
    if (f"{dataset_identifier}" in exp) and ("DataCreation_SPEC" in exp)
]

experiments_to_run = []

N_GENERATED_IMAGES_PER_CLASS = 2_000
N_REAL_IMAGES_PER_CLASS = 3_000

n_images_per_class = [(3000, 2000), (4000, 1000), (5000, 0)]
for gen_img_pc, real_img_pc in n_images_per_class:
    N_GENERATED_IMAGES_PER_CLASS = gen_img_pc
    N_REAL_IMAGES_PER_CLASS = real_img_pc

    for ex in creation_experiments:
        # Extract number of trained generators
        match1 = re.search(r"(\d+)_GEN", ex)
        generators_trained = int(match1.group(1)) if match1 else None
        #
        # Extract the generator used
        match2 = re.search(r"_GEN_(\d+)", ex)
        generator_used = int(match2.group(1)) if match2 else None

        experiments_to_run.append(
            StratifiedClassifierExperiment(
                name=f"{experiment_name}_{generators_trained}_used_generator_{generator_used}__images_real_{N_REAL_IMAGES_PER_CLASS}_gen_{N_GENERATED_IMAGES_PER_CLASS}",
                # name=f"{experiment_name}_BASE__images_real_{N_REAL_IMAGES_PER_CLASS}_gen_{N_GENERATED_IMAGES_PER_CLASS}",
                epochs=50,
                dataset="mnist",
                creation_experiment_path=experiments_path / ex,
                tf_dataset_load_func=tf.keras.datasets.mnist.load_data,
                number_of_generated_images_per_class={
                    i: N_GENERATED_IMAGES_PER_CLASS for i in range(10)
                },
                number_of_real_images_per_class={
                    i: N_REAL_IMAGES_PER_CLASS for i in range(10)
                },
                experiments_base_path="./experiments/MNIST_STRATIFIED_CLASSIFIERS",
            )
        )


queue = ExperimentQueue()
for exp in experiments_to_run:
    queue.add_experiment(exp)
queue.run_all()
