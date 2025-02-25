import tensorflow as tf
from experiment.experiment_queue import ExperimentQueue
from experiment.experiments.classifier import StratifiedClassifierExperiment

experiments = [
    StratifiedClassifierExperiment(
        name="STRATIFIED_MNIST",
        epochs=10,
        dataset="mnist",
        creation_experiment_path="C:\\Users\\NiXoN\\Desktop\\_thesis\\mad_gan_thesis\\code\\experiments\\2025-02-12_MADGAN_MNIST_5_GEN_DataCreation_SPEC_GEN_0",
        tf_dataset_load_func=tf.keras.datasets.mnist.load_data,
        number_of_generated_images_per_class={i: 2000 for i in range(10)},
        number_of_real_images_per_class={i: 2000 for i in range(10)},
    )
]

queue = ExperimentQueue()
for exp in experiments:
    queue.add_experiment(exp)
queue.run_all()
