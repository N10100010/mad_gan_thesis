import tensorflow as tf
from experiment.experiment_queue import ExperimentQueue
from experiment.experiments.fid_id_score import ScoringExperiment
from model_definitions.classifiers import FashionMNISTClassifier, MNISTClassifier

experiments = [
    ScoringExperiment(
        name="ScoringExperiment_MNIST",
        classifier_class=MNISTClassifier,
        model_path="C:\\Users\\NiXoN\\Desktop\\_thesis\\mad_gan_thesis\\code\\experiments\\2025-02-20_CLASSFIER_MNIST\\checkpoints\\best_weights.h5",
        generation_experiment_path="C:\\Users\\NiXoN\\Desktop\\_thesis\\mad_gan_thesis\\code\\experiments\\2025-02-12_MADGAN_MNIST_5_GEN_DataCreation_SPEC_GEN_4",
        n_generated_images=100,
    ),
    ScoringExperiment(
        name="ScoringExperiment_FashionMNIST",
        classifier_class=FashionMNISTClassifier,
        model_path="C:\\Users\\NiXoN\\Desktop\\_thesis\\mad_gan_thesis\\code\\experiments\\2025-02-21_CLASSFIER_FashionMNIST\\checkpoints\\best_weights.h5",
        generation_experiment_path="C:\\Users\\NiXoN\\Desktop\\_thesis\\mad_gan_thesis\\code\\experiments\\2025-02-12_MADGAN_FASHION_MNIST_5_GEN_DataCreation_SPEC_GEN_4",
        n_generated_images=100,
    ),
    ScoringExperiment(
        name="ScoringExperiment_CIFAR",
        classifier=tf.keras.applications.InceptionV3(
            weights="imagenet", include_top=False, pooling="avg"
        ),
        generation_experiment_path="C:\\Users\\NiXoN\\Desktop\\_thesis\\mad_gan_thesis\\code\\experiments\\2025-02-03_generative_creation_test_cifar10",
        n_generated_images=100,
    ),
]

queue = ExperimentQueue()
for exp in experiments:
    queue.add_experiment(exp)
queue.run_all()
