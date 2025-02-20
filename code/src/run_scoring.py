from experiment.experiment_queue import ExperimentQueue
from experiment.experiments.fid_id_score import ScoringExperiment
from model_definitions.classifiers import MNISTClassifier

experiments = [
    ScoringExperiment(
        name="ScoringExperiment_MNIST_VanillaGAN",
        classifier_class=MNISTClassifier,
        model_path="C:\\Users\\NiXoN\\Desktop\\_thesis\\mad_gan_thesis\\code\\experiments\\2025-02-20_CLASSFIER_MNIST\\checkpoints\\best_weights.h5",
        # generation_experiment_path="C:\\Users\\NiXoN\\Desktop\\_thesis\\mad_gan_thesis\\code\\experiments\\2025-02-18___CIFAR_GENERATIVE_VanillaGAN_Experiment"
        generation_experiment_path="C:\\Users\\NiXoN\\Desktop\\_thesis\\mad_gan_thesis\\code\\experiments\\2025-02-12_MADGAN_MNIST_5_GEN_DataCreation_SPEC_GEN_4",
    )
]

queue = ExperimentQueue()
for exp in experiments:
    queue.add_experiment(exp)
queue.run_all()
