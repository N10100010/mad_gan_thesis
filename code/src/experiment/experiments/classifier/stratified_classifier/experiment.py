from pathlib import Path
from typing import Callable

import numpy as np
import tensorflow as tf
from classification_metrics.f1 import F1Score
from datasets.dataset_creator import DatasetCreator
from experiment.base_experiments.base_experiment import BaseExperiment
from model_definitions.classifiers import (
    BaseClassifier,
    CIFAR10Classifier,
    FashionMNISTClassifier,
    MNISTClassifier,
)
from monitors.classification import ClassificationSaveCallback
from utils.plotting import plot_classifier_training_history


class StratifiedClassifierExperiment(BaseExperiment):
    epochs: int = 50
    batch_size: int = 64
    num_classes: int = 10

    # for DatasetCreator
    dataset: str
    creation_experiment_path: str
    tf_dataset_load_func: Callable
    number_of_generated_images_per_class: dict
    number_of_real_images_per_class: dict

    classifier = None
    lr: float = 1e-4

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def _setup(self):
        self.dsc = DatasetCreator(
            dataset=self.dataset,
            experiment_folder_path=self.creation_experiment_path,
            tf_dataset_load_func=self.tf_dataset_load_func,
            number_of_generated_images_per_class=self.number_of_generated_images_per_class,
            number_of_real_images_per_class=self.number_of_real_images_per_class,
        )

    def _load_data(self):
        self.train_x, self.train_y, self.test_x, self.test_y = self.dsc.create_dataset()

        if (
            len(self.test_x.shape) == 3
        ): 
            self.test_x = np.expand_dims(
                self.test_x, axis=-1
            ) 

        y_train_cat = tf.keras.utils.to_categorical(self.train_y, self.num_classes)
        y_test_cat = tf.keras.utils.to_categorical(self.test_y, self.num_classes)

        self.test_dataset = (
            tf.data.Dataset.from_tensor_slices((self.test_x, self.test_y))
            .batch(self.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

        print("Train X shape:", self.train_x.shape, "dtype:", self.train_x.dtype)
        print("Train Y shape:", self.train_y.shape, "dtype:", self.train_y.dtype)

        self.train_generator = train_datagen.flow(
            self.train_x, y_train_cat, batch_size=self.batch_size, shuffle=False
        )
        self.test_generator = test_datagen.flow(
            self.test_x, y_test_cat, batch_size=self.batch_size, shuffle=False
        )

    def _initialize_models(self):
        if self.dataset == BaseClassifier.MNIST:
            self.classifier = MNISTClassifier()
        elif self.dataset == BaseClassifier.FASHION_MNIST:
            self.classifier = FashionMNISTClassifier()
        elif self.dataset == BaseClassifier.CIFAR10:
            self.classifier = CIFAR10Classifier()

        self.classifier.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
            metrics=[
                "accuracy",
                F1Score(name="f1_score", num_classes=self.num_classes),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),  # Added Recall metric
            ],
            loss="categorical_crossentropy",
        )

    def _run(self):
        checkpoint_path = self.dir_path / "checkpoints" / "best_weights.h5"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        custom_save_callback = ClassificationSaveCallback(checkpoint_path)

        self.history = self.classifier.fit(
            self.train_generator,
            epochs=self.epochs,
            validation_data=self.test_generator,
            callbacks=[
                custom_save_callback,
                tf.keras.callbacks.ReduceLROnPlateau(patience=2),
            ],
        )

    def _save_results(self):
        model_weights_path = Path(self.dir_path, "final_model.weights.h5")
        self.classifier.save_weights(model_weights_path)
        self.logger.info(f"Model saved to: {model_weights_path}")

        history_path = Path(self.dir_path, "training_history.npy")
        np.save(history_path, self.history.history)
        self.logger.info(f"Training history saved to: {history_path}")

        plot_classifier_training_history(
            history=self.history,
            path=self.dir_path,
        )

    def _save_results(self):
        model_weights_path = Path(self.dir_path, "final_model.weights.h5")
        self.classifier.save_weights(model_weights_path)
        self.logger.info(f"Model saved to: {model_weights_path}")

        history_path = Path(self.dir_path, "training_history.npy")
        np.save(history_path, self.history.history)
        self.logger.info(f"Training history saved to: {history_path}")

        plot_classifier_training_history(
            history=self.history,
            path=self.dir_path,
        )
        self._save_classification_report()

    def _save_classification_report(self):
        from sklearn.metrics import classification_report

        if self.dataset == "mnist":
            labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        elif self.dataset == "fashion_mnist":
            labels = [
                "T-shirt/top",
                "Trouser",
                "Pullover",
                "Dress",
                "Coat",
                "Sandal",
                "Shirt",
                "Sneaker",
                "Bag",
                "Ankle boot",
            ]
        else:
            labels = [
                "Airplane",
                "Automobile",
                "Bird",
                "Cat",
                "Deer",
                "Dog",
                "Frog",
                "Horse",
                "Ship",
                "Truck",
            ]

        y_true = np.concatenate([y for _, y in self.test_dataset], axis=0)
        y_pred = np.argmax(self.classifier.predict(self.test_dataset), axis=1)

        report = classification_report(y_true, y_pred, target_names=labels)

        report_path = Path(self.dir_path, "classification_report.txt")
        with open(report_path, "w") as f:
            f.write(report)
        self.logger.info(f"Classification report saved to: {report_path}")
