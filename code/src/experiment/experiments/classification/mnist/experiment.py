from pathlib import Path

import numpy as np
import tensorflow as tf
from classification_metrics import F1Score
from experiment.base_experiments import BaseExperiment
from model_definitions.classifiers import MNISTClassifier
from monitors.classification import ClassificationSaveCallback
from utils.plotting import plot_classifier_training_history


class CLASS_MNIST_Experiment(BaseExperiment):
    epochs: int = 10

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for k, v in kwargs.items():
            setattr(self, k, v)

    def _setup(self):
        pass

    def _load_data(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        # Preprocess data
        x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
        x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

        # Create TensorFlow datasets
        train_dataset = (
            tf.data.Dataset.from_tensor_slices((x_train, y_train))
            .shuffle(1000)
            .batch(32)
        )
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    def _initialize_models(self):
        self.classifier = MNISTClassifier()
        self.classifier.compile(
            optimizer=tf.keras.optimizers.Adam(),
            # Uses default loss (sparse categorical crossentropy with logits)
            metrics=["accuracy", F1Score(name="f1_score")],
        )

    def _run(self):
        checkpoint_path = self.dir_path / "checkpoints" / "best_weights.h5"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        custom_save_callback = ClassificationSaveCallback(checkpoint_path)

        self.history = self.classifier.fit(
            self.train_dataset,
            epochs=self.epochs,
            validation_data=self.test_dataset,
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

        self._save_classification_report()

    def _save_classification_report(self):
        from sklearn.metrics import classification_report

        y_true = np.concatenate([y for x, y in self.test_dataset], axis=0)
        y_pred = np.argmax(self.classifier.predict(self.test_dataset), axis=1)

        report = classification_report(
            y_true,
            y_pred,
            target_names=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
        )

        report_path = Path(self.dir_path, "classification_report.txt")
        with open(report_path, "w") as f:
            f.write(report)
        self.logger.info(f"Classification report saved to: {report_path}")
