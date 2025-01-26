from pathlib import Path

import numpy as np
import tensorflow as tf
from experiment.base_experiments.base_experiment import BaseExperiment
from model_definitions.classifiers import CIFAR10Classifier
from utils.plotting import plot_classifier_training_history


class CLASS_CIFAR10_Experiment(BaseExperiment):
    epochs: int = 50  # More epochs for complex dataset
    batch_size: int = 128

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def _setup(self):
        pass

    def _load_data(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

        # Normalize and convert to float32
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0

        # No reshaping needed for CIFAR-10 (already 32x32x3)
        self.train_dataset = (
            tf.data.Dataset.from_tensor_slices((x_train, y_train))
            .shuffle(5000)
            .batch(self.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        self.test_dataset = (
            tf.data.Dataset.from_tensor_slices((x_test, y_test))
            .batch(self.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

    def _initialize_models(self):
        self.classifier = CIFAR10Classifier()
        self.classifier.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            metrics=["accuracy"],
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        )

    def _run(self):
        checkpoint_path = self.dir_path / "checkpoints" / "best_weights.h5"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=True,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
        )

        self.history = self.classifier.fit(
            self.train_dataset,
            epochs=self.epochs,
            validation_data=self.test_dataset,
            callbacks=[checkpoint, tf.keras.callbacks.ReduceLROnPlateau(patience=2)],
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
            target_names=[
                "airplane",
                "automobile",
                "bird",
                "cat",
                "deer",
                "dog",
                "frog",
                "horse",
                "ship",
                "truck",
            ],
        )

        report_path = Path(self.dir_path, "classification_report.txt")
        with open(report_path, "w") as f:
            f.write(report)
        self.logger.info(f"Classification report saved to: {report_path}")
