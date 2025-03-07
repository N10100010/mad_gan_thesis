from pathlib import Path

import numpy as np
import tensorflow as tf
from classification_metrics import F1Score
from experiment.base_experiments.base_experiment import BaseExperiment
from model_definitions.classifiers import FashionMNISTClassifier
from monitors.classification import ClassificationSaveCallback
from utils.plotting import plot_classifier_training_history


class CLASS_FashionMNIST_Experiment(BaseExperiment):
    epochs: int = 15  # Increased for more complex data
    batch_size: int = 64
    num_classes: int = 10

    traditional_data_augmentation: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def _setup(self):
        pass

    def _load_data(self):
        (
            (x_train, y_train),
            (
                x_test,
                y_test,
            ),
        ) = tf.keras.datasets.fashion_mnist.load_data()

        # Preprocess data
        x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") - 127.5 / 127.5
        x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") - 127.5 / 127.5

        # needed for classification_report
        self.test_dataset = (
            tf.data.Dataset.from_tensor_slices((x_test, y_test))
            .batch(self.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        # Convert labels to categorical (one-hot encoding)
        y_train = tf.keras.utils.to_categorical(y_train, self.num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, self.num_classes)

        if self.traditional_data_augmentation:
            train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=15,
                width_shift_range=0.1,
                height_shift_range=0.1,
            )
        else:
            train_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

        # Create generators
        self.train_generator = train_datagen.flow(
            x_train, y_train, batch_size=self.batch_size
        )
        self.test_generator = test_datagen.flow(
            x_test, y_test, batch_size=self.batch_size, shuffle=False
        )

    def _initialize_models(self):
        self.classifier = FashionMNISTClassifier()
        self.classifier.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
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
        model_weights_path = Path(self.dir_path, "final_model.h5")
        self.classifier.model.save(model_weights_path)
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
            ],
        )

        report_path = Path(self.dir_path, "classification_report.txt")
        with open(report_path, "w") as f:
            f.write(report)
        self.logger.info(f"Classification report saved to: {report_path}")
