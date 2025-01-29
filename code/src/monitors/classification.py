from pathlib import Path

import numpy as np
import tensorflow as tf
from utils.logging import setup_logger


class ClassificationSaveCallback(tf.keras.callbacks.Callback):
    combined_score_weights = {
        "val_f1_score": 0.5,
        "val_accuracy": 0.3,
        "val_roc_auc": 0.2,
        "val_precision": 0.05,
        "val_recall": 0.05,
    }

    def __init__(self, save_path: Path):
        super(ClassificationSaveCallback, self).__init__()
        self.save_path = save_path
        self.best_score = -np.inf
        self.logger = setup_logger(name="SaveCallback")

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return

        accuracy = logs.get("val_accuracy", 0)
        f1 = logs.get("val_f1_score", 0)
        roc_auc = logs.get("val_roc_auc", 0)
        precision = logs.get("val_precision", 0)
        recall = logs.get("val_recall", 0)

        # Weights are chosen based on the importance of each metric (arbitrary reasoning)
        combined_score = (
            (0.5 * f1)
            + (0.3 * accuracy)
            + (0.2 * roc_auc)
            + (0.05 * precision)
            + (0.05 * recall)
        )

        self.logger.info(f"Epoch {epoch + 1}: Combined Score = {combined_score:.4f}")

        # Save the model if the combined score improves
        if combined_score > self.best_score:
            self.best_score = combined_score
            self.model.save_weights(self.save_path)
            self.logger.info(f"New best model saved with score {self.best_score:.4f}")
