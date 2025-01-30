import tensorflow as tf


class F1Score(tf.keras.metrics.Metric):
    def __init__(self, num_classes, average="macro", name="f1_score", **kwargs):
        """
        Initialize the F1Score metric.

        Args:
            num_classes (int): Number of classes in the classification problem.
            average (str): Method to compute the F1 score.
                           Options: 'macro' (default), 'micro', or None.
                           If None, returns F1 score for each class.
            name (str): Name of the metric.
            **kwargs: Additional arguments for the parent class.
        """
        super(F1Score, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.average = average

        # Variables to accumulate true positives, false positives, and false negatives
        self.true_positives = self.add_weight(
            name="tp", initializer="zeros", shape=(num_classes,)
        )
        self.false_positives = self.add_weight(
            name="fp", initializer="zeros", shape=(num_classes,)
        )
        self.false_negatives = self.add_weight(
            name="fn", initializer="zeros", shape=(num_classes,)
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Update the state of the metric with new data.

        Args:
            y_true (tf.Tensor): Ground truth labels (integer-encoded or one-hot encoded).
            y_pred (tf.Tensor): Predicted labels (integer-encoded or one-hot encoded).
            sample_weight (tf.Tensor): Optional sample weights.
        """
        # Convert predictions to integer-encoded labels if necessary
        if y_pred.shape.ndims > 1:
            y_pred = tf.argmax(y_pred, axis=-1)
        if y_true.shape.ndims > 1:
            y_true = tf.argmax(y_true, axis=-1)

        # Confusion matrix
        confusion_matrix = tf.math.confusion_matrix(
            y_true,
            y_pred,
            num_classes=self.num_classes,
            dtype=tf.float32,  # Ensure float32
        )

        # Update true positives, false positives, and false negatives
        tp = tf.cast(tf.linalg.diag_part(confusion_matrix), dtype=tf.float32)
        fp = tf.cast(tf.reduce_sum(confusion_matrix, axis=0) - tp, dtype=tf.float32)
        fn = tf.cast(tf.reduce_sum(confusion_matrix, axis=1) - tp, dtype=tf.float32)

        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)

    def result(self):
        """
        Compute and return the F1 score.

        Returns:
            tf.Tensor: F1 score(s) based on the `average` parameter.
        """
        # Precision and Recall
        precision = self.true_positives / (
            self.true_positives + self.false_positives + tf.keras.backend.epsilon()
        )
        recall = self.true_positives / (
            self.true_positives + self.false_negatives + tf.keras.backend.epsilon()
        )

        # F1 Score
        f1 = (
            2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
        )

        if self.average == "macro":
            return tf.reduce_mean(f1)
        elif self.average == "micro":
            # Micro-average: aggregate TP, FP, FN across all classes
            total_tp = tf.reduce_sum(self.true_positives)
            total_fp = tf.reduce_sum(self.false_positives)
            total_fn = tf.reduce_sum(self.false_negatives)
            micro_precision = total_tp / (
                total_tp + total_fp + tf.keras.backend.epsilon()
            )
            micro_recall = total_tp / (total_tp + total_fn + tf.keras.backend.epsilon())
            micro_f1 = (
                2
                * (micro_precision * micro_recall)
                / (micro_precision + micro_recall + tf.keras.backend.epsilon())
            )
            return micro_f1
        elif self.average is None:
            return f1
        else:
            raise ValueError("Unsupported average type. Use 'macro', 'micro', or None.")

    def reset_state(self):
        """
        Reset the state of the metric.
        """
        self.true_positives.assign(tf.zeros(self.num_classes))
        self.false_positives.assign(tf.zeros(self.num_classes))
        self.false_negatives.assign(tf.zeros(self.num_classes))
