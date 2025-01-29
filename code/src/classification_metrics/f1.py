import tensorflow as tf


class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name="f1_score", num_classes=10, **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.true_positives = self.add_weight(
            name="tp", shape=(num_classes,), initializer="zeros"
        )
        self.false_positives = self.add_weight(
            name="fp", shape=(num_classes,), initializer="zeros"
        )
        self.false_negatives = self.add_weight(
            name="fn", shape=(num_classes,), initializer="zeros"
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert logits to class indices
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.cast(tf.squeeze(y_true, axis=-1), tf.int64)

        # One-hot encode y_true for correct class-wise comparison
        y_true_one_hot = tf.one_hot(y_true, depth=self.num_classes)
        y_pred_one_hot = tf.one_hot(y_pred, depth=self.num_classes)

        # Compute per-class true positives, false positives, and false negatives
        tp = tf.reduce_sum(y_true_one_hot * y_pred_one_hot, axis=0)
        fp = tf.reduce_sum((1 - y_true_one_hot) * y_pred_one_hot, axis=0)
        fn = tf.reduce_sum(y_true_one_hot * (1 - y_pred_one_hot), axis=0)

        # Accumulate the values
        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)

    def result(self):
        precision = self.true_positives / (
            self.true_positives + self.false_positives + tf.keras.backend.epsilon()
        )
        recall = self.true_positives / (
            self.true_positives + self.false_negatives + tf.keras.backend.epsilon()
        )

        f1_per_class = (
            2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
        )

        # Compute the mean F1-score across all classes
        return tf.reduce_mean(f1_per_class)

    def reset_state(self):
        self.true_positives.assign(tf.zeros_like(self.true_positives))
        self.false_positives.assign(tf.zeros_like(self.false_positives))
        self.false_negatives.assign(tf.zeros_like(self.false_negatives))
