import tensorflow as tf


class BaseClassifier(tf.keras.Model):
    # metrics = [
    #             tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
    #             # tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top_5_accuracy"),  # not used at the moment
    #             # tf.keras.metrics.Precision(name="precision"),
    #             # tf.keras.metrics.Recall(name="recall"),
    #             # tf.keras.metrics.AUC(name="roc_auc"),
    #             # F1Score(name="f1_score"),
    #         ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compile(self, optimizer, loss=None, metrics=None, **kwargs):
        # Set default loss and metrics if not provided
        if loss is None:
            # Default assumes subclass models output logits (no softmax activation)
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        if metrics is None:
            metrics = [
                tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            ]
            # metrics = self.metrics
        super().compile(optimizer=optimizer, loss=loss, metrics=metrics, **kwargs)

    @tf.function
    def train_step(self, data):
        # Standard training loop compatible with any input/output structure
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(
                x, training=True
            )  # Forward pass (uses subclass-defined layers)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def call(self, inputs):
        """Subclasses MUST implement this to define their forward pass."""
        raise NotImplementedError("Subclasses must implement the call method.")
