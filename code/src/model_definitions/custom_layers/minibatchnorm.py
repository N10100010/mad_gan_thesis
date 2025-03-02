import tensorflow as tf


class MinibatchDiscrimination(tf.keras.layers.Layer):
    """Encourages diversity by allowing discriminator to compare samples in a batch."""

    def __init__(self, num_kernels=5, kernel_dim=10):
        super(MinibatchDiscrimination, self).__init__()
        self.num_kernels = num_kernels
        self.kernel_dim = kernel_dim

    def build(self, input_shape):
        self.T = self.add_weight(
            shape=(input_shape[-1], self.num_kernels * self.kernel_dim),
            initializer="random_normal",
            trainable=True,
        )

    def call(self, x):
        activation = tf.matmul(x, self.T)
        activation = tf.reshape(activation, (-1, self.num_kernels, self.kernel_dim))
        diffs = tf.expand_dims(activation, 3) - tf.expand_dims(activation, 2)
        abs_diffs = tf.reduce_sum(tf.abs(diffs), axis=-1)
        exp_diffs = tf.exp(-abs_diffs)
        minibatch_features = tf.reduce_sum(exp_diffs, axis=2)
        return tf.concat([x, minibatch_features], axis=-1)
