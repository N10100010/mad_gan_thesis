import tensorflow as tf

def generators_loss_function(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    The loss function for generators.

    Parameters
    ----------
    y_true : tf.Tensor
        The true labels of the data.
    y_pred : tf.Tensor
        The predicted labels of the data.

    Returns
    -------
    tf.Tensor
        The loss of the generators.
    """
    
    logarithm = -tf.math.log(y_pred[:, -1] + 1e-15)
    return tf.reduce_mean(logarithm, axis=-1)
