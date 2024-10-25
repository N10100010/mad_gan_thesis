import tensorflow as tf 

def generators_loss_function(y_true, y_pred): 
    """Follows the details, described int he paper: https://arxiv.org/abs/1704.02906

    Args:
        y_true (actual): -
        y_pred (predicted): -

    Returns:
        Tensor: -
    """
    logarithm = -tf.math.log(y_pred[:,-1] + 1e-15)
    return tf.reduce_mean(logarithm, axis=-1)
