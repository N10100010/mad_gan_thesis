import tensorflow as tf


def preprocess_image(img_path, target_size=(32, 32, 3), scale_minus_1_1: bool = False):
    """
    Preprocess an image for inference.

    Args:
        img_path (str): Path to the image file.
        target_size (tuple): Target size of the image (height, width).

    Returns:
        np.array: Preprocessed image with shape (1, height, width, channels).
    """
    
    # weird, but this is to laod colored or gray images with tf...image.load_img function
    # if not specifically defined, it'll load 3-channel iamges by default
    target_cmap = "grayscale" if target_size[2] == 1 else "rgb"
    target_size = target_size[0:2]

    img = tf.keras.preprocessing.image.load_img(img_path, target_size=target_size, color_mode=target_cmap)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array - 127.5 / 127.5

    if scale_minus_1_1 == (-1.0, 1.0):
        img_array = (img_array * 2.0) - 1.0

    # Add a batch dimension
    img_array = tf.expand_dims(img_array, axis=0)

    return img_array
