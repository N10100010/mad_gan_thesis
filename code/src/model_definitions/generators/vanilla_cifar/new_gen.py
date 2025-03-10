import tensorflow as tf

def define_generator(latent_dim, ngf=128, nc=3):
    inputs = tf.keras.Input(shape=(1, 1, latent_dim), name="input")
    
    x = tf.keras.layers.Conv2DTranspose(ngf * 8, kernel_size=4, strides=1, padding="valid", use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    x = tf.keras.layers.Conv2DTranspose(ngf * 4, kernel_size=4, strides=2, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    x = tf.keras.layers.Conv2DTranspose(ngf * 2, kernel_size=4, strides=2, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    x = tf.keras.layers.Conv2DTranspose(ngf, kernel_size=4, strides=2, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    outputs = tf.keras.layers.Conv2DTranspose(nc, kernel_size=4, strides=1, padding="same", use_bias=False, activation="tanh")(x)
    
    return tf.keras.Model(inputs, outputs, name="Generator")

# Example usage
# generator = define_generator(100)
# generator.summary()
# x = generator(tf.random.normal([1, 1, 1, 100]))
# print(x.shape)