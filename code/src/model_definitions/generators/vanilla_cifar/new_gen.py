import tensorflow as tf


def define_generator(latent_dim, ngf=128, nc=3):
    # nz = 100   # Latent vector size
    # ngf = 64   # Feature map depth in the generator
    # nc = 3     # Number of channels (RGB images)

    model = tf.keras.Sequential(
        [
            # Input Z, going into a transposed convolution
            tf.keras.layers.Conv2DTranspose(
                ngf * 8,
                kernel_size=4,
                strides=1,
                padding="valid",
                use_bias=False,
                input_shape=(1, 1, latent_dim),
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            # State size: (ngf*8) x 4 x 4
            tf.keras.layers.Conv2DTranspose(
                ngf * 4, kernel_size=4, strides=2, padding="same", use_bias=False
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            # State size: (ngf*4) x 8 x 8
            tf.keras.layers.Conv2DTranspose(
                ngf * 2, kernel_size=4, strides=2, padding="same", use_bias=False
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            # State size: (ngf*2) x 16 x 16
            tf.keras.layers.Conv2DTranspose(
                ngf, kernel_size=4, strides=2, padding="same", use_bias=False
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            # State size: (ngf) x 32 x 32
            tf.keras.layers.Conv2DTranspose(
                nc,
                kernel_size=4,
                strides=1,
                padding="same",
                use_bias=False,
                activation="tanh",
            ),
            # State size: (nc) x 32 x 32
        ]
    )
    return model

    # # Input Z, going into a transposed convolution
    # inputs = tf.keras.Input(shape=latent_dim, name="input")
    # x = tf.keras.layers.Conv2DTranspose(ngf * 8, kernel_size=4, strides=1, padding="valid", use_bias=False,)(inputs)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.LeakyReLU()(x)
    # # State size: (ngf*8) x 4 x 4
    # x = tf.keras.layers.Conv2DTranspose(ngf * 4, kernel_size=4, strides=2, padding="same", use_bias=False)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.LeakyReLU()(x)
    # # State size: (ngf*4) x 8 x 8
    # x = tf.keras.layers.Conv2DTranspose(ngf * 2, kernel_size=4, strides=2, padding="same", use_bias=False)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.LeakyReLU()(x)
    # # State size: (ngf*2) x 16 x 16
    # x = tf.keras.layers.Conv2DTranspose(ngf, kernel_size=4, strides=2, padding="same", use_bias=False)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.LeakyReLU()(x)
    # # State size: (ngf) x 32 x 32
    # x = tf.keras.layers.Conv2DTranspose(nc, kernel_size=4, strides=1, padding="same", use_bias=False, activation="tanh")(x)
    # # State size: (nc) x 32 x 32


#
# return tf.keras.Model(inputs, x, name="Generator")

# Create the model
# generator = define_generator(100)
# generator.summary()
# x = generator(tf.random.normal([1, 1, 1, 100]))
# print(x.shape)
