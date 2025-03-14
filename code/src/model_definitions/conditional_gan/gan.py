import tensorflow as tf

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(
        tf.ones_like(real_output) * 0.9, real_output
    )  # Label smoothing
    fake_loss = cross_entropy(
        tf.zeros_like(fake_output) + 0.1, fake_output
    )  # Noisy labels
    return real_loss + fake_loss


def generator_loss(fake_output):
    return cross_entropy(
        tf.ones_like(fake_output), fake_output
    )  # Wants fake to be real


class ConditionalGAN(tf.keras.Model):
    def __init__(
        self,
        generator,
        discriminator,
        latent_dim: int,
        n_classes: int,
        input_noise_stddev: float = 0.1,
    ):
        super(ConditionalGAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.batch_size = 64
        self.input_noise_stddev = input_noise_stddev

    def compile(self, generator_optimizer, discriminator_optimizer, **kwargs):
        super(ConditionalGAN, self).compile(**kwargs)
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer

    def train_step(self, data):
        real_images, labels = data
        batch_size = tf.shape(real_images)[0]

        batch_z = tf.random.normal(shape=(batch_size, self.latent_dim))

        fake_images = self.generator(
            [batch_z, tf.reshape(labels, (-1, 1))], training=True
        )

        # Train Discriminator
        with tf.GradientTape() as disc_tape:
            real_output = self.discriminator(
                [real_images, tf.reshape(labels, (-1, 1))], training=True
            )
            fake_output = self.discriminator(
                [fake_images, tf.reshape(labels, (-1, 1))], training=True
            )
            disc_loss = discriminator_loss(real_output, fake_output)

        disc_gradients = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_variables
        )
        self.discriminator_optimizer.apply_gradients(
            zip(disc_gradients, self.discriminator.trainable_variables)
        )

        # Train Generator
        with tf.GradientTape() as gen_tape:
            fake_images = self.generator(
                [batch_z, tf.reshape(labels, (-1, 1))], training=True
            )
            fake_output = self.discriminator(
                [fake_images, tf.reshape(labels, (-1, 1))], training=True
            )
            gen_loss = generator_loss(fake_output)

        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(
            zip(gen_gradients, self.generator.trainable_variables)
        )

        return {"d_loss": disc_loss, "g_loss": gen_loss}
