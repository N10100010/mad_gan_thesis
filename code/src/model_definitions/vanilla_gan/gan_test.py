import tensorflow as tf

# Binary Cross-Entropy Loss with logits
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    # Label smoothing: real labels are 0.9 and fake labels are 0.1.
    real_loss = cross_entropy(tf.ones_like(real_output) * 0.9, real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output) + 0.1, fake_output)
    return real_loss + fake_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


class VanillaGAN(tf.keras.Model):
    def __init__(self, generator, discriminator, latent_dim):
        super(VanillaGAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        self.batch_size = 64  # You may want to update this dynamically

    def compile(self, generator_optimizer, discriminator_optimizer, **kwargs):
        super(VanillaGAN, self).compile(**kwargs)
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer

    def train_step(self, real_images):
        # real_images: A batch of real images from the dataset.
        batch_size = tf.shape(real_images)[0]

        # Sample noise with shape (batch_size, 1, 1, latent_dim)
        batch_z = tf.random.normal(shape=(batch_size, 1, 1, self.latent_dim))

        # --------- Update Discriminator ---------
        with tf.GradientTape() as disc_tape:
            fake_images = self.generator(batch_z, training=True)
            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(fake_images, training=True)
            disc_loss = discriminator_loss(real_output, fake_output)
        disc_gradients = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_variables
        )
        self.discriminator_optimizer.apply_gradients(
            zip(disc_gradients, self.discriminator.trainable_variables)
        )

        # --------- Update Generator (first update) ---------
        with tf.GradientTape() as gen_tape:
            fake_images = self.generator(batch_z, training=True)
            fake_output = self.discriminator(fake_images, training=True)
            gen_loss_1 = generator_loss(fake_output)
        gen_gradients = gen_tape.gradient(
            gen_loss_1, self.generator.trainable_variables
        )
        self.generator_optimizer.apply_gradients(
            zip(gen_gradients, self.generator.trainable_variables)
        )

        # --------- Update Generator (second update) ---------
        # Using the same noise batch to ensure discriminator loss doesn't vanish
        with tf.GradientTape() as gen_tape:
            fake_images = self.generator(batch_z, training=True)
            fake_output = self.discriminator(fake_images, training=True)
            gen_loss_2 = generator_loss(fake_output)
        gen_gradients = gen_tape.gradient(
            gen_loss_2, self.generator.trainable_variables
        )
        self.generator_optimizer.apply_gradients(
            zip(gen_gradients, self.generator.trainable_variables)
        )

        # Average the generator loss over the two updates for reporting.
        gen_loss_avg = (gen_loss_1 + gen_loss_2) / 2.0

        # Optionally, additional metrics can be computed for logging.
        return {"d_loss": disc_loss, "g_loss": gen_loss_avg}
