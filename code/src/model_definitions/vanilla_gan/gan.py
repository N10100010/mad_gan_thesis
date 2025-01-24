import tensorflow as tf

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


class VanillaGAN(tf.keras.Model):
    def __init__(self, generator, discriminator, latent_dim):
        super(VanillaGAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        self.batch_size = 256

    def compile(self, generator_optimizer, discriminator_optimizer, loss_fn):
        super(VanillaGAN, self).compile()
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.loss_fn = loss_fn
        self.d_metric = tf.keras.metrics.Mean(name="d_loss")
        self.g_metric = tf.keras.metrics.Mean(name="g_loss")

    @tf.function
    def train_step(self, real_images):
        noise = tf.random.normal([self.batch_size, self.latent_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            # Get discriminator outputs first
            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            # Create labels after outputs are defined
            real_labels = tf.ones_like(real_output) * 0.9  # Label smoothing
            fake_labels = tf.zeros_like(fake_output) + 0.1

            # Calculate losses
            real_loss = self.loss_fn(real_labels, real_output)
            fake_loss = self.loss_fn(fake_labels, fake_output)
            disc_loss = real_loss + fake_loss

            # Generator tries to make fake outputs appear real
            gen_loss = self.loss_fn(real_labels, fake_output)  # Use real_labels here

        # Gradient clipping
        gradients_of_discriminator = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_variables
        )
        gradients_of_discriminator = [
            tf.clip_by_value(g, -0.01, 0.01) for g in gradients_of_discriminator
        ]

        gradients_of_generator = gen_tape.gradient(
            gen_loss, self.generator.trainable_variables
        )

        # Apply gradients
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables)
        )
        self.generator_optimizer.apply_gradients(
            zip(gradients_of_generator, self.generator.trainable_variables)
        )

        # Update metrics
        self.d_metric.update_state(disc_loss)
        self.g_metric.update_state(gen_loss)

        return {"d_loss": self.d_metric.result(), "g_loss": self.g_metric.result()}
