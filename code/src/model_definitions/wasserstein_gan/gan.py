import tensorflow as tf


class WassersteinGAN(tf.keras.Model):
    batch_size = 256

    def __init__(self, generator, discriminator, latent_dim, gp_weight=10.0):
        super(WassersteinGAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        self.gp_weight = gp_weight  # Gradient penalty weight (λ)

    def compile(self, generator_optimizer, discriminator_optimizer):
        super(WassersteinGAN, self).compile()
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer

    def gradient_penalty(self, real_images, fake_images):
        """Compute the gradient penalty for WGAN-GP."""
        batch_size = tf.shape(real_images)[0]
        epsilon = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
        interpolated_images = epsilon * real_images + (1 - epsilon) * fake_images

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated_images)
            pred = self.discriminator(interpolated_images, training=True)

        gradients = gp_tape.gradient(pred, interpolated_images)
        gradients_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
        gp = tf.reduce_mean((gradients_norm - 1.0) ** 2)  # Penalize deviations from 1
        return gp

    @tf.function
    def train_step(self, real_images):
        noise = tf.random.normal([self.batch_size, self.latent_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            # Wasserstein loss for generator and discriminator
            gen_loss = -tf.reduce_mean(fake_output)  # Maximize D(fake)
            disc_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(
                real_output
            )  # Maximize D(real) - D(fake)

            # Add gradient penalty
            gp = self.gradient_penalty(real_images, generated_images)
            disc_loss += self.gp_weight * gp

        # Compute gradients
        gradients_of_generator = gen_tape.gradient(
            gen_loss, self.generator.trainable_variables
        )
        gradients_of_discriminator = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_variables
        )

        # Apply gradients
        self.generator_optimizer.apply_gradients(
            zip(gradients_of_generator, self.generator.trainable_variables)
        )
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables)
        )

        return {"d_loss": disc_loss, "g_loss": gen_loss, "gp": gp}
