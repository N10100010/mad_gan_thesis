import tensorflow as tf

# Binary Cross-Entropy Loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(
        tf.ones_like(real_output) * 0.9, real_output
    )  # Label smoothing
    fake_loss = cross_entropy(
        tf.zeros_like(fake_output) + 0.1, fake_output
    )  # Label smoothing
    return real_loss + fake_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


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


class VanillaGAN(tf.keras.Model):
    def __init__(self, generator, discriminator, latent_dim, unroll_steps=5):
        super(VanillaGAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        self.batch_size = 256
        self.unroll_steps = unroll_steps

    def compile(self, generator_optimizer, discriminator_optimizer):
        super(VanillaGAN, self).compile()
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.d_metric = tf.keras.metrics.Mean(name="d_loss")
        self.g_metric = tf.keras.metrics.Mean(name="g_loss")

    @tf.function
    def train_step(self, real_images):
        noise = tf.random.normal([self.batch_size, self.latent_dim])

        # **Unrolled GAN: Train the discriminator multiple times before updating the generator**
        for _ in range(self.unroll_steps):
            with tf.GradientTape() as disc_tape:
                generated_images = self.generator(noise, training=True)

                # Add **instance noise** to real and fake images
                real_images_noisy = real_images + tf.random.normal(
                    shape=tf.shape(real_images), stddev=0.1
                )
                generated_images_noisy = generated_images + tf.random.normal(
                    shape=tf.shape(generated_images), stddev=0.1
                )

                real_output = self.discriminator(real_images_noisy, training=True)
                fake_output = self.discriminator(generated_images_noisy, training=True)

                disc_loss = discriminator_loss(real_output, fake_output)

            gradients_of_discriminator = disc_tape.gradient(
                disc_loss, self.discriminator.trainable_variables
            )
            gradients_of_discriminator = [
                tf.clip_by_value(g, -0.01, 0.01) for g in gradients_of_discriminator
            ]  # Gradient clipping
            self.discriminator_optimizer.apply_gradients(
                zip(gradients_of_discriminator, self.discriminator.trainable_variables)
            )

        # **Feature Matching: Match statistics of real and fake features**
        with tf.GradientTape() as gen_tape:
            generated_images = self.generator(noise, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            real_features = tf.reduce_mean(
                self.discriminator(real_images, training=True), axis=0
            )
            fake_features = tf.reduce_mean(
                self.discriminator(generated_images, training=True), axis=0
            )
            feature_matching_loss = tf.reduce_mean(
                tf.abs(real_features - fake_features)
            )

            gen_loss = (
                generator_loss(fake_output) + 0.1 * feature_matching_loss
            )  # Encourage feature diversity

        gradients_of_generator = gen_tape.gradient(
            gen_loss, self.generator.trainable_variables
        )
        self.generator_optimizer.apply_gradients(
            zip(gradients_of_generator, self.generator.trainable_variables)
        )

        self.d_metric.update_state(disc_loss)
        self.g_metric.update_state(gen_loss)

        return {"d_loss": self.d_metric.result(), "g_loss": self.g_metric.result()}
