import tensorflow as tf
from tensorflow import keras


class CMADGAN(keras.Model):
    def __init__(
        self,
        latent_dim,
        condition_dim,
        data_shape,
        num_generators,
        diversity_weight,
        define_generator,
        define_discriminator,
    ):
        super(CMADGAN, self).__init__()
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.data_shape = data_shape
        self.num_generators = num_generators
        self.diversity_weight = diversity_weight
        self.build_generator = define_generator
        self.build_discriminator = define_discriminator

        # Use CIFAR-10 specific architectures
        self.discriminator = self.build_discriminator(
            self.data_shape, self.condition_dim
        )
        self.generators = [
            self.build_generator(
                self.latent_dim,
                self.condition_dim,
                self.data_shape,
                name=f"Generator_{i}",
            )
            for i in range(self.num_generators)
        ]

        # Loss functions (same logic)
        self.cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)
        self.cosine_similarity_loss = lambda x, y: -tf.reduce_mean(
            tf.losses.cosine_similarity(
                tf.reshape(x, [tf.shape(x)[0], -1]),
                tf.reshape(y, [tf.shape(y)[0], -1]),
                axis=-1,
            )
        )

    def compile(self, d_optimizer, g_optimizers):
        super(CMADGAN, self).compile()
        self.d_optimizer = d_optimizer
        if (
            not isinstance(g_optimizers, list)
            or len(g_optimizers) != self.num_generators
        ):
            raise ValueError(
                f"g_optimizers must be a list of {self.num_generators} optimizers."
            )
        self.g_optimizers = g_optimizers

    # calculate_diversity_loss remains the same as the MNIST version
    def calculate_diversity_loss(self, fake_samples_list):
        total_similarity = 0.0
        num_pairs = 0
        for i in range(self.num_generators):
            for j in range(i + 1, self.num_generators):
                similarity = self.cosine_similarity_loss(
                    fake_samples_list[i], fake_samples_list[j]
                )
                total_similarity += similarity
                num_pairs += 1
        if num_pairs == 0:
            return tf.constant(0.0)
        avg_similarity = total_similarity / float(num_pairs)
        return avg_similarity

    # train_step remains the same logic as the MNIST version
    @tf.function
    def train_step(self, data):
        real_samples, conditions = data
        batch_size = tf.shape(real_samples)[0]
        noise = tf.random.normal(shape=(batch_size, self.latent_dim))

        # --- Train Discriminator ---
        with tf.GradientTape() as tape_d:
            real_output = self.discriminator([real_samples, conditions], training=True)
            d_loss_real = self.cross_entropy(tf.ones_like(real_output), real_output)

            d_loss_fake_total = 0.0
            fake_samples_list_no_grad = [
                gen([noise, conditions], training=False) for gen in self.generators
            ]

            for fake_samples in fake_samples_list_no_grad:
                fake_output = self.discriminator(
                    [fake_samples, conditions], training=True
                )
                d_loss_fake_total += self.cross_entropy(
                    tf.zeros_like(fake_output), fake_output
                )

            d_loss_fake = d_loss_fake_total / self.num_generators
            d_loss = d_loss_real + d_loss_fake

        grads_d = tape_d.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(
            zip(grads_d, self.discriminator.trainable_variables)
        )

        # --- Train Generators ---
        with tf.GradientTape() as tape_g:
            fake_samples_list = []
            gen_adv_losses = []
            for i in range(self.num_generators):
                fake_samples = self.generators[i]([noise, conditions], training=True)
                fake_samples_list.append(fake_samples)
                fake_output = self.discriminator(
                    [fake_samples, conditions], training=False
                )
                adv_loss = self.cross_entropy(tf.ones_like(fake_output), fake_output)
                gen_adv_losses.append(adv_loss)

            diversity_loss = self.calculate_diversity_loss(fake_samples_list)
            total_generator_loss = (
                tf.reduce_sum(gen_adv_losses) + self.diversity_weight * diversity_loss
            )

        all_gen_trainable_vars = [
            var for gen in self.generators for var in gen.trainable_variables
        ]
        grads_g = tape_g.gradient(total_generator_loss, all_gen_trainable_vars)

        var_index = 0
        for i in range(self.num_generators):
            num_vars = len(self.generators[i].trainable_variables)
            gen_grads = grads_g[var_index : var_index + num_vars]
            self.g_optimizers[i].apply_gradients(
                zip(gen_grads, self.generators[i].trainable_variables)
            )
            var_index += num_vars

        history = {
            "d_loss": d_loss,
            "g_adv_loss": tf.reduce_mean(gen_adv_losses),
            "g_div_loss": diversity_loss,
            "g_total_loss": total_generator_loss / self.num_generators,
        }
        for i in range(self.num_generators):
            history[f"g_loss_{i}"] = gen_adv_losses[i]

        return history

    def generate(self, noise, conditions):
        """Generate samples from all generators given noise and conditions."""
        generated_samples = [
            gen([noise, conditions], training=False) for gen in self.generators
        ]
        return generated_samples
