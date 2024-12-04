from typing import Dict, List
import tensorflow as tf 

class MADGAN(tf.keras.Model):
    """
    Multi-Agent Diverse Generative Adversarial Network (MADGAN)
    
    A MADGAN consists of multiple generator models and a single discriminator model.
    
    Parameters
    ----------
    discriminator : tf.keras.Model
        The discriminator model.
    generators : list of tf.keras.Model
        A list of generator models.
    latent_dim : int
        The dimensionality of the latent space.
    n_gen : int
        The number of generator models.
    """
    def __init__(self, discriminator: tf.keras.Model, generators: List[tf.keras.Model], latent_dim: int, n_gen: int) -> None:
        """
        Initialize the MADGAN model.

        Parameters
        ----------
        discriminator : tf.keras.Model
            The discriminator model.
        generators : list of tf.keras.Model
            A list of generator models.
        latent_dim : int
            The dimensionality of the latent space.
        n_gen : int
            The number of generator models.
        
        Returns
        -------
        None
        """
        super(MADGAN, self).__init__()
        self.discriminator = discriminator
        self.generators = generators
        self.latent_dim = latent_dim
        self.n_gen = n_gen

    def compile(
        self, 
        d_optimizer: tf.keras.optimizers.Optimizer, 
        g_optimizer: tf.keras.optimizers.Optimizer, 
        d_loss_fn: tf.keras.losses.Loss, 
        g_loss_fn: tf.keras.losses.Loss
    ) -> None:
        """
        Compile the MADGAN model.

        Parameters
        ----------
        d_optimizer : tf.keras.optimizers.Optimizer
            The optimizer for the discriminator model.
        g_optimizer : tf.keras.optimizers.Optimizer
            The optimizer for the generator models.
        d_loss_fn : tf.keras.losses.Loss
            The loss function for the discriminator model.
        g_loss_fn : tf.keras.losses.Loss
            The loss function for the generator models.

        Returns
        -------
        None
        """
        super(MADGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def train_step(self, data: tf.Tensor) -> Dict[str, tf.Tensor]:
        """
        A single training iteration for the MADGAN model.

        Parameters
        ----------
        data : tf.Tensor
            A tensor of shape `(batch_size, height, width, channels)` that contains the input data.

        Returns
        -------
        dict[str, tf.Tensor]
            A dictionary of loss values, with keys "d_loss", "g_loss0", ..., "g_lossN" where N is the number of generators.
        """
        X = data
        
        batch_size = tf.shape(X)[0]
        random_latent_vectors = generate_latent_points(self.latent_dim, batch_size//self.n_gen, self.n_gen)
        print(self.latent_dim, batch_size//self.n_gen, self.n_gen)
        # Decode them to fake generator output
        x_generator = []
        for g in range(self.n_gen):
            x_generator.append(self.generators[g](random_latent_vectors[g]))
        
        # Combine them with real samples
        combined_samples = tf.concat([x_generator[g] for g in range(self.n_gen)] + 
                                     [X], 
                                     axis=0
                                     )
        # Assemble labels discriminating real from fake samples
        labels = tf.concat(
            [
                tf.one_hot(g * tf.ones(batch_size//self.n_gen, dtype=tf.int32), self.n_gen + 1) 
                for g in range(self.n_gen)
            ] + 
            [tf.one_hot(self.n_gen * tf.ones(batch_size, dtype=tf.int32), self.n_gen + 1)], 
            axis=0
        )

        # Add random noise to the labels. important trick
        labels += 0.05 * tf.random.uniform(shape = tf.shape(labels), minval = -1, maxval = 1)
        
        #######################
        # Train Discriminator #
        #######################
        
        # make weights in the discriminator trainable
        with tf.GradientTape() as tape:
            # Discriminator forward pass
            predictions = self.discriminator(combined_samples)
            
            # Compute the loss value
            d_loss = self.d_loss_fn(labels, predictions)
            
        # Compute gradients
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        # Update weights
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))
        
        #######################
        #   Train Generator   #
        #######################
        
        # Assemble labels that say are "all real samples" to try to fool the disc during gen training
        misleading_labels =  tf.one_hot(self.n_gen * tf.ones(batch_size//self.n_gen, dtype=tf.int32), self.n_gen + 1)
        g_loss_list = []
        fake_image = []
        
        for g in range(self.n_gen):
            with tf.GradientTape() as tape:
                # Generator[g] and discriminator forward pass
                predictions = self.discriminator(self.generators[g](random_latent_vectors[g]))
                
                # Compute the loss value
                g_loss = self.g_loss_fn(misleading_labels, predictions)
                
            # Compute gradients
            grads = tape.gradient(g_loss, self.generators[g].trainable_weights)
            # Update weights
            self.g_optimizer[g].apply_gradients(zip(grads, self.generators[g].trainable_weights))
            g_loss_list.append(g_loss)
            
        mydict = {f"g_loss{g}": g_loss_list[g] for g in range(self.n_gen)}
        mydict.update({"d_loss": d_loss})
        return mydict
