import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# --- Configuration ---
LATENT_DIM = 128       # Increased latent dim slightly
CONDITION_DIM = 10     # 10 classes for CIFAR-10
DATA_SHAPE = (32, 32, 3) # Shape of the CIFAR-10 image data
NUM_GENERATORS = 3     # Number of diverse generators
LEARNING_RATE_G = 0.0002
LEARNING_RATE_D = 0.0002
BETA_1 = 0.5           # Adam optimizer parameter
DIVERSITY_WEIGHT = 0.4 # Lambda coefficient for the diversity loss term (Needs Tuning!)
BATCH_SIZE = 64        # Smaller batch size might be needed depending on GPU memory
EPOCHS = 100           # CIFAR-10 needs more epochs than MNIST (adjust as needed)
SAVE_INTERVAL = 10     # Save generated image grid every N epochs
OUTPUT_DIR = "cmadgan_cifar10_output" # Directory for results

# CIFAR-10 Class Names
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

# --- Create Output Directory ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Use GPU if available ---
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("Using GPU")
    except RuntimeError as e:
        print(e)
else:
    print("Using CPU")

# --- Define Deeper CNN Network Architectures for CIFAR-10 ---

def build_generator_cnn_cifar(latent_dim, condition_dim, data_shape, name="Generator_CNN_CIFAR"):
    """Builds a Conditional Generator Model for CIFAR-10 using Conv2DTranspose."""
    noise_input = keras.Input(shape=(latent_dim,), name="noise_input")
    condition_input = keras.Input(shape=(condition_dim,), name="condition_input")

    merged_input = layers.Concatenate()([noise_input, condition_input])

    # Project and reshape to start convolution at 4x4
    start_filters = 256 # Can be adjusted (e.g., 512)
    x = layers.Dense(4 * 4 * start_filters)(merged_input)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Reshape((4, 4, start_filters))(x) # Shape: (4, 4, start_filters)

    # Upsample to 8x8
    x = layers.Conv2DTranspose(start_filters // 2, kernel_size=4, strides=2, padding='same')(x) # (8, 8, 128)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    # Upsample to 16x16
    x = layers.Conv2DTranspose(start_filters // 4, kernel_size=4, strides=2, padding='same')(x) # (16, 16, 64)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    # Upsample to 32x32
    x = layers.Conv2DTranspose(start_filters // 8, kernel_size=4, strides=2, padding='same')(x) # (32, 32, 32)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)


    # Final Conv layer to get 3 channel image (RGB)
    # Use tanh activation for pixel values in [-1, 1]
    output = layers.Conv2D(data_shape[-1], kernel_size=5, strides=1, padding='same', activation='tanh')(x) # (32, 32, 3)

    model = keras.Model([noise_input, condition_input], output, name=name)
    return model

def build_discriminator_cnn_cifar(data_shape, condition_dim, name="Discriminator_CNN_CIFAR"):
    """Builds a Conditional Discriminator Model for CIFAR-10 using Conv2D."""
    data_input = keras.Input(shape=data_shape, name="data_input") # (32, 32, 3)
    condition_input = keras.Input(shape=(condition_dim,), name="condition_input") # (10,)

    # Process condition: Embed and reshape for concatenation
    # Project condition to match spatial dims * channels of an early layer or just spatially
    # Let's try reshaping to 32x32x1 for simple concatenation
    cond_embedding_size = data_shape[0] * data_shape[1] # 32*32 = 1024
    c = layers.Dense(cond_embedding_size)(condition_input)
    c = layers.Reshape((data_shape[0], data_shape[1], 1))(c) # Shape: (32, 32, 1)

    # Concatenate processed condition with image data along the channel axis
    merged_input = layers.Concatenate(axis=-1)([data_input, c]) # Shape: (32, 32, 3+1=4)

    start_filters = 64 # Can be adjusted

    # Downsample 32x32 -> 16x16
    x = layers.Conv2D(start_filters, kernel_size=4, strides=2, padding='same')(merged_input) # (16, 16, 64)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.3)(x)

    # Downsample 16x16 -> 8x8
    x = layers.Conv2D(start_filters * 2, kernel_size=4, strides=2, padding='same')(x) # (8, 8, 128)
    # Using LayerNormalization instead of BatchNormalization can sometimes help in Discriminators
    x = layers.LayerNormalization()(x) # Optional: Experiment with LayerNorm/BatchNorm
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.3)(x)

    # Downsample 8x8 -> 4x4
    x = layers.Conv2D(start_filters * 4, kernel_size=4, strides=2, padding='same')(x) # (4, 4, 256)
    x = layers.LayerNormalization()(x) # Optional
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.3)(x)

    # Flatten and add Dense layers
    x = layers.Flatten()(x)
    # x = layers.Dense(128)(x) # Optional intermediate dense layer
    # x = layers.LeakyReLU(alpha=0.2)(x)

    # Output layer: Single logit
    output = layers.Dense(1)(x)

    model = keras.Model([data_input, condition_input], output, name=name)
    return model

# --- Define the CMAD-GAN Model (using the same class structure) ---

class CMADGAN(keras.Model):
    def __init__(self, latent_dim, condition_dim, data_shape, num_generators, diversity_weight):
        super(CMADGAN, self).__init__()
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.data_shape = data_shape
        self.num_generators = num_generators
        self.diversity_weight = diversity_weight

        # Use CIFAR-10 specific architectures
        self.discriminator = build_discriminator_cnn_cifar(self.data_shape, self.condition_dim)
        self.generators = [
            build_generator_cnn_cifar(self.latent_dim, self.condition_dim, self.data_shape, name=f"Generator_{i}")
            for i in range(self.num_generators)
        ]

        # Loss functions (same logic)
        self.cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)
        self.cosine_similarity_loss = lambda x, y: -tf.reduce_mean(
             tf.losses.cosine_similarity(tf.reshape(x, [tf.shape(x)[0], -1]),
                                         tf.reshape(y, [tf.shape(y)[0], -1]), axis=-1)
        )

    def compile(self, d_optimizer, g_optimizers):
        super(CMADGAN, self).compile()
        self.d_optimizer = d_optimizer
        if not isinstance(g_optimizers, list) or len(g_optimizers) != self.num_generators:
             raise ValueError(f"g_optimizers must be a list of {self.num_generators} optimizers.")
        self.g_optimizers = g_optimizers

    # calculate_diversity_loss remains the same as the MNIST version
    def calculate_diversity_loss(self, fake_samples_list):
        total_similarity = 0.0
        num_pairs = 0
        for i in range(self.num_generators):
            for j in range(i + 1, self.num_generators):
                similarity = self.cosine_similarity_loss(fake_samples_list[i], fake_samples_list[j])
                total_similarity += similarity
                num_pairs += 1
        if num_pairs == 0: return tf.constant(0.0)
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
            fake_samples_list_no_grad = [gen([noise, conditions], training=False) for gen in self.generators]

            for fake_samples in fake_samples_list_no_grad:
                fake_output = self.discriminator([fake_samples, conditions], training=True)
                d_loss_fake_total += self.cross_entropy(tf.zeros_like(fake_output), fake_output)

            d_loss_fake = d_loss_fake_total / self.num_generators
            d_loss = d_loss_real + d_loss_fake

        grads_d = tape_d.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(grads_d, self.discriminator.trainable_variables))

        # --- Train Generators ---
        with tf.GradientTape() as tape_g:
            fake_samples_list = []
            gen_adv_losses = []
            for i in range(self.num_generators):
                fake_samples = self.generators[i]([noise, conditions], training=True)
                fake_samples_list.append(fake_samples)
                fake_output = self.discriminator([fake_samples, conditions], training=False)
                adv_loss = self.cross_entropy(tf.ones_like(fake_output), fake_output)
                gen_adv_losses.append(adv_loss)

            diversity_loss = self.calculate_diversity_loss(fake_samples_list)
            total_generator_loss = tf.reduce_sum(gen_adv_losses) + self.diversity_weight * diversity_loss

        all_gen_trainable_vars = [var for gen in self.generators for var in gen.trainable_variables]
        grads_g = tape_g.gradient(total_generator_loss, all_gen_trainable_vars)

        var_index = 0
        for i in range(self.num_generators):
            num_vars = len(self.generators[i].trainable_variables)
            gen_grads = grads_g[var_index : var_index + num_vars]
            self.g_optimizers[i].apply_gradients(zip(gen_grads, self.generators[i].trainable_variables))
            var_index += num_vars

        return {
            "d_loss": d_loss,
            "g_adv_loss": tf.reduce_mean(gen_adv_losses),
            "g_div_loss": diversity_loss,
            "g_total_loss": total_generator_loss / self.num_generators
        }

    def generate(self, noise, conditions):
        """Generate samples from all generators given noise and conditions."""
        generated_samples = [gen([noise, conditions], training=False) for gen in self.generators]
        return generated_samples

# --- Load and Prepare CIFAR-10 Dataset ---
print("\n--- Loading and Preparing CIFAR-10 Dataset ---")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Combine train and test sets for more data
all_images = np.concatenate([x_train, x_test], axis=0).astype("float32")
all_labels = np.concatenate([y_train, y_test], axis=0).flatten().astype("int32") # Flatten labels

# Normalize images to [-1, 1]
all_images = (all_images - 127.5) / 127.5

# One-hot encode labels
all_labels_one_hot = tf.one_hot(all_labels, depth=CONDITION_DIM)

# Create tf.data.Dataset
dataset = tf.data.Dataset.from_tensor_slices((all_images, all_labels_one_hot))
dataset = dataset.shuffle(buffer_size=len(all_images)).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

print(f"Dataset created with {len(all_images)} images.")
print(f"Image shape: {all_images.shape[1:]}, Label (one-hot) shape: {all_labels_one_hot.shape[1:]}")

# --- Prepare Optimizers ---
d_optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE_D, beta_1=BETA_1)
g_optimizers = [
    keras.optimizers.Adam(learning_rate=LEARNING_RATE_G, beta_1=BETA_1)
    for _ in range(NUM_GENERATORS)
]

# --- Instantiate and Compile the CMAD-GAN ---
cmadgan = CMADGAN(
    latent_dim=LATENT_DIM,
    condition_dim=CONDITION_DIM,
    data_shape=DATA_SHAPE,
    num_generators=NUM_GENERATORS,
    diversity_weight=DIVERSITY_WEIGHT
)
cmadgan.compile(d_optimizer=d_optimizer, g_optimizers=g_optimizers)

print("\n--- Model Summary (Discriminator - CIFAR10) ---")
cmadgan.discriminator.summary()
print("\n--- Model Summary (Generator 0 - CIFAR10) ---")
cmadgan.generators[0].summary()


# --- Function to Save Generated CIFAR-10 Images ---
def save_generated_images_cifar(epoch, generated_sample_sets, class_names, examples=10, dim=(NUM_GENERATORS, 10), figsize=(12, NUM_GENERATORS*1.2)):
    """Saves a grid of generated CIFAR-10 images for each generator."""
    fig, axes = plt.subplots(dim[0], dim[1], figsize=figsize)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    for i in range(dim[0]): # Loop through generators
        for j in range(dim[1]): # Loop through examples (classes 0-9)
            img = generated_sample_sets[i][j] # Get the j-th sample from the i-th generator
            img = tf.reshape(img, DATA_SHAPE) # Reshape just in case
            # De-normalize from [-1, 1] to [0, 1] for display
            img = (img + 1.0) / 2.0
            img = tf.clip_by_value(img, 0.0, 1.0) # Ensure values are valid for imshow
            axes[i, j].imshow(img) # Display color image
            axes[i, j].axis('off')
            if j == 0: # Add generator label
                 axes[i, j].text(-0.1, 0.5, f'Gen {i}', horizontalalignment='right', verticalalignment='center', transform=axes[i, j].transAxes, fontsize=10)
            if i == 0: # Add class label
                 axes[i, j].set_title(f'{class_names[j]}', fontsize=9, rotation=20, ha='left', va='bottom')


    plt.suptitle(f'CMAD-GAN CIFAR-10 Generated Images - Epoch {epoch+1}', fontsize=16)
    save_path = os.path.join(OUTPUT_DIR, f"cifar10_epoch_{epoch+1:04d}.png")
    plt.savefig(save_path)
    print(f"Saved generated image grid to {save_path}")
    plt.close(fig)


# --- Training Loop ---
print("\n--- Starting Training (CIFAR-10) ---")
# Fixed noise and conditions for consistent visualization
fixed_noise = tf.random.normal(shape=(CONDITION_DIM, LATENT_DIM)) # Generate one sample per class
fixed_conditions = tf.one_hot(tf.range(CONDITION_DIM), depth=CONDITION_DIM) # Conditions 0-9

total_start_time = time.time()
history = {'d_loss': [], 'g_adv_loss': [], 'g_div_loss': [], 'g_total_loss': []}

for epoch in range(EPOCHS):
    epoch_start_time = time.time()
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    d_loss_epoch, g_adv_epoch, g_div_epoch, g_total_epoch = [], [], [], []

    for step, (real_batch, condition_batch) in enumerate(dataset):
        step_start_time = time.time()
        losses = cmadgan.train_step((real_batch, condition_batch))
        d_loss_epoch.append(losses['d_loss'].numpy())
        g_adv_epoch.append(losses['g_adv_loss'].numpy())
        g_div_epoch.append(losses['g_div_loss'].numpy())
        g_total_epoch.append(losses['g_total_loss'].numpy())

        if step % 100 == 0: # Print progress less frequently for potentially longer epochs
             step_time = time.time() - step_start_time
             print(f"  Step {step}/{len(dataset)}: "
                  f"d_loss={losses['d_loss']:.4f}, "
                  f"g_adv={losses['g_adv_loss']:.4f}, "
                  f"g_div={losses['g_div_loss']:.4f}, "
                  f"g_total={losses['g_total_loss']:.4f} "
                  f"({step_time*100:.2f} sec/100 steps)")

    # Log average losses for the epoch
    avg_d_loss = np.mean(d_loss_epoch)
    avg_g_adv = np.mean(g_adv_epoch)
    avg_g_div = np.mean(g_div_epoch)
    avg_g_total = np.mean(g_total_epoch)
    history['d_loss'].append(avg_d_loss)
    history['g_adv_loss'].append(avg_g_adv)
    history['g_div_loss'].append(avg_g_div)
    history['g_total_loss'].append(avg_g_total)

    epoch_time = time.time() - epoch_start_time
    print(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds.")
    print(f"  Avg Losses: D={avg_d_loss:.4f}, G_Adv={avg_g_adv:.4f}, G_Div={avg_g_div:.4f}, G_Total={avg_g_total:.4f}")

    # Save generated images periodically and at the end
    if (epoch + 1) % SAVE_INTERVAL == 0 or (epoch + 1) == EPOCHS:
        generated_sample_sets = cmadgan.generate(fixed_noise, fixed_conditions)
        save_generated_images_cifar(epoch, generated_sample_sets, CIFAR10_CLASSES, examples=CONDITION_DIM)


print(f"\nTotal training time: {time.time() - total_start_time:.2f} seconds.")

# --- Plot Loss History ---
plt.figure(figsize=(12, 5))
plt.plot(history['d_loss'], label='Discriminator Loss')
plt.plot(history['g_total_loss'], label='Total Generator Loss')
plt.plot(history['g_adv_loss'], label='Generator Adv Loss', linestyle=':')
plt.plot(history['g_div_loss'], label='Generator Div Loss', linestyle=':')
plt.title('CMAD-GAN CIFAR-10 Training Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
loss_plot_path = os.path.join(OUTPUT_DIR, "cifar10_training_losses.png")
plt.savefig(loss_plot_path)
print(f"Saved loss history plot to {loss_plot_path}")
plt.show()

print("\n--- Training Finished (CIFAR-10) ---")