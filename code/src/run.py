import tensorflow as tf
from tensorflow.keras import layers, models, datasets
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
import os
import json

# Set random seed for reproducibility
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# Load and preprocess CIFAR-10 dataset
def load_real_samples():
    (trainX, _), (_, _) = datasets.cifar10.load_data()
    X = trainX.astype('float32')
    X = (X - 127.5) / 127.5  # Normalize to [-1, 1]
    return X

# Define the standalone discriminator model
def define_discriminator(in_shape=(32, 32, 3)):
    model = models.Sequential()
    model.add(layers.Conv2D(64, (3,3), padding='same', input_shape=in_shape))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv2D(128, (3,3), strides=(2,2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv2D(128, (3,3), strides=(2,2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(1, activation='sigmoid'))
    opt = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

# Define the standalone generator model
def define_generator(latent_dim):
    model = models.Sequential()
    n_nodes = 128 * 8 * 8
    model.add(layers.Dense(n_nodes, input_dim=latent_dim))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Reshape((8, 8, 128)))
    model.add(layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv2D(3, (3,3), activation='tanh', padding='same'))
    return model

# Define the combined generator and discriminator model, for updating the generator
def define_gan(generator, discriminator):
    discriminator.trainable = False
    model = models.Sequential()
    model.add(generator)
    model.add(discriminator)
    opt = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model

# Select real samples
def generate_real_samples(dataset, n_samples):
    ix = np.random.randint(0, dataset.shape[0], n_samples)
    X = dataset[ix]
    y = np.ones((n_samples, 1))
    return X, y

# Generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    x_input = np.random.randn(latent_dim * n_samples)
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input

# Use the generator to generate n fake examples
def generate_fake_samples(generator, latent_dim, n_samples):
    x_input = generate_latent_points(latent_dim, n_samples)
    X = generator.predict(x_input, verbose=0)
    y = np.zeros((n_samples, 1))
    return X, y

# Scale images to the [0, 1] range for InceptionV3
def scale_images(images, new_shape):
    images_list = []
    for image in images:
        new_image = tf.image.resize(image, new_shape).numpy()
        images_list.append(new_image)
    return np.asarray(images_list)

# Calculate the FID between two sets of images
def calculate_fid(model, images1, images2):
    images1 = preprocess_input(images1)
    images2 = preprocess_input(images2)
    act1 = model.predict(images1)
    act2 = model.predict(images2)
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

# Calculate the Inception Score for a set of images
def calculate_inception_score(model, images, n_split=10, eps=1E-16):
    scores = []
    n_part = int(np.floor(images.shape[0] / n_split))
    for i in range(n_split):
        ix_start, ix_end = i * n_part, (i + 1) * n_part
        subset = images[ix_start:ix_end]
        subset = preprocess_input(subset)
        p_yx = model.predict(subset)
        p_yx = np.clip(p_yx, eps, 1 - eps)
        p_y = np.expand_dims(p_yx.mean(axis=0), 0)
        kl_div = p_yx * (np.log(p_yx) - np.log(p_y))
        sum_kl_div = kl_div.sum(axis=1)
        avg_kl_div = np.mean(sum_kl_div)
        is_score = np.exp(avg_kl_div)
        scores.append(is_score)
    is_avg, is_std = np.mean(scores), np.std(scores)
    return is_avg, is_std

# Train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=256, eval_interval=10):
    bat_per_epo = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    history = {'d_loss_real': [], 'd_loss_fake': [], 'g_loss': [], 'fid': [], 'is_avg': [], 'is_std': []}
    inception_model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))
    for i in range(n_epochs):
        for j in range(bat_per_epo):
            # Generate real and fake samples
            X_real, y_real = generate_real_samples(dataset, half_batch)
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)

            # Train the discriminator (real classified as ones, fake as zeros)
            d_loss_real, _ = d_model.train_on_batch(X_real, y_real)
            d_loss_fake, _ = d_model.train_on_batch(X_fake, y_fake)

            # Update the generator via the GAN model (discriminator is frozen)
            g_loss = gan_model.train_on_batch(generate_latent_points(latent_dim, n_batch), np.ones((n_batch, 1)))

            # Store losses
            history['d_loss_real'].append(d_loss_real)
            history['d_loss_fake'].append(d_loss_fake)
            history['g_loss'].append(g_loss)

            # Evaluate FID and Inception Score at interval
            if (j + 1) % eval_interval == 0:
                # Generate fake samples to evaluate the FID and Inception score
                X_fake_eval, _ = generate_fake_samples(g_model, latent_dim, n_batch)
                
                # Rescale images for FID calculation
                X_fake_rescaled = scale_images(X_fake_eval, (299, 299))
                X_real_rescaled = scale_images(X_real, (299, 299))

                

                # Calculate FID
                fid_value = calculate_fid(inception_model, X_real_rescaled, X_fake_rescaled)
                history['fid'].append(fid_value)

                # Calculate Inception Score
                is_avg, is_std = calculate_inception_score(inception_model, X_fake_rescaled)
                history['is_avg'].append(is_avg)
                history['is_std'].append(is_std)

                print(f"Epoch {i+1}/{n_epochs}, Batch {j+1}/{bat_per_epo}, "
                      f"D Loss Real: {d_loss_real:.4f}, D Loss Fake: {d_loss_fake:.4f}, "
                      f"G Loss: {g_loss:.4f}, FID: {fid_value:.4f}, IS Avg: {is_avg:.4f}, IS Std: {is_std:.4f}")

        # Save model weights at each epoch (optional)
        # g_model.save(f"generator_epoch_{i+1}.h5")
        # d_model.save(f"discriminator_epoch_{i+1}.h5")

    return history

# Plot training history and metrics
def plot_history(history, output_dir):
    # Plot losses
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(history['d_loss_real'], label='Discriminator Real Loss')
    plt.plot(history['d_loss_fake'], label='Discriminator Fake Loss')
    plt.plot(history['g_loss'], label='Generator Loss')
    plt.title('Losses')
    plt.legend()

    # Plot FID and Inception Score
    plt.subplot(2, 2, 2)
    plt.plot(history['fid'], label='FID')
    plt.title('FID over Epochs')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(history['is_avg'], label='Inception Score (avg)')
    plt.plot(history['is_std'], label='Inception Score (std)')
    plt.title('Inception Score over Epochs')
    plt.legend()

    # Save the plot
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()

# Save training history to a JSON file
def save_history(history, output_dir):
    history_file = os.path.join(output_dir, 'training_history.json')
    with open(history_file, 'w') as f:
        json.dump(history, f)

# Main execution
def main():
    latent_dim = 100
    n_epochs = 100
    n_batch = 256
    eval_interval = 10  # Every 10 batches

    # Load CIFAR-10 dataset
    dataset = load_real_samples()

    # Define the models
    d_model = define_discriminator()
    g_model = define_generator(latent_dim)
    gan_model = define_gan(g_model, d_model)

    # Train the models and track history
    history = train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs, n_batch, eval_interval)

    # Plot and save training history
    output_dir = './gan_is_fid_score_test'
    plot_history(history, output_dir)
    save_history(history, output_dir)
    print("Training complete. History and plots saved.")

if __name__ == "__main__":
    main()

 


# dsc = DatasetCreator(
#     created_images_folder=Path(
#         "C:\\Users\\NiXoN\\Desktop\\_thesis\\mad_gan_thesis\\code\\experiments\\2025-02-03_generative_creation_test_cifar10"
#     ),
#     tf_dataset_load_func=tf.keras.datasets.cifar10.load_data,
#     number_of_generated_images_per_class={
#         0: 1,
#         1: 1,
#         2: 1,
#         3: 1,
#         4: 1,
#         5: 1,
#         6: 1,
#         7: 1,
#         8: 1,
#         9: 1,
#     },
#     number_of_real_images_per_class={
#         0: 1,
#         1: 1,
#         2: 1,
#         3: 1,
#         4: 1,
#         5: 1,
#         6: 1,
#         7: 1,
#         8: 1,
#         9: 1,
#     },
# )
# # Get the dataset without any batching
# ds = dsc.create_dataset()
#
# Path("./test").mkdir(exist_ok=True)
#
# def denormalize_image(image):
#     """
#     Convert image from arbitrary range to [0, 255] range.
#     This handles both the [-1, 1] normalization and any outliers.
#     """
#     # First clip the values to a reasonable range
#     image = tf.clip_by_value(image, -1.0, 1.0)
#     # Convert from [-1, 1] to [0, 1]
#     image = (image + 1.0) / 2.0
#     # Convert to [0, 255] range and ensure uint8 type
#     # I dont like that this rounds the values, but it's the simplest way
#     image = tf.cast(image * 255, tf.uint8)
#     return image
#
# for i, (image, label) in enumerate(ds):
#     print(f"\nProcessing image {i}:")
#     print(f"Image shape: {image.shape}")
#     print(f"Image value range: [{tf.reduce_min(image):.2f}, {tf.reduce_max(image):.2f}]")
#     print(f"Label: {label.numpy()}")
#
#     save_image = denormalize_image(image)
#     save_path = str(Path("./test") / f"image_{i}.png")
#     fig, ax = plt.subplots()
#     ax.imshow(save_image.numpy())
#     ax.set_title(str(label.numpy()))
#     ax.axis('off')
#     fig.savefig(save_path, bbox_inches='tight')
#     plt.close(fig)
#
#     # tf.keras.utils.save_img(save_path, save_image.numpy())
#
#
