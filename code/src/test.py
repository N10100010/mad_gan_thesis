import matplotlib.pyplot as plt
import numpy as np

# Load the model architecture (generator and discriminator)
from model_definitions.conditional_gan.gan import ConditionalGAN
from model_definitions.discriminators.conditional_cifar.disc import (
    define_discriminator as define_discriminator_cifar,
)
from model_definitions.generators.conditional_cifar.gen import (
    define_generator as define_generator_cifar,
)

# Define the generator and discriminator
generator = define_generator_cifar(latent_dim=100, n_classes=10)
discriminator = define_discriminator_cifar(n_classes=10)

# Create the Conditional GAN model
gan = ConditionalGAN(
    generator=generator,
    discriminator=discriminator,
    latent_dim=100,
    n_classes=10,
)
gan.built = True  # Set the model as built

# Load the model weights
gan.load_weights(
    "C:\\Users\\NiXoN\\Desktop\\_thesis\\mad_gan_thesis\\code\\experiments\\CONDITIONAL_GAN_MODELS\\2025-03-14_CIFAR_ConditionalGAN_Experiment_latent_100\\checkpoints\\backup_epoch_4.h5"
)

# Generate a random latent vector and a label (for example, label 3)
latent_vector = np.random.normal(size=(1, 100))  # latent_dim = 100
label = np.array([3])  # Choose a label (between 0 and 9 for CIFAR-10)

# Reshape label to match the input expected by the model
label = np.reshape(label, (-1, 1))

# Generate an image from the generator using the latent vector and label
generated_image = generator([latent_vector, label], training=False)
generated_image = (generated_image * 127.5 + 127.5).numpy().astype(np.uint8)

# Display the generated image using matplotlib
plt.imshow(generated_image[0])  # .numpy() to convert the tensor to a numpy array
plt.axis("off")  # Turn off axis labels
plt.show()
