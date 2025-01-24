import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.datasets import mnist

# Hyperparameters
latent_dim = 100
num_classes = 10
num_generators = 3  # K
batch_size = 64
lr = 0.0002
epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Shared layers for all generators
class SharedGeneratorLayers(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU()
        )
    
    def forward(self, z, y):
        z = torch.cat([z, y], dim=1)
        return self.main(z)

# Individual generator with shared base + unique output layer
class Generator(nn.Module):
    def __init__(self, shared_layers):
        super().__init__()
        self.shared = shared_layers
        self.output_layer = nn.Sequential(
            nn.Linear(1024, 28*28),
            nn.Tanh()
        )
    
    def forward(self, z, y):
        shared_output = self.shared(z, y)
        return self.output_layer(shared_output).view(-1, 1, 28, 28)

# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(28*28, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2)
        )
        self.real_fake = nn.Linear(256, 1)
        self.classifier = nn.Linear(256, num_classes)
        self.gen_id = nn.Linear(256, num_generators)
    
    def forward(self, x):
        x = x.view(-1, 28*28)
        features = self.features(x)
        return (
            self.real_fake(features),
            self.classifier(features),
            self.gen_id(features)
        )

# Function to monitor and save generated images
def monitor_generators(generators, epoch, num_images=5):
    generators = [gen.to(device) for gen in generators]
    generators = [gen.eval() for gen in generators]
    
    # Create a directory to save images if it doesn't exist
    save_dir = "generated_images"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Create a figure to display images
    fig, axes = plt.subplots(num_generators, num_images, figsize=(15, 3 * num_generators))
    fig.suptitle(f"Generated Images at Epoch {epoch}", fontsize=16)
    
    # Generate images for each generator
    for i, gen in enumerate(generators):
        for j in range(num_images):
            # Random noise and fixed class label
            z = torch.randn(1, latent_dim).to(device)
            y = torch.tensor([j % num_classes]).to(device)  # Cycle through classes
            y_onehot = nn.functional.one_hot(y, num_classes).float()
            
            # Generate image
            with torch.no_grad():
                gen_img = gen(z, y_onehot).cpu().squeeze().numpy()
            
            # Plot image
            ax = axes[i, j]
            ax.imshow(gen_img, cmap='gray')
            ax.axis('off')
            if j == 0:
                ax.set_title(f"Generator {i+1}\nClass {y.item()}")
            else:
                ax.set_title(f"Class {y.item()}")
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"epoch_{epoch:03d}.png"))
    plt.close()

# Load MNIST dataset using TensorFlow
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize and reshape the data
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0
x_train = x_train.reshape(-1, 1, 28, 28)
x_test = x_test.reshape(-1, 1, 28, 28)

# Convert to PyTorch tensors
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train).long()
x_test = torch.from_numpy(x_test)
y_test = torch.from_numpy(y_test).long()

# Create PyTorch datasets and dataloaders
train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize models
shared_layers = SharedGeneratorLayers(latent_dim, num_classes).to(device)
generators = [Generator(shared_layers).to(device) for _ in range(num_generators)]
discriminator = Discriminator().to(device)

# Optimizers
all_generator_params = list(shared_layers.parameters())
for gen in generators:
    all_generator_params += list(gen.output_layer.parameters())
G_optimizer = optim.Adam(all_generator_params, lr=lr)
D_optimizer = optim.Adam(discriminator.parameters(), lr=lr)

# Loss functions
adv_loss = nn.BCEWithLogitsLoss()
class_loss = nn.CrossEntropyLoss()
kl_loss = nn.KLDivLoss(reduction='batchmean')

# Adjust learning rates
G_optimizer = optim.Adam(all_generator_params, lr=0.0001, betas=(0.5, 0.999))
D_optimizer = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

# Add feature matching loss
def feature_matching_loss(real_features, fake_features):
    return torch.mean(torch.abs(real_features - fake_features))

# Inside the training loop
for epoch in range(epochs):
    for real_imgs, real_labels in train_loader:
        real_imgs = real_imgs.to(device)
        real_labels = real_labels.to(device)
        batch_size = real_imgs.size(0)
        
        # Convert labels to one-hot
        real_y = nn.functional.one_hot(real_labels, num_classes).float()
        
        # Train Discriminator
        D_optimizer.zero_grad()
        
        # Real images
        real_logits, real_class, _ = discriminator(real_imgs)
        d_real_loss = adv_loss(real_logits, torch.ones(batch_size, 1).to(device))
        d_class_real = class_loss(real_class, real_labels)
        
        # Fake images
        fake_imgs = []
        gen_ids = []
        fake_labels = []
        for i in range(num_generators):
            z = torch.randn(batch_size, latent_dim).to(device)
            y = torch.randint(0, num_classes, (batch_size,)).to(device)
            y_onehot = nn.functional.one_hot(y, num_classes).float()
            fake = generators[i](z, y_onehot)
            fake_imgs.append(fake)
            gen_ids.extend([i]*batch_size)
            fake_labels.append(y)
        
        fake_imgs = torch.cat(fake_imgs)
        gen_ids = torch.tensor(gen_ids).to(device)
        fake_labels = torch.cat(fake_labels)
        
        fake_logits, fake_class, fake_gen = discriminator(fake_imgs.detach())
        d_fake_loss = adv_loss(fake_logits, torch.zeros(fake_imgs.size(0), 1).to(device))
        d_class_fake = class_loss(fake_class, fake_labels)
        d_gen_loss = class_loss(fake_gen, gen_ids)
        
        # Total D loss
        D_total = d_real_loss + d_fake_loss + d_class_real + d_class_fake + d_gen_loss
        D_total.backward()
        D_optimizer.step()
        
        # Train Generators
        G_optimizer.zero_grad()
        total_g_loss = 0
        for i, gen in enumerate(generators):
            z = torch.randn(batch_size, latent_dim).to(device)
            y = torch.randint(0, num_classes, (batch_size,)).to(device)
            y_onehot = nn.functional.one_hot(y, num_classes).float()
            gen_fake = gen(z, y_onehot)
            
            fake_logits, fake_class, fake_gen = discriminator(gen_fake)
            
            # Adversarial loss
            g_adv = adv_loss(fake_logits, torch.ones(batch_size, 1).to(device))
            # Class loss
            g_class = class_loss(fake_class, y)
            # Diversity loss (KL divergence between fake_gen and uniform)
            uniform = torch.ones_like(fake_gen) / num_generators
            g_div = kl_loss(fake_gen.log_softmax(dim=1), uniform)
            # Feature matching loss
            real_features = discriminator.features(real_imgs)
            fake_features = discriminator.features(gen_fake)
            g_feat = feature_matching_loss(real_features, fake_features)
            
            total_g_loss += (g_adv + g_class + g_div + g_feat)
        
        # Backpropagate total loss and update all parameters
        total_g_loss.backward()
        G_optimizer.step()
    
    # Monitor generators at the end of each epoch
    if (epoch + 1) % 5 == 0:  # Monitor every 5 epochs
        monitor_generators(generators, epoch + 1)