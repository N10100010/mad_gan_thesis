import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10

# Set global plotting style for thesis-level aesthetics
plt.style.use('seaborn-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.figsize': (6, 4),
    'figure.dpi': 300,
    'axes.edgecolor': 'black',
    'axes.linewidth': 1,
    'axes.labelweight': 'bold',
    'text.usetex': False,  # Set to True if you compile with LaTeX
})

# Helper function to create a clean bar plot
def plot_distribution(labels, dataset_name, class_names=None, save_path=None):
    labels = labels.flatten()
    classes, counts = np.unique(labels, return_counts=True)

    fig, ax = plt.subplots()
    bars = ax.bar(classes, counts, color='#4C72B0', edgecolor='black')

    if class_names:
        ax.set_xticks(classes)
        ax.set_xticklabels(class_names, rotation=45, ha='right')

    ax.set_title(f"{dataset_name} – Class Distribution", pad=10, weight='bold')
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of Samples")
    ax.set_axisbelow(True)
    ax.spines[['right', 'top']].set_visible(False)
    ax.grid(axis='y', linestyle='--', linewidth=0.7)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

# Load datasets
(x_mnist, y_mnist), _ = mnist.load_data()
(x_fashion, y_fashion), _ = fashion_mnist.load_data()
(x_cifar, y_cifar), _ = cifar10.load_data()

# Define class names
fashion_labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                  "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
cifar_labels = ["Airplane", "Automobile", "Bird", "Cat", "Deer",
                "Dog", "Frog", "Horse", "Ship", "Truck"]

# Create and save the plots
plot_distribution(y_mnist, "MNIST", save_path="C:\\Users\\NiXoN\\Desktop\\_thesis\\mad_gan_thesis\\latex\\master_thesis\\abb\\used_datasets_class_dist\\mnist_class_distribution.png")
plot_distribution(y_fashion, "Fashion-MNIST", fashion_labels, "C:\\Users\\NiXoN\\Desktop\\_thesis\\mad_gan_thesis\\latex\\master_thesis\\abb\\used_datasets_class_dist\\fashion_mnist_class_distribution.png")
plot_distribution(y_cifar, "CIFAR-10", cifar_labels, "C:\\Users\\NiXoN\\Desktop\\_thesis\\mad_gan_thesis\\latex\\master_thesis\\abb\\used_datasets_class_dist\\cifar10_class_distribution.png")
