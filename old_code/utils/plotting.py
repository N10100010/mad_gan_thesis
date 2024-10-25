from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf 
from keras.callbacks import History

import os

from .logger import get_logger

logger = get_logger()



# Function to create a GIF from a list of image paths
def create_gif_from_png(image_folder: str, output_path: str, duration=500) -> None:
    """Generates a animated GIF, given png's in a folder, defined by image_folder. The output GIF will be saved at output_path.

    Args:
        image_folder (str): Path to png's
        output_path (str): Path to save the GIF to.
        duration (int, optional): Duration to show each png in the resulting GIF. Defaults to 500.
    """
    image_files = sorted([os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.endswith('.png')])

    frames = []
    for image_name in image_files:
        img = Image.open(image_name)
        frames.append(img)

    # Save the frames as an animated GIF
    frames[0].save(output_path, format='GIF', append_images=frames[1:], save_all=True, duration=duration, loop=0)



def plot_mad_gan_history(history: History, output_path: str = None, save: bool = True) -> None:
    """Plots the history of a MAD-GAN set. AKA: N generators and 1 discriminator.

    Args:
        history (keras.callbacks.History): _description_
        save (bool, optional): _description_. Defaults to True.
    """
    
    if save: 
        if not output_path: 
            logger.error("Called plot_training_history to save the output, without defining output_path.")
            raise ValueError("output_path is not defined.")
    
    # Extract losses from history
    history_dict = history.history
    generator_losses = []
    discriminator_loss = None
    
    
    # Separate generator losses and discriminator loss
    for key in history_dict.keys():
        if 'g_loss' in key:
            generator_losses.append((key, history_dict[key]))
        elif key == 'd_loss':
            discriminator_loss = history_dict[key]
    
    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Plot generator losses
    for gen_loss_key, gen_loss_values in generator_losses:
        plt.plot(gen_loss_values, label=gen_loss_key)
    
    # Plot discriminator loss
    if discriminator_loss is not None:
        plt.plot(discriminator_loss, label='d_loss', linewidth=2, linestyle='--', color='black')
    
    plt.title('Training Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    if save: 
        plt.savefig(f'{output_path}', dpi=200, format="png")
        
    
    # Display the plot
    plt.show()


def plot_generators_examples(
    n_rows: int, n_cols: int, 
    random_latent_vectors: list, 
    data, generators: list,  
    dir_name: str, 
    epoch: int, 
    save: bool = False, show: bool = False,
) -> None: 
    # Create a figure and a grid of subplots
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(12, 8))
    fig.suptitle(f'Epoch: {epoch}', fontsize=20)
    # Flatten the axes array to iterate over individual subplots
    axes = axes.flatten()

    # Iterate over the subplots
    for i, ax in enumerate(axes):
        # Calculate the current row based on index
        current_row = i // n_cols
        
        # Determine if we're plotting real or generated data
        if (i + 1) % n_cols == 0: 
            # Plot real data
            if current_row < len(data):
                ax.imshow((data[current_row, :, :,] * 127.5 + 127.5) / 255, cmap='gray')
                ax.set_title("REAL")
            else:
                print(f"Skipping real data plot for row {current_row}: Index out of bounds.")
        else: 
            # Plot generated data
            generator_index = i % (n_cols - 1)
            generated_sample = generators[generator_index](random_latent_vectors[generator_index])
            ax.imshow((generated_sample[current_row, :, :,] * 127.5 + 127.5) / 255, cmap='gray')
            ax.set_title(f"FAKE (Gen {generator_index + 1})")
        
        # Turn off axis labels for clarity
        ax.axis('off')
    
    # Adjust layout and spacing
    fig.tight_layout()

    if save: 
        plt.savefig(f'{dir_name}/image_at_epoch_{(epoch + 1):04}.png', dpi=200, format="png")
    #if show: 
    #    plt.show()
