import json
import os
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .logging import setup_logger

logger = setup_logger(name="Plotting")


def plot_labels_histogram(
    data,
    title: str,
    suptitle: str = None,
    x_tick_labels: list = [str(_) for _ in range(10)],
    show_bar_counts: bool = True,
    show: bool = True,
    save: bool = False,
):
    plot_config_path = Path(
        "C:\\Users\\NiXoN\\Desktop\\_thesis\\mad_gan_thesis\\code\\src\\plott_configs\\config.json"
    )
    plot_config = None
    with open(plot_config_path) as f:
        plot_config = json.load(f)

    plt.rcParams.update(
        {
            "font.family": plot_config.get("font_family", "serif"),
            "font.size": plot_config.get("font_size", 12),
        }
    )

    counts, bin_edges, bars = plt.hist(
        data,
        bins=np.arange(11) - 0.5,
        edgecolor=plot_config.get("histogram_edge_color", "black"),
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.xticks(bin_centers, labels=x_tick_labels)
    plt.title(title, fontsize=10)
    plt.suptitle(
        suptitle,
    )

    if show_bar_counts:
        plt.bar_label(bars, fontsize=10, label_type="edge")

    if show:
        plt.show()


def process_experiment_folder(experiments_root: Path, plot_config_path: Path) -> None:
    """
    Recursively searches for .npy files in the given experiment folder and
    calls `plot_training_history` for each one, saving plots in the same folder.

    Args:
        experiments_root (Path): Root directory containing experiment subfolders.
        plot_config_path (Path): Path to the JSON file defining plot styles.

    Returns:
        None
    """
    experiments_root = Path(experiments_root)

    # Traverse all subdirectories
    for root, _, files in os.walk(experiments_root):
        root_path = Path(root)

        # Check for .npy files in the current folder
        for file in files:
            if file.endswith(".npy"):
                history_path = root_path / file
                save_path = root_path / "plots"

                logger.info(f"Processing: {history_path}")  # Debugging output

                # Call the plot function
                plot_training_history(history_path, plot_config_path, save_path)


def plot_training_history(
    history_path: Path, plot_config_path: Path, save_path: Path = None
) -> None:
    """
    Loads training history and plot configuration, then generates separate plots for each metric.

    Args:
        history_path (Path): Path to the .npy file containing the training history.
        plot_config_path (Path): Path to the JSON file defining plot styles.
        save_path (Path, optional): Directory where plots should be saved. If None, plots are only displayed.

    Returns:
        None
    """
    with open(plot_config_path, "r") as f:
        config = json.load(f)

    history = np.load(history_path, allow_pickle=True).item()

    plt.rcParams.update(
        {
            "font.family": config.get("font_family", "serif"),
            "font.size": config.get("font_size", 12),
        }
    )

    training_metrics = [key for key in history.keys() if not key.startswith("val_")]

    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

    for metric in training_metrics:
        plt.figure(figsize=tuple(config["figsize"]), dpi=config.get("dpi", 100))

        plt.plot(
            history[metric], label=f"Train {metric}", **config["line_styles"]["train"]
        )

        val_metric = f"val_{metric}"
        if val_metric in history:
            plt.plot(
                history[val_metric],
                label=f"Validation {metric}",
                **config["line_styles"]["val"],
            )

        plt.title(metric.replace("_", " ").capitalize())
        plt.xlabel(config["xlabel"])
        plt.ylabel(config["ylabel"])

        if config["grid"]:
            plt.grid(True)
        if config["legend"]:
            plt.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path / f"{metric}.png", dpi=config.get("dpi", 300))
        else:
            plt.show()

        plt.close()  # Close the figure to free memory


def plot_classifier_training_history(
    history: Dict, save: bool = True, path: Path = None, display: bool = False
) -> None:
    """
    Visualizes training history with metrics and loss curves.

    Parameters:
    history (Dict): History object returned from model.fit()
    save (bool): Whether to save the plot (default: True)
    path (str): Path to save the plot (default: 'training_history.png')
    display (bool): Whether to show the plot (default: False)
    """
    # Extract metrics from history
    metrics = [k for k in history.history.keys() if not k.startswith("val_")]
    n_metrics = len(metrics)

    if n_metrics == 0:
        raise ValueError("No training history found in the provided object")

    plt.figure(figsize=(12, 6 * n_metrics))

    # Create subplots for each metric
    for i, metric in enumerate(metrics, 1):
        plt.subplot(n_metrics, 1, i)

        # Plot training values
        plt.plot(history.history[metric], label=f"Training {metric}")

        # Plot validation values if available
        val_metric = f"val_{metric}"
        if val_metric in history.history:
            plt.plot(history.history[val_metric], label=f"Validation {metric}")

        plt.title(f"{metric.capitalize()} Curve")
        plt.ylabel(metric.capitalize())
        plt.xlabel("Epoch")
        plt.legend()

    plt.tight_layout()

    if save:
        if path is None:
            dir_name = Path.cwd()
        else:
            dir_name = Path(path)
        dir_name.mkdir(exist_ok=True, parents=True)
        plt.savefig(f"{dir_name}/training_history.png", dpi=200, format="png")

    if display:
        plt.show()

    plt.close()


def plot_gan_training_history(
    history, save: bool = True, path: Path = None, display: bool = False
):
    """
    Plot training history of GAN.

    Parameters
    ----------
    history : dict
        History of GAN training.
    save : bool, optional
        Whether to save the plot or not. Defaults to True.
    path : str, optional
        Path to save the plot. Defaults to None, which means the plot is saved
        in the current working directory.
    display : bool, optional
        Whether to display the plot or not. Defaults to False.

    """

    history_dict = history.history
    generator_losses = []
    discriminator_loss = None

    # Separate generator losses and discriminator loss
    for key in history_dict.keys():
        if "g_loss" in key:
            generator_losses.append((key, history_dict[key]))
        elif key == "d_loss":
            discriminator_loss = history_dict[key]

    # Plotting
    plt.figure(figsize=(10, 6))

    # Plot generator losses
    for i, (_, gen_loss_values) in enumerate(generator_losses):
        plt.plot(gen_loss_values, label=f"Generator Loss {i}", linewidth=2)

    # Plot discriminator loss
    if discriminator_loss is not None:
        plt.plot(
            discriminator_loss,
            label="Discriminator Loss",
            linewidth=2,
            linestyle="--",
            color="black",
        )

    # Add equilibrium line
    plt.plot(
        [0.5] * len(discriminator_loss),
        label="Discriminator Equilibrium [0.5]",
        linewidth=1,
        linestyle="-",
        color="red",
    )

    # Set y-axis ticks to specific values
    # y_ticks = [0, 0.2, 0.4, 0.45, 0.5, 0.55, 0.6, 0.8, 1, 2, 3] + list(range(4, 26))
    # plt.yticks(y_ticks)

    plt.title("Training Losses")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    if display:
        plt.show()

    if save:
        if path is None:
            dir_name = Path.cwd()
        else:
            dir_name = Path(path)
        dir_name.mkdir(exist_ok=True, parents=True)
        plt.savefig(f"{dir_name}/training_history.png", dpi=200, format="png")


def plot_generators_examples(
    n_rows: int,
    n_cols: int,
    random_latent_vectors: list,
    data: np.ndarray,
    generators: list,
    dir_name: Path,
    epoch: int,
    samples_subfolder: str = "generators_examples",
    save: bool = True,
    show: bool = False,
) -> None:
    """
    Plot a grid of images generated by the generator(s) at the given epoch.
    """
    # Create a figure and a grid of subplots
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(12, 8))
    fig.suptitle(f"Epoch: {epoch}", fontsize=20)
    # Flatten the axes array to iterate over individual subplots
    axes = axes.flatten()

    generator_index = 0
    # Iterate over the subplots
    for ax_index, ax in enumerate(axes):
        # Determine if we're plotting real or generated data
        if (ax_index + 1) % n_cols == 0:
            # Plot real data
            ax.imshow(
                (data[np.random.randint(data.shape[0]), :, :] * 127.5 + 127.5) / 255,
                cmap="gray",
            )
            ax.set_title("Real (random)")
            generator_index = 0
        else:
            # Plot generated data
            generated_sample = generators[generator_index](
                random_latent_vectors[generator_index]
            )
            ax.imshow(
                (
                    generated_sample[
                        ax_index // n_cols,
                        :,
                        :,
                    ]
                    * 127.5
                    + 127.5
                )
                / 255,
                cmap="gray",
            )
            ax.set_title(f"FAKE (Gen {generator_index + 1})")
            generator_index += 1

        # Turn off axis labels for clarity
        ax.axis("off")
    fig.tight_layout()

    if save:
        Path(dir_name / samples_subfolder).mkdir(exist_ok=True, parents=True)
        plt.savefig(
            dir_name / samples_subfolder / f"image_at_epoch_{(epoch + 1):04}.png",
            dpi=200,
            format="png",
        )
        plt.close()
    if show:
        plt.show()


# Function to create a GIF from a list of image paths
def generate_gan_training_gif(image_folder, output_gif, duration=500):
    image_files = sorted(
        [
            os.path.join(image_folder, file)
            for file in os.listdir(image_folder)
            if file.endswith(".png")
        ]
    )

    frames = []
    for image_name in image_files:
        img = Image.open(image_name)
        frames.append(img)

    # Save the frames as an animated GIF
    frames[0].save(
        output_gif,
        format="GIF",
        append_images=frames[1:],
        save_all=True,
        duration=duration,
        loop=0,
    )
