from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class ConditionalGANMonitor(tf.keras.callbacks.Callback):
    def __init__(
        self,
        random_latent_vectors: list,
        data: np.ndarray,
        n_classes: int,
        latent_dim: int = 128,
        dir_name: Path = "Model",
        samples_subfolder: str = "generators_examples",
        generator_example_freq: int = 10,
        save: bool = True,
    ):
        super().__init__()

        self.data = data
        self.random_latent_vectors = random_latent_vectors
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.dir_name = dir_name
        self.samples_subfolder = samples_subfolder
        self.generator_example_freq = generator_example_freq
        self.save = save

    def create_generator_images(self, epoch: int) -> None:
        fig, axes = plt.subplots(nrows=10, ncols=10, figsize=(10, 10))
        fig.suptitle(f"Epoch: {epoch}", fontsize=20)

        # Generate fake images (9 per class)
        labels = tf.reshape(tf.repeat(tf.range(10), repeats=9), (-1, 1))
        fake_images = self.model.generator(
            [self.random_latent_vectors, labels], training=False
        )

        # Ensure fake images are in numpy format
        fake_images = fake_images.numpy()

        # Extract one real image per class from dataset
        real_images_dict = {}
        for batch_nr, (imgs, lbls) in enumerate(self.data):
            for i in range(len(imgs)):
                lbl = int(lbls[i].numpy())
                if lbl not in real_images_dict:
                    real_images_dict[lbl] = imgs[i].numpy()
                    break

            if len(real_images_dict) == 10:
                break  # Stop when we have one real image per class

        # Convert dict to sorted list (matching row indices 0-9)
        real_images = np.array([real_images_dict[i] for i in range(10)])
        fake_images = (fake_images * 127.5 + 127.5).astype(np.uint8)
        real_images = (real_images * 127.5 + 127.5).astype(np.uint8)

        # Determine if images are grayscale or color
        cmap = "gray" if fake_images.shape[-1] == 1 else None

        # Plot images
        for row in range(10):
            axes[row, 0].set_ylabel(f"Class {row}", fontsize=12, labelpad=10)
            for col in range(9):
                axes[row, col].imshow(
                    np.clip(fake_images[row * 9 + col], 0, 255), cmap=cmap
                )
                axes[row, col].axis("off")

            # Add real image in the last column
            axes[row, 9].imshow(np.clip(real_images[row], 0, 255), cmap=cmap)
            axes[row, 9].axis("off")

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)  # Adjust to fit the title
        # show = True
        # if show:
        #     plt.show()

        if self.save:
            Path(self.dir_name / self.samples_subfolder).mkdir(
                exist_ok=True, parents=True
            )
            plt.savefig(
                self.dir_name
                / self.samples_subfolder
                / f"image_at_epoch_{(epoch + 1):04}.png",
                dpi=200,
                format="png",
            )
            plt.close()

    def on_epoch_end(self, epoch: int, logs: dict = None) -> None:
        if epoch % self.generator_example_freq == 0:
            self.create_generator_images(epoch)
