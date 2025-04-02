import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from model_definitions.classifiers.base import BaseClassifier
from scoring_metrics import calculate_fid_score, calculate_inception_score
from utils.logging import setup_logger


class ScoreCMADGANMonitor(tf.keras.callbacks.Callback):
    logger = setup_logger(name="ScoreCMADGANMonitor", prefix="\n MONITOR: ")

    def __init__(
        self,
        dir_name: str,
        latent_dim: int,
        dataset: str,  # Should match BaseClassifier constants e.g., BaseClassifier.CIFAR10
        total_epochs: int,
        num_samples_for_scoring: int = 5000,  # Standard is often higher (e.g., 10k or 50k for FID)
        score_calculation_freq: int = 5,  # Calculate less frequently by default
    ):
        super().__init__()

        self.dir_name = Path(dir_name)
        self.latent_dim = latent_dim
        self.dataset = dataset
        self.total_epochs = total_epochs
        self.score_calculation_freq = score_calculation_freq
        self.num_samples_for_scoring = num_samples_for_scoring

        # Determine image shape based on dataset name
        if self.dataset == BaseClassifier.CIFAR10:
            self.image_data_shape = (32, 32, 3)
            # self.n_classes_assumed = 10 # Set based on dataset if not directly available from model
        elif self.dataset == BaseClassifier.MNIST:
            self.image_data_shape = (28, 28, 1)
            # self.n_classes_assumed = 10
        else:
            # Default or raise error if dataset unknown
            self.logger.warning(
                f"Unknown dataset '{self.dataset}'. Assuming MNIST shape/classes."
            )
            self.image_data_shape = (28, 28, 1)
            # self.n_classes_assumed = 10
            # raise ValueError(f"Dataset '{self.dataset}' not recognized for image shape determination.")

        # Setup directories and files
        (self.dir_name / "scores").mkdir(parents=True, exist_ok=True)
        self.scores_folder = self.dir_name / "scores"
        self.scores_file = self.scores_folder / "metrics.json"

        # Delete the file for restarts to avoid appending to old runs
        if self.scores_file.exists():
            self.logger.warning(f"Deleting existing scores file: {self.scores_file}")
            try:
                self.scores_file.unlink()
            except OSError as e:
                self.logger.error(f"Error deleting scores file: {e}")

        # Check model compatibility later in on_epoch_end when self.model is set

    def on_epoch_end(self, epoch: int, logs: dict = None) -> None:
        # Check if the model structure is compatible (has generators and condition_dim)
        if not hasattr(self.model, "generators") or not isinstance(
            self.model.generators, list
        ):
            self.logger.error(
                "Model does not have a 'generators' list attribute. Cannot run Monitor."
            )
            return
        if not hasattr(self.model, "condition_dim"):
            self.logger.error(
                "Model does not have a 'condition_dim' attribute. Cannot run Monitor."
            )
            return
            # Alternatively, use self.n_classes_assumed set in __init__

        n_classes = self.model.condition_dim  # Get number of classes from the model

        # Calculate scores every `score_calculation_freq` epochs OR on the very last epoch
        if (epoch + 1) % self.score_calculation_freq != 0 and (
            epoch + 1
        ) != self.total_epochs:
            self.logger.info(
                f"Epoch {epoch+1}/{self.total_epochs}: Skipping score calculation."
            )
            return

        self.logger.info(f"Epoch {epoch+1}/{self.total_epochs}: Calculating scores...")

        # --- Generate Latent Vectors and Labels (Batch) ---
        samples_per_class = (
            self.num_samples_for_scoring // n_classes
        )  # Integer division
        actual_num_samples = samples_per_class * n_classes  # Ensure perfect balance

        if actual_num_samples != self.num_samples_for_scoring:
            self.logger.warning(
                f"Adjusted number of samples for scoring from {self.num_samples_for_scoring} "
                f"to {actual_num_samples} for balanced classes."
            )
        if actual_num_samples == 0:
            self.logger.error(
                f"Cannot generate 0 samples (num_samples_for_scoring too low for {n_classes} classes). Skipping."
            )
            return

        random_latent_vectors = tf.random.normal(
            shape=(actual_num_samples, self.latent_dim)
        )

        # Generate balanced labels across classes
        uniform_labels_int = np.repeat(np.arange(n_classes), samples_per_class)
        # Shuffle labels to avoid potential sequential bias if generator is sensitive
        np.random.shuffle(uniform_labels_int)
        uniform_labels = tf.convert_to_tensor(uniform_labels_int, dtype=tf.int32)
        one_hot_labels = tf.one_hot(uniform_labels, depth=n_classes)

        scores_by_gen = {}
        # --- Iterate Through Generators ---
        for gen_nr, generator in enumerate(self.model.generators):
            self.logger.info(
                f"  Generating samples and scoring for Generator {gen_nr}..."
            )

            try:
                # Generate samples in one batch call
                # Assumes generator takes a list [noise, condition] as input
                generated_samples_tensor = generator(
                    [random_latent_vectors, one_hot_labels], training=False
                )
                generated_samples = generated_samples_tensor.numpy()

                # --- Post-processing ---
                # Clip to expected range (e.g., [-1, 1] for tanh output)
                generated_samples = np.clip(generated_samples, -1.0, 1.0)

                # Ensure correct shape (N, H, W, C)
                expected_shape = (actual_num_samples,) + self.image_data_shape
                if generated_samples.shape != expected_shape:
                    self.logger.warning(
                        f"Gen {gen_nr}: Output shape {generated_samples.shape} != expected {expected_shape}. Attempting reshape."
                    )
                    try:
                        generated_samples = generated_samples.reshape(expected_shape)
                    except ValueError as e:
                        self.logger.error(
                            f"Gen {gen_nr}: Reshape failed ({e}). Skipping scoring."
                        )
                        continue  # Skip this generator

                # --- De-normalize and Prepare for Scoring ---
                # Convert [-1, 1] to [0, 1] (common input range for Inception models)
                generated_samples_01 = (generated_samples + 1.0) / 2.0

                # Repeat grayscale channel if necessary (Inception V3 expects 3 channels)
                if generated_samples_01.shape[-1] == 1:
                    images_for_scoring = np.repeat(generated_samples_01, 3, axis=-1)
                    self.logger.info(
                        f"  Gen {gen_nr}: Repeated grayscale channel to 3 channels."
                    )
                elif generated_samples_01.shape[-1] == 3:
                    images_for_scoring = generated_samples_01
                else:
                    self.logger.error(
                        f"Gen {gen_nr}: Generated samples have unexpected channel dimension: {generated_samples_01.shape[-1]}. Skipping scoring."
                    )
                    continue  # Skip this generator

                # Optional: Convert to uint8 [0, 255] if scoring functions strictly require it
                # images_for_scoring_uint8 = (images_for_scoring * 255).astype(np.uint8)
                # Use images_for_scoring_uint8 below if needed

                # --- Calculate Scores ---
                self.logger.info(f"  Gen {gen_nr}: Calculating FID...")
                # Make sure calculate_fid_score is correctly implemented
                # - Takes images in expected range ([0,1] or [0,255])
                # - Knows how to find/use pre-calculated real data stats for the specified `dataset`
                fid_score = calculate_fid_score(
                    generated_images=images_for_scoring, dataset=self.dataset
                )

                self.logger.info(f"  Gen {gen_nr}: Calculating IS...")
                # Make sure calculate_inception_score takes images in expected range
                is_score, is_std = calculate_inception_score(
                    generated_images=images_for_scoring
                )

                # Store scores
                fid_score_float = float(
                    fid_score
                )  # Ensure it's a standard float for JSON
                is_score_float = float(is_score)
                is_std_float = float(is_std)

                self.logger.info(
                    f"Epoch {epoch+1} - Generator {gen_nr}: FID: {fid_score_float:.4f}, IS: {is_score_float:.4f} +/- {is_std_float:.4f}"
                )
                scores_by_gen[gen_nr] = {
                    "epoch": epoch + 1,  # Store epoch number (1-based for clarity)
                    "FID": fid_score_float,
                    "IS": is_score_float,
                    "IS_std": is_std_float,
                }

            except Exception as e:
                self.logger.error(
                    f"Error during scoring for Generator {gen_nr} at epoch {epoch+1}: {e}",
                    exc_info=True,
                )
                # Continue to the next generator if one fails

        # --- Save Scores to JSON ---
        if not scores_by_gen:  # Handle case where no scores were calculated
            self.logger.warning(
                f"No scores were calculated for epoch {epoch+1}. Skipping JSON update."
            )
            return

        if self.scores_file.exists():
            try:
                with open(self.scores_file, "r") as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                self.logger.warning(
                    f"Could not decode existing scores file {self.scores_file}. Starting fresh."
                )
                data = {}
            except Exception as e:
                self.logger.error(
                    f"Error reading scores file {self.scores_file}: {e}. Starting fresh.",
                    exc_info=True,
                )
                data = {}
        else:
            data = {}

        # Use string for epoch key in JSON, store 1-based epoch
        data[str(epoch + 1)] = scores_by_gen

        try:
            with open(self.scores_file, "w") as f:
                json.dump(data, f, indent=4)  # Standard floats should be fine now
        except Exception as e:
            self.logger.error(
                f"Error writing scores to {self.scores_file}: {e}", exc_info=True
            )

        # --- Plot Scores at the End ---
        if epoch + 1 == self.total_epochs:
            self.logger.info("Plotting final scores...")
            try:
                self.plot_scores(data)
            except Exception as e:
                self.logger.error(f"Error plotting scores: {e}", exc_info=True)

    def plot_scores(self, data):
        """Plots FID and IS scores loaded from the results data dictionary."""
        # Data keys are string epochs, values are dicts {gen_nr: scores_dict}
        try:
            # Convert string epoch keys to integers for sorting
            epochs = sorted([int(e) for e in data.keys()])
            if not epochs:
                self.logger.warning("No epochs found in score data for plotting.")
                return

            # Determine number of generators from the first epoch's data
            first_epoch_data = data[str(epochs[0])]
            num_generators = len(first_epoch_data)
            if num_generators == 0:
                self.logger.warning("No generator data found for plotting.")
                return

            # --- FID Scores Plot (Single Figure) ---
            plt.figure(figsize=(12, 6))
            for gen in range(num_generators):
                # Ensure data exists for this generator across all logged epochs
                fid_scores = [
                    data[str(epoch)].get(gen, {}).get("FID", np.nan) for epoch in epochs
                ]
                plt.plot(
                    epochs,
                    fid_scores,
                    marker=".",
                    linestyle="-",
                    label=f"Generator {gen}",
                )

            plt.xlabel("Epochs")
            plt.ylabel("FID Score (Lower is Better)")
            plt.title(f"FID Scores Over Epochs ({self.dataset.upper()})")
            plt.legend(loc="upper right")
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.xticks(epochs)  # Ensure all logged epochs are marked

            fid_save_path = self.scores_folder / f"{self.dataset}_FID.png"
            plt.savefig(fid_save_path)
            self.logger.info(f"Saved FID plot to: {fid_save_path}")
            plt.close()  # Close the figure

            # --- IS Scores Plot (Subplots for each Generator) ---
            fig, axes = plt.subplots(
                num_generators,
                1,
                figsize=(12, 4 * num_generators),
                sharex=True,
                squeeze=False,
            )
            axes = axes.flatten()  # Ensure axes is always flat array

            for gen in range(num_generators):
                is_scores = np.array(
                    [
                        data[str(epoch)].get(gen, {}).get("IS", np.nan)
                        for epoch in epochs
                    ]
                )
                is_stds = np.array(
                    [
                        data[str(epoch)].get(gen, {}).get("IS_std", np.nan)
                        for epoch in epochs
                    ]
                )

                valid_idx = ~np.isnan(is_scores)  # Plot only where data exists
                epochs_valid = np.array(epochs)[valid_idx]
                is_scores_valid = is_scores[valid_idx]
                is_stds_valid = is_stds[valid_idx]

                axes[gen].plot(
                    epochs_valid,
                    is_scores_valid,
                    marker=".",
                    linestyle="-",
                    label="IS Mean",
                    color=f"C{gen}",
                )
                axes[gen].fill_between(
                    epochs_valid,
                    is_scores_valid - is_stds_valid,
                    is_scores_valid + is_stds_valid,
                    alpha=0.2,
                    color=f"C{gen}",
                    label="IS +/- Std Dev",
                )

                axes[gen].set_ylabel("IS Score (Higher is Better)")
                axes[gen].set_title(f"Inception Score - Generator {gen}")
                axes[gen].legend(loc="lower right")
                axes[gen].grid(True, linestyle="--", alpha=0.6)

            axes[-1].set_xlabel("Epochs")
            plt.xticks(epochs)  # Ensure all logged epochs are marked on x-axis
            fig.suptitle(
                f"Inception Scores Over Epochs ({self.dataset.upper()})",
                fontsize=16,
                y=1.02,
            )
            plt.tight_layout(
                rect=[0, 0.03, 1, 0.98]
            )  # Adjust layout to prevent title overlap

            is_save_path = self.scores_folder / f"{self.dataset}_IS.png"
            plt.savefig(is_save_path)
            self.logger.info(f"Saved IS plot to: {is_save_path}")
            plt.close(fig)  # Close the figure

        except Exception as e:
            self.logger.error(f"Failed to plot scores: {e}", exc_info=True)
