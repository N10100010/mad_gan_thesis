import json
from pathlib import Path

import numpy as np
from experiment.base_experiments.base_experiment import BaseExperiment
from model_definitions.mad_gan.mad_gan import MADGAN
from utils.plotting import generate_gan_training_gif, plot_gan_training_history


class BaseMADGANExperiment(BaseExperiment):
    n_gen: int = None
    latent_dim: int = None
    size_dataset: int = None
    batch_size: int = None
    epochs: int = None
    steps_per_epoch: int = None
    generate_after_epochs: int = None
    generator_training_samples_subfolder: str = None
    madgan: MADGAN = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for k, v in kwargs.items():
            setattr(self, k, v)

    def _save_results(self):
        model_weights_path = Path(self.dir_path, "final_model.weights.h5")
        self.madgan.save_weights(model_weights_path)
        self.logger.info(f"Model saved to: {model_weights_path}")

        history_path = Path(self.dir_path, "training_history.npy")
        np.save(history_path, self.history.history)
        self.logger.info(f"Training history saved to: {history_path}")

        plot_gan_training_history(
            history=self.history,
            path=self.dir_path,
        )

    def _final(self):
        generate_gan_training_gif(
            image_folder=self.dir_path / self.generator_training_samples_subfolder,
            output_gif=Path(self.dir_path, "generator_training.gif"),
            duration=300,
        )

    @classmethod
    def load_from_path(cls, path: Path):
        try:
            json_path = Path(path, "metadata.json")
            with open(json_path, "r") as f:
                metadata = json.load(f)
        except Exception as e:
            raise Exception(f"Error loading from metadata file {json_path}: {e}")
        return cls(**metadata)

    def load_model_weights(self, checkpoint: int = None):
        self._initialize_models()
        self.madgan.built = True
        if isinstance(self.dir_path, str):
            self.dir_path = Path(self.dir_path)

        if checkpoint is not None:
            file_path = (
                self.dir_path / "checkpoints" / f"backup_epoch_{checkpoint}.weights.h5"
            )
        else:
            file_path = self.dir_path / "final_model.weights.h5"
        if not file_path.exists():
            raise Exception(f"Model weights not found at {file_path}")

        self.madgan.load_weights(file_path)
