import json
import os
import traceback
import warnings
from abc import ABC, ABCMeta, abstractmethod
from datetime import datetime
from pathlib import Path

import tensorflow as tf
from utils import setup_logger

# Suppress TensorFlow INFO and WARNING messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Suppress specific warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*scale_identity_multiplier.*")


class AutoSuperMeta(ABCMeta):
    """
    Wrapper to automatically add super() calls to methods. As of now, this is solely for ease of logging.
    However, this functionality can be used for greater benefits.

    Scenario: Writing into a directory
        The base method could ensure, that the directory exists before writing into it.
        This could be achived by calling super().write(),
        BUT it would require to explicitly mention the super call.

    Adjusting the super-classes __new__ method and explicitly calling the super method
    AND then calling the sub-classes method achieves this behaviour automatically.

    QUESTION: Indirections could be an interesting point here, that may loose some performance.
    """

    def __new__(cls, name, bases, dct):
        for attr_name, attr_value in dct.items():
            if callable(attr_value) and attr_name not in ("__init__",):
                original_method = attr_value

                # Use a closure to wrap methods
                def make_wrapper(original_method, attr_name):
                    def wrapper(self, *args, **kwargs):
                        # Avoid recursive wrapping
                        for base in bases:
                            base_method = getattr(base, attr_name, None)
                            if base_method:
                                base_method(self, *args, **kwargs)
                        # Call the current method
                        return original_method(self, *args, **kwargs)

                    return wrapper

                # Replace the method with the wrapped version
                dct[attr_name] = make_wrapper(original_method, attr_name)
        return super().__new__(cls, name, bases, dct)


class BaseExperiment(ABC, metaclass=AutoSuperMeta):
    """
    Abstract base class for running experiments.

    Provides a basic structure for defining how an experiment should be run and
    how the results should be saved, etc.
    """

    whitelist_attributes = [
        "n_images",
        "save",
        "results_json_file_path",
        "latent_dim",
        "epochs",
        "image_data_shape",
        "experiments_base_path",
        "experiment_start_time",
        "experiment_end_time",
        "experiment_name",
        "classifier",
        "n_gen",
        "size_dataset",
        "generator_training_samples_subfolder",
        "batch_size",
        "steps_per_epoch",
        "experiment_duration",
        "dir_path",
        "name",
        "meta_json_filename",
        "save_raw_image",
        "created_images_folder_path",
        "experiment_suffix",
        "classifier_class",
        "model_path",
        "use_generator",
        "labels_json_file_name",
        "unique_labels",
    ]


    def __init__(
        self,
        name: str,
        experiments_base_path: str = "./experiments",
        experiment_suffix: str = "",
        **kwargs,
    ):
        self.name: str = name
        self.experiments_base_path: str = experiments_base_path
        self.experiment_suffix: str = experiment_suffix
        self.experiment_start_time: datetime = None
        self.experiment_end_time: datetime = None
        self.experiment_duration: datetime = None

        self.meta_json_filename: str = "metadata.json"

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.logger = setup_logger(name=name)
        self._setup()

        self.dir_path: Path = None

    def run(self):
        """
        Run the experiment.

        This method runs the experiment from start to finish. It will create a
        directory for the experiment, check if a GPU is available, load the
        data, initialize the models, run the experiment, and save the results.

        If an error occurs during the experiment, it will be logged and the
        experiment will exit.

        :raises Exception: If an error occurs during the experiment.
        """
        try:
            self.logger.info(f"################# Running experiment {self.name}...")
            self.dir_path: Path = self._create_experiment_directory()
            self._check_gpu()
            self._load_data()
            self._initialize_models()
            ## TODO: could nest _save_results with _run, to pass history, etc OR
            ## set self.history in the _run method's implementation
            self._run()
            self._save_results()
            self._save_metadata_file()
            self._final()
            self.logger.info(f"################# Experiment {self.name} completed.")
        except Exception as e:
            self.logger.error(f"Error running experiment {self.name}: {e}")
            self.logger.error(traceback.format_exc())

    @abstractmethod
    def _run(self):
        """
        Define the main logic for running the experiment.

        This method should contain the code that is executed when the experiment
        is run.
        """
        self.experiment_start_time = datetime.now()

        self.logger.info("################# Running")
        self.logger.info(
            f"################# Experiment Start Time: {self.experiment_start_time}"
        )

        pass

    @abstractmethod
    def _setup(self):
        """
        Define how the experiment should be set up.

        This method should be overridden by subclasses to define how the
        experiment should be set up before running the experiment.
        """
        self.logger.info("################# Setup")
        pass

    @abstractmethod
    def _save_results(self):
        """
        Define how results should be saved.

        This method should be overridden by subclasses to define how the results
        of the experiment should be saved. This may include saving model weights,
        training history, results of statistical tests, etc.
        """

        self.experiment_end_time = datetime.now()
        self.experiment_duration = self.experiment_end_time - self.experiment_start_time
        self.logger.info("################# Saving results")
        self.logger.info(
            f"################# Experiment End Time: {self.experiment_end_time}"
        )

        self.logger.info(f"################# Duration: {self.experiment_duration}")

        pass

    @abstractmethod
    def _load_data(self):
        """
        Define how data should be loaded.

        This method should be overridden by subclasses to define how the data
        should be loaded for the experiment.
        """
        self.logger.info("################# Loading data")
        pass

    @abstractmethod
    def _initialize_models(self):
        """
        Define how models should be initialized.

        This method should be overridden by subclasses to define how the models
        should be initialized for the experiment.
        """
        self.logger.info("################# Initializing Models")
        pass

    def _check_gpu(self):
        """
        Check if a GPU is available and log whether a GPU or CPU is being used.
        """
        self.logger.debug(
            "Num GPUs Available: ", len(tf.config.list_physical_devices("GPU"))
        )
        if tf.test.gpu_device_name() == "/device:GPU:0":
            self.logger.info("################# Using a GPU")
        else:
            self.logger.info("################# Using a CPU")

    def _create_experiment_directory(self) -> Path:
        """
        Creates a directory for the experiment based on the current date, experiment name,
        and an optional experiment suffix.

        The directory is created under the specified base path for experiments. If the directory
        already exists, it will not be recreated. A log entry is created to indicate the directory
        path.

        Returns:
            Path: The path to the created experiment directory.
        """

        current_date = datetime.now().strftime("%Y-%m-%d")
        dir_name = (
            Path(self.experiments_base_path)
            / f"{current_date}_{self.name}{'_' + self.experiment_suffix if self.experiment_suffix else ''}"
        )
        dir_name.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"################# Experiment directory: {dir_name}")
        return dir_name

    @classmethod
    def load_from_path(cls, path: Path):
        try:
            json_path = Path(path, "metadata.json")
            with open(json_path, "r") as f:
                metadata = json.load(f)
        except Exception as e:
            raise Exception(f"Error loading from metadata file {json_path}: {e}")
        return cls(**metadata)

    def _save_metadata_file(self):
        """
        Creates a metadata file (`metadata.json`) for the experiment,
        saving all relevant attributes of the implementing class to a JSON file.

        The file is created in the experiment's directory.
        """

        if not self.dir_path:
            raise ValueError(
                "Experiment directory is not set. Run `_create_experiment_directory` first."
            )

        metadata = {
            "experiment_name": self.name,
            "experiment_end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        def is_serializable(value):
            """
            Determines if a value is serializable to JSON.
            Returns True for basic types (int, float, str, bool, None).
            """
            return isinstance(value, (int, float, str, bool, type(None)))

        # Combine instance and class attributes
        for cls in self.__class__.__mro__:  # Traverse the class hierarchy
            for attr_name, attr_value in cls.__dict__.items():
                if (
                    not attr_name.startswith("_")
                    and attr_name in self.whitelist_attributes
                    and not callable(attr_value)
                ):
                    # Use the actual attribute value from the instance
                    attr = getattr(self, attr_name, None)
                    metadata[attr_name] = attr if is_serializable(attr) else str(attr)

        # Add instance-specific attributes
        for attr_name, attr_value in vars(self).items():
            if not attr_name.startswith("_") and attr_name in self.whitelist_attributes:
                metadata[attr_name] = (
                    attr_value if is_serializable(attr_value) else str(attr_value)
                )

        # Write metadata to a file
        metadata_file = self.dir_path / self.meta_json_filename
        with metadata_file.open("w") as f:
            json.dump(metadata, f, indent=4, sort_keys=True)

        self.logger.info(f"################# Metadata saved to {metadata_file}")
        return metadata_file

    def _final(self):
        """
        Define any final operations that should be performed after the experiment is completed.
        """
        pass
