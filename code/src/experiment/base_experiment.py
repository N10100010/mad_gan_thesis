import os
from abc import ABC, ABCMeta, abstractmethod
from datetime import datetime
from pathlib import Path

import tensorflow as tf
from utils import setup_logger

# Set environment variable to reduce TF logging to warning level
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


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

                # Use a closure to capture the method and attribute name
                def make_wrapper(original_method, attr_name):
                    def wrapper(self, *args, **kwargs):
                        # Call the base class method if it exists
                        for base in bases:
                            if hasattr(base, attr_name):
                                getattr(super(type(self), self), attr_name)(
                                    *args, **kwargs
                                )
                        # Call the current method
                        return original_method(self, *args, **kwargs)

                    return wrapper

                dct[attr_name] = make_wrapper(original_method, attr_name)
        return super().__new__(cls, name, bases, dct)


class BaseExperiment(ABC, metaclass=AutoSuperMeta):
    """
    Abstract base class for running experiments.

    Provides a basic structure for defining how an experiment should be run and
    how the results should be saved, etc.
    """

    def __init__(
        self,
        name: str,
        experiments_base_path: str = "./experiments",
        experiment_suffix: str = "",
    ):
        self.name: str = name
        self.experiments_base_path: str = experiments_base_path
        self.experiment_suffix: str = experiment_suffix
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
            self.dir_path = self._create_experiment_directory()
            self._check_gpu()
            self._load_data()
            self._initialize_models()
            ## TODO: could nest _save_results with _run, to pass history, etc OR
            ## set self.history in the _run method's implementation
            self._run()
            self._save_results()
            self.logger.info(f"################# Experiment {self.name} completed.")
        except Exception as e:
            self.logger.error(f"Error running experiment {self.name}: {e}")

    @abstractmethod
    def _run(self):
        """
        Define the main logic for running the experiment.

        This method should contain the code that is executed when the experiment
        is run.
        """
        self.logger.info("################# Running")
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
        self.logger.info("################# Saving results")
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

    def _create_metadata_file(self):
        """
        Creates a metadata file for the experiment based on the current date, experiment name,
        and an optional experiment suffix.

        The metadata file is created under the specified base path for experiments. If the file
        already exists, it will not be overwritten. A log entry is created to indicate the file
        path.

        Returns:
            Path: The path to the created metadata file.
        """
        # self.meta
        pass
