# Set environment variable to reduce TF logging to warning level
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from abc import ABC, abstractmethod
from utils import setup_logger

def call_super(func):
    def wrapper(self, *args, **kwargs):
        # Call the base class method
        base_method = getattr(super(type(self), self), func.__name__, None)
        if base_method:
            base_method(*args, **kwargs)
        # Call the overridden method
        return func(self, *args, **kwargs)
    return wrapper

class BaseExperiment(ABC):
    """
    Abstract base class for running experiments.

    Provides a basic structure for defining how an experiment should be run and
    how the results should be saved.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.logger = setup_logger(name = name)
        self._setup()

    def run(self):
        self.logger.info(f"Running experiment {self.name}...")
        self._check_gpu()
        self._load_data()
        self._initialize_models()
        self._run()
        self._save_results()
        self.logger.info(f"Experiment {self.name} completed.")

    @abstractmethod
    def _run(self):
        """
        Define the main logic for running the experiment.

        This method should contain the code that is executed when the experiment
        is run.
        """
        self.logger.info("Running")
        pass
    
    @abstractmethod
    def _setup(self):
        """
        Define how the experiment should be set up.

        This method should be overridden by subclasses to define how the
        experiment should be set up before running the experiment.
        """
        self.logger.info("Setup")
        pass
        
        
    @abstractmethod
    def _save_results(self):
        """
        Define how results should be saved.

        This method should be overridden by subclasses to define how the results
        of the experiment should be saved. This may include saving model weights,
        training history, results of statistical tests, etc.
        """
        self.logger.info("Saving results")
        pass
    
    @abstractmethod
    def _load_data(self):
        """
        Define how data should be loaded.
        
        This method should be overridden by subclasses to define how the data
        should be loaded for the experiment.
        """
        self.logger.info("Loading data")
        pass
    
    @abstractmethod
    def _initialize_models(self):
        """
        Define how models should be initialized.
        
        This method should be overridden by subclasses to define how the models
        should be initialized for the experiment.
        """
        self.logger.info("INitializing Models")
        pass



    def _check_gpu(self):
        self.logger.debug("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        if tf.test.gpu_device_name() == '/device:GPU:0':
            self.logger.info("Using a GPU")
        else:
            self.logger.info("Using a CPU")
            