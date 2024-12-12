# Set environment variable to reduce TF logging to warning level
import os
from abc import ABC, ABCMeta, abstractmethod

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from abc import ABC, abstractmethod
from utils import setup_logger

class AutoSuperMeta(ABCMeta):
    def __new__(cls, name, bases, dct):
        for attr_name, attr_value in dct.items():
            if callable(attr_value) and attr_name not in ('__init__',):
                original_method = attr_value

                # Use a closure to capture the method and attribute name
                def make_wrapper(original_method, attr_name):
                    def wrapper(self, *args, **kwargs):
                        # Call the base class method if it exists
                        for base in bases:
                            if hasattr(base, attr_name):
                                getattr(super(type(self), self), attr_name)(*args, **kwargs)
                        # Call the current method
                        return original_method(self, *args, **kwargs)
                    return wrapper

                dct[attr_name] = make_wrapper(original_method, attr_name)
        return super().__new__(cls, name, bases, dct)

class BaseExperiment(ABC, metaclass=AutoSuperMeta):
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
        self.logger.info(f"################# Running experiment {self.name}...")
        self._check_gpu()
        self._load_data()
        self._initialize_models()
        self._run()
        self._save_results()
        self.logger.info(f"################# Experiment {self.name} completed.")

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
        self.logger.debug("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        if tf.test.gpu_device_name() == '/device:GPU:0':
            self.logger.info("################# Using a GPU")
        else:
            self.logger.info("################# Using a CPU")
            