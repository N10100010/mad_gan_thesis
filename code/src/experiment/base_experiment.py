import tensorflow as tf

from abc import ABC, abstractmethod
from utils import setup_logger

class BaseExperiment(ABC):
    """
    Abstract base class for running experiments.

    Provides a basic structure for defining how an experiment should be run and
    how the results should be saved.
    """
    
    def __init__(self, name: str):
        logger = setup_logger(name = "")
        self._setup()

    def run(self):
        logger.info(f"Running experiment {self.name}...")
        self._check_gpu()
        self._load_data()
        self._initialize_models()
        self._run()
        self._save_results()
        logger.info(f"Experiment {self.name} completed.")

    @abstractmethod
    def _run(self):
        """
        Define the main logic for running the experiment.

        This method should contain the code that is executed when the experiment
        is run.
        """
        pass
    
    @abstractmethod
    def _setup(self):
        """
        Define how the experiment should be set up.

        This method should be overridden by subclasses to define how the
        experiment should be set up before running the experiment.
        """
        pass
        
        
    @abstractmethod
    def _save_results(self):
        """
        Define how results should be saved.

        This method should be overridden by subclasses to define how the results
        of the experiment should be saved. This may include saving model weights,
        training history, results of statistical tests, etc.
        """
        pass
    
    def _load_data(self):
        """
        Define how data should be loaded.
        
        This method should be overridden by subclasses to define how the data
        should be loaded for the experiment.
        """
        pass
        
    def _initialize_models(self):
        """
        Define how models should be initialized.
        
        This method should be overridden by subclasses to define how the models
        should be initialized for the experiment.
        """
        pass



    def _check_gpu(self):
        logger.debug("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        if tf.test.gpu_device_name() == '/device:GPU:0':
            logger.info("Using a GPU")
        else:
            logger.info("Using a CPU")
            