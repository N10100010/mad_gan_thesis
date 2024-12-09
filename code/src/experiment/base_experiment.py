# experiments/base_experiment.py
from abc import ABC, abstractmethod

class BaseExperiment(ABC):
    """
    Abstract base class for running experiments.

    Provides a basic structure for defining how an experiment should be run and
    how the results should be saved.
    """

    def run(self):
        self._check_gpu()
        self._load_data()
        self._initialize_models()
        self._run()
        self._save_results()
        print("Experiment completed.")

    @abstractmethod
    def _run(self):
        """
        Define the main logic for running the experiment.

        This method should contain the code that is executed when the experiment
        is run.
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
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        if tf.test.gpu_device_name() == '/device:GPU:0':
            print("Using a GPU")
        else:
            print("Using a CPU")
            