import gc
from queue import Queue

import tensorflow as tf
from utils import setup_logger


class ExperimentQueue:
    def __init__(self):
        self.logger = setup_logger(name="Queue")
        self.queue = Queue()

    def add_experiment(self, experiment):
        self.logger.info(f"### ADDED EXPERIMENT {experiment.name}...")
        self.queue.put(experiment)

    def run_all(self):
        self.logger.info("### RUNNING ALL EXPERIMENTS...")

        self.successful_experiments = []
        self.failed_experiments = []

        while not self.queue.empty():
            experiment = self.queue.get()
            self.logger.info(f"### RUNNING {experiment.name}...")

            try:
                experiment.run()
                self.logger.info(f"### COMPLETED {experiment.name}.")
                self.successful_experiments.append(experiment.name)
            except Exception as e:
                self.logger.error(f"### ERROR in {experiment.name}: {str(e)}")
                self.failed_experiments.append(experiment.name)
            finally:
                # Memory management
                self.logger.info("### CLEARING SESSION AND COLLECTING GARBAGE...")
                tf.keras.backend.clear_session()  # Clear TensorFlow session
                gc.collect()  # Run garbage collector

            self.queue.task_done()

        if self.failed_experiments:
            self.logger.warning(
                f"### {len(self.failed_experiments)} EXPERIMENTS FAILED: {self.failed_experiments}"
            )
        else:
            self.logger.info(
                f"### ALL {len(self.successful_experiments)} EXPERIMENTS COMPLETED."
            )
