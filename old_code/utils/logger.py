import logging
from logging.handlers import RotatingFileHandler

class SingletonLogger:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(SingletonLogger, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialize_logger()
        return cls._instance

    def _initialize_logger(self):
        # Create the logger
        self.logger = logging.getLogger('PROJECT_LOGGER')
        self.logger.setLevel(logging.DEBUG)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)

        # File handler with rotation
        # file_handler = RotatingFileHandler('app.log', maxBytes=5*1024*1024, backupCount=5)
        # file_handler.setLevel(logging.INFO)

        # Define formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Attach formatter to handlers
        console_handler.setFormatter(formatter)
        # file_handler.setFormatter(formatter)

        # Attach handlers to logger
        self.logger.addHandler(console_handler)
        # self.logger.addHandler(file_handler)

    def get_logger(self):
        return self.logger

# Singleton access method
def get_logger():
    return SingletonLogger().get_logger()
