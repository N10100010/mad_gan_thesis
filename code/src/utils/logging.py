import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)


class LoggerSingleton:
    _instances = {}

    def __new__(cls, name, *args, **kwargs):
        if name not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[name] = instance
            instance._initialize(name, *args, **kwargs)
        return cls._instances[name]

    def _initialize(self, name, log_file="app.log", level=logging.INFO):
        self.logger = logging.getLogger(name)
        if not self.logger.hasHandlers():  # Prevent duplicate handlers
            self.logger.setLevel(level)

            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_handler.setFormatter(
                logging.Formatter(f"%(asctime)s - {name} - %(levelname)s - %(message)s")
            )
            self.logger.addHandler(console_handler)

            # NOT MANDATORY FOR NOW: Rotating file handler
            # file_handler = RotatingFileHandler(log_file, maxBytes=10**6, backupCount=3)
            # file_handler.setLevel(level)
            # file_handler.setFormatter(logging.Formatter(f"%(asctime)s - {name} - %(levelname)s - %(message)s"))
            # self.logger.addHandler(file_handler)


# Helper function
def setup_logger(name="App", log_file="app.log", level=logging.INFO):
    return LoggerSingleton(name, log_file=log_file, level=level).logger
