from .logger import get_logger
import tensorflow as tf 

logger = get_logger()


def gpu_detection() -> list: 
    """_summary_

    Returns:
        list: _description_
    """
    import tensorflow as tf 
    
    physical_devices = tf.config.list_physical_devices('GPU')
    
    if len(physical_devices) > 0: 
        logger.info(f"Number of GPU's available: {len(physical_devices)}. GPU's available: ")
        for gpu in physical_devices: 
            logger.info(f"{gpu}")
    else: 
        logger.info(f"No GPU detected. Using CPU instead")
        
    
    