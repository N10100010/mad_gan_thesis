from typing import Type, TypeVar

import yaml

T = TypeVar('T')

def load_pydantic_object(path: str, pydantic_class: Type[T]) -> T: 
    """Given a path to a config-file, the function tries to load a pydantic model.

    Args:
        path (str): Path to config
        cls (Type[T]): Pydantic class

    Returns:
        T: Pydantic instance
    """
    with open(path, 'r') as f: 
        data = yaml.safe_load(f)
        
    conf = pydantic_class(**data)
    
    return conf
