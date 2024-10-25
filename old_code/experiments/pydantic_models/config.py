from pydantic import BaseModel, Field, conint, conlist
from typing import List, Optional, Literal, Dict


# General configuration
class GeneralConfig(BaseModel):
    experiment_name: str
    seed: conint(ge=0)  # Seed should be a non-negative integer
    device: Literal["cuda", "cpu"]  # Only "cuda" and "cpu" are allowed


# Layer configuration for shared and separate layers
class LayerConfig(BaseModel):
    type: str
    units: Optional[int] = None  # Some layers don't have 'units'
    bias: Optional[bool] = None
    input_shape: Optional[List[int]] = None
    values: Optional[List[int]] = None
    filter: Optional[List[int]] = None
    stride: Optional[List[int]] = None
    padding: Optional[str] = None
    rate: Optional[float] = None
    activation: Optional[str] = None
    shape: Optional[List[int]] = None  # Used for Input layer


# Generators configuration
class GeneratorsConfig(BaseModel):
    num_generators: conint(ge=1)
    latent_dim: conint(ge=1)
    shared_layers: List[LayerConfig]
    separate_layers: Dict[str, List[LayerConfig]]  # 'before' and 'after' sections


# Discriminator configuration
class DiscriminatorConfig(BaseModel):
    layers: List[LayerConfig]


# Model configuration combining generators and discriminator
class MadGanConfig(BaseModel):
    num_generators: conint(ge=1)
    latent_dim: conint(ge=1)    
    num_classes: conint(ge=1)
    generators: GeneratorsConfig
    discriminator: DiscriminatorConfig


# Main configuration combining general and model configurations
class Config(BaseModel):
    general: GeneralConfig
    model: MadGanConfig
    # Uncomment below lines if training and data sections are added
    # training: TrainingConfig
    # data: DataConfig


# Example Usage:
# from pydantic import ValidationError
# import yaml
#
# with open('path_to_yaml.yaml', 'r') as file:
#     config_data = yaml.safe_load(file)
#
# try:
#     config = Config(**config_data)
# except ValidationError as e:
#     print(e.json())
