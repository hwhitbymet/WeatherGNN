import yaml
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    num_epochs: int
    early_stopping_patience: int

@dataclass
class DataConfig:
    train_period: dict
    validation_period: dict
    test_period: dict
    num_workers: int
    zarr_dataset_path: str
    init_params_cache: str
    splits_cache_dir: str

@dataclass
class ModelConfig:
    """Configuration for the predictive model and its components"""
    n_lat: int
    n_lon: int
    n_pressure_levels: int
    n_variables: int
    n_sphere_points: int
    latent_size: int
    num_layers: int
    num_message_passing_steps: int
    max_distance_degrees: float

    @property
    def n_spatial_nodes(self) -> int:
        return self.n_lat * self.n_lon
    
    @property
    def n_features(self) -> int:
        return self.n_variables * self.n_pressure_levels

class Configuration:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        self.model = ModelConfig(**config_dict['model'])
        self.training = TrainingConfig(**config_dict['training'])
        self.data = DataConfig(**config_dict['data'])

    @classmethod
    def load(cls, config_path: str) -> 'Configuration':
        return cls(config_path)