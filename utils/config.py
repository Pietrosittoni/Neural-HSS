import os
import yaml
from typing import Any, Dict

class Config:
    def __init__(self, config_dict: Dict[str, Any]):
        self.__dict__.update(config_dict)
    
    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        if 'base_config' in config_dict:
            base_path = os.path.join(os.path.dirname(path), config_dict.pop('base_config'))
            with open(base_path, 'r') as f:
                base_dict = yaml.safe_load(f)
                base_dict.update(config_dict)
                config_dict = base_dict
        return cls(config_dict)