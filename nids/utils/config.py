"""Configuration management utilities."""

import yaml
from pathlib import Path
from typing import Dict, Any, Union


import os

def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load YAML configuration file with SOC validations.
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
            config = config if config is not None else {}
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse YAML config: {e}")

    # Recursive ENV Override (Format: NIDS_SECTION_KEY)
    def _override_from_env(d, prefix="NIDS"):
        for k, v in d.items():
            env_key = f"{prefix}_{k.upper()}"
            if isinstance(v, dict):
                _override_from_env(v, env_key)
            else:
                env_val = os.getenv(env_key)
                if env_val is not None:
                    # Cast env string to original type
                    type_cast = type(v) if v is not None else str
                    try:
                        d[k] = type_cast(env_val)
                    except ValueError:
                        d[k] = env_val

    _override_from_env(config)
    
    # Core SOC Validation
    if 'models' in config and 'confidence_threshold' in config['models']:
        assert 0.0 <= config['models']['confidence_threshold'] <= 1.0, "Invalid conf threshold"

    return config


def save_config(config: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        output_path: Path to save YAML file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries.
    Later configs override earlier ones.
    
    Args:
        *configs: Variable number of config dictionaries
        
    Returns:
        Merged configuration dictionary
    """
    merged = {}
    for config in configs:
        merged.update(config)
    return merged
