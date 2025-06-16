import yaml
import logging
from pathlib import Path
from typing import Union, Tuple
from pydantic import ValidationError

from .schemas import SystemConfig, ModelConfig


class ConfigLoader:
    """Simple configuration loader with validation"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def load_and_validate(self, system_config_path: Union[str, Path], model_config_path: Union[str, Path]) -> Tuple[SystemConfig, ModelConfig]:
        """
        Load and validate configuration files from YAML files.

        Args:
            system_config_path: Path to YAML configuration file for OFDM-related parameters
            model_config_path: Path to YAML configuration file for model-related parameters


        Returns:
            ModelConfig: Validated model configuration object
            SystemConfig: Validated system configuration object

        Raises:
            FileNotFoundError: If one of the config files doesn't exist
            ValueError: If configuration validation fails
        """
        system_config_path = Path(system_config_path)
        model_config_path = Path(model_config_path)

        if not system_config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {system_config_path}")

        if not model_config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {model_config_path}")

        try:
            with open(system_config_path, 'r') as f:
                system_raw_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse YAML file {system_config_path}: {e}")

        try:
            with open(model_config_path, 'r') as f:
                model_raw_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse YAML file {model_config_path}: {e}")

        try:
            system_config = SystemConfig(**system_raw_config)
            self.logger.info(f"Successfully loaded system config from {system_config_path}")
        except ValidationError as e:
            raise ValueError(f"Configuration validation for {system_config_path} failed:\n{e}")
        if system_config:
            try:
                model_config = ModelConfig(system_config, **model_raw_config)
                self.logger.info(f"Successfully loaded model config from {model_config_path}")
            except ValidationError as e:
                raise ValueError(f"Configuration validation for {model_config_path} failed:\n{e}")

        return system_config, model_config



def load_config(system_config_path: Union[str, Path], model_config_path: Union[str, Path]) -> Tuple[SystemConfig, ModelConfig]:
    """Convenience function to load and validate config."""
    config_loader = ConfigLoader()
    return config_loader.load_and_validate(system_config_path, model_config_path)
