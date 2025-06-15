import yaml
import logging
from pathlib import Path
from typing import Union
from pydantic import ValidationError

from .schemas import SystemConfig


class ConfigLoader:
    """Simple configuration loader with validation"""

    @staticmethod
    def load_and_validate(config_path: Union[str, Path]) -> SystemConfig:
        """
        Load and validate configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            ModelConfig: Validated configuration object

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If configuration validation fails
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path, 'r') as f:
                raw_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse YAML file {config_path}: {e}")

        try:
            config = SystemConfig(**raw_config)
            logging.getLogger(__name__).info(f"Successfully loaded config from {config_path}")
            return config
        except ValidationError as e:
            raise ValueError(f"Configuration validation failed:\n{e}")


def load_config(config_path: Union[str, Path]) -> SystemConfig:
    """Convenience function to load and validate config."""
    return ConfigLoader.load_and_validate(config_path)