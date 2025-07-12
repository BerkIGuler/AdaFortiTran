"""
Command line argument parser for OFDM channel estimation model training.

This module provides functionality for parsing and validating command-line arguments
used in training OFDM channel estimation models. It defines the available parameters,
their types, default values, and validation rules to ensure proper configuration
of training runs.
"""

from pathlib import Path
import argparse
from pydantic import BaseModel, Field, model_validator
from typing import Self


class TrainingArguments(BaseModel):
    """Container for OFDM model training arguments.

    Stores, validates, and provides access to all parameters needed for
    training an OFDM channel estimation model.

    Attributes:
        # Model Configuration
        model_name: Supports linear, adafortitran, or fortitran training
        system_config_path: Path to OFDM system configuration file
        model_config_path: Path to model configuration file

        # Dataset Paths
        train_set: Path to training dataset directory
        val_set: Path to validation dataset directory
        test_set: Path to test dataset directory

        # Experiment Settings
        exp_id: Experiment identifier string
        python_log_level: Logging verbosity level
        tensorboard_log_dir: Directory for tensorboard logs
        python_log_dir: Directory for python logging files

        # Training Hyperparameters
        batch_size: Number of samples per batch
        lr: Learning rate for optimizer
        max_epoch: Maximum number of training epochs
        patience: Early stopping patience in epochs

        # Evaluation
        test_every_n: Number of epochs between test evaluations
    """

    # Model Configuration
    model_name: str = Field(..., description="Model type to train")
    system_config_path: Path = Field(..., description="Path to OFDM system configuration file")
    model_config_path: Path = Field(..., description="Path to model configuration file")

    # Dataset Paths
    train_set: Path = Field(..., description="Training dataset folder path")
    val_set: Path = Field(..., description="Validation dataset folder path")
    test_set: Path = Field(..., description="Test dataset folder path")

    # Experiment Settings
    exp_id: str = Field(..., description="Experiment identifier for log folder naming")
    python_log_level: str = Field(default="INFO", description="Logger level for python logging module")
    tensorboard_log_dir: Path = Field(default=Path("runs"), description="Directory for tensorboard logs")
    python_log_dir: Path = Field(default=Path("logs"), description="Directory for python logging files")

    # Training Hyperparameters
    batch_size: int = Field(default=64, gt=0, description="Training batch size")
    lr: float = Field(default=1e-3, gt=0, description="Initial learning rate")
    max_epoch: int = Field(default=10, gt=0, description="Maximum number of training epochs")
    patience: int = Field(default=3, gt=0, description="Early stopping patience (epochs)")

    # Evaluation
    test_every_n: int = Field(default=10, gt=0, description="Test model every N epochs")

    @model_validator(mode='after')
    def validate_paths(self) -> Self:
        """Validate path-related arguments.

        Checks that the config files exist and have the correct extension.

        Raises:
            ValueError: If the config files don't exist or aren't YAML files
        """
        if not self.system_config_path.exists():
            raise ValueError(f"System config file not found: {self.system_config_path}")

        if not self.system_config_path.suffix == '.yaml':
            raise ValueError(f"System config file must be a .yaml file: {self.system_config_path}")

        if not self.model_config_path.exists():
            raise ValueError(f"Model config file not found: {self.model_config_path}")

        if not self.model_config_path.suffix == '.yaml':
            raise ValueError(f"Model config file must be a .yaml file: {self.model_config_path}")

        return self


def parse_arguments() -> TrainingArguments:
    """Parse command-line arguments for training an OFDM channel estimation model.

    Sets up an argument parser with all required and optional arguments,
    processes the command line input, and returns a validated TrainingArguments
    object with all parameters needed for model training.

    Returns:
        TrainingArguments: Validated arguments for model training

    Raises:
        ValueError: If validation fails for any arguments
        SystemExit: If argument parsing fails (raised by argparse)
    """

    parser = argparse.ArgumentParser(
        description='Train an OFDM channel estimation model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    required = parser.add_argument_group('required arguments')
    required.add_argument(
        '--model_name',
        type=str,
        required=True,
        choices=['linear', 'adafortitran', 'fortitran'],
        help='Model type to train (linear, adafortitran, or fortitran)'
    )
    required.add_argument(
        '--system_config_path',
        type=Path,
        required=True,
        help='Path to YAML file containing OFDM system parameters'
    )
    required.add_argument(
        '--model_config_path',
        type=Path,
        required=True,
        help='Path to YAML file containing model architecture parameters'
    )
    required.add_argument(
        '--train_set',
        type=Path,
        required=True,
        help='Training dataset folder path'
    )
    required.add_argument(
        '--val_set',
        type=Path,
        required=True,
        help='Validation dataset folder path'
    )
    required.add_argument(
        '--test_set',
        type=Path,
        required=True,
        help='Test dataset folder path'
    )
    required.add_argument(
        '--exp_id',
        type=str,
        required=True,
        help='Experiment identifier for log folder naming'
    )

    # Optional arguments
    optional = parser.add_argument_group('optional arguments')
    optional.add_argument(
        '--python_log_level',
        type=str,
        default="INFO",
        help='Logger level for python logging module'
    )
    optional.add_argument(
        '--tensorboard_log_dir',
        type=Path,
        default="runs",
        help='Directory for tensorboard logs'
    )
    optional.add_argument(
        '--python_log_dir',
        type=Path,
        default="logs",
        help='Directory for python logging files'
    )
    optional.add_argument(
        '--test_every_n',
        type=int,
        default=10,
        help='Test model every N epochs'
    )
    optional.add_argument(
        '--max_epoch',
        type=int,
        default=10,
        help='Maximum number of training epochs'
    )
    optional.add_argument(
        '--patience',
        type=int,
        default=3,
        help='Early stopping patience (epochs)'
    )
    optional.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Training batch size'
    )

    optional.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='Initial learning rate'
    )


    args = parser.parse_args()

    # Create and validate TrainingArguments
    return TrainingArguments(**vars(args))
