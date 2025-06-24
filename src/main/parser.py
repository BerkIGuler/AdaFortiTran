"""
Command line argument parser for OFDM channel estimation model training.

This module provides functionality for parsing and validating command-line arguments
used in training OFDM channel estimation models. It defines the available parameters,
their types, default values, and validation rules to ensure proper configuration
of training runs.
"""

from dataclasses import dataclass
from pathlib import Path
import argparse
from enum import Enum


class LossType(Enum):
    """Enumeration of supported loss functions."""
    MSE = "mse"
    MAE = "mae"
    HUBER = "huber"


@dataclass
class TrainingArguments:
    """Container for OFDM model training arguments.

    Stores, validates, and provides access to all parameters needed for
    training an OFDM channel estimation model.

    Attributes:
        # Model Configuration
        model_name: Supports Linear, AdaFortiTran, or FortiTran training
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

        # Training Hyperparameters
        batch_size: Number of samples per batch
        lr: Learning rate for optimizer
        max_epoch: Maximum number of training epochs
        patience: Early stopping patience in epochs
        loss_type: Type of loss function to use
        return_type: Type of data to return from dataset

        # Hardware & Evaluation
        cuda: CUDA device index
        test_every_n: Number of epochs between test evaluations
    """

    # Model Configuration
    model_name: str
    system_config_path: Path
    model_config_path: Path

    # Dataset Paths
    train_set: Path
    val_set: Path
    test_set: Path

    # Experiment Settings
    exp_id: str
    python_log_level: str = "INFO"
    tensorboard_log_dir: Path = Path("runs")

    # Training Hyperparameters
    batch_size: int = 64
    lr: float = 1e-3
    max_epoch: int = 10
    patience: int = 3
    loss_type: LossType = LossType.MSE
    return_type: str = "complex"

    # Hardware & Evaluation
    cuda: int = 0
    test_every_n: int = 10

    def __post_init__(self) -> None:
        """Validate arguments after initialization.

        Runs multiple validation checks on the provided arguments to ensure
        they are consistent and valid for training.

        Raises:
            ValueError: If any validation check fails
        """
        self._validate_paths()
        self._validate_numeric_args()

    def _validate_paths(self) -> None:
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

    def _validate_numeric_args(self) -> None:
        """Validate numeric arguments.

        Ensures that all numeric parameters have appropriate values:
        - test_every_n, max_epoch, patience, batch_size, lr must be positive
        - cuda must be non-negative

        Raises:
            ValueError: If any numeric argument has an invalid value
        """
        if self.test_every_n <= 0:
            raise ValueError(f"test_every_n must be positive, got: {self.test_every_n}")

        if self.max_epoch <= 0:
            raise ValueError(f"max_epoch must be positive, got: {self.max_epoch}")

        if self.patience <= 0:
            raise ValueError(f"patience must be positive, got: {self.patience}")

        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got: {self.batch_size}")

        if self.cuda < 0:
            raise ValueError(f"cuda must be non-negative, got: {self.cuda}")

        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got: {self.lr}")


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
        choices=['Linear', 'AdaFortiTran', 'FortiTran'],
        help='Model type to train (Linear, AdaFortiTran, or FortiTran)'
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
        '--cuda',
        type=int,
        default=0,
        help='CUDA device index (0 for single GPU)'
    )
    optional.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='Initial learning rate'
    )
    optional.add_argument(
        '--loss_type',
        type=str,
        default="mse",
        choices=['mse', 'mae', 'huber'],
        help='Loss function type'
    )
    optional.add_argument(
        '--return_type',
        type=str,
        default="complex",
        choices=['complex', 'real'],
        help='Type of data to return from dataset'
    )

    args = parser.parse_args()

    # Convert loss_type string to enum
    args.loss_type = LossType(args.loss_type)

    # Create and validate TrainingArguments
    return TrainingArguments(**vars(args))