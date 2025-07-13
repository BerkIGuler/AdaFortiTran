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
from typing import Self, Optional


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
        weight_decay: Weight decay for optimizer
        gradient_clip_val: Gradient clipping value
        use_mixed_precision: Whether to use mixed precision training

        # Evaluation
        test_every_n: Number of epochs between test evaluations
        
        # Checkpointing
        save_checkpoints: Whether to save model checkpoints
        save_best_only: Whether to save only the best model
        save_every_n_epochs: Save checkpoint every N epochs
        resume_from_checkpoint: Path to checkpoint to resume from
        
        # Data Loading
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for faster GPU transfer
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
    weight_decay: float = Field(default=0.0, ge=0.0, description="Weight decay for optimizer")
    gradient_clip_val: Optional[float] = Field(default=None, gt=0, description="Gradient clipping value")
    use_mixed_precision: bool = Field(default=False, description="Whether to use mixed precision training")

    # Evaluation
    test_every_n: int = Field(default=10, gt=0, description="Test model every N epochs")

    # Checkpointing
    save_checkpoints: bool = Field(default=True, description="Whether to save model checkpoints")
    save_best_only: bool = Field(default=True, description="Whether to save only the best model")
    save_every_n_epochs: Optional[int] = Field(default=None, gt=0, description="Save checkpoint every N epochs")
    resume_from_checkpoint: Optional[Path] = Field(default=None, description="Path to checkpoint to resume from")

    # Data Loading
    num_workers: int = Field(default=4, ge=0, description="Number of data loading workers")
    pin_memory: bool = Field(default=True, description="Whether to pin memory for faster GPU transfer")

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

        # Validate checkpoint path if provided
        if self.resume_from_checkpoint is not None:
            if not self.resume_from_checkpoint.exists():
                raise ValueError(f"Checkpoint file not found: {self.resume_from_checkpoint}")
            if not self.resume_from_checkpoint.suffix == '.pt':
                raise ValueError(f"Checkpoint file must be a .pt file: {self.resume_from_checkpoint}")

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

    # Training hyperparameters
    training = parser.add_argument_group('training hyperparameters')
    training.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Training batch size'
    )
    training.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='Initial learning rate'
    )
    training.add_argument(
        '--max_epoch',
        type=int,
        default=10,
        help='Maximum number of training epochs'
    )
    training.add_argument(
        '--patience',
        type=int,
        default=3,
        help='Early stopping patience (epochs)'
    )
    training.add_argument(
        '--weight_decay',
        type=float,
        default=0.0,
        help='Weight decay for optimizer'
    )
    training.add_argument(
        '--gradient_clip_val',
        type=float,
        default=None,
        help='Gradient clipping value (disabled if not specified)'
    )
    training.add_argument(
        '--use_mixed_precision',
        action='store_true',
        help='Use mixed precision training (requires PyTorch >= 1.6)'
    )

    # Evaluation settings
    evaluation = parser.add_argument_group('evaluation settings')
    evaluation.add_argument(
        '--test_every_n',
        type=int,
        default=10,
        help='Test model every N epochs'
    )

    # Checkpointing settings
    checkpointing = parser.add_argument_group('checkpointing settings')
    checkpointing.add_argument(
        '--save_checkpoints',
        action='store_true',
        default=True,
        help='Save model checkpoints'
    )
    checkpointing.add_argument(
        '--no_save_checkpoints',
        action='store_false',
        dest='save_checkpoints',
        help='Disable saving model checkpoints'
    )
    checkpointing.add_argument(
        '--save_best_only',
        action='store_true',
        default=True,
        help='Save only the best model based on validation loss'
    )
    checkpointing.add_argument(
        '--save_every_n_epochs',
        type=int,
        default=None,
        help='Save checkpoint every N epochs (in addition to best model)'
    )
    checkpointing.add_argument(
        '--resume_from_checkpoint',
        type=Path,
        default=None,
        help='Path to checkpoint file to resume training from'
    )

    # Data loading settings
    data_loading = parser.add_argument_group('data loading settings')
    data_loading.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of data loading workers'
    )
    data_loading.add_argument(
        '--pin_memory',
        action='store_true',
        default=True,
        help='Pin memory for faster GPU transfer'
    )
    data_loading.add_argument(
        '--no_pin_memory',
        action='store_false',
        dest='pin_memory',
        help='Disable pin memory'
    )

    # Logging settings
    logging_group = parser.add_argument_group('logging settings')
    logging_group.add_argument(
        '--python_log_level',
        type=str,
        default="INFO",
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Logger level for python logging module'
    )
    logging_group.add_argument(
        '--tensorboard_log_dir',
        type=Path,
        default="runs",
        help='Directory for tensorboard logs'
    )
    logging_group.add_argument(
        '--python_log_dir',
        type=Path,
        default="logs",
        help='Directory for python logging files'
    )

    args = parser.parse_args()

    # Create and validate TrainingArguments
    return TrainingArguments(**vars(args))
