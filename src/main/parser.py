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
from typing import Self, Optional, Literal
import torch


class TrainingArguments(BaseModel):
    """Container for OFDM model training arguments.

    Stores, validates, and provides access to all parameters needed for
    training an OFDM channel estimation model.

    Attributes:
        # Model Configuration
        model_name: Supports linear, adafortitran, or fortitran training
        system_config_path: Path to OFDM system configuration file
        model_config_path: Path to model configuration file (not required for linear model)
        device: Computing device (cpu, cuda, cuda:N, mps, or auto)

        # Dataset Paths
        train_set: Path to training dataset directory
        val_set: Path to validation dataset directory

        # Experiment Settings
        exp_id: Experiment identifier string used for logging and checkpointing
        python_log_level: Logging verbosity level for python logging module
        tensorboard_log_dir: Directory for tensorboard logs
        python_log_dir: Directory for python logging files

        # Training Hyperparameters
        batch_size: Number of samples per mini-batch
        lr: Learning rate for optimizer
        max_epoch: Maximum number of training epochs
        patience: Early stopping patience in epochs
        weight_decay: Weight decay for optimizer (i.e. lambda for L2 regularization on the model weights)
        gradient_clip_val: Gradient clipping value (i.e. max norm for gradient clipping)
        use_mixed_precision: Whether to use mixed precision training
        
        # Checkpointing
        save_checkpoints: Whether to save model checkpoints
        save_best_only: Whether to save only the best model
        save_every_n_epochs: Save checkpoint every N training epochs
        resume_from_checkpoint: Path to checkpoint to resume training from
        
        # Data Loading
        num_workers: Number of data loading workers for parallel data loading
        pin_memory: Whether to pin memory for faster GPU data transfer
    """

    # Model Configuration
    model_name: Literal['linear', 'adafortitran', 'fortitran'] = Field(..., description="Model type to train (linear, adafortitran, or fortitran)")
    system_config_path: Path = Field(..., description="Path to OFDM system configuration file (YAML file)")
    model_config_path: Optional[Path] = Field(default=None, description="Path to model configuration file (YAML file); not required for linear model")
    device: str = Field(default="auto", description="Computing device (cpu, cuda, cuda:0, mps, or auto)")

    # Dataset Paths
    train_set: Path = Field(..., description="Training dataset folder path")
    val_set: Path = Field(..., description="Validation dataset folder path")

    # Experiment Settings
    exp_id: str = Field(..., description="Experiment identifier for log folder naming")
    python_log_level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = Field(default="INFO", description="Logger level for python logging module")
    tensorboard_log_dir: Path = Field(default=Path("runs"), description="Directory for tensorboard logs")
    python_log_dir: Path = Field(default=Path("logs"), description="Directory for python logging files")

    # Training Hyperparameters
    batch_size: int = Field(default=64, gt=0, description="Training batch size")
    lr: float = Field(default=1e-3, gt=0, description="Initial learning rate")
    max_epoch: int = Field(default=10, gt=0, description="Maximum number of training epochs")
    patience: int = Field(default=3, gt=0, description="Early stopping patience (epochs)")
    weight_decay: float = Field(default=0.0, ge=0.0, description="Weight decay for optimizer (L2 regularization on the model weights)")
    gradient_clip_val: Optional[float] = Field(default=None, gt=0, description="Gradient clipping value")
    use_mixed_precision: bool = Field(default=False, description="Whether to use mixed precision training")

    # Checkpointing
    save_checkpoints: bool = Field(default=True, description="Whether to save model checkpoints")
    save_best_only: bool = Field(default=True, description="Whether to save only the best model")
    save_every_n_epochs: Optional[int] = Field(default=None, gt=0, description="Save checkpoint every N training epochs")
    resume_from_checkpoint: Optional[Path] = Field(default=None, description="Path to checkpoint to resume training from")

    # Data Loading
    num_workers: int = Field(default=4, ge=0, description="Number of data loading workers")
    pin_memory: bool = Field(default=True, description="Whether to pin memory for faster GPU data transfer")

    @model_validator(mode='after')
    def validate_arguments(self) -> Self:
        """Validate training arguments.

        Checks paths, device, hyperparameters, and logical consistency.

        Raises:
            ValueError: If validation fails
        """
        # Validate system config path
        if not self.system_config_path.exists():
            raise ValueError(f"System configuration file not found: {self.system_config_path}")
        if not self.system_config_path.suffix == '.yaml':
            raise ValueError(f"System configuration file must be a .yaml file: {self.system_config_path}")

        # Model config is required for non-linear models
        if self.model_name != 'linear':
            if self.model_config_path is None:
                raise ValueError("model_config_path is required unless model_name is 'linear'")
            if not self.model_config_path.exists():
                raise ValueError(f"Model configuration file not found: {self.model_config_path}")
            if not self.model_config_path.suffix == '.yaml':
                raise ValueError(f"Model configuration file must be a .yaml file: {self.model_config_path}")

        # Validate checkpoint path if provided
        if self.resume_from_checkpoint is not None:
            if not self.resume_from_checkpoint.exists():
                raise ValueError(f"Checkpoint file not found: {self.resume_from_checkpoint}")
            if not self.resume_from_checkpoint.suffix == '.pt':
                raise ValueError(f"Checkpoint file must be a .pt file: {self.resume_from_checkpoint}")

        # Validate dataset paths exist
        if not self.train_set.exists():
            raise ValueError(f"Training dataset not found: {self.train_set}")
        if not self.val_set.exists():
            raise ValueError(f"Validation dataset not found: {self.val_set}")

        # Validate and resolve device
        self._validate_and_resolve_device()

        return self
    
    def _validate_and_resolve_device(self) -> None:
        """Validate and resolve the device string.
        
        Handles 'auto' selection, validates CUDA and MPS availability,
        and checks CUDA device IDs.
        
        Raises:
            ValueError: If device is invalid or unavailable
        """
        device_str = self.device.lower()

        # Handle 'auto' - automatically select best available device
        if device_str == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = 'mps'  # Apple Silicon (MPS)
            else:
                self.device = 'cpu'
            return

        # CPU is always valid
        if device_str == 'cpu':
            return

        # Validate CUDA devices
        if device_str.startswith('cuda'):
            if not torch.cuda.is_available():
                raise ValueError("CUDA is not available on this system")

            # Handle specific CUDA device (e.g., 'cuda:0', 'cuda:1')
            if ':' in device_str:
                try:
                    device_id = int(device_str.split(':')[1])
                    if device_id >= torch.cuda.device_count():
                        available_devices = list(range(torch.cuda.device_count()))
                        raise ValueError(
                            f"CUDA device {device_id} not available. "
                            f"Available CUDA devices: {available_devices}"
                        )
                except (ValueError, IndexError) as e:
                    if "invalid literal" in str(e):
                        raise ValueError(f"Invalid CUDA device format: {device_str}")
                    raise
            return

        # Validate MPS (Apple Silicon)
        if device_str == 'mps':
            if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                raise ValueError("MPS is not available/detected on this system")
            return

        # If we get here, the device is not recognized
        available_devices = ['cpu']
        if torch.cuda.is_available():
            cuda_devices = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
            available_devices.extend(['cuda'] + cuda_devices)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            available_devices.append('mps')

        raise ValueError(
            f"Unsupported device: '{self.device}'. "
            f"Available devices: {available_devices}"
        )


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
        '--exp_id',
        type=str,
        required=True,
        help='Experiment identifier for log folder naming'
    )

    # Optional argument; path for model configuration file
    # required for non-linear models in TrainingArguments class
    parser.add_argument(
        '--model_config_path',
        type=Path,
        default=None,
        help='Path to YAML with model architecture (required for fortitran/adafortitran; optional for linear)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Computing device: cpu, cuda, cuda:N, mps, or auto (default: auto selects best available)'
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
        help='Weight decay for optimizer (L2 regularization on the model weights)'
    )
    training.add_argument(
        '--gradient_clip_val',
        type=float,
        default=None,
        help='Gradient clipping value (disabled if not specified); recommended value: 1.0'
    )
    training.add_argument(
        '--use_mixed_precision',
        action='store_true',
        default=False,
        help='Use mixed precision training'
    )

    # Checkpointing settings
    checkpointing = parser.add_argument_group('checkpointing settings')
    checkpointing.add_argument(
        '--save_best_only',
        action='store_true',
        default=False,
        help='Save only the best model based on validation loss (default: save all models)'
    )
    checkpointing.add_argument(
        '--save_every_n_epochs',
        type=int,
        default=10,
        help='Save checkpoint every N training epochs (in addition to best model)'
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
