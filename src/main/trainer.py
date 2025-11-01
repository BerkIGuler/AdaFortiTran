"""
OFDM channel estimation model training module.

This module provides functionality for training and evaluating deep learning models
for OFDM channel estimation tasks. It includes a ModelTrainer class that handles
the complete training workflow, including model initialization, data loading,
training loop management, evaluation, and result logging.
"""

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from typing import Dict, Tuple, Type, Union, Optional, List, Protocol
import logging
from tqdm import tqdm
from dataclasses import dataclass
from pathlib import Path
from abc import ABC, abstractmethod

from .parser import TrainingArguments
from src.data import MatDataset, get_test_dataloaders
from src.models import LinearEstimator, AdaFortiTranEstimator, FortiTranEstimator
from src.utils import (
    EarlyStopping,
    get_ls_mse_per_folder,
    get_model_details,
    get_test_stats_plot,
    get_error_images,
    concat_complex_channel,
    to_db
)
from src.config.schemas import SystemConfig, ModelConfig

# A union type representing supported model classes
ModelType = Union[LinearEstimator, AdaFortiTranEstimator, FortiTranEstimator]


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    train_loss: float
    val_loss: float
    epoch: int
    learning_rate: float


@dataclass
class TestResults:
    """Container for test results."""
    ds_stats: Dict[int, float]
    mds_stats: Dict[int, float]
    snr_stats: Dict[int, float]


class Callback(ABC):
    """Base class for training callbacks."""
    
    @abstractmethod
    def on_epoch_begin(self, epoch: int) -> None:
        """Called at the beginning of each epoch."""
        pass
    
    @abstractmethod
    def on_epoch_end(self, epoch: int, metrics: TrainingMetrics) -> None:
        """Called at the end of each epoch."""
        pass
    
    @abstractmethod
    def on_training_begin(self) -> None:
        """Called at the beginning of training."""
        pass
    
    @abstractmethod
    def on_training_end(self) -> None:
        """Called at the end of training."""
        pass


class CheckpointCallback(Callback):
    """Callback for saving model checkpoints."""
    
    def __init__(self, save_dir: Path, save_best_only: bool = True, 
                 save_every_n_epochs: Optional[int] = None):
        self.save_dir = save_dir
        self.save_best_only = save_best_only
        self.save_every_n_epochs = save_every_n_epochs
        self.best_val_loss = float('inf')
        self.trainer = None
        
    def set_trainer(self, trainer: 'ModelTrainer') -> None:
        """Set the trainer reference."""
        self.trainer = trainer
        
    def on_epoch_begin(self, epoch: int) -> None:
        pass
        
    def on_epoch_end(self, epoch: int, metrics: TrainingMetrics) -> None:
        if self.trainer is None:
            return
            
        # Save best model
        if self.save_best_only and metrics.val_loss < self.best_val_loss:
            self.best_val_loss = metrics.val_loss
            self.trainer.save_checkpoint(
                epoch, metrics, 
                checkpoint_dir=self.save_dir / "best"
            )
            
        # Save every N epochs
        if (self.save_every_n_epochs is not None and 
            (epoch + 1) % self.save_every_n_epochs == 0):
            self.trainer.save_checkpoint(
                epoch, metrics, 
                checkpoint_dir=self.save_dir / "periodic"
            )
            
    def on_training_begin(self) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def on_training_end(self) -> None:
        pass


class TensorBoardCallback(Callback):
    """Callback for TensorBoard logging."""
    
    def __init__(self, writer: SummaryWriter):
        self.writer = writer
        
    def on_epoch_begin(self, epoch: int) -> None:
        pass
        
    def on_epoch_end(self, epoch: int, metrics: TrainingMetrics) -> None:
        self.writer.add_scalar('Loss/Train', metrics.train_loss, metrics.epoch + 1)
        self.writer.add_scalar('Loss/Val', metrics.val_loss, metrics.epoch + 1)
        self.writer.add_scalar('Learning_Rate', metrics.learning_rate, metrics.epoch + 1)
        
    def on_training_begin(self) -> None:
        pass
        
    def on_training_end(self) -> None:
        self.writer.close()


class TrainingLoop:
    """Handles the core training loop logic."""
    
    def __init__(self, model: ModelType, optimizer: optim.Optimizer, 
                 scheduler: optim.lr_scheduler.LRScheduler, 
                 loss_fn: nn.Module, device: torch.device, scaler: Optional[torch.cuda.amp.GradScaler] = None,
                 gradient_clip_val: Optional[float] = None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.scaler = scaler
        self.gradient_clip_val = gradient_clip_val
        
    def _compute_loss(self, estimated_channel: torch.Tensor, 
                     ideal_channel: torch.Tensor) -> torch.Tensor:
        """Compute loss between estimated and ideal channels."""
        return self.loss_fn(
            concat_complex_channel(estimated_channel),
            concat_complex_channel(ideal_channel)
        )

    def _forward_pass(self, batch: Tuple[torch.Tensor, torch.Tensor, Tuple], 
                     model: ModelType) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform forward pass through the model."""
        estimated_channel, ideal_channel, meta_data = batch
        
        # All models now handle complex input directly
        if isinstance(model, AdaFortiTranEstimator):
            # AdaFortiTran uses meta_data for channel adaptation
            estimated_channel = model(estimated_channel, meta_data)
        else:
            # Linear and FortiTran models don't use meta_data
            estimated_channel = model(estimated_channel)
            
        return estimated_channel, ideal_channel.to(model.device)

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        train_loss = 0.0
        self.model.train()
        num_samples = 0
        
        for batch in train_loader:
            self.optimizer.zero_grad()
            estimated_channel, ideal_channel = self._forward_pass(batch, self.model)
            
            if self.scaler:
                with torch.cuda.amp.autocast():
                    loss = self._compute_loss(estimated_channel, ideal_channel)
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.gradient_clip_val:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss = self._compute_loss(estimated_channel, ideal_channel)
                loss.backward()
                
                # Gradient clipping
                if self.gradient_clip_val:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                
                self.optimizer.step()
            
            batch_size = batch[0].size(0)
            train_loss += (2 * loss.item() * batch_size)
            num_samples += batch_size
            
        self.scheduler.step()
        return train_loss / num_samples

    def evaluate(self, eval_loader: DataLoader) -> float:
        """Evaluate the model."""
        val_loss = 0.0
        self.model.eval()
        num_samples = 0
        
        with torch.no_grad():
            for batch in eval_loader:
                estimated_channel, ideal_channel = self._forward_pass(batch, self.model)
                
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        loss = self._compute_loss(estimated_channel, ideal_channel)
                else:
                    loss = self._compute_loss(estimated_channel, ideal_channel)
                
                batch_size = batch[0].size(0)
                val_loss += (2 * loss.item() * batch_size)
                num_samples += batch_size
                
        return val_loss / num_samples


class ModelEvaluator:
    """Handles model evaluation and testing."""
    
    def __init__(self, model: ModelType, device: torch.device, logger: logging.Logger):
        self.model = model
        self.device = device
        self.logger = logger
        
    def _forward_pass(self, batch: Tuple[torch.Tensor, torch.Tensor, Tuple], 
                     model: ModelType) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform forward pass through the model."""
        estimated_channel, ideal_channel, meta_data = batch
        
        if isinstance(model, AdaFortiTranEstimator):
            estimated_channel = model(estimated_channel, meta_data)
        else:
            estimated_channel = model(estimated_channel)
            
        return estimated_channel, ideal_channel.to(model.device)

    def predict_channels(self, test_dataloaders: List[Tuple[str, DataLoader]]) -> Dict[int, Dict]:
        """Predict channels for visualization."""
        channels = {}
        sorted_loaders = sorted(
            test_dataloaders,
            key=lambda x: int(x[0].split("_")[1])
        )
        
        for name, test_dataloader in sorted_loaders:
            with torch.no_grad():
                batch = next(iter(test_dataloader))
                estimated_channels, ideal_channels = self._forward_pass(batch, self.model)
                
            var, val = name.split("_")
            channels[int(val)] = {
                "estimated_channel": estimated_channels[0],
                "ideal_channel": ideal_channels[0]
            }
        return channels

    def get_test_stats(self, test_dataloaders: List[Tuple[str, DataLoader]], 
                      loss_fn: nn.Module) -> Dict[int, float]:
        """Get test statistics for a set of dataloaders."""
        stats = {}
        sorted_loaders = sorted(
            test_dataloaders,
            key=lambda x: int(x[0].split("_")[1])
        )
        
        for name, test_dataloader in sorted_loaders:
            var, val = name.split("_")
            test_loss = self._evaluate_dataloader(test_dataloader, loss_fn)
            db_error = to_db(test_loss)
            self.logger.info(f"{var}:{val} Test MSE: {db_error:.4f} dB")
            stats[int(val)] = db_error
        return stats

    def _evaluate_dataloader(self, dataloader: DataLoader, loss_fn: nn.Module) -> float:
        """Evaluate a single dataloader."""
        total_loss = 0.0
        num_samples = 0
        self.model.eval()
        
        with torch.no_grad():
            for batch in dataloader:
                estimated_channel, ideal_channel = self._forward_pass(batch, self.model)
                loss = loss_fn(
                    concat_complex_channel(estimated_channel),
                    concat_complex_channel(ideal_channel)
                )
                
                batch_size = batch[0].size(0)
                total_loss += (2 * loss.item() * batch_size)
                num_samples += batch_size
                
        return total_loss / num_samples


class ModelTrainer:
    """Handles the training and evaluation of deep learning models.

    This class manages the complete lifecycle of model training, including:
    - Model initialization and configuration
    - Optimizer and loss function setup
    - Data loading and preprocessing
    - Training loop execution
    - Performance evaluation
    - Result logging and visualization via TensorBoard

    Attributes:
        MODEL_REGISTRY: Dictionary mapping model names to model classes
        system_config: OFDM system configuration
        model_config: OFDM model configuration
        args: Training arguments parsed from command line
        device: PyTorch device for computation
        writer: TensorBoard SummaryWriter for logging
        model: Initialized Torch model instance
        optimizer: Torch optimizer for training
        scheduler: Learning rate scheduler for training
        early_stopper: Helper for early stopping
        train_loader: DataLoader for training set (used for training)
        val_loader: DataLoader for validation set (used for validation)
        test_loaders: Dictionary of test set DataLoaders (used for testing)
        logger: Logger instance for logging messages
        training_loop: TrainingLoop instance for core training logic
        evaluator: ModelEvaluator instance for evaluation logic
        callbacks: List of training callbacks
    """

    MODEL_REGISTRY: Dict[str, Type[ModelType]] = {
        "linear": LinearEstimator,
        "adafortitran": AdaFortiTranEstimator,
        "fortitran": FortiTranEstimator,
    }

    EXP_LR_GAMMA = 0.995

    def __init__(self, system_config: SystemConfig, model_config: ModelConfig, args: TrainingArguments):
        """
        Initialize the ModelTrainer.

        Args:
            system_config: OFDM system configuration dictionary from YAML file
            model_config: OFDM model configuration dictionary from YAML file
            args: Validated training arguments parsed from command line
        """
        self.system_config = system_config
        self.model_config = model_config
        self.args = args
        self.device = torch.device(model_config.device)
        self.writer = self._setup_tensorboard()
        self.logger = logging.getLogger(__name__)

        self.model = self._initialize_model()
        
        # Initialize optimizer with weight decay
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.EXP_LR_GAMMA)
        self.early_stopper = EarlyStopping(patience=args.patience)
        self.training_loss = nn.MSELoss()

        # Initialize mixed precision training if requested
        self.scaler = None
        if args.use_mixed_precision and self.device.type == 'cuda':
            self.scaler = torch.cuda.amp.GradScaler()
            self.logger.info("Mixed precision training enabled")

        self.train_loader, self.val_loader, self.test_loaders = self._get_dataloaders()
        
        # Initialize components
        self.training_loop = TrainingLoop(
            self.model, self.optimizer, self.scheduler, self.training_loss, 
            self.device, self.scaler, self.args.gradient_clip_val
        )
        self.evaluator = ModelEvaluator(self.model, self.device, self.logger)
        
        # Initialize callbacks
        self.callbacks = self._setup_callbacks()
        
        # Resume from checkpoint if specified
        if args.resume_from_checkpoint is not None:
            self._resume_from_checkpoint(args.resume_from_checkpoint)

    def _setup_callbacks(self) -> List[Callback]:
        """Set up training callbacks."""
        callbacks = []
        
        # TensorBoard callback
        callbacks.append(TensorBoardCallback(self.writer))
        
        # Checkpoint callback (only if checkpointing is enabled)
        if self.args.save_checkpoints:
            checkpoint_dir = self.args.tensorboard_log_dir / f"{self.args.model_name}_{self.args.exp_id}"
            checkpoint_callback = CheckpointCallback(
                save_dir=checkpoint_dir,
                save_best_only=self.args.save_best_only,
                save_every_n_epochs=self.args.save_every_n_epochs
            )
            checkpoint_callback.set_trainer(self)
            callbacks.append(checkpoint_callback)
        
        return callbacks

    def _setup_tensorboard(self) -> SummaryWriter:
        """Set up TensorBoard logging.

        Creates a unique log directory based on model name and experiment ID.

        Returns:
            Initialized SummaryWriter for TensorBoard logging

        Raises:
            RuntimeError: If experiment directory already exists
        """
        log_path = self.args.tensorboard_log_dir / f"{self.args.model_name}_{self.args.exp_id}"
        if log_path.exists():
            raise RuntimeError(f"Experiment {log_path} already exists")

        return SummaryWriter(str(log_path))

    def _initialize_model(self) -> ModelType:
        """Initialize the model based on configuration.

        Creates an instance of the appropriate model class from the registry,
        logs model summary information, and returns the initialized model.

        Returns:
            Initialized model instance of the specified type
        """
        if self.args.model_name not in self.MODEL_REGISTRY:
            raise ValueError(f"Unknown model name: {self.args.model_name}. Available: {list(self.MODEL_REGISTRY.keys())}")
        
        model_class = self.MODEL_REGISTRY[self.args.model_name]
        model = model_class(self.system_config, self.model_config)
        
        num_params, model_summary = get_model_details(model)
        self.logger.info("\n" + model_summary)
        self.logger.info(f"Model name: {self.args.model_name} | Number of parameters: {num_params}")
        self.writer.add_text("Model Summary", model_summary)
        self.writer.add_text("Number of Parameters", str(num_params))
        return model

    def _get_dataloaders(self) -> Tuple[DataLoader, DataLoader, dict[str, list[tuple[str, DataLoader]]]]:
        """Get training, validation, and test dataloaders."""
        # Training and validation dataloaders
        train_dataset = MatDataset(self.args.train_set, self.system_config.pilot)
        val_dataset = MatDataset(self.args.val_set, self.system_config.pilot)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory and self.device.type == 'cuda'
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,  # No need to shuffle validation data
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory and self.device.type == 'cuda'
        )
        
        # Test dataloaders
        test_loaders = {
            "DS": get_test_dataloaders(
                self.args.test_set / "DS_test_set",
                self.system_config.pilot,
                self.args.batch_size
            ),
            "MDS": get_test_dataloaders(
                self.args.test_set / "MDS_test_set",
                self.system_config.pilot,
                self.args.batch_size
            ),
            "SNR": get_test_dataloaders(
                self.args.test_set / "SNR_test_set",
                self.system_config.pilot,
                self.args.batch_size
            ),
        }
        return train_loader, val_loader, test_loaders

    def _log_test_results(self, epoch: int, test_stats: Dict[str, Dict]) -> None:
        """Log test results to TensorBoard.

        Creates and logs visualizations for model performance across different test conditions.

        Args:
            epoch: Current training epoch
            test_stats: Dictionary of test statistics for the model
        """
        for key in ("DS", "MDS", "SNR"):
            # Plot test statistics
            self.writer.add_figure(
                tag=f"MSE vs. {key} (Epoch:{epoch + 1})",
                figure=get_test_stats_plot(
                    x_name=key,
                    stats=[test_stats[key]],
                    methods=[self.args.model_name]
                )
            )

            # Plot error images
            predicted_channels = self.evaluator.predict_channels(self.test_loaders[key])
            self.writer.add_figure(
                tag=f"{key} Error Images (Epoch:{epoch + 1})",
                figure=get_error_images(
                    key,
                    predicted_channels,
                    show=False
                )
            )

    def _run_tests(self, epoch: int) -> TestResults:
        """Run tests and log results.

        Evaluates the model on all test datasets and logs performance metrics and visualizations.

        Args:
            epoch: Current training epoch
            
        Returns:
            TestResults containing all test statistics
        """
        ds_stats = self.evaluator.get_test_stats(self.test_loaders["DS"], self.training_loss)
        mds_stats = self.evaluator.get_test_stats(self.test_loaders["MDS"], self.training_loss)
        snr_stats = self.evaluator.get_test_stats(self.test_loaders["SNR"], self.training_loss)

        test_stats = {
            "DS": ds_stats,
            "MDS": mds_stats,
            "SNR": snr_stats
        }

        self._log_test_results(epoch, test_stats)
        
        return TestResults(ds_stats, mds_stats, snr_stats)

    def _log_final_metrics(self, final_epoch: int) -> None:
        """Log final training metrics and hyperparameters.

        Records hyperparameters used in training and final performance metrics
        across all test conditions for experiment tracking.

        Args:
            final_epoch: The index of the final training epoch
        """
        str_params = {k: str(v) for k, v in vars(self.args).items()}
        self.writer.add_hparams(
            hparam_dict=str_params,
            metric_dict={"last_epoch": final_epoch + 1},
            run_name="."
        )

        try:
            for key in ("DS", "MDS", "SNR"):
                ds_stats, mds_stats, snr_stats = self._get_all_test_stats()
                ls_stats = {
                    "DS": get_ls_mse_per_folder(self.args.test_set / "DS_test_set"),
                    "MDS": get_ls_mse_per_folder(self.args.test_set / "MDS_test_set"),
                    "SNR": get_ls_mse_per_folder(self.args.test_set / "SNR_test_set")
                }

                if key == "DS":
                    stats = ds_stats
                elif key == "MDS":
                    stats = mds_stats
                else:
                    stats = snr_stats

                for val in stats.keys():
                    self.writer.add_scalars(
                        key,
                        {
                            "LS": ls_stats[key][val],
                            self.args.model_name: stats[val]
                        },
                        val
                    )
        except Exception as e:
            self.writer.add_text("Error", f"Failed to log final test results: {str(e)}")

    def _get_all_test_stats(self) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float]]:
        """Get all test statistics."""
        ds_stats = self.evaluator.get_test_stats(self.test_loaders["DS"], self.training_loss)
        mds_stats = self.evaluator.get_test_stats(self.test_loaders["MDS"], self.training_loss)
        snr_stats = self.evaluator.get_test_stats(self.test_loaders["SNR"], self.training_loss)
        return ds_stats, mds_stats, snr_stats

    def save_checkpoint(self, epoch: int, metrics: TrainingMetrics, 
                       checkpoint_dir: Optional[Path] = None) -> None:
        """Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            metrics: Current training metrics
            checkpoint_dir: Directory to save checkpoint (defaults to tensorboard log dir)
        """
        if checkpoint_dir is None:
            checkpoint_dir = self.args.tensorboard_log_dir / f"{self.args.model_name}_{self.args.exp_id}"
        
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss': metrics.train_loss,
            'val_loss': metrics.val_loss,
            'learning_rate': metrics.learning_rate,
            'system_config': self.system_config,
            'model_config': self.model_config,
            'args': self.args
        }
        
        # Save scaler state if using mixed precision
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: Path) -> int:
        """Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Epoch number of loaded checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load scaler state if it exists
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint['epoch']

    def _resume_from_checkpoint(self, checkpoint_path: Path) -> None:
        """Resume training from a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        start_epoch = self.load_checkpoint(checkpoint_path)
        self.logger.info(f"Resuming training from epoch {start_epoch}")
        
        # Update the early stopper with the best loss from checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'val_loss' in checkpoint:
            self.early_stopper.min_loss = checkpoint['val_loss']
            self.logger.info(f"Early stopper initialized with validation loss: {checkpoint['val_loss']:.4f}")

    def train(self) -> None:
        """Execute the training loop.

        Runs the complete training process including:
        - Training and validation for each epoch
        - Periodic testing based on test_every_n
        - Early stopping when validation loss plateaus
        - Logging final metrics and results
        """
        # Notify callbacks that training is beginning
        for callback in self.callbacks:
            callback.on_training_begin()
            
        last_epoch = 0
        pbar = tqdm(range(self.args.max_epoch), desc="Training")
        
        for epoch in pbar:
            last_epoch = epoch
            
            # Notify callbacks that epoch is beginning
            for callback in self.callbacks:
                callback.on_epoch_begin(epoch)
            
            # Training step
            train_loss = self.training_loop.train_epoch(self.train_loader)
            
            # Validation step
            val_loss = self.training_loop.evaluate(self.val_loader)
            
            # Create metrics object
            metrics = TrainingMetrics(
                train_loss=train_loss,
                val_loss=val_loss,
                epoch=epoch,
                learning_rate=self.optimizer.param_groups[0]['lr']
            )
            
            # Notify callbacks that epoch has ended
            for callback in self.callbacks:
                callback.on_epoch_end(epoch, metrics)

            # Update progress bar with loss info
            pbar.set_description(
                f"Epoch {epoch + 1}/{self.args.max_epoch} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

            if self.early_stopper.early_stop(val_loss):
                pbar.write(f"Early stopping triggered at epoch {epoch + 1}")
                break

            # Periodic testing
            if (epoch + 1) % self.args.test_every_n == 0:
                message = f"Test results after epoch {epoch + 1}:\n" + 50 * "-"
                pbar.write(message)
                self._run_tests(epoch)
                
        self._log_final_metrics(last_epoch)
        
        # Notify callbacks that training has ended
        for callback in self.callbacks:
            callback.on_training_end()


def train(system_config: SystemConfig, model_config: ModelConfig, args: TrainingArguments) -> None:
    """
    Train an OFDM channel estimation model.

    This is the main entry point for model training. It initializes a ModelTrainer
    with the specified configuration and runs the training process.

    Args:
        system_config: OFDM system configuration dictionary from YAML file
        model_config: OFDM model configuration dictionary from YAML file
        args: Validated training arguments containing all necessary parameters
              for model training, including dataset paths, hyperparameters,
              and logging configuration
    """
    trainer = ModelTrainer(system_config, model_config, args)
    trainer.train()
