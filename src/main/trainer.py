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
from typing import Dict, Tuple, Type, Union
import logging
from tqdm import tqdm

from .parser import TrainingArguments
from src.data.dataset import MatDataset, get_test_dataloaders
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
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.EXP_LR_GAMMA)
        self.early_stopper = EarlyStopping(patience=args.patience)

        self.training_loss = nn.MSELoss()

        self.train_loader, self.val_loader, self.test_loaders = self._get_dataloaders()

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
        pilot_dims = [self.system_config.pilot.num_scs, self.system_config.pilot.num_symbols]
        # Training and validation dataloaders
        train_dataset = MatDataset(
            self.args.train_set,
            pilot_dims
        )
        val_dataset = MatDataset(
            self.args.val_set,
            pilot_dims
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            shuffle=True
        )
        test_loaders = {
            "DS": get_test_dataloaders(
                self.args.test_set / "DS_test_set",
                pilot_dims,
                self.args.batch_size
            ),
            "MDS": get_test_dataloaders(
                self.args.test_set / "MDS_test_set",
                pilot_dims,
                self.args.batch_size
            ),
            "SNR": get_test_dataloaders(
                self.args.test_set / "SNR_test_set",
                pilot_dims,
                self.args.batch_size
            ),
        }
        return train_loader, val_loader, test_loaders

    def _log_test_results(
            self,
            epoch: int,
            test_stats: Dict[str, Dict]
    ) -> None:
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
            predicted_channels = self._predict_channels(self.test_loaders[key])
            self.writer.add_figure(
                tag=f"{key} Error Images (Epoch:{epoch + 1})",
                figure=get_error_images(
                    key,
                    predicted_channels,
                    show=False
                )
            )

    def _run_tests(self, epoch: int) -> None:
        """Run tests and log results.

        Evaluates the model on all test datasets and logs performance metrics and visualizations.

        Args:
            epoch: Current training epoch
        """
        ds_stats, mds_stats, snr_stats = self._get_all_test_stats()

        test_stats = {
            "DS": ds_stats,
            "MDS": mds_stats,
            "SNR": snr_stats
        }

        self._log_test_results(epoch, test_stats)

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

    def _compute_loss(self, estimated_channel, ideal_channel, loss_fn):
        return loss_fn(
            concat_complex_channel(estimated_channel),
            concat_complex_channel(ideal_channel)
        )

    def _forward_pass(self, batch, model):
        estimated_channel, ideal_channel, meta_data = batch
        
        # All models now handle complex input directly
        if isinstance(model, AdaFortiTranEstimator):
            # AdaFortiTran uses meta_data for channel adaptation
            estimated_channel = model(estimated_channel, meta_data)
        else:
            # Linear and FortiTran models don't use meta_data
            estimated_channel = model(estimated_channel)
            
        return estimated_channel, ideal_channel.to(model.device)

    def _train_epoch(self):
        train_loss = 0.0
        self.model.train()
        num_samples = 0
        for batch in self.train_loader:
            self.optimizer.zero_grad()
            estimated_channel, ideal_channel = self._forward_pass(batch, self.model)
            output = self._compute_loss(estimated_channel, ideal_channel, self.training_loss)
            output.backward()
            self.optimizer.step()
            batch_size = batch[0].size(0)
            train_loss += (2 * output.item() * batch_size)
            num_samples += batch_size
        self.scheduler.step()
        train_loss /= num_samples
        return train_loss

    def _eval_model(self, eval_dataloader):
        val_loss = 0.0
        self.model.eval()
        num_samples = 0
        with torch.no_grad():
            for batch in eval_dataloader:
                estimated_channel, ideal_channel = self._forward_pass(batch, self.model)
                output = self._compute_loss(estimated_channel, ideal_channel, self.training_loss)
                batch_size = batch[0].size(0)
                val_loss += (2 * output.item() * batch_size)
                num_samples += batch_size
        val_loss /= num_samples
        return val_loss

    def _predict_channels(self, test_dataloaders):
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

    def _get_test_stats(self, test_dataloaders):
        stats = {}
        sorted_loaders = sorted(
            test_dataloaders,
            key=lambda x: int(x[0].split("_")[1])
        )
        for name, test_dataloader in sorted_loaders:
            var, val = name.split("_")
            test_loss = self._eval_model(test_dataloader)
            db_error = to_db(test_loss)
            self.logger.info(f"{var}:{val} Test MSE: {db_error:.4f} dB")
            stats[int(val)] = db_error
        return stats

    def _get_all_test_stats(self):
        ds_stats = self._get_test_stats(self.test_loaders["DS"])
        mds_stats = self._get_test_stats(self.test_loaders["MDS"])
        snr_stats = self._get_test_stats(self.test_loaders["SNR"])
        return ds_stats, mds_stats, snr_stats

    def train(self) -> None:
        """Execute the training loop.

        Runs the complete training process including:
        - Training and validation for each epoch
        - Periodic testing based on test_every_n
        - Early stopping when validation loss plateaus
        - Logging final metrics and results
        """
        last_epoch = 0
        pbar = tqdm(range(self.args.max_epoch), desc="Training")
        for epoch in pbar:
            last_epoch = epoch
            # Training step
            train_loss = self._train_epoch()
            self.writer.add_scalar('Loss/Train', train_loss, epoch + 1)

            # Validation step
            val_loss = self._eval_model(self.val_loader)
            self.writer.add_scalar('Loss/Val', val_loss, epoch + 1)

            # Update progress bar with loss info
            pbar.set_description(
                f"Epoch {epoch + 1}/{self.args.max_epoch} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            if self.early_stopper.early_stop(val_loss):
                pbar.write(f"Early stopping triggered at epoch {epoch + 1}")
                break

            # Periodic testing
            if (epoch + 1) % self.args.test_every_n == 0:
                message = f"Test results after epoch {epoch + 1}:\n" + 50 * "-"
                pbar.write(message)
                self._run_tests(epoch)
        self._log_final_metrics(last_epoch)
        self.writer.close()


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
