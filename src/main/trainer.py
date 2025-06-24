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
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Tuple, Type, Union

from .parser import TrainingArguments
from src.data.dataset import MatDataset, get_test_dataloaders
from src.models import LinearEstimator, AdaFortiTranEstimator, FortiTranEstimator
from src.utils import (
    EarlyStopping,
    get_ls_mse_per_folder,
    get_model_details,
    get_test_stats_plot,
    get_error_images
)
from src.main.train_helpers import (
    get_all_test_stats,
    train_epoch,
    eval_model,
    predict_channels
)

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
        args: Training arguments
        device: PyTorch device for computation
        writer: TensorBoard SummaryWriter for logging
        model: Initialized model instance
        optimizer: Torch optimizer for training
        scheduler: Learning rate scheduler
        early_stopper: Helper for early stopping
        train_loader: DataLoader for training set
        val_loader: DataLoader for validation set
        test_loaders: Dictionary of test set DataLoaders
    """

    MODEL_REGISTRY: Dict[str, Type[ModelType]] = {
        "linear": LinearEstimator,
        "adafortitran": AdaFortiTranEstimator,
        "fortitran": FortiTranEstimator,
    }

    def __init__(self, system_config: Dict, args: TrainingArguments):
        """
        Initialize the ModelTrainer.

        Args:
            config: Model configuration dictionary from YAML file
            args: Validated training arguments
        """
        self.system_config = system_config
        self.args = args
        self.device = torch.device(f"cuda:{args.cuda}")
        self.writer = self._setup_tensorboard()

        self.model = self._initialize_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.995)
        self.early_stopper = EarlyStopping(patience=args.patience)

        self.training_loss = self._get_loss_function()
        self.comparison_loss = nn.MSELoss()  # used for test set evaluation

        self.train_loader, self.val_loader, self.test_loaders = self._get_dataloaders()

    def _get_loss_function(self) -> nn.Module:
        """Get the appropriate loss function based on arguments.

        Returns:
            The selected PyTorch loss function based on args.loss_type

        Raises:
            ValueError: If an unsupported loss type is specified
        """
        if self.args.loss_type == LossType.MSE:
            return nn.MSELoss()
        elif self.args.loss_type == LossType.MAE:
            return nn.L1Loss()
        elif self.args.loss_type == LossType.HUBER:
            return nn.HuberLoss()
        else:
            raise ValueError(f"Unsupported loss type: {self.args.loss_type}")

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
        model_class = self.MODEL_REGISTRY[self.args.model_name]
        model = model_class(self.device, self.config, vars(self.args))

        num_params, model_summary = get_model_details(model)
        print(model_summary)
        print(f"Model name: {self.config['model_name']}\nNumber of parameters: {num_params}")
        self.writer.add_text("Number of Parameters", str(num_params))

        return model

    def _get_dataloaders(self) -> Tuple[DataLoader, DataLoader, dict[str, list[tuple[str, DataLoader]]]]:
        """Initialize all required dataloaders.

        Creates DataLoader instances for:
        - Training dataset
        - Validation dataset
        - Test datasets grouped by test condition (DS, MDS, SNR)

        Returns:
            Tuple containing (train_loader, val_loader, test_loaders_dict)
        """
        # Training and validation dataloaders
        train_dataset = MatDataset(
            self.args.train_set,
            self.args.pilot_dims,
            return_type=self.config["return_type"]
        )
        val_dataset = MatDataset(
            self.args.val_set,
            self.args.pilot_dims,
            return_type=self.config["return_type"]
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

        # Test dataloaders
        test_loaders = {
            "DS": get_test_dataloaders(
                self.args.test_set / "DS_test_set",
                vars(self.args),
                self.config["return_type"]
            ),
            "MDS": get_test_dataloaders(
                self.args.test_set / "MDS_test_set",
                vars(self.args),
                self.config["return_type"]
            ),
            "SNR": get_test_dataloaders(
                self.args.test_set / "SNR_test_set",
                vars(self.args),
                self.config["return_type"]
            ),
        }

        return train_loader, val_loader, test_loaders

    def _log_test_results(
            self,
            epoch: int,
            test_stats: Dict[str, Dict],
            ls_stats: Dict[str, Dict]
    ) -> None:
        """Log test results to TensorBoard.

        Creates and logs visualizations comparing model performance against
        baseline LS estimator across different test conditions.

        Args:
            epoch: Current training epoch
            test_stats: Dictionary of test statistics for the model
            ls_stats: Dictionary of test statistics for the LS baseline
        """
        for key in ("DS", "MDS", "SNR"):
            # Plot test statistics
            self.writer.add_figure(
                tag=f"MSE vs. {key} (Epoch:{epoch + 1})",
                figure=get_test_stats_plot(
                    x_name=key,
                    stats=[test_stats[key], ls_stats[key]],
                    methods=[self.config["model_name"], "LS"]
                )
            )

            # Plot error images
            predicted_channels = predict_channels(
                self.model,
                self.test_loaders[key]
            )
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

        Evaluates the model on all test datasets, compares with LS baseline,
        and logs performance metrics and visualizations.

        Args:
            epoch: Current training epoch
        """
        ds_stats, mds_stats, snr_stats = get_all_test_stats(
            self.model,
            self.test_loaders,
            self.comparison_loss
        )

        ls_stats = {
            "DS": get_ls_mse_per_folder(self.args.test_set / "DS_test_set"),
            "MDS": get_ls_mse_per_folder(self.args.test_set / "MDS_test_set"),
            "SNR": get_ls_mse_per_folder(self.args.test_set / "SNR_test_set")
        }

        test_stats = {
            "DS": ds_stats,
            "MDS": mds_stats,
            "SNR": snr_stats
        }

        self._log_test_results(epoch, test_stats, ls_stats)

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
                ds_stats, mds_stats, snr_stats = get_all_test_stats(
                    self.model,
                    self.test_loaders,
                    self.comparison_loss
                )
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
                            self.config["model_name"]: stats[val]
                        },
                        val
                    )
        except Exception as e:
            self.writer.add_text("Error", f"Failed to log final test results: {str(e)}")

    def train(self) -> None:
        """Execute the training loop.

        Runs the complete training process including:
        - Training and validation for each epoch
        - Periodic testing based on test_every_n
        - Early stopping when validation loss plateaus
        - Logging final metrics and results
        """
        try:
            from tqdm import tqdm
            use_tqdm = True
        except ImportError:
            use_tqdm = False
            print("tqdm not found, progress bar will not be displayed")

        epoch = None

        # Create progress bar if tqdm is available
        if use_tqdm:
            pbar = tqdm(range(self.args.max_epoch), desc="Training")
        else:
            pbar = range(self.args.max_epoch)

        for epoch in pbar:
            # Training step
            train_loss = train_epoch(
                self.model,
                self.optimizer,
                self.training_loss,
                self.scheduler,
                self.train_loader
            )
            self.writer.add_scalar('Loss/Train', train_loss, epoch + 1)

            # Validation step
            val_loss = eval_model(self.model, self.val_loader, self.training_loss)
            self.writer.add_scalar('Loss/Val', val_loss, epoch + 1)

            # Update progress bar with loss info if tqdm is available
            if use_tqdm:
                pbar.set_description(
                    f"Epoch {epoch + 1}/{self.args.max_epoch} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            if self.early_stopper.early_stop(val_loss):
                if use_tqdm:
                    pbar.write(f"Early stopping triggered at epoch {epoch + 1}")
                else:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                break

            # Periodic testing
            if (epoch + 1) % self.args.test_every_n == 0:
                message = f"Test results after epoch {epoch + 1}:\n" + 50 * "-"
                if use_tqdm:
                    pbar.write(message)
                else:
                    print(message)
                self._run_tests(epoch)

        self._log_final_metrics(epoch)
        self.writer.close()


def train(config: Dict, args: TrainingArguments) -> None:
    """
    Train an OFDM channel estimation model.

    This is the main entry point for model training. It initializes a ModelTrainer
    with the specified configuration and runs the training process.

    Args:
        config: Model configuration dictionary loaded from YAML file,
                containing model architecture and training parameters
        args: Validated training arguments containing all necessary parameters
              for model training, including dataset paths, hyperparameters,
              and logging configuration
    """
    trainer = ModelTrainer(config, args)
    trainer.train()
