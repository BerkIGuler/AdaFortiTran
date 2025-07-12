"""
Training helper functions for OFDM channel estimation models.

This module provides utility functions for training, evaluating, and testing
deep learning models for OFDM channel estimation tasks. It includes functions
for performing training epochs, model evaluation, prediction generation,
and performance statistics calculation across different test conditions.
"""

from typing import Dict, List, Tuple, Union, Callable
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ExponentialLR
from src.utils import to_db, concat_complex_channel

# Type aliases
ComplexTensor = torch.Tensor  # Complex tensor
BatchType = Tuple[ComplexTensor, ComplexTensor, Union[Dict, None]]
TestDataLoadersType = List[Tuple[str, DataLoader]]
StatsType = Dict[int, float]


def get_all_test_stats(
        model: nn.Module,
        test_dataloaders: Dict[str, TestDataLoadersType],
        loss_fn: Callable
) -> Tuple[StatsType, StatsType, StatsType]:
    """
    Evaluate model on all test datasets.

    Calculates performance statistics (MSE in dB) for a model across different
    test conditions: Delay Spread (DS), Max Doppler Shift (MDS), and
    Signal-to-Noise Ratio (SNR).

    Args:
        model: Model to evaluate
        test_dataloaders: Dictionary containing DataLoader objects for test sets:
            - "DS": Delay Spread test set
            - "MDS": Max Doppler Shift test set
            - "SNR": Signal-to-Noise Ratio test set
        loss_fn: Loss function for evaluation

    Returns:
        Tuple containing statistics (MSE in dB) for DS, MDS, and SNR test sets,
        where each set of statistics is a dictionary mapping parameter values to MSE
    """
    ds_stats = get_test_stats(model, test_dataloaders["DS"], loss_fn)
    mds_stats = get_test_stats(model, test_dataloaders["MDS"], loss_fn)
    snr_stats = get_test_stats(model, test_dataloaders["SNR"], loss_fn)
    return ds_stats, mds_stats, snr_stats


def get_test_stats(
        model: nn.Module,
        test_dataloaders: TestDataLoadersType,
        loss_fn: Callable
) -> StatsType:
    """
    Evaluate model on provided test dataloaders.

    Calculates performance statistics (MSE in dB) for a model on a
    specific set of test conditions.

    Args:
        model: Model to evaluate
        test_dataloaders: List of (name, DataLoader) tuples for test sets,
                         where names are in format "parameter_value"
        loss_fn: Loss function for evaluation

    Returns:
        Dictionary mapping test parameter values (as integers) to MSE values in dB
    """
    stats: StatsType = {}
    sorted_loaders = sorted(
        test_dataloaders,
        key=lambda x: int(x[0].split("_")[1])
    )

    for name, test_dataloader in sorted_loaders:
        var, val = name.split("_")
        test_loss = eval_model(model, test_dataloader, loss_fn)
        db_error = to_db(test_loss)
        print(f"{var}:{val} Test MSE: {db_error:.4f} dB")
        stats[int(val)] = db_error

    return stats


def eval_model(
        model: nn.Module,
        eval_dataloader: DataLoader,
        loss_fn: Callable
) -> float:
    """
    Evaluate model on given dataloader.

    Calculates the average loss for a model on a dataset without
    performing parameter updates.

    Args:
        model: Model to evaluate
        eval_dataloader: DataLoader containing evaluation data
        loss_fn: Loss function for computing error

    Returns:
        Average validation loss (adjusted for complex values)

    Notes:
        Loss is multiplied by 2 to account for complex-valued matrices being
        represented as real-valued matrices of double size.
    """
    val_loss = 0.0
    model.eval()

    with torch.no_grad():
        for batch in eval_dataloader:
            estimated_channel, ideal_channel = _forward_pass(batch, model)
            output = _compute_loss(estimated_channel, ideal_channel, loss_fn)
            val_loss += (2 * output.item() * batch[0].size(0))

    val_loss /= sum(len(batch[0]) for batch in eval_dataloader)
    return val_loss


def predict_channels(
        model: nn.Module,
        test_dataloaders: TestDataLoadersType
) -> Dict[int, Dict[str, ComplexTensor]]:
    """
    Generate channel predictions for test datasets.

    Creates predictions for a sample from each test dataset to enable
    visualization and error analysis.

    Args:
        model: Model to use for predictions
        test_dataloaders: List of (name, DataLoader) tuples for test sets,
                         where names are in format "parameter_value"

    Returns:
        Dictionary mapping test parameter values (as integers) to dictionaries containing
        estimated and ideal channels for a single sample
    """
    channels: Dict[int, Dict[str, ComplexTensor]] = {}
    sorted_loaders = sorted(
        test_dataloaders,
        key=lambda x: int(x[0].split("_")[1])
    )

    for name, test_dataloader in sorted_loaders:
        with torch.no_grad():
            batch = next(iter(test_dataloader))
            estimated_channels, ideal_channels = _forward_pass(batch, model)

        var, val = name.split("_")
        channels[int(val)] = {
            "estimated_channel": estimated_channels[0],
            "ideal_channel": ideal_channels[0]
        }

    return channels


def train_epoch(
        model: nn.Module,
        optimizer: Optimizer,
        loss_fn: Callable,
        scheduler: ExponentialLR,
        train_dataloader: DataLoader
) -> float:
    """
    Train model for one epoch.

    Performs a complete training iteration over the dataset, including:
    - Forward pass through the model
    - Loss calculation
    - Backpropagation
    - Parameter updates
    - Learning rate scheduling

    Args:
        model: Model to train
        optimizer: Optimizer for updating model parameters
        loss_fn: Loss function for computing error
        scheduler: Learning rate scheduler
        train_dataloader: DataLoader containing training data

    Returns:
        Average training loss for the epoch (adjusted for complex values)

    Notes:
        Loss is multiplied by 2 to account for complex-valued matrices being
        represented as real-valued matrices of double size.
    """
    train_loss = 0.0
    model.train()

    for batch in train_dataloader:
        optimizer.zero_grad()
        estimated_channel, ideal_channel = _forward_pass(batch, model)
        output = _compute_loss(estimated_channel, ideal_channel, loss_fn)
        output.backward()
        optimizer.step()
        train_loss += (2 * output.item() * batch[0].size(0))

    scheduler.step()
    train_loss /= sum(len(batch[0]) for batch in train_dataloader)
    return train_loss


def _forward_pass(batch: BatchType, model: nn.Module) -> Tuple[ComplexTensor, ComplexTensor]:
    """
    Perform forward pass through model.

    Processes input data through the appropriate model based on its type,
    handling different input requirements for different model architectures.

    Args:
        batch: Tuple containing (estimated_channel, ideal_channel, metadata)
        model: Model to use for processing

    Returns:
        Tuple of (processed_estimated_channel, ideal_channel)

    Raises:
        ValueError: If model type is not recognized
    """
    estimated_channel, ideal_channel, meta_data = batch

    # All models now handle complex input directly
    if hasattr(model, 'use_channel_adaptation') and model.use_channel_adaptation:
        # AdaFortiTran uses meta_data for channel adaptation
        estimated_channel = model(estimated_channel, meta_data)
    else:
        # Linear and FortiTran models don't use meta_data
        estimated_channel = model(estimated_channel)

    return estimated_channel, ideal_channel.to(model.device)


def _compute_loss(
        estimated_channel: ComplexTensor,
        ideal_channel: ComplexTensor,
        loss_fn: Callable
) -> torch.Tensor:
    """
    Calculate loss between estimated and ideal channels.

    Computes the loss between model output and ground truth using the specified
    loss function, with appropriate handling of complex values.

    Args:
        estimated_channel: Estimated channel from model
        ideal_channel: Ground truth ideal channel
        loss_fn: Loss function to compute error

    Returns:
        Computed loss value as a scalar tensor
    """
    return loss_fn(
        concat_complex_channel(estimated_channel),
        concat_complex_channel(ideal_channel)
    )
