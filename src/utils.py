"""
Utility functions for OFDM channel estimation.

This module provides various utility functions for processing, visualizing,
and analyzing OFDM channel estimation data, including complex channel matrices,
error calculations, model statistics, and visualization tools for
performance evaluation.
"""

from pathlib import Path
from typing import Optional, Union
import re
import os

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import torch


class EarlyStopping:
    """Handles early stopping logic for training.

    Monitors validation loss during training and signals when to stop
    training if the loss has not improved for a specified number of epochs.

    Attributes:
        patience: Number of epochs to wait before stopping training
        remaining_patience: Current remaining patience counter
        min_loss: Minimum validation loss observed so far
    """

    def __init__(self, patience: int = 3):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping
        """
        self.patience = patience
        self.remaining_patience = patience
        self.min_loss: Optional[float] = None

    def early_stop(self, loss: float) -> bool:
        """
        Check if training should stop.

        Args:
            loss: Current validation loss

        Returns:
            Whether to stop training
        """
        if self.min_loss is None:
            self.min_loss = loss
            return False

        if loss < self.min_loss:
            self.min_loss = loss
            self.remaining_patience = self.patience
            return False

        self.remaining_patience -= 1
        return self.remaining_patience == 0


def extract_values(file_name):
    """
    Extract channel information from a file name.

    Parses file names with format:
    '{file_number}_SNR-{snr}_DS-{delay_spread}_DOP-{doppler}_N-{pilot_freq}_{channel_type}.mat'

    Example:
        For filename "1_SNR-20_DS-50_DOP-500_N-3_TDL-A.mat":
        - file_number: 1
        - snr: 20 (Signal-to-Noise Ratio in dB)
        - delay_spread: 50 (Delay Spread)
        - doppler: 500 (Maximum Doppler Shift)
        - pilot_freq: 3 (Pilot placement frequency)
        - channel_type: TDL-A (Channel model type)

    Args:
        file_name: The file name from which to extract channel information

    Returns:
        tuple: A tuple containing:
            - file_number (torch.Tensor): The file number (sequential identifier)
            - snr (torch.Tensor): Signal-to-noise ratio value in dB
            - delay_spread (torch.Tensor): Delay spread value
            - max_doppler_shift (torch.Tensor): Maximum Doppler shift value
            - pilot_placement_frequency (torch.Tensor): Pilot placement frequency
            - channel_type (list): The channel type (e.g., ['TDL-A'])

    Raises:
        ValueError: If the file name does not match the expected pattern
    """
    pattern = r'(\d+)_SNR-(\d+)_DS-(\d+)_DOP-(\d+)_N-(\d+)_([A-Z\-]+)\.mat'
    match = re.match(pattern, file_name)
    if match:
        file_no = torch.tensor([int(match.group(1))], dtype=torch.float)
        snr_value = torch.tensor([int(match.group(2))], dtype=torch.float)
        ds_value = torch.tensor([int(match.group(3))], dtype=torch.float)
        dop_value = torch.tensor([int(match.group(4))], dtype=torch.float)
        n = torch.tensor([int(match.group(5))], dtype=torch.float)
        channel_type = [match.group(6)]
        return file_no, snr_value, ds_value, dop_value, n, channel_type
    else:
        raise ValueError("Cannot extract file information.")


def get_error_images(variable, channel_data, show=False):
    """
    Create visualizations of channel estimation errors.

    Generates a figure with error heatmaps for different channel conditions,
    showing the absolute difference between estimated and ideal channels.

    Args:
        variable: Name of the variable being visualized (e.g., 'SNR', 'DS')
        channel_data: Dictionary mapping parameter values to dictionaries
                     containing 'estimated_channel' and 'ideal_channel'
        show: Whether to display the figure immediately (default: False)

    Returns:
        matplotlib.figure.Figure: The generated figure with error heatmaps
    """
    # Create a figure with 7 subplots
    fig, axes = plt.subplots(1, len(channel_data), figsize=(20, 6))

    # Plot each subplot with consistent color scaling
    for i, (key, channels) in enumerate(channel_data.items()):
        # Calculate absolute error between estimated and ideal channels

        estimated_channel = channels['estimated_channel']
        ideal_channel = channels['ideal_channel']

        error_matrix = torch.abs(estimated_channel - ideal_channel)
        error_numpy = error_matrix.detach().cpu().numpy()

        # Plot in the corresponding subplot with shared colormap limits
        ax = axes[i]
        cax = ax.imshow(error_numpy, cmap='viridis', aspect=14 / 120, vmin=0, vmax=1)
        ax.set_title(f"{variable} = {key}")
        ax.set_xlabel('Columns (14)')
        ax.set_ylabel('Rows (120)')

    # Create a new axis for the color bar to the right of the subplots
    cbar_ax = fig.add_axes((0.92, 0.15, 0.02, 0.7))  # [left, bottom, width, height]
    fig.colorbar(cax, cax=cbar_ax, label='Error Magnitude')

    # Adjust layout to prevent overlapping labels
    fig.tight_layout(rect=(0, 0, 0.9, 1))  # Leave space for the color bar on the right

    # Show the figure if `show` is True
    if show:
        plt.show()

    # Return the main figure
    return fig


def concat_complex_channel(channel_matrix):
    """
    Convert a complex channel matrix into a real matrix by concatenating real and imaginary parts.

    Transforms a complex tensor into a real-valued tensor by concatenating
    the real and imaginary components along the specified dimension.

    Args:
        channel_matrix: Complex channel matrix

    Returns:
        Real-valued channel matrix with concatenated real and imaginary parts
    """
    real_channel_m = torch.real(channel_matrix)
    imag_channel_m = torch.imag(channel_matrix)
    cat_channel_m = torch.cat((real_channel_m, imag_channel_m), dim=1)
    return cat_channel_m





def get_test_stats_plot(x_name, stats, methods, show=False):
    """
    Plot test statistics for multiple methods as line graphs.

    Creates a line plot comparing performance metrics (e.g., MSE) across
    different conditions or parameters for multiple methods.

    Args:
        x_name: Label for the x-axis (e.g., 'SNR', 'DS', 'Epoch')
        stats: List of dictionaries where each dictionary maps x-values to
               performance metrics for a specific method
        methods: List of method names corresponding to each entry in stats
        show: Whether to display the plot immediately (default: False)

    Returns:
        matplotlib.figure.Figure: The generated figure object

    Raises:
        AssertionError: If stats and methods lists have different lengths
    """
    assert len(stats) == len(methods), "Provided stats and methods do not have the same length."
    fig = plt.figure()
    symbols = iter(["*", "x", "+", "D", "v", "^"])
    for stat in stats:
        try:
            symbol = next(symbols)
        except StopIteration:
            symbols = iter(["o", "*", "x", "+", "D", "v", "^"])
            symbol = next(symbols)

        kv_pairs = sorted(list(stat.items()), key=lambda x: x[0])
        x_vals = []
        y_vals = []
        for key, value in kv_pairs:
            x_vals.append(key)
            y_vals.append(value)

        plt.plot(x_vals, y_vals, f"{symbol}--")
        plt.xlabel(x_name)
        plt.ylabel("MSE Error (dB)")
        plt.grid()
    plt.legend(methods)
    if show:
        plt.show()
    return fig


def to_db(val):
    """
    Convert values to decibels (dB).

    Applies the formula 10 * log10(val) to convert values to the decibel scale.

    Args:
        val: Input value or array to convert to dB (must be positive)

    Returns:
        The input value(s) converted to decibels
    """
    return 10 * np.log10(val)


def mse(x, y):
    """
    Calculate mean squared error (MSE) in dB between two complex arrays.

    Computes the average squared magnitude of the difference between
    two complex arrays and converts the result to decibels.

    Args:
        x: First complex numpy array
        y: Second complex numpy array (same shape as x)

    Returns:
        MSE in decibels (dB) between the two arrays
    """
    mse_xy = np.mean(np.square(np.abs(x - y)))
    mse_xy_db = to_db(mse_xy)
    return mse_xy_db


def get_ls_mse_per_folder(folders_dir: Union[Path, str]):
    """
    Calculate average MSE for LS estimates in each subfolder.

    For each subfolder in the specified directory, calculates the average
    mean squared error between least-squares channel estimates and ideal
    channel values across all .mat files in that subfolder.

    Args:
        folders_dir: Path to directory containing subfolders with .mat files
                    Each subfolder should be named 'prefix_val' where val is an integer

    Returns:
        Dictionary mapping integer values from subfolder names to average MSE values in dB

    Notes:
        - Each .mat file should contain a 3D matrix 'H' where:
          - H[:,:,0] is the ideal channel
          - H[:,:,2] is the LS channel estimate
        - Subfolders are sorted by the integer in their names
    """

    mse_sums = {}
    folders = os.listdir(folders_dir)
    folders = sorted(folders, key=lambda x: int(x.split("_")[1]))
    for folder in folders:
        _, val = folder.split("_")
        mse_sum = 0
        folder_size = len(os.listdir(os.path.join(folders_dir, folder)))
        for file in os.listdir(os.path.join(folders_dir, folder)):
            mat_data = sio.loadmat(os.path.join(folders_dir, folder, file))['H']
            ls_estimate = mat_data[:, :, 2]
            ideal = mat_data[:, :, 0]
            mse_sum += mse(ls_estimate, ideal)
        mse_sum /= folder_size
        mse_sums[int(val)] = mse_sum
    return mse_sums

def get_model_details(model):
    """
    Get parameter counts and structure details for a PyTorch model.

    Analyzes a PyTorch model to determine the total number of trainable
    parameters and creates a formatted table showing the parameter count
    for each named parameter in the model.

    Args:
        model: PyTorch model to analyze

    Returns:
        tuple containing:
            - total_params: Total number of trainable parameters
            - table: PrettyTable showing parameter counts by module
    """

    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    return total_params, table
