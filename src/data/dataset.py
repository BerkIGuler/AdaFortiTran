"""Module for loading and processing .mat files containing channel matrices for PyTorch.

This module expects .mat files with a specific naming convention and internal structure:

File Naming Convention:
    Files must follow the pattern: {file_number}_SNR-{snr}_DS-{delay_spread}_DOP-{doppler}_N-{pilot_freq}_{channel_type}.mat
    
    Example: 1_SNR-20_DS-50_DOP-500_N-3_TDL-A.mat
    - file_number: Sequential file identifier
    - SNR: Signal-to-Noise Ratio in dB
    - DS: Delay Spread in nanoseconds
    - DOP: Maximum Doppler Shift in Hz
    - pilot_freq: Pilot placement frequency (i.e. every Nth subcarrier is a pilot)
    - channel_type: Channel model type (e.g., TDL-A, TDL-B, etc.)

File Content Structure:
    Each .mat file must contain a variable 'H' with shape [subcarriers, symbols, 2]:
    - H[:, :, 0]: Complex valued ground truth channel
    - H[:, :, 1]: Complex valued least squares (LS) channel estimate at pilot positions with zeros at non-pilot positions (used as input to models)

The dataset extracts pilot values from the LS estimates and provides metadata from the filename for adaptive channel estimation models.
"""
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader

from src.config.schemas import PilotParams
from src.utils import extract_values

__all__ = ['MatDataset', 'get_test_dataloaders']


class MatDataset(Dataset):
    """Dataset for loading and formatting .mat files containing channel matrices.

    Processes .mat files containing channel estimation data and converts them into
    PyTorch complex tensors for channel estimation tasks.

    Expected File Format:
        - Files must be named according to the pattern: 
          {file_number}_SNR-{snr}_DS-{delay_spread}_DOP-{doppler}_N-{pilot_freq}_{channel_type}.mat
        - Each .mat file must contain a variable 'H' with shape [subcarriers, symbols, 2]
        - H[:, :, 0]: complex valued ground truth channel matrix
        - H[:, :, 1]: complex valued least-squares (LS) channel estimate at pilot positions with zeros at non-pilot positions

    Returns:
        For each sample, returns a tuple of:
        - LS channel estimate at pilot positions(complex tensor, shape [pilot_subcarriers, pilot_symbols])
        - Ground truth channel estimate (complex tensor, shape [ofdm_subcarriers, ofdm_symbols])
        - Metadata tuple: (file_number, snr, delay_spread, doppler, pilot_freq, channel_type)
    """

    def __init__(
            self,
            data_dir: Union[str, Path],
            pilot_params: PilotParams,
            transform: Optional[Callable] = None
    ) -> None:
        """Initialize the MatDataset.

        Args:
            data_dir: Path to the directory containing the dataset (should contain .mat files).
            pilot_params: PilotParams object containing validated pilot dimensions.
            transform: Optional transformation to apply to samples.

        Raises:
            FileNotFoundError: If data_dir doesn't exist.
            ValueError: If no .mat files are found in data_dir.
        """
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)
        
        self.data_dir = data_dir
        self.pilot_params = pilot_params
        self.transform = transform

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        self.file_list = list(self.data_dir.glob("*.mat"))
        if not self.file_list:
            raise ValueError(f"No .mat files found in {self.data_dir}")

    def __len__(self) -> int:
        """Return the total number of files in the dataset.

        Returns:
            Integer count of .mat files in the dataset directory
        """
        return len(self.file_list)

    def _process_channel_data(
            self,
            mat_data: dict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process channel data and extract pilot values from sparse (pilot positions only) LS-estimated channel matrix.

        Extracts pilot values (non-zero entries) from LS channel estimates,
        returning complex-valued tensors for both estimate and ground truth.

        Args:
            mat_data: Loaded .mat file data containing 'H' variable with shape [subcarriers, symbols, 3]

        Returns:
            Tuple of (pilot LS estimate at pilot positions (complex tensor, shape [pilot_scs, pilot_symbols]),
                      ground truth channel matrix (complex tensor, shape [ofdm_scs, ofdm_symbols]))

        Raises:
            ValueError: If the data format is unexpected or processing fails
        """
        try:
            # Extract ground truth channel
            h_ideal = torch.tensor(mat_data['H'][:, :, 0], dtype=torch.cfloat)
            
            # LS channel estimate at pilot positions with zeros at non-pilot positions
            hzero_ls = torch.tensor(mat_data['H'][:, :, 1], dtype=torch.cfloat)

            # Remove zero entries, keep only channel at pilot values
            zero_complex = torch.complex(torch.tensor(0.0), torch.tensor(0.0))
            # Boolean masking extracts in row-major order: all OFDM symbols for subcarrier 0, then subcarrier 1, etc.
            # Extracted order: [sc0_sym0, sc0_sym1, ..., sc0_symN, sc1_sym0, sc1_sym1, ..., sc1_symN, ...]
            hp_ls = hzero_ls[hzero_ls != zero_complex]

            # Validate expected number of pilot values (number of non-zero entries)
            expected_pilots = self.pilot_params.num_scs * self.pilot_params.num_symbols
            if hp_ls.numel() != expected_pilots:
                raise ValueError(
                    f"Expected {expected_pilots} pilot values, got {hp_ls.numel()}"
                )

            # Reshape to pilot grid dimensions [subcarriers, symbols]
            # Row-major extraction matches [subcarriers x symbols] layout directly
            hp_ls = hp_ls.view(
                self.pilot_params.num_scs,
                self.pilot_params.num_symbols
            )

            return hp_ls, h_ideal

        except Exception as e:
            raise ValueError(f"Error processing channel data: {str(e)}")

    def __getitem__(
            self,
            idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple]:
        """Load and process a .mat file at the given index.

        Args:
            idx: Index of the file to load.

        Returns:
            Tuple containing:
                - Pilot LS channel estimate (complex tensor, shape [pilot_subcarriers, pilot_symbols])
                - Ground truth channel estimate (complex tensor, shape [ofdm_subcarriers, ofdm_symbols])
                - Metadata tuple: (file_number, snr, delay_spread, doppler, pilot_freq, channel_type)
                  All metadata values are torch.Tensor except channel_type which is a list

        Raises:
            ValueError: If file format is invalid or processing fails.
            IndexError: If idx is out of range.
        """
        if not 0 <= idx < len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")

        try:
            mat_data = sio.loadmat(self.file_list[idx])
            if 'H' not in mat_data or mat_data['H'].shape[-1] < 2:
                raise ValueError("Invalid .mat file format: missing required data")

            # Process channel data to extract pilot estimates
            h_est, h_ideal = self._process_channel_data(mat_data)

            # Extract metadata from filename
            meta_data = extract_values(self.file_list[idx].name)
            if meta_data is None:
                raise ValueError(f"Unrecognized filename format: {self.file_list[idx].name}")

            # Apply optional transforms
            if self.transform:
                h_est = self.transform(h_est)
                h_ideal = self.transform(h_ideal)

            return h_est, h_ideal, meta_data

        except Exception as e:
            raise ValueError(f"Error processing file {self.file_list[idx]}: {str(e)}")


def get_test_dataloaders(
        dataset_dir: Union[str, Path],
        pilot_params: PilotParams,
        batch_size: int
) -> List[Tuple[str, DataLoader]]:
    """Create DataLoaders for each subdirectory in the dataset directory.

    Automatically discovers and creates appropriate DataLoader instances for
    all subdirectories in the specified dataset directory, useful for testing
    across multiple test conditions or scenarios.

    Expected Directory Structure:
        dataset_dir/
        ├── DS_50/          # Delay Spread = 50
        │   ├── 1_SNR-20_DS-50_DOP-500_N-3_TDL-A.mat
        │   ├── 2_SNR-20_DS-50_DOP-500_N-3_TDL-A.mat
        │   └── ...
        ├── DS_100/         # Delay Spread = 100
        │   ├── 1_SNR-20_DS-100_DOP-500_N-3_TDL-A.mat
        │   └── ...
        ├── SNR_10/         # SNR = 10 dB
        │   ├── 1_SNR-10_DS-50_DOP-500_N-3_TDL-A.mat
        │   └── ...
        └── ...

    Each subdirectory should contain .mat files with the naming convention:
    {file_number}_SNR-{snr}_DS-{delay_spread}_DOP-{doppler}_N-{pilot_freq}_{channel_type}.mat

    Args:
        dataset_dir: Path to main directory containing dataset subdirectories
        pilot_params: PilotParams object containing validated pilot dimensions
        batch_size: Number of samples per batch

    Returns:
        List of tuples containing (subdirectory_name, corresponding_dataloader)

    Raises:
        FileNotFoundError: If dataset_dir doesn't exist
        ValueError: If no valid subdirectories are found
    """
    if isinstance(dataset_dir, str):
        dataset_dir = Path(dataset_dir)
    
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    subdirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
    if not subdirs:
        raise ValueError(f"No subdirectories found in {dataset_dir}")

    test_datasets = [
        (
            subdir.name,
            MatDataset(
                subdir,
                pilot_params
            )
        )
        for subdir in subdirs
    ]

    return [
        (name, DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,  # no shuffling for testing
            num_workers=0
        ))
        for name, dataset in test_datasets
    ]