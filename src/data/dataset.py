"""Module for loading and processing .mat files containing channel estimates for PyTorch."""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader

from src.utils import extract_values

__all__ = ['MatDataset', 'get_test_dataloaders']


@dataclass
class PilotDimensions:
    """Container for pilot signal dimensions.

    Stores and validates the dimensions of pilot signals used in channel estimation.

    Attributes:
        num_subcarriers: Number of subcarriers in the pilot signal
        num_ofdm_symbols: Number of OFDM symbols in the pilot signal
    """
    num_subcarriers: int
    num_ofdm_symbols: int

    def __post_init__(self):
        """Validate dimensions after initialization.

        Raises:
            ValueError: If either dimension is not a positive integer
        """
        if self.num_subcarriers <= 0 or self.num_ofdm_symbols <= 0:
            raise ValueError("Pilot dimensions must be positive integers")

    def as_tuple(self) -> Tuple[int, int]:
        """Return dimensions as a tuple.

        Returns:
            Tuple of (num_subcarriers, num_ofdm_symbols)
        """
        return self.num_subcarriers, self.num_ofdm_symbols


class MatDataset(Dataset):
    """Dataset for loading and formatting .mat files containing channel estimates.

    Processes .mat files containing channel estimation data and converts them into
    PyTorch complex tensors for channel estimation tasks.
    """

    def __init__(
            self,
            data_dir: Union[str, Path],
            pilot_dims: List[int],
            transform: Optional[Callable] = None
    ) -> None:
        """Initialize the MatDataset.

        Args:
            data_dir: Path to the directory containing the dataset.
            pilot_dims: Dimensions of pilot data as [num_subcarriers, num_ofdm_symbols].
            transform: Optional transformation to apply to samples.

        Raises:
            ValueError: If pilot dimensions are invalid.
            FileNotFoundError: If data_dir doesn't exist.
        """
        self.data_dir = Path(data_dir)
        self.pilot_dims = PilotDimensions(*pilot_dims)
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
            h_ideal: torch.Tensor,
            mat_data: dict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process channel data and extract pilot values from LS estimates.

        Extracts pilot values from LS channel estimates with zero entries removed,
        returning complex-valued tensors for both estimate and ground truth.

        Args:
            h_ideal: Ground truth channel tensor
            mat_data: Loaded .mat file data

        Returns:
            Tuple of (pilot LS estimate, ground truth channel)

        Raises:
            ValueError: If the data format is unexpected or processing fails
        """
        try:
            # Extract LS channel estimate with zero entries
            hzero_ls = torch.tensor(mat_data['H'][:, :, 1], dtype=torch.cfloat)

            # Remove zero entries, keep only pilot values
            zero_complex = torch.complex(torch.tensor(0.0), torch.tensor(0.0))
            hp_ls = hzero_ls[hzero_ls != zero_complex]

            # Validate expected number of pilot values
            expected_pilots = self.pilot_dims.num_subcarriers * self.pilot_dims.num_ofdm_symbols
            if hp_ls.numel() != expected_pilots:
                raise ValueError(
                    f"Expected {expected_pilots} pilot values, got {hp_ls.numel()}"
                )

            # Reshape to pilot grid dimensions [subcarriers, symbols]
            hp_ls = hp_ls.unsqueeze(dim=1).view(
                self.pilot_dims.num_ofdm_symbols,
                self.pilot_dims.num_subcarriers
            ).t()

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
                - Pilot LS channel estimate (complex tensor)
                - Ground truth channel estimate (complex tensor)
                - Metadata extracted from filename

        Raises:
            ValueError: If file format is invalid or processing fails.
            IndexError: If idx is out of range.
        """
        if not 0 <= idx < len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")

        try:
            # Load .mat file
            mat_data = sio.loadmat(self.file_list[idx])
            if 'H' not in mat_data or mat_data['H'].shape[-1] < 3:
                raise ValueError("Invalid .mat file format: missing required data")

            # Extract ground truth channel
            h_ideal = torch.tensor(mat_data['H'][:, :, 0], dtype=torch.cfloat)

            # Process channel data to extract pilot estimates
            h_est, h_ideal = self._process_channel_data(h_ideal, mat_data)

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
        params: dict
) -> List[Tuple[str, DataLoader]]:
    """Create DataLoaders for each subdirectory in the dataset directory.

    Automatically discovers and creates appropriate DataLoader instances for
    all subdirectories in the specified dataset directory, useful for testing
    across multiple test conditions or scenarios.

    Args:
        dataset_dir: Path to main directory containing dataset subdirectories
        params: Configuration parameters including:
            - pilot_dims: List of [num_subcarriers, num_ofdm_symbols]
            - batch_size: Number of samples per batch

    Returns:
        List of tuples containing (subdirectory_name, corresponding_dataloader)

    Raises:
        FileNotFoundError: If dataset_dir doesn't exist
        ValueError: If params are invalid or no valid subdirectories are found
    """
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    if not isinstance(params, dict) or "pilot_dims" not in params or "batch_size" not in params:
        raise ValueError("params must be a dict containing 'pilot_dims' and 'batch_size'")

    subdirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
    if not subdirs:
        raise ValueError(f"No subdirectories found in {dataset_dir}")

    test_datasets = [
        (
            subdir.name,
            MatDataset(
                subdir,
                params["pilot_dims"]
            )
        )
        for subdir in subdirs
    ]

    return [
        (name, DataLoader(
            dataset,
            batch_size=params["batch_size"],
            shuffle=False,
            num_workers=0
        ))
        for name, dataset in test_datasets
    ]