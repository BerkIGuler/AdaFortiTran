"""
Learned linear estimator module for OFDM channel estimation.

This module implements an estimator for transforming channel estimates at
pilot signals to complete channel estimates using a learned linear transformation.
"""

from typing import Tuple
import logging
import torch
import torch.nn as nn

from src.config.schemas import SystemConfig


class LinearEstimator(nn.Module):
    """Learned MMSE estimator.

    Attributes:
        device (torch.device): Target device for computation
        config (SystemConfig): Validated configuration object
        ofdm_size (Tuple[int, int]): Dimensions of OFDM frame as (height, width)
            height (int): number of sub-carriers
            width (int): number of OFDM symbols
        pilot_size (Tuple[int, int]): Dimensions of pilot signal as (height, width)
            height (int): number of pilots across sub-carriers
            width (int): number of pilots across OFDM symbols
    """

    def __init__(self, config: SystemConfig) -> None:
        """Initialize the MMSE estimator.

        Args:
            config: Validated SystemConfig object containing OFDM system parameters
        """
        super().__init__()

        self.config = config
        self.device = torch.device(config.device)
        self.logger = logging.getLogger(__name__)

        # Extract dimensions from validated config
        self.ofdm_size = (config.ofdm.num_scs, config.ofdm.num_symbols)
        self.pilot_size = (config.pilot.num_scs, config.pilot.num_symbols)

        # Calculate feature dimensions
        in_feature_dim = config.pilot.num_scs * config.pilot.num_symbols
        out_feature_dim = config.ofdm.num_scs * config.ofdm.num_symbols

        self.logger.info(f"Initializing LinearEstimator:")
        self.logger.info(f"  OFDM size: {self.ofdm_size}")
        self.logger.info(f"  Pilot size: {self.pilot_size}")
        self.logger.info(f"  Input features: {in_feature_dim}")
        self.logger.info(f"  Output features: {out_feature_dim}")
        self.logger.info(f"  Device: {self.device}")

        # Create linear layer
        self.linear = nn.Linear(in_feature_dim, out_feature_dim)
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MMSE estimator.

        Args:
            x: Input tensor containing pilot signals with shape
               (batch_size, pilot_size[0], pilot_size[1])

        Returns:
            Estimated OFDM signal tensor with shape
            (batch_size, ofdm_size[0], ofdm_size[1])
        """
        # pytorch does nothin if input is already on correct device
        x = x.to(self.device)
        self.logger.debug(f"Input shape: {x.size()}")

        # Validate input shape
        expected_shape = (x.size(0), self.pilot_size[0], self.pilot_size[1])
        if x.size() != expected_shape:
            raise ValueError(
                f"Expected input shape {expected_shape}, got {x.size()}"
            )

        # Flatten input for linear transformation
        x = torch.flatten(x, start_dim=1)
        self.logger.debug(f"Flattened shape: {x.size()}")

        # Apply linear transformation
        x = self.linear(x)
        self.logger.debug(f"Linear output shape: {x.size()}")

        # Reshape to OFDM dimensions
        x = x.reshape(-1, self.ofdm_size[0], self.ofdm_size[1])
        self.logger.debug(f"Reshaped output shape: {x.size()}")

        return x

    def get_config(self) -> SystemConfig:
        """Get the configuration used by this estimator.

        Returns:
            SystemConfig: The configuration object
        """
        return self.config

    def __repr__(self) -> str:
        """String representation of the estimator."""
        return (
            f"LinearEstimator(\n"
            f"  ofdm_size={self.ofdm_size},\n"
            f"  pilot_size={self.pilot_size},\n"
            f"  device={self.device}\n"
            f")"
        )