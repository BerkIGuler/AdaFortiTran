"""
Learned linear estimator module for OFDM channel estimation.

This module implements an estimator for transforming channel estimates at
pilot signals to complete channel estimates using a learned linear transformation.
"""

from typing import Tuple
import logging
import torch
import torch.nn as nn

from src.config.schemas import SystemConfig, ModelConfig


class LinearEstimator(nn.Module):
    """Learned MMSE estimator.

    Find W such that W*h_pilot = h_hat, where h_hat is the estimated channel by stochastic gradient descent on |h_hat - h_ideal|^2

    Attributes:
        device (torch.device): Target device for computation
        system_config (SystemConfig): Validated configuration object for OFDM system parameters
        model_config (ModelConfig): Validated configuration object for model parameters
        ofdm_size (Tuple[int, int]): Dimensions of OFDM frame as (num_subcarriers, num_symbols)
            num_subcarriers (int): number of sub-carriers
            num_symbols (int): number of OFDM symbols
        pilot_size (Tuple[int, int]): Dimensions of pilot signal as (num_subcarriers, num_symbols)
            num_subcarriers (int): number of pilots across sub-carriers
            num_symbols (int): number of pilots across OFDM symbols
    """

    def __init__(self, system_config: SystemConfig, model_config: ModelConfig) -> None:
        """Initialize the MMSE estimator.

        Args:
            system_config: Validated SystemConfig object containing OFDM system parameters
            model_config: Validated ModelConfig object containing model parameters
        """
        super().__init__()

        self.system_config = system_config
        self.model_config = model_config
        self.device = torch.device(model_config.device)
        self.logger = logging.getLogger(__name__)

        # Extract dimensions from validated config
        self.ofdm_size = (system_config.ofdm.num_scs, system_config.ofdm.num_symbols)
        self.pilot_size = (system_config.pilot.num_scs, system_config.pilot.num_symbols)

        # Calculate feature dimensions
        in_feature_dim = system_config.pilot.num_scs * system_config.pilot.num_symbols
        out_feature_dim = system_config.ofdm.num_scs * system_config.ofdm.num_symbols

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
        # pytorch does nothing if input is already on correct device
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

    def __repr__(self) -> str:
        """String representation of the estimator."""
        return (
            f"LinearEstimator(\n"
            f"  ofdm_size={self.ofdm_size},\n"
            f"  pilot_size={self.pilot_size},\n"
            f"  device={self.device}\n"
            f")"
        )