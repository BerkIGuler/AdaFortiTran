from torch import nn
import torch
import logging

from src.config.schemas import SystemConfig, ModelConfig


class FortiTranEstimator(nn.Module):
    """A DL-based Channel Estimator based on a hybrid convolutional + transformers model"""
    def __init__(self, system_config: SystemConfig, model_config: ModelConfig) -> None:
        """Initialize the FortiTranEstimator.

        Args:
            system_config: SystemConfig object containing OFDM system parameters
            system_config: ModelConfig object containing model parameters
        """
        super().__init__()

        self.system_config = system_config
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