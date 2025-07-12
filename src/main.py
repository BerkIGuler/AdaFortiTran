#!/usr/bin/env python3
"""
Main entry point for OFDM channel estimation model training.

This script provides the command-line interface for training OFDM channel estimation
models. It loads configuration files, parses command-line arguments, and initiates
the training process.

Dataset Requirements:
    The training script expects datasets with the following structure:
    
    Training/Validation Sets:
        Directory containing .mat files with naming convention:
        {file_number}_SNR-{snr}_DS-{delay_spread}_DOP-{doppler}_N-{pilot_freq}_{channel_type}.mat
        
        Example: 1_SNR-20_DS-50_DOP-500_N-3_TDL-A.mat
    
    Test Sets:
        Directory with subdirectories for different test conditions:
        test_set/
        ├── DS_test_set/     # Delay Spread tests
        │   ├── DS_50/
        │   ├── DS_100/
        │   └── ...
        ├── SNR_test_set/    # SNR tests
        │   ├── SNR_10/
        │   ├── SNR_20/
        │   └── ...
        └── MDS_test_set/    # Multi-Doppler tests
            ├── DOP_200/
            ├── DOP_400/
            └── ...
    
    Each .mat file must contain variable 'H' with shape [subcarriers, symbols, 3]:
    - H[:, :, 0]: Ground truth channel
    - H[:, :, 1]: LS channel estimate with zeros for non-pilot positions
    - H[:, :, 2]: Unused (reserved)
"""

import logging
import sys
from pathlib import Path

from src.main.parser import parse_arguments
from src.main.trainer import train
from src.config import load_config
from src.config.schemas import ModelConfig


def setup_logging(log_level: str) -> None:
    """Set up logging configuration.
    
    Args:
        log_level: Logging level string (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('training.log')
        ]
    )


def main() -> None:
    """Main entry point for the training script."""
    try:
        # Parse command-line arguments
        args = parse_arguments()
        
        # Set up logging
        setup_logging(args.python_log_level)
        logger = logging.getLogger(__name__)
        
        logger.info("Starting OFDM channel estimation model training")
        logger.info(f"Model: {args.model_name}")
        logger.info(f"System config: {args.system_config_path}")
        logger.info(f"Model config: {args.model_config_path}")
        logger.info(f"Experiment ID: {args.exp_id}")
        
        # Load and validate configurations
        logger.info("Loading configuration files...")
        system_config, model_config = load_config(
            args.system_config_path, 
            args.model_config_path
        )
        
        logger.info("Configuration loaded successfully")
        logger.info(f"OFDM dimensions: {system_config.ofdm.num_scs} subcarriers x {system_config.ofdm.num_symbols} symbols")
        logger.info(f"Pilot dimensions: {system_config.pilot.num_scs} subcarriers x {system_config.pilot.num_symbols} symbols")
        if model_config.model_type == "linear":
            logger.info(f"Linear model with device: {model_config.device}")
        else:
            logger.info(f"Model architecture: {model_config.num_layers} layers, {model_config.model_dim} dimensions")
        
        # Start training
        logger.info("Initializing training...")
        train(system_config, model_config, args)
        
        logger.info("Training completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 