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
    - H[:, :, 0]: Ground truth channel (complex-valued channel matrix)
    - H[:, :, 1]: LS channel estimate with zeros for non-pilot positions (complex-valued) - used as input to models
    - H[:, :, 2]: Bilinear interpolated LS channel estimate (complex-valued) - available but currently unused
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

from src.main.parser import parse_arguments
from src.main.trainer import train
from src.config import load_config
from src.config.schemas import ModelConfig


def setup_logging(log_level: str, log_dir: Path, exp_id: str) -> None:
    """Set up logging configuration.
    
    Args:
        log_level: Logging level string (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory path for log files
        exp_id: Experiment identifier for log file naming
    """
    # Create logs directory if it doesn't exist
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log file path using exp_id for easy matching
    log_file = log_dir / f"training_{exp_id}.log"
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file)
        ]
    )


def main() -> None:
    """Main entry point for the training script."""
    try:
        # Parse command-line arguments
        args = parse_arguments()
        
        # Set up logging
        setup_logging(args.python_log_level, args.python_log_dir, args.exp_id)
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
        
        # Validate model type consistency
        expected_model_types = {
            "linear": "linear",
            "fortitran": "fortitran", 
            "adafortitran": "adafortitran"
        }
        
        if args.model_name not in expected_model_types:
            raise ValueError(f"Unknown model name: {args.model_name}. Expected one of: {list(expected_model_types.keys())}")
        
        if model_config.model_type != expected_model_types[args.model_name]:
            raise ValueError(f"Model type mismatch: config specifies '{model_config.model_type}' but model name is '{args.model_name}'")
        
        logger.info("Configuration loaded successfully")
        logger.info(f"OFDM dimensions: {system_config.ofdm.num_scs} subcarriers x {system_config.ofdm.num_symbols} symbols")
        logger.info(f"Pilot dimensions: {system_config.pilot.num_scs} subcarriers x {system_config.pilot.num_symbols} symbols")
        
        # Log model-specific information
        if model_config.model_type == "linear":
            logger.info(f"Linear model with device: {model_config.device}")
        elif model_config.model_type == "fortitran":
            logger.info(f"FortiTran model: {model_config.num_layers} layers, {model_config.model_dim} dimensions")
            logger.info(f"Channel adaptation: disabled")
        elif model_config.model_type == "adafortitran":
            logger.info(f"AdaFortiTran model: {model_config.num_layers} layers, {model_config.model_dim} dimensions")
            logger.info(f"Channel adaptation: enabled")
            logger.info(f"Adaptive token length: {model_config.adaptive_token_length}")
        else:
            logger.warning(f"Unknown model type: {model_config.model_type}")
        
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