#!/usr/bin/env python3
"""
Main entry point for OFDM channel estimation model training.

This script provides the command-line interface for training OFDM channel estimation
models. It loads configuration files, parses command-line arguments, and initiates
the training process.
"""

import logging
import sys
from pathlib import Path

from src.main.parser import parse_arguments
from src.main.trainer import train
from src.config import load_config


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
        if model_config is not None:
            logger.info(f"Model architecture: {model_config.num_layers} layers, {model_config.model_dim} dimensions")
        else:
            logger.info("Using Linear model (no model config required)")
        
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