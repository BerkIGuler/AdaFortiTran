#!/usr/bin/env python3
"""
Standalone evaluation script for OFDM channel estimation models.

This script evaluates a trained model checkpoint on the test set. It should ONLY be run
once you have finalized your model selection based on validation performance. This ensures
proper separation between model development and final evaluation.

Best Practice:
    1. Train multiple models using src/main.py
    2. Select the best model based on VALIDATION performance
    3. Run this evaluation script ONCE on the selected model to get final test metrics

Usage:
    python src/evaluate.py \\
        --checkpoint_path runs/adafortitran_experiment/best/checkpoint_epoch_50.pt \\
        --test_set data/test \\
        --batch_size 128 \\
        --output_dir results/final_evaluation

The script will:
    - Load the model checkpoint and configuration
    - Evaluate on DS (Delay Spread), MDS (Doppler), and SNR test sets
    - Generate performance plots and error visualizations
    - Save results to the specified output directory
"""

import argparse
import logging
import torch
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.data import get_test_dataloaders
from src.models import LinearEstimator, AdaFortiTranEstimator, FortiTranEstimator
from src.utils import (
    get_ls_mse_per_folder,
    get_test_stats_plot,
    get_error_images,
    concat_complex_channel,
    to_db
)
from src.config.schemas import SystemConfig, ModelConfig


# Model registry
MODEL_REGISTRY = {
    "linear": LinearEstimator,
    "adafortitran": AdaFortiTranEstimator,
    "fortitran": FortiTranEstimator,
}


def setup_logging() -> logging.Logger:
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> Tuple[nn.Module, SystemConfig, ModelConfig, str]:
    """Load model checkpoint and configurations.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        Tuple of (model, system_config, model_config, model_name)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    system_config = checkpoint['system_config']
    model_config = checkpoint['model_config']
    
    # Get model name from args or model_config
    if 'args' in checkpoint and hasattr(checkpoint['args'], 'model_name'):
        model_name = checkpoint['args'].model_name
    elif hasattr(model_config, 'model_type'):
        model_name = model_config.model_type
    else:
        raise ValueError("Cannot determine model type from checkpoint")
    
    # Initialize model
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model name: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
    
    model_class = MODEL_REGISTRY[model_name]
    model = model_class(system_config, model_config)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, system_config, model_config, model_name


def forward_pass(coarse_estimated_channel: torch.Tensor, 
                model: nn.Module,
                meta_data: Optional[Tuple] = None) -> torch.Tensor:
    """Perform forward pass through the model.
    
    Args:
        coarse_estimated_channel: LS channel estimate at pilot positions
        model: Model to perform forward pass with
        meta_data: Optional metadata for AdaFortiTran models
        
    Returns:
        Estimated channel after forward pass
    """
    if isinstance(model, AdaFortiTranEstimator):
        if meta_data is not None:
            return model(coarse_estimated_channel, meta_data)
        else:
            raise ValueError("AdaFortiTranEstimator requires meta_data but it was not provided")
    else:
        return model(coarse_estimated_channel)


def evaluate_dataloader(model: nn.Module, 
                        dataloader: DataLoader, 
                        loss_fn: nn.Module,
                        device: torch.device) -> float:
    """Evaluate model on a single dataloader.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader to evaluate on
        loss_fn: Loss function
        device: Device to run evaluation on
        
    Returns:
        Average loss over the dataset
    """
    total_loss = 0.0
    num_samples = 0
    model.eval()
    
    with torch.no_grad():
        for batch in dataloader:
            estimated_channel_input, ideal_channel, meta_data = batch
            estimated_channel = forward_pass(estimated_channel_input, model, meta_data)
            loss = loss_fn(
                concat_complex_channel(estimated_channel),
                concat_complex_channel(ideal_channel)
            )
            
            batch_size = batch[0].size(0)
            total_loss += (2 * loss.item() * batch_size)
            num_samples += batch_size
            
    return total_loss / num_samples


def get_test_stats(model: nn.Module,
                  test_dataloaders: List[Tuple[str, DataLoader]], 
                  loss_fn: nn.Module,
                  device: torch.device,
                  logger: logging.Logger) -> Dict[int, float]:
    """Get test statistics for a set of dataloaders.
    
    Args:
        model: Model to evaluate
        test_dataloaders: List of (name, dataloader) tuples
        loss_fn: Loss function
        device: Device to run evaluation on
        logger: Logger instance
        
    Returns:
        Dictionary mapping condition values to test errors (in dB)
    """
    stats = {}
    sorted_loaders = sorted(
        test_dataloaders,
        key=lambda x: int(x[0].split("_")[1])
    )
    
    for name, test_dataloader in tqdm(sorted_loaders, desc="Evaluating conditions"):
        var, val = name.split("_")
        test_loss = evaluate_dataloader(model, test_dataloader, loss_fn, device)
        db_error = to_db(test_loss)
        logger.info(f"{var}:{val} Test MSE: {db_error:.4f} dB")
        stats[int(val)] = db_error
    
    return stats


def predict_channels(model: nn.Module,
                    test_dataloaders: List[Tuple[str, DataLoader]],
                    device: torch.device) -> Dict[int, Dict]:
    """Predict channels for visualization.
    
    Args:
        model: Model to use for prediction
        test_dataloaders: List of (name, dataloader) tuples
        device: Device to run prediction on
        
    Returns:
        Dictionary mapping condition values to channel predictions
    """
    channels = {}
    sorted_loaders = sorted(
        test_dataloaders,
        key=lambda x: int(x[0].split("_")[1])
    )
    
    model.eval()
    with torch.no_grad():
        for name, test_dataloader in sorted_loaders:
            batch = next(iter(test_dataloader))
            estimated_channel_input, ideal_channels, meta_data = batch
            estimated_channels = forward_pass(estimated_channel_input, model, meta_data)
            
            var, val = name.split("_")
            channels[int(val)] = {
                "estimated_channel": estimated_channels[0],
                "ideal_channel": ideal_channels[0]
            }
    
    return channels


def save_results(output_dir: Path,
                model_name: str,
                ds_stats: Dict[int, float],
                mds_stats: Dict[int, float],
                snr_stats: Dict[int, float],
                test_set: Path,
                ds_dataloaders: List[Tuple[str, DataLoader]],
                mds_dataloaders: List[Tuple[str, DataLoader]],
                snr_dataloaders: List[Tuple[str, DataLoader]],
                model: nn.Module,
                device: torch.device,
                logger: logging.Logger) -> None:
    """Save evaluation results and visualizations.
    
    Args:
        output_dir: Directory to save results
        model_name: Name of the model
        ds_stats: Delay Spread test statistics
        mds_stats: Multi-Doppler Shift test statistics
        snr_stats: SNR test statistics
        test_set: Path to test set directory
        ds_dataloaders: Delay Spread dataloaders
        mds_dataloaders: Multi-Doppler dataloaders
        snr_dataloaders: SNR dataloaders
        model: Model instance
        device: Device
        logger: Logger instance
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save statistics to text file
    with open(output_dir / "test_results.txt", "w") as f:
        f.write("=" * 80 + "\n")
        f.write("FINAL TEST SET EVALUATION RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Model: {model_name}\n\n")
        
        f.write("Delay Spread (DS) Test Results (dB MSE):\n")
        f.write("-" * 40 + "\n")
        for val, error in sorted(ds_stats.items()):
            f.write(f"DS={val:3d} ns: {error:7.4f} dB\n")
        
        f.write("\nMulti-Doppler Shift (MDS) Test Results (dB MSE):\n")
        f.write("-" * 40 + "\n")
        for val, error in sorted(mds_stats.items()):
            f.write(f"DOP={val:4d} Hz: {error:7.4f} dB\n")
        
        f.write("\nSignal-to-Noise Ratio (SNR) Test Results (dB MSE):\n")
        f.write("-" * 40 + "\n")
        for val, error in sorted(snr_stats.items()):
            f.write(f"SNR={val:2d} dB: {error:7.4f} dB\n")
    
    logger.info(f"Test results saved to {output_dir / 'test_results.txt'}")
    
    # Get LS baseline statistics
    try:
        ls_ds_stats = get_ls_mse_per_folder(test_set / "DS_test_set")
        ls_mds_stats = get_ls_mse_per_folder(test_set / "MDS_test_set")
        ls_snr_stats = get_ls_mse_per_folder(test_set / "SNR_test_set")
        include_ls = True
    except Exception as e:
        logger.warning(f"Could not compute LS baseline: {e}")
        include_ls = False
    
    # Generate and save plots
    test_configs = [
        ("DS", ds_stats, ds_dataloaders, ls_ds_stats if include_ls else None),
        ("MDS", mds_stats, mds_dataloaders, ls_mds_stats if include_ls else None),
        ("SNR", snr_stats, snr_dataloaders, ls_snr_stats if include_ls else None),
    ]
    
    for key, stats, dataloaders, ls_stats in test_configs:
        # Plot performance comparison
        methods = [model_name]
        all_stats = [stats]
        
        if include_ls and ls_stats is not None:
            methods.insert(0, "LS")
            all_stats.insert(0, ls_stats)
        
        fig = get_test_stats_plot(
            x_name=key,
            stats=all_stats,
            methods=methods
        )
        fig.savefig(output_dir / f"{key}_performance.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved {key} performance plot to {output_dir / f'{key}_performance.png'}")
        
        # Plot error images
        predicted_channels = predict_channels(model, dataloaders, device)
        fig = get_error_images(key, predicted_channels, show=False)
        fig.savefig(output_dir / f"{key}_error_images.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved {key} error images to {output_dir / f'{key}_error_images.png'}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate a trained OFDM channel estimation model on the test set',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--checkpoint_path',
        type=Path,
        required=True,
        help='Path to model checkpoint (.pt file)'
    )
    parser.add_argument(
        '--test_set',
        type=Path,
        required=True,
        help='Path to test dataset directory (should contain DS_test_set, MDS_test_set, SNR_test_set)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=Path('results/evaluation'),
        help='Directory to save evaluation results'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device to use (cpu, cuda, cuda:N, mps, or auto)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of data loading workers'
    )
    
    return parser.parse_args()


def resolve_device(device_str: str) -> torch.device:
    """Resolve device string to torch.device."""
    if device_str == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    return torch.device(device_str)


def main():
    """Main evaluation function."""
    args = parse_args()
    logger = setup_logging()
    
    # Validate paths
    if not args.checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
    if not args.test_set.exists():
        raise FileNotFoundError(f"Test set not found: {args.test_set}")
    
    # Resolve device
    device = resolve_device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint_path}")
    model, system_config, model_config, model_name = load_checkpoint(args.checkpoint_path, device)
    logger.info(f"Loaded model: {model_name}")
    
    # Load test dataloaders
    logger.info("Loading test datasets...")
    ds_dataloaders = get_test_dataloaders(
        args.test_set / "DS_test_set",
        system_config.pilot,
        args.batch_size,
        num_workers=args.num_workers
    )
    mds_dataloaders = get_test_dataloaders(
        args.test_set / "MDS_test_set",
        system_config.pilot,
        args.batch_size,
        num_workers=args.num_workers
    )
    snr_dataloaders = get_test_dataloaders(
        args.test_set / "SNR_test_set",
        system_config.pilot,
        args.batch_size,
        num_workers=args.num_workers
    )
    logger.info("Test datasets loaded successfully")
    
    # Define loss function
    loss_fn = nn.MSELoss()
    
    # Run evaluation
    logger.info("=" * 80)
    logger.info("Starting Test Set Evaluation")
    logger.info("=" * 80)
    
    logger.info("\nEvaluating Delay Spread (DS) robustness...")
    ds_stats = get_test_stats(model, ds_dataloaders, loss_fn, device, logger)
    
    logger.info("\nEvaluating Multi-Doppler Shift (MDS) robustness...")
    mds_stats = get_test_stats(model, mds_dataloaders, loss_fn, device, logger)
    
    logger.info("\nEvaluating Signal-to-Noise Ratio (SNR) robustness...")
    snr_stats = get_test_stats(model, snr_dataloaders, loss_fn, device, logger)
    
    # Save results
    logger.info("\nSaving results...")
    save_results(
        output_dir=args.output_dir,
        model_name=model_name,
        ds_stats=ds_stats,
        mds_stats=mds_stats,
        snr_stats=snr_stats,
        test_set=args.test_set,
        ds_dataloaders=ds_dataloaders,
        mds_dataloaders=mds_dataloaders,
        snr_dataloaders=snr_dataloaders,
        model=model,
        device=device,
        logger=logger
    )
    
    logger.info("=" * 80)
    logger.info("Evaluation Complete!")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

