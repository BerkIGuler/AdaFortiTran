# AdaFortiTran: Adaptive Transformer Model for Robust OFDM Channel Estimation

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Official implementation of [AdaFortiTran: An Adaptive Transformer Model for Robust OFDM Channel Estimation](https://ieeexplore.ieee.org/document/11160810) accepted at ICC 2025, Montreal, Canada.

## Overview

AdaFortiTran is a novel, compact, and adaptive transformer-based channel estimation model for SISO OFDM systems. AdaFortiTran dynamically adapts to channel conditions such as SNR, delay spread, Doppler shift. The model combines a custom-designed deep upsampling network with multi-head self-attention (MHSA) and convolutional operators, along with a channel-aware adaptation mechanism embedded into the MHSA calculation, to achieve competitive performance across diverse wireless environments. In the absence of priors on the channel conditions, we resort to FortiTran, which is parameter-free, i.e., lacks the adaptation capability, but still demonstrates impressive results beyond competing methods.

> **Note on Data:** This repository includes sample data (`data/sample_data/`) for demonstration purposes only. To generate the full dataset used in the paper (144k samples with train/val/test splits), please use the companion repository: [OFDMChannelGenerator](https://github.com/BerkIGuler/OFDMChannelGenerator).

## Architecture

This repository implements three models:

1. **Linear Estimator**: Simple learned linear estimator baseline (single fully-connected layer without activation)
2. **FortiTran**: Base channel estimator based on MHSA and convolutional operators w/o channel adaptivity
3. **AdaFortiTran**: Adaptive version of FortiTran with channel condition awareness

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/AdaFortiTran.git
   cd AdaFortiTran
   git lfs pull  # to pull the large files tracked by LFS (sample data)
   ```

2. **Make sure to have CUDA properly installed**: If you have a CUDA-compatible GPU, you should install and configure the necessary drivers/kernels for accelerated computing on GPU(s). 

   **Note:** This repository is tested for CUDA and CPU only.

3. **Install the package**:
   ```bash
   pip install -e .
   ```

   This installs the package in development mode with all dependencies:
   - `torch` - PyTorch deep learning framework
   - `pydantic` - Configuration validation
   - `pyyaml` - YAML configuration parsing
   - `scipy` - MATLAB file I/O for `.mat` datasets
   - `tqdm` - Progress bars
   - `matplotlib` - Visualization
   - `prettytable` - Formatted console output
   - `tensorboard` - Training metrics visualization

4. **Verify CUDA availability** (optional):
   ```python
   import torch
   print(torch.cuda.is_available())
   ```

## Quick Start with Example Notebooks

The `examples/` directory contains Jupyter notebooks that provides simple walkthrough of the workflow:

### 01_data_setup.ipynb — Data Preparation

Extracts and prepares the sample dataset for training:
- Extracts `data/sample_data/sample_files.zip` containing 1000 `.mat` files
- Splits data into 80% training (800 samples) / 20% validation (200 samples)
- Copies files to `data/train/` and `data/val/` directories
- Preprocesses `.mat` files to keep only the required channels (`H[:,:,0:2]`)
- Visualizes sample channel data (perfect channel vs. LS estimate)

**Note**: The sample data is included for demonstration purposes only. Please contact the author(s) for the datasets used in the original paper. 

**Run this notebook first before training.**

### 02_training.ipynb — Model Training

Demonstrates training all three models via the training script:
- Configurable hyperparameters (batch size, learning rate, epochs, patience)
- Trains **Linear**, **FortiTran**, and **AdaFortiTran** models sequentially
- Saves checkpoints to `runs/{model_name}_{exp_id}/`
- Logs training metrics to TensorBoard

### 03_inference.ipynb — Inference & Visualization

Loads trained models and performs inference:
- Discovers and lists available checkpoints from `runs/`
- Loads a trained model from checkpoint
- Runs inference on validation data
- Visualizes channel estimation results (ground truth vs. predicted vs. error)
- Computes **Normalized MSE (NMSE)** in linear and dB scale:
  
  $$\text{NMSE} = \frac{\|\hat{H} - H\|^2}{\|H\|^2}$$

- Evaluates performance across the entire validation set

**Note:** We use val set here for demonstration purposes. The inference should be performed on the unseen test data.
**Prerequisites:** Run `01_data_setup.ipynb` and `02_training.ipynb` first.

## Project Structure

```
AdaFortiTran/
├── config/                    # Configuration files
│   ├── system_config.yaml     # OFDM system parameters
│   ├── adafortitran.yaml      # AdaFortiTran model config
│   └── fortitran.yaml         # FortiTran model config
├── data/                      # Dataset directory
│   ├── sample_data/           # Sample data for demos
│   │   ├── sample_files.zip   # 1000 sample .mat files
│   │   └── extracted/         # Extracted sample files
│   ├── train/                 # Training data
│   ├── val/                   # Validation data
│   └── test/                  # Test data organized by evaluation scenarios
│       ├── DS_test_set/       # Delay Spread robustness tests (7 conditions)  
│       ├── MDS_test_set/      # Max. Doppler Shift tests (7 conditions)
│       └── SNR_test_set/      # Signal-to-Noise Ratio tests (7 conditions)
├── examples/                  # Jupyter notebook tutorials
│   ├── 01_data_setup.ipynb    # Data extraction and preparation
│   ├── 02_training.ipynb      # Model training demo
│   └── 03_inference.ipynb     # Inference and visualization
├── logs/                      # Training log files
├── runs/                      # Checkpoints and TensorBoard logs
├── scripts/                   # Utility scripts
│   ├── add_gitkeep.py         # Add .gitkeep files to empty directories
│   └── upload_to_huggingface.py  # Dataset upload utility
├── src/                       # Source code
│   ├── train.py               # Training script (main entry point)
│   ├── evaluate.py            # Standalone test set evaluation script
│   ├── utils.py               # Utility functions
│   ├── main/                  # Training pipeline
│   │   ├── trainer.py         # Unified model training
│   │   └── parser.py          # Command-line argument parser
│   ├── models/                # Model implementations
│   │   ├── adafortitran.py    # AdaFortiTran model (extends FortiTran)
│   │   ├── fortitran.py       # FortiTran model
│   │   ├── linear.py          # Linear model
│   │   └── blocks/            # Model building blocks
│   │       ├── channel_adaptivity.py  # Channel adaptation mechanisms
│   │       ├── encoders.py    # Transformer encoders
│   │       ├── enhancers.py   # Feature enhancement modules
│   │       ├── patch_processors.py    # Patch processing layers
│   │       └── positional_encodings.py  # Positional encoding implementations
│   ├── data/                  # Data loading
│   │   └── dataset.py         # MatDataset class for .mat file handling
│   └── config/                # Configuration management
│       ├── config_loader.py   # YAML configuration loader
│       └── schemas.py         # Pydantic validation schemas
├── pyproject.toml             # Package configuration and dependencies
├── requirements.txt           # Alternative dependency list
├── LICENSE                    # MIT License
└── README.md                  # This file
```

## Configuration

### System Configuration (`config/system_config.yaml`)

Defines OFDM system parameters:

```yaml
ofdm:
  num_scs: 120      # Number of subcarriers
  num_symbols: 14   # Number of OFDM symbols

pilot:
  num_scs: 40       # Number of pilot subcarriers
  num_symbols: 2    # Number of pilot symbols
```

*Note:* Update these parameters to match your dataset. If there is a mismatch, you will get an error.

### Model Configuration

#### FortiTran (`config/fortitran.yaml`)

```yaml
model_type: 'fortitran'
patch_size: [3, 2]            # Patch dimensions [height, width]
num_layers: 6                 # Number of transformer layers
model_dim: 32                 # Model dimension (embedding size)
num_head: 4                   # Number of self-attention heads
activation: 'gelu'            # Activation function in MLP blocks
dropout: 0.1                  # Dropout rate
max_seq_len: 512              # Maximum sequence length
pos_encoding_type: 'learnable'  # Positional encoding type
```

#### AdaFortiTran (`config/adafortitran.yaml`)

```yaml
model_type: 'adafortitran'
patch_size: [3, 2]            # Patch dimensions [height, width]
num_layers: 6                 # Number of transformer layers
model_dim: 32                 # Model dimension (embedding size)
num_head: 4                   # Number of self-attention heads
activation: 'gelu'            # Activation function in MLP blocks
dropout: 0.1                  # Dropout rate
max_seq_len: 512              # Maximum sequence length
pos_encoding_type: 'learnable'  # Positional encoding type
channel_adaptivity_hidden_sizes: [7, 42, 560]  # MLP hidden sizes for adaptation
adaptive_token_length: 6      # Adaptive token vector length
```

## Training

### Sample Training Scripts

**Train AdaFortiTran with default settings:**

```bash
python src/train.py \
    --model_name adafortitran \
    --system_config_path config/system_config.yaml \
    --model_config_path config/adafortitran.yaml \
    --train_set data/train \
    --val_set data/val \
    --exp_id my_experiment
```

**Train with full configurability:**

```bash
python src/train.py \
    --model_name adafortitran \
    --system_config_path config/system_config.yaml \
    --model_config_path config/adafortitran.yaml \
    --train_set data/train \
    --val_set data/val \
    --exp_id advanced_experiment \
    --batch_size 128 \
    --lr 5e-4 \
    --max_epoch 100 \
    --patience 10 \
    --weight_decay 1e-4 \
    --gradient_clip_val 1.0 \
    --use_mixed_precision \
    --save_every_n_epochs 5 \
    --num_workers 8
```

### Training Options

| Feature | Description | Default |
|---------|-------------|---------|
| `--batch_size` | Training batch size | 64 |
| `--lr` | Initial learning rate | 1e-3 |
| `--max_epoch` | Maximum number of training epochs | 10 |
| `--patience` | Early stopping patience (epochs) | 3 |
| `--weight_decay` | Weight decay for optimizer (L2 regularization) | 0.0 |
| `--gradient_clip_val` | Gradient clipping value | None |
| `--use_mixed_precision` | Enable mixed precision training | False |
| `--save_best_only` | Save only best model | True |
| `--save_every_n_epochs` | Save checkpoint every N epochs | None |
| `--resume_from_checkpoint` | Resume from checkpoint | None |
| `--num_workers` | Data loading workers | 4 |
| `--pin_memory` | Pin memory for faster GPU transfer | True |
| `--device` | Computing device (cpu, cuda, mps, auto) | auto |

### Training Different Models

**Linear Estimator:**
```bash
python src/train.py \
    --model_name linear \
    --system_config_path config/system_config.yaml \
    --train_set data/train \
    --val_set data/val \
    --exp_id linear_baseline
```

**FortiTran:**
```bash
python src/train.py \
    --model_name fortitran \
    --system_config_path config/system_config.yaml \
    --model_config_path config/fortitran.yaml \
    --train_set data/train \
    --val_set data/val \
    --exp_id fortitran_experiment
```

**AdaFortiTran:**
```bash
python src/train.py \
    --model_name adafortitran \
    --system_config_path config/system_config.yaml \
    --model_config_path config/adafortitran.yaml \
    --train_set data/train \
    --val_set data/val \
    --exp_id adafortitran_experiment
```

### Resume Training

```bash
python src/train.py \
    --model_name adafortitran \
    --system_config_path config/system_config.yaml \
    --model_config_path config/adafortitran.yaml \
    --train_set data/train \
    --val_set data/val \
    --exp_id resumed_experiment \
    --resume_from_checkpoint runs/adafortitran_experiment/best/checkpoint_epoch_50.pt
```

### Callback System

The training pipeline includes an extensible callback system:

- **TensorBoard Logging**: Automatic metric tracking and visualization
- **Checkpoint Management**: Flexible checkpoint saving strategies
- **Custom Callbacks**: Easy to add new logging or monitoring systems

## Dataset Format

### Expected File Structure

```
data/
├── sample_data/
│   └── sample_files.zip     # Sample data for demos (1000 files)
├── train/
│   ├── 1_SNR-20_DS-50_DOP-500_N-3_TDL-A.mat
│   ├── 2_SNR-20_DS-50_DOP-500_N-3_TDL-A.mat
│   └── ...
├── val/
│   └── ...
└── test/
    ├── DS_test_set/          # Delay Spread robustness evaluation
    │   ├── DS_50/            # 50 ns delay spread
    │   ├── DS_100/           # 100 ns delay spread  
    │   ├── DS_150/           # 150 ns delay spread
    │   ├── DS_200/           # 200 ns delay spread
    │   ├── DS_250/           # 250 ns delay spread
    │   ├── DS_300/           # 300 ns delay spread
    │   └── DS_350/           # 350 ns delay spread
    ├── SNR_test_set/         # Signal-to-Noise Ratio robustness evaluation
    │   ├── SNR_0/            # 0 dB SNR
    │   ├── SNR_5/            # 5 dB SNR
    │   ├── SNR_10/           # 10 dB SNR
    │   ├── SNR_15/           # 15 dB SNR
    │   ├── SNR_20/           # 20 dB SNR
    │   ├── SNR_25/           # 25 dB SNR
    │   └── SNR_30/           # 30 dB SNR
    └── MDS_test_set/         # Multi-Doppler Shift robustness evaluation
        ├── DOP_200/          # 200 Hz Doppler frequency
        ├── DOP_400/          # 400 Hz Doppler frequency
        ├── DOP_600/          # 600 Hz Doppler frequency
        ├── DOP_800/          # 800 Hz Doppler frequency
        ├── DOP_1000/         # 1000 Hz Doppler frequency
        ├── DOP_1200/         # 1200 Hz Doppler frequency
        └── DOP_1400/         # 1400 Hz Doppler frequency
```

**Note:**
### Test Set Organization

Each test set evaluates model robustness under specific channel conditions:

- **DS_test_set (Delay Spread)**: Evaluates performance across different multipath delay spreads (50-350 ns), simulating various indoor/outdoor propagation environments from small rooms to large urban areas.

- **SNR_test_set (Signal-to-Noise Ratio)**: Tests model resilience to noise across SNR levels from 0-30 dB, covering challenging low-SNR scenarios to high-quality channel conditions.

- **MDS_test_set (Multi-Doppler Shift)**: Assesses adaptation to mobility-induced Doppler shifts (200-1400 Hz), representing scenarios from pedestrian movement to high-speed vehicular communication.

Each subdirectory contains `.mat` files following the same naming convention and data format as training/validation sets, but with fixed channel conditions corresponding to the test scenario.

### File Naming Convention

Files must follow the pattern:
```
{file_number}_SNR-{snr}_DS-{delay_spread}_DOP-{doppler}_N-{pilot_freq}_{channel_type}.mat
```

Example: `1_SNR-20_DS-50_DOP-500_N-3_TDL-A.mat`

### Data Format

Each `.mat` file must contain a variable `H` with shape `[# OFDM subcarriers, # OFDM symbols, 2]`:
- `H[:, :, 0]`: Complex-valued ground truth channel matrix
- `H[:, :, 1]`: Least squares estimate of the channel at pilot positions (zeros for non-pilot positions)

## Monitoring and Logging

### TensorBoard Integration

Training automatically logs metrics to TensorBoard:

```bash
tensorboard --logdir runs/
```

Then open http://localhost:6006 in your browser.

Available metrics/logs:
- Training/validation loss
- Learning rate
- Model hyperparameters

### Log Files

Training logs are saved to:
- `logs/training_{model_name}_{exp_id}_{timestamp}.log`: Python logging output
- `runs/{model_name}_{exp_id}_{timestamp}/`: TensorBoard logs and checkpoints

Both files share the same timestamp, making it easy to match logs with TensorBoard runs. For example:
- Log file: `logs/training_adafortitran_my_exp_20260109_143052.log`
- TensorBoard: `runs/adafortitran_my_exp_20260109_143052/`

## Testing and Evaluation

### Evaluation Script

After training, evaluate your model on the test set:

```bash
python src/evaluate.py \
    --checkpoint_path runs/adafortitran_experiment/best/checkpoint_epoch_50.pt \
    --test_set data/test \
    --batch_size 128 \
    --output_dir results/final_evaluation
```

The evaluation script will:
- Load your trained model checkpoint
- Evaluate across all test conditions:
  - **DS (Delay Spread)**: 7 conditions from 50-350 ns testing multipath robustness
  - **SNR (Signal-to-Noise Ratio)**: 7 levels from 0-30 dB testing noise resilience  
  - **MDS (Multi-Doppler Shift)**: 7 frequencies from 200-1400 Hz testing mobility adaptation
- Generate performance plots and error visualizations
- Save results to the specified output directory

### Evaluation Script Options

```bash
python src/evaluate.py --help
```

Key arguments:
- `--checkpoint_path`: Path to trained model checkpoint (required)
- `--test_set`: Path to test dataset directory (required)
- `--batch_size`: Batch size for evaluation (default: 64)
- `--output_dir`: Directory to save evaluation results
- `--device`: Computing device (cpu, cuda, mps, or auto)
- `--num_workers`: Number of data loading workers

## Citation

If you use part of this code in your research, please cite:

```bibtex
@inproceedings{GulJaf2025,
  author={Guler, Berkay and Jafarkhani, Hamid},
  booktitle={ICC 2025 - IEEE International Conference on Communications}, 
  title={AdaFortiTran: An Adaptive Transformer Model for Robust OFDM Channel Estimation}, 
  year={2025},
  volume={},
  number={},
  pages={3797-3802},
  keywords={Deep learning;Doppler shift;Wireless communication;Adaptation models;OFDM;Channel estimation;Computer architecture;Transformers;Delays;Signal to noise ratio;channel estimation;OFDM;Transformer;Attention;Deep learning},
  doi={10.1109/ICC52391.2025.11160810}}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2025 [Berkay Guler/University of California, Irvine]
