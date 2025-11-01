# AdaFortiTran: Adaptive Transformer Model for Robust OFDM Channel Estimation

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-red.svg)](https://pytorch.org/)

Official implementation of [AdaFortiTran: An Adaptive Transformer Model for Robust OFDM Channel Estimation](https://ieeexplore.ieee.org/document/11160810) accepted at ICC 2025, Montreal, Canada.

## Overview

AdaFortiTran is a novel adaptive transformer-based model for SISO OFDM channel estimation that dynamically adapts to channel conditions (e.g. SNR, delay spread, Doppler shift). The model combines a custom-designed deep upsampling network with multi head self attention (MHSA) and convolutional operators and a channel-aware adaptation mechanism embedded into MHSA calculation to achieve competitive performance across diverse wireless environments.

## Architecture

This repository implements three models:

1. **Linear Estimator**: Simple learned linear estimator baseline (single fully-connected layer without non-linear activations)
2. **FortiTran**: Base channel estimator based on MHSA and convolutional operators w/o channel adaptivity
3. **AdaFortiTran**: Adaptive version of FortiTran with channel condition awareness

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/AdaFortiTran.git
   cd AdaFortiTran
   ```
2. **Make sure to have CUDA properly installed**: If you have a CUDA-compatible GPU, you should install and configure the necessary drivers/kernels for accelerated computing on GPU(s).  

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

*Note:* One simple way to see if CUDA is accessible through PyTorch is to run the following python script.
```
import torch
print(torch.cuda.is_available())
```

### Sample Training scripts

*To train an AdaFortiTran model with default settings:*

```bash
python3 src/main.py \
    --model_name adafortitran \
    --system_config_path config/system_config.yaml \
    --model_config_path config/adafortitran.yaml \
    --train_set data/train \
    --val_set data/val \
    --test_set data/test \
    --exp_id my_experiment
```

*To train an AdaFortiTran model with maximal configurability:*

```bash
python src/main.py \
    --model_name adafortitran \
    --system_config_path config/system_config.yaml \
    --model_config_path config/adafortitran.yaml \
    --train_set data/train \
    --val_set data/val \
    --test_set data/test \
    --exp_id advanced_experiment \
    --batch_size 128 \
    --lr 5e-4 \
    --max_epoch 100 \
    --patience 10 \
    --weight_decay 1e-4 \
    --gradient_clip_val 1.0 \
    --use_mixed_precision \
    --save_every_n_epochs 5 \
    --num_workers 8 \
    --test_every_n 5
```

## Project Structure
```
AdaFortiTran/
├── config/                    # Configuration files
│   ├── system_config.yaml     # configurable OFDM system parameters
│   ├── adafortitran.yaml      # default AdaFortiTran model config
│   ├── fortitran.yaml         # default FortiTran model config
├── data/                      # Dataset directory
│   ├── train/                 # Training data
│   ├── val/                   # Validation data
│   └── test/                  # Test data organized by evaluation scenarios (see below for more details on that)
│       ├── DS_test_set/       # Delay Spread robustness tests (7 conditions)  
│       ├── MDS_test_set/      # Max. Doppler Shift tests (7 conditions)
│       └── SNR_test_set/      # Signal-to-Noise Ratio tests (7 conditions)
├── src/                       # Source code
│   ├── main/                  # Training pipeline
│   │   ├── trainer.py         # Unified model training
│   │   └── parser.py          # Command-line argument parser
│   ├── models/                # Model implementations
│   │   ├── adafortitran.py    # AdaFortiTran model (extends FortiTran)
│   │   ├── fortitran.py       # FortiTran model
│   │   ├── linear.py          # Linear model
│   │   └── blocks/            # Model building blocks
│   ├── data/                  # Data loading
│   │   └── dataset.py         # Data handling/processing
│   ├── config/                # Configuration management
│   │   ├── config_loader.py   # YAML configuration loader
│   │   └── schemas.py         # Pydantic validation schemas
│   └── utils.py               # Utility functions
├── requirements.txt           # Python dependencies
├── README.md                  # This file
```

## Configuration

### System Configuration (`config/system_config.yaml`)

Defines OFDM system parameters:

```yaml
ofdm:
  num_scs: 120      # Number of subcarriers
  num_symbols: 14   # Number of OFDM symbols

pilot:
  num_scs: 12       # Number of pilot subcarriers
  num_symbols: 2    # Number of pilot symbols
```

*Note:* You should update those parameters based on your dataset.

### Model Configuration (`config/adafortitran.yaml`)

Defines the AdaFortiTran architecture parameters:

```yaml
model_type: 'adafortitran'            # should not be changed
patch_size: [3, 2]                    # Patch dimensions
num_layers: 6                         # number of transformer layers
model_dim: 128                        # model dimension (i.e. dimension after the input projection is applied to each patch)
num_head: 4                           # Number of self-attention heads
activation: 'gelu'                    # Activation function (used within the MLP block of the transformer encoder)
dropout: 0.1                          # Dropout rate (for MLP block of the transformer encoder)
max_seq_len: 512                      # Maximum sequence length (should be >= number of patches)
pos_encoding_type: 'learnable'        # Positional encoding type
channel_adaptivity_hidden_sizes: [7, 42, 560]  # hidden sizes of the MLP used for adaptation to channel condition
adaptive_token_length: 6              # Adaptive token vector (concatenated with each flattened patch) length
```

## Training Features

### Training Options

| Feature | Description | Default |
|---------|-------------|---------|
| `--use_mixed_precision` | Enable mixed precision training | False |
| `--gradient_clip_val` | Gradient clipping value | None |
| `--weight_decay` | Weight decay for optimizer | 0.0 |
| `--save_checkpoints` | Enable model checkpointing | True |
| `--save_best_only` | Save only best model | True |
| `--resume_from_checkpoint` | Resume from checkpoint | None |
| `--num_workers` | Data loading workers | 4 |
| `--pin_memory` | Pin memory for GPU | True |

### Callback System

The training pipeline includes an extensible callback system:

- **TensorBoard Logging**: Automatic metric tracking and visualization
- **Checkpoint Management**: Flexible checkpoint saving strategies
- **Custom Callbacks**: Easy to add new logging or monitoring systems

## Dataset Format

### Expected File Structure

```
data/
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

Each `.mat` file must contain a variable `H` with shape `[# OFDM subcarriers, # OFDM symbols, 3]`:
- `H[:, :, 0]`: complex valued ground truth channel matrix
- `H[:, :, 1]`: least square estimate of the channel at pilot positions and zeros for non-pilot positions
- `H[:, :, 2]`: Reserved for future use

## Usage Examples

### Training Different Models

**Linear Estimator**:
```bash
python src/main.py \
    --model_name linear \
    --system_config_path config/system_config.yaml \
    --train_set data/train \
    --val_set data/val \
    --test_set data/test \
    --exp_id linear_baseline
```

**FortiTran**:
```bash
python src/main.py \
    --model_name fortitran \
    --system_config_path config/system_config.yaml \
    --model_config_path config/fortitran.yaml \
    --train_set data/train \
    --val_set data/val \
    --test_set data/test \
    --exp_id fortitran_experiment
```

**AdaFortiTran**:
```bash
python src/main.py \
    --model_name adafortitran \
    --system_config_path config/system_config.yaml \
    --model_config_path config/adafortitran.yaml \
    --train_set data/train \
    --val_set data/val \
    --test_set data/test \
    --exp_id adafortitran_experiment
```

### Resume Training

```bash
python src/main.py \
    --model_name adafortitran \
    --system_config_path config/system_config.yaml \
    --model_config_path config/adafortitran.yaml \
    --train_set data/train \
    --val_set data/val \
    --test_set data/test \
    --exp_id resumed_experiment \
    --resume_from_checkpoint runs/adafortitran_experiment/best/checkpoint_epoch_50.pt
```

## Monitoring and Logging

### TensorBoard Integration

Training automatically logs metrics to TensorBoard:

```bash
tensorboard --logdir runs/
```

Available metrics/logs:
- Training/validation loss
- Learning rate
- Test performance across conditions (logged once training completes)
- Error visualizations
- Model hyperparameters

### Log Files

Training logs are saved to:
- `logs/training_{exp_id}.log`: Python logging output
- `runs/{model_name}_{exp_id}/`: TensorBoard logs and checkpoints

## Testing and Evaluation

### Automatic Testing

The training pipeline, once training finishes, automatically evaluates models across comprehensive test scenarios:

- **DS (Delay Spread)**: 7 conditions from 50-350 ns testing multipath robustness
- **SNR (Signal-to-Noise Ratio)**: 7 levels from 0-30 dB testing noise resilience  
- **MDS (Multi-Doppler Shift)**: 7 frequencies from 200-1400 Hz testing mobility adaptation

Results are logged per test condition, enabling detailed robustness analysis across different wireless environments.

### Manual Evaluation

```python
from src.models import AdaFortiTranEstimator
from src.config import load_config

# Load configurations
system_config, model_config = load_config(
    'config/system_config.yaml', 
    'config/adafortitran.yaml'
)

# Initialize model
model = AdaFortiTranEstimator(system_config, model_config)

# Load checkpoint
checkpoint = torch.load('checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Evaluate
model.eval()
# ... evaluation code
```

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
