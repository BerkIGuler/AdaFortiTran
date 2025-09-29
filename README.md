# AdaFortiTran: Adaptive Transformer Model for Robust OFDM Channel Estimation

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-red.svg)](https://pytorch.org/)

Official implementation of [AdaFortiTran: An Adaptive Transformer Model for Robust OFDM Channel Estimation](https://ieeexplore.ieee.org/document/11160810) accepted at ICC 2025, Montreal, Canada.

## 📖 Overview

AdaFortiTran is a novel adaptive transformer-based model for OFDM channel estimation that dynamically adapts to varying channel conditions (SNR, delay spread, Doppler shift). The model combines the power of transformer architectures with channel-aware adaptation mechanisms to achieve robust performance across diverse wireless environments.

### Key Features
- **🔄 Adaptive Architecture**: Dynamically adapts to channel conditions using meta-information
- **⚡ High Performance**: State-of-the-art results on OFDM channel estimation tasks
- **🧠 Transformer-Based**: Leverages attention mechanisms for long-range dependencies
- **🎯 Robust**: Maintains performance across varying SNR, delay spread, and Doppler conditions
- **🚀 Production Ready**: Comprehensive training pipeline with advanced features

## 🏗️ Architecture

The project implements three model variants:

1. **Linear Estimator**: Simple learned linear estimator baseline
2. **FortiTran**: Base transformer-based channel estimator
3. **AdaFortiTran**: Adaptive version of FortiTran with channel condition awareness

### Model Comparison

| Model | Channel Adaptation | Complexity | Performance |
|-------|-------------------|------------|-------------|
| Linear | ❌ | Low | Baseline |
| FortiTran | ❌ | Medium | Good |
| AdaFortiTran | ✅ | High | **Best** |

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/AdaFortiTran.git
   cd AdaFortiTran
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Basic Training

Train an AdaFortiTran model with default settings:

```bash
python src/main.py \
    --model_name adafortitran \
    --system_config_path config/system_config.yaml \
    --model_config_path config/adafortitran.yaml \
    --train_set data/train \
    --val_set data/val \
    --test_set data/test \
    --exp_id my_experiment
```

### Advanced Training

Use all available features for optimal performance:

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

## 📁 Project Structure

```
AdaFortiTran/
├── config/                     # Configuration files
│   ├── system_config.yaml     # OFDM system parameters
│   ├── adafortitran.yaml      # AdaFortiTran model config
│   ├── fortitran.yaml         # FortiTran model config
│   └── linear.yaml            # Linear model config
├── data/                      # Dataset directory
│   ├── train/                 # Training data
│   ├── val/                   # Validation data
│   └── test/                  # Test data (DS, MDS, SNR sets)
├── src/                       # Source code
│   ├── main/                  # Training pipeline
│   │   ├── trainer.py         # Enhanced ModelTrainer
│   │   └── parser.py          # Command-line argument parser
│   ├── models/                # Model implementations
│   │   ├── adafortitran.py    # AdaFortiTran model
│   │   ├── fortitran.py       # FortiTran model
│   │   ├── linear.py          # Linear model
│   │   └── blocks/            # Model building blocks
│   ├── data/                  # Data loading
│   │   └── dataset.py         # Dataset and DataLoader classes
│   ├── config/                # Configuration management
│   │   ├── config_loader.py   # YAML configuration loader
│   │   └── schemas.py         # Pydantic validation schemas
│   └── utils.py               # Utility functions
├── requirements.txt           # Python dependencies
├── README.md                  # This file
```

## ⚙️ Configuration

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

### Model Configuration (`config/adafortitran.yaml`)

Defines model architecture parameters:

```yaml
model_type: 'adafortitran'
patch_size: [3, 2]                    # Patch dimensions
num_layers: 6                         # Transformer layers
model_dim: 128                        # Model dimension
num_head: 4                           # Attention heads
activation: 'gelu'                    # Activation function
dropout: 0.1                          # Dropout rate
max_seq_len: 512                      # Maximum sequence length
pos_encoding_type: 'learnable'        # Positional encoding
channel_adaptivity_hidden_sizes: [7, 42, 560]  # Adaptation layers
adaptive_token_length: 6              # Adaptive token length
```

## 🎯 Training Features

### Advanced Training Options

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

### Performance Optimizations

- **Mixed Precision Training**: Faster training on modern GPUs
- **Optimized Data Loading**: Configurable workers and memory pinning
- **Gradient Clipping**: Stable training with configurable clipping
- **Early Stopping**: Automatic training termination on plateau

## 📊 Dataset Format

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
    ├── DS_test_set/          # Delay Spread tests
    │   ├── DS_50/
    │   ├── DS_100/
    │   └── ...
    ├── SNR_test_set/         # SNR tests
    │   ├── SNR_10/
    │   ├── SNR_20/
    │   └── ...
    └── MDS_test_set/         # Multi-Doppler tests
        ├── DOP_200/
        ├── DOP_400/
        └── ...
```

### File Naming Convention

Files must follow the pattern:
```
{file_number}_SNR-{snr}_DS-{delay_spread}_DOP-{doppler}_N-{pilot_freq}_{channel_type}.mat
```

Example: `1_SNR-20_DS-50_DOP-500_N-3_TDL-A.mat`

### Data Format

Each `.mat` file must contain a variable `H` with shape `[subcarriers, symbols, 3]`:
- `H[:, :, 0]`: Ground truth channel (complex values)
- `H[:, :, 1]`: LS channel estimate with zeros for non-pilot positions
- `H[:, :, 2]`: Reserved for future use

## 🔧 Usage Examples

### Training Different Models

**Linear Estimator**:
```bash
python src/main.py \
    --model_name linear \
    --system_config_path config/system_config.yaml \
    --model_config_path config/linear.yaml \
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

## 📈 Monitoring and Logging

### TensorBoard Integration

Training automatically logs metrics to TensorBoard:

```bash
tensorboard --logdir runs/
```

Available metrics:
- Training/validation loss
- Learning rate
- Test performance across conditions
- Error visualizations
- Model hyperparameters

### Log Files

Training logs are saved to:
- `logs/training_{exp_id}.log`: Python logging output
- `runs/{model_name}_{exp_id}/`: TensorBoard logs and checkpoints

## 🧪 Testing and Evaluation

### Automatic Testing

The training pipeline automatically evaluates models on:
- **DS (Delay Spread)**: Varying delay spread conditions
- **SNR**: Different signal-to-noise ratios
- **MDS (Multi-Doppler)**: Various Doppler shift scenarios

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

## 🔬 Research and Development

### Adding Custom Callbacks

```python
from src.main.trainer import Callback, TrainingMetrics

class CustomCallback(Callback):
    def on_epoch_end(self, epoch: int, metrics: TrainingMetrics) -> None:
        # Custom logic here
        print(f"Epoch {epoch}: Train Loss = {metrics.train_loss:.4f}")
```

### Extending Models

The modular architecture makes it easy to add new model variants:

```python
from src.models.fortitran import BaseFortiTranEstimator

class CustomEstimator(BaseFortiTranEstimator):
    def __init__(self, system_config, model_config):
        super().__init__(system_config, model_config, use_channel_adaptation=True)
        # Add custom components
```

## 📚 Citation

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

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2025 [Berkay Guler/University of California, Irvine]
