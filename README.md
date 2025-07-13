---
language:
- en
tags:
- pytorch
- transformer
- channel-estimation
- ofdm
- wireless
- adaptive
license: mit
datasets:
- custom
metrics:
- mse
---

# AdaFortiTran: Adaptive Transformer Model for Robust OFDM Channel Estimation

## Model Description

AdaFortiTran is a novel adaptive transformer-based model for OFDM channel estimation that dynamically adapts to varying channel conditions (SNR, delay spread, Doppler shift). The model combines the power of transformer architectures with channel-aware adaptation mechanisms to achieve robust performance across diverse wireless environments.

## Key Features

- **🔄 Adaptive Architecture**: Dynamically adapts to channel conditions using meta-information
- **⚡ High Performance**: State-of-the-art results on OFDM channel estimation tasks
- **🧠 Transformer-Based**: Leverages attention mechanisms for long-range dependencies
- **🎯 Robust**: Maintains performance across varying SNR, delay spread, and Doppler conditions
- **🚀 Production Ready**: Comprehensive training pipeline with advanced features

## Architecture

The project implements three model variants:

1. **Linear Estimator**: Simple learned linear transformation baseline
2. **FortiTran**: Fixed transformer-based channel estimator
3. **AdaFortiTran**: Adaptive transformer with channel condition awareness

## Usage

### Installation

```bash
pip install -r requirements.txt
```

### Training

```bash
python src/main.py     --model_name adafortitran     --system_config_path config/system_config.yaml     --model_config_path config/adafortitran.yaml     --train_set data/train     --val_set data/val     --test_set data/test     --exp_id my_experiment
```

## Citation

If you use this model in your research, please cite:

```bibtex
@misc{guler2025adafortitranadaptivetransformermodel,
      title={AdaFortiTran: An Adaptive Transformer Model for Robust OFDM Channel Estimation}, 
      author={Berkay Guler and Hamid Jafarkhani},
      year={2025},
      eprint={2505.09076},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.09076}, 
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
