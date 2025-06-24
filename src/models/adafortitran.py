from .fortitran import BaseFortiTranEstimator
from src.config.schemas import SystemConfig, ModelConfig


class AdaFortiTranEstimator(BaseFortiTranEstimator):
    """
    Adaptive Hybrid CNN-Transformer Channel Estimator for OFDM Systems with channel adaptation.

    This model extends the base estimator with channel adaptation capabilities,
    incorporating channel conditions (SNR, delay spread, Doppler shift) into
    the estimation process through conditional attention mechanisms.
    """

    def __init__(self, system_config: SystemConfig, model_config: ModelConfig) -> None:
        """
        Initialize the AdaFortiTranEstimator.

        Args:
            system_config: OFDM system configuration (subcarriers, symbols, pilot arrangement)
            model_config: Model architecture configuration (patch size, layers, etc.)
        """
        super().__init__(system_config, model_config, use_channel_adaptation=True)