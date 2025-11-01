from pydantic import BaseModel, Field, model_validator
from typing import Self, Tuple, List, Optional, Literal
import torch


class OFDMParams(BaseModel):
    # ... means required (i.e. no default value)
    # gt=0 means greater than 0
    num_scs: int = Field(..., gt=0, description="Number of OFDM subcarriers")
    num_symbols: int = Field(..., gt=0, description="Number of OFDM symbols")


class PilotParams(BaseModel):
    # ... means required (i.e. no default value)
    # gt=0 means greater than 0
    num_scs: int = Field(..., gt=0, description="Number of pilots across sub-carriers")
    num_symbols: int = Field(..., gt=0, description="Number of pilots across OFDM symbols")


class SystemConfig(BaseModel):
    """System configuration for OFDM and pilot parameters.
    
    Validates that pilot parameters (subcarriers and symbols) do not exceed
    the corresponding OFDM parameters.
    """
    ofdm: OFDMParams
    pilot: PilotParams

    @model_validator(mode='after')  # validates after all fields are initialized
    def validate_pilot_constraints(self) -> Self:
        """Ensure pilot parameters don't exceed OFDM parameters."""
        if self.pilot.num_scs > self.ofdm.num_scs:
            raise ValueError(
                f"Pilot sub-carriers ({self.pilot.num_scs}) cannot exceed "
                f"OFDM sub-carriers ({self.ofdm.num_scs})"
            )

        if self.pilot.num_symbols > self.ofdm.num_symbols:
            raise ValueError(
                f"Pilot symbols ({self.pilot.num_symbols}) cannot exceed "
                f"OFDM symbols ({self.ofdm.num_symbols})"
            )
        return self

    model_config = {"extra": "forbid"}  # forbid extra fields


class BaseConfig(BaseModel):
    """Base configuration class with device validation."""
    
    device: str = Field(default="cpu", description="Computing device to use")

    @model_validator(mode='after')  # validates after all fields are initialized
    def validate_device(self) -> Self:
        """Validate that the specified device is available."""
        device_str = self.device.lower()

        # automatically selects best available device
        if device_str == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = 'mps'  # Apple Silicon (MPS)
            else:
                self.device = 'cpu'
            return self

        elif device_str == 'cpu':
            return self

        # Validate CUDA devices
        elif device_str.startswith('cuda'):
            if not torch.cuda.is_available():
                raise ValueError("CUDA is not available on this system")

            # Handle specific CUDA device (e.g., 'cuda:0', 'cuda:1')
            if ':' in device_str:
                try:
                    device_id = int(device_str.split(':')[1])
                    if device_id >= torch.cuda.device_count():
                        available_devices = list(range(torch.cuda.device_count()))
                        raise ValueError(
                            f"CUDA device {device_id} not available. "
                            f"Available CUDA devices: {available_devices}"
                        )
                except (ValueError, IndexError) as e:
                    if "invalid literal" in str(e):
                        raise ValueError(f"Invalid CUDA device format: {device_str}")
                    raise

            return self

        # Validate MPS (Apple Silicon)
        if device_str == 'mps':
            if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                raise ValueError("MPS is not available/detected on this system")
            return self

        # If we get here, the device is not recognized
        available_devices = ['cpu']
        if torch.cuda.is_available():
            cuda_devices = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
            available_devices.extend(['cuda'] + cuda_devices)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            available_devices.append('mps')

        raise ValueError(
            f"Unsupported device: '{self.device}'. "
            f"Available devices: {available_devices}"
        )


class ModelConfig(BaseConfig):
    """Configuration for model architecture and training parameters.
    
    Validates model-specific requirements (e.g., AdaFortiTran requires
    adaptive_token_length and channel_adaptivity_hidden_sizes).
    """

    model_type: Literal["linear", "fortitran", "adafortitran"] = Field(
        default="fortitran",
        description="Type of model (linear, fortitran, or adafortitran)"
    )
    patch_size: Tuple[int, int] = Field(..., description="Patch size as (subcarriers_per_patch, symbols_per_patch)")
    num_layers: int = Field(..., gt=0, description="Number of transformer layers")
    model_dim: int = Field(..., gt=0, description="Model dimension")
    num_head: int = Field(..., gt=0, description="Number of attention heads")
    activation: Literal["relu", "gelu"] = Field(
        default="gelu", 
        description="Activation function used within the transformer's MLP block"
    )
    dropout: float = Field(default=0.1, ge=0.0, le=1.0, description="Dropout rate used within the transformer's FFN")
    max_seq_len: int = Field(default=512, gt=0, description="Maximum sequence length")
    pos_encoding_type: Literal["learnable", "sinusoidal"] = Field(
        default="learnable", 
        description="Positional encoding type"
    )
    adaptive_token_length: Optional[int] = Field(
        default=None, 
        gt=0, 
        description="Adaptive token length (required for AdaFortiTran)"
    )
    channel_adaptivity_hidden_sizes: Optional[List[int]] = Field(
        default=None, 
        description="Hidden sizes for channel adaptation MLP (required for AdaFortiTran)"
    )

    @model_validator(mode='after')
    def validate_model_specific_requirements(self) -> Self:
        """Validate model-specific configuration requirements."""
        if self.model_type == "linear":
            # Linear model only needs device, no additional validation required
            pass
        elif self.model_type == "adafortitran":
            if self.channel_adaptivity_hidden_sizes is None:
                raise ValueError(
                    "channel_adaptivity_hidden_sizes is required for AdaFortiTran model"
                )
            if self.adaptive_token_length is None:
                raise ValueError(
                    "adaptive_token_length is required for AdaFortiTran model"
                )
        elif self.model_type == "fortitran":
            if self.channel_adaptivity_hidden_sizes is not None:
                raise ValueError(
                    "channel_adaptivity_hidden_sizes should not be provided for FortiTran model"
                )
            if self.adaptive_token_length is not None:
                raise ValueError(
                    "adaptive_token_length should not be provided for FortiTran model"
                )
        
        return self

    model_config = {"extra": "forbid"}  # forbid extra fields
