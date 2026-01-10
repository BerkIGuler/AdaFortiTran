from pydantic import BaseModel, Field, model_validator
from typing import Self, Tuple, List, Optional, Literal


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
    """Base configuration class for model configurations.
    
    Note: This class contains only model-specific parameters.
    Experiment-specific parameters (device, batch_size, etc.) are handled
    separately in TrainingArguments and passed via command line.
    """
    pass


class ModelConfig(BaseConfig):
    """Configuration for model architecture and training parameters.
    
    Validates model-specific requirements (e.g., AdaFortiTran requires
    adaptive_token_length and channel_adaptivity_hidden_sizes).
    """

    model_type: Literal["linear", "fortitran", "adafortitran"] = Field(
        default="linear",
        description="Type of model (linear, fortitran, or adafortitran)"
    )
    # Optional fields for transformer-based models (not required for linear)
    patch_size: Optional[Tuple[int, int]] = Field(default=None, description="Patch size as (subcarriers_per_patch, symbols_per_patch)")
    num_layers: Optional[int] = Field(default=None, gt=0, description="Number of transformer layers")
    model_dim: Optional[int] = Field(default=None, gt=0, description="Model dimension")
    num_head: Optional[int] = Field(default=None, gt=0, description="Number of attention heads")
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
        elif self.model_type in ["fortitran", "adafortitran"]:
            # Transformer-based models require these fields
            required_fields = ["patch_size", "num_layers", "model_dim", "num_head"]
            for field in required_fields:
                if getattr(self, field) is None:
                    raise ValueError(f"{field} is required for {self.model_type} model")
            
            # AdaFortiTran-specific requirements
            if self.model_type == "adafortitran":
                if self.channel_adaptivity_hidden_sizes is None:
                    raise ValueError(
                        "channel_adaptivity_hidden_sizes is required for AdaFortiTran model"
                    )
                if self.adaptive_token_length is None:
                    raise ValueError(
                        "adaptive_token_length is required for AdaFortiTran model"
                    )
            # FortiTran-specific constraints
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
