from pydantic import BaseModel, Field, model_validator
from typing import Self, Tuple
import torch


class OFDMParams(BaseModel):
    num_scs: int = Field(..., gt=0, description="Number of sub-carriers")
    num_symbols: int = Field(..., gt=0, description="Number of OFDM symbols")


class PilotParams(BaseModel):
    num_scs: int = Field(..., gt=0, description="Number of pilots across sub-carriers")
    num_symbols: int = Field(..., gt=0, description="Number of pilots across OFDM symbols")


class ModelParams(BaseModel):
    patch_size: Tuple[int, int] = Field(default=(10, 4), description="Patch size as (height, width)")
    num_layers: int = Field(default=6, gt=0, description="Number of model layers")
    device: str = Field(default="cpu", description="Device to use")

    @model_validator(mode='after')
    def validate_device(self) -> Self:
        """Validate that the specified device is available."""
        device_str = self.device.lower()

        # Handle 'auto' case - automatically select best available device
        if device_str == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = 'mps'  # Apple Silicon
            else:
                self.device = 'cpu'
            return self

        # Validate CPU
        if device_str == 'cpu':
            return self

        # Validate CUDA devices
        if device_str.startswith('cuda'):
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
                raise ValueError("MPS is not available on this system")
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


class SystemConfig(BaseModel):
    ofdm: OFDMParams
    pilot: PilotParams

    @model_validator(mode='after')
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

    model_config = {"extra": "forbid"}


class ModelConfig(BaseModel):
    system: SystemConfig
    model: ModelParams

    @model_validator(mode='after')
    def validate_patch_constraints(self) -> Self:
        """Ensure patch size is compatible with OFDM dimensions."""
        patch_height, patch_width = self.model.patch_size

        if patch_height > self.system.ofdm.num_symbols:
            raise ValueError(
                f"Patch height ({patch_height}) cannot exceed "
                f"OFDM symbols ({self.system.ofdm.num_symbols})"
            )

        if patch_width > self.system.ofdm.num_scs:
            raise ValueError(
                f"Patch width ({patch_width}) cannot exceed "
                f"OFDM sub-carriers ({self.system.ofdm.num_scs})"
            )

        # Check if OFDM dimensions are divisible by patch size for clean patching
        if self.system.ofdm.num_symbols % patch_height != 0:
            raise ValueError(
                f"OFDM symbols ({self.system.ofdm.num_symbols}) must be divisible "
                f"by patch height ({patch_height}) for clean patching"
            )

        if self.system.ofdm.num_scs % patch_width != 0:
            raise ValueError(
                f"OFDM sub-carriers ({self.system.ofdm.num_scs}) must be divisible "
                f"by patch width ({patch_width}) for clean patching"
            )

        return self

    model_config = {"extra": "forbid"}
