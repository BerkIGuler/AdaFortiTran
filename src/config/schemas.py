from pydantic import BaseModel, Field, model_validator
from typing import Self, Tuple


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
        pass





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