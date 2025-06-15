from pydantic import BaseModel, Field, model_validator
from typing import Self


class OFDMParams(BaseModel):
    num_scs: int = Field(..., gt=0, description="Number of sub-carriers")
    num_symbols: int = Field(..., gt=0, description="Number of OFDM symbols")


class PilotParams(BaseModel):
    num_scs: int = Field(..., gt=0, description="Number of pilots across sub-carriers")
    num_symbols: int = Field(..., gt=0, description="Number of pilots across OFDM symbols")


class SystemConfig(BaseModel):
    ofdm: OFDMParams
    pilot: PilotParams
    device: str = Field(
        default="cpu",
        pattern=r"^(cpu|cuda(:\d+)?)$",  # Updated regex to allow cuda:x
        description="Target device (cpu, cuda, or cuda:x where x is device index)"
    )

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
