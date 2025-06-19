import torch
from torch import nn
import logging

from src.config.schemas import SystemConfig, ModelConfig
from src.models.blocks import ConvEnhancer, PatchEmbedding, InversePatchEmbedding, TransformerEncoderForChannels


class FortiTranEstimator(nn.Module):
    """
    Hybrid CNN-Transformer Channel Estimator for OFDM Systems.

    This model performs channel estimation by:
    1. Upsampling pilot symbols to full OFDM grid size
    2. Applying convolutional enhancement for spatial features
    3. Converting to patch embeddings for transformer processing
    4. Using transformer encoder to capture long-range dependencies
    5. Reconstructing spatial representation and applying residual connections
    6. Final convolutional refinement for high-quality channel estimates
    """

    def __init__(self, system_config: SystemConfig, model_config: ModelConfig) -> None:
        """
        Initialize the FortiTranEstimator.

        Args:
            system_config: OFDM system configuration (subcarriers, symbols, pilot arrangement)
            model_config: Model architecture configuration (patch size, layers, etc.)
        """
        super().__init__()

        self.system_config = system_config
        self.model_config = model_config
        self.device = torch.device(model_config.device)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Cache key dimensions for efficiency
        self._setup_dimensions()

        # Initialize model components
        self._build_architecture()

        # Move model to specified device
        self.to(self.device)

        self._log_initialization_info()

    def _setup_dimensions(self) -> None:
        """Calculate and cache key dimensions from configuration."""
        # OFDM grid dimensions
        self.ofdm_size = (
            self.system_config.ofdm.num_scs,
            self.system_config.ofdm.num_symbols
        )

        # Pilot arrangement dimensions
        self.pilot_size = (
            self.system_config.pilot.num_scs,
            self.system_config.pilot.num_symbols
        )

        # Feature dimensions for linear layers
        self.pilot_features = self.pilot_size[0] * self.pilot_size[1]
        self.ofdm_features = self.ofdm_size[0] * self.ofdm_size[1]

        # Patch processing dimensions
        self.patch_length = (
                self.model_config.patch_size[0] * self.model_config.patch_size[1]
        )

    def _build_architecture(self) -> None:
        """Construct the model architecture components."""
        # 1. Pilot-to-OFDM upsampling
        self.pilot_upsampler = nn.Linear(self.pilot_features, self.ofdm_features)
        # 2. Initial convolutional enhancement
        self.initial_enhancer = ConvEnhancer()

        # 3. Patch embedding for transformer processing
        self.patch_embedder = PatchEmbedding(self.model_config.patch_size)

        # 4. Transformer encoder for sequence modeling
        self.transformer_encoder = TransformerEncoderForChannels(
            input_dim=self.patch_length,
            output_dim=self.patch_length,
            model_dim=self.model_config.model_dim,
            num_head=self.model_config.num_head,
            activation=self.model_config.activation,
            dropout=self.model_config.dropout,
            num_layers=self.model_config.num_layers,
            max_len=self.model_config.max_seq_len,
            pos_encoding_type=self.model_config.pos_encoding_type,
        )

        # 5. Patch reconstruction
        self.patch_reconstructor = InversePatchEmbedding(
            self.ofdm_size,
            self.model_config.patch_size
        )

        # 6. Final convolutional refinement
        self.final_refiner = ConvEnhancer()

    def _log_initialization_info(self) -> None:
        """Log model initialization details."""
        self.logger.info("FortiTranEstimator initialized successfully:")
        self.logger.info(f"  OFDM grid: {self.ofdm_size[0]}×{self.ofdm_size[1]} = {self.ofdm_features} elements")
        self.logger.info(f"  Pilot grid: {self.pilot_size[0]}×{self.pilot_size[1]} = {self.pilot_features} elements")
        self.logger.info(f"  Patch size: {self.model_config.patch_size}")
        self.logger.info(f"  Model dimension: {self.model_config.model_dim}")
        self.logger.info(f"  Transformer layers: {self.model_config.num_layers}")
        self.logger.info(f"  Device: {self.device}")

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.logger.info(f"  Total parameters: {total_params:,}")
        self.logger.info(f"  Trainable parameters: {trainable_params:,}")

    def forward(self, pilot_symbols: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for channel estimation.

        Args:
            pilot_symbols: Complex pilot symbols of shape [batch, pilot_scs, pilot_symbols]

        Returns:
            Estimated channel matrix of shape [batch, ofdm_scs, ofdm_symbols]
        """
        # Ensure input is on correct device
        pilot_symbols = pilot_symbols.to(self.device)

        # Process real and imaginary parts separately
        real_estimate = self._forward_real_valued(pilot_symbols.real)
        imag_estimate = self._forward_real_valued(pilot_symbols.imag)

        # Combine into complex tensor
        channel_estimate = torch.complex(real_estimate, imag_estimate)

        return channel_estimate

    def _forward_real_valued(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process real-valued input through the estimation pipeline.

        Args:
            x: Real-valued input tensor [batch, pilot_features] or [batch, pilot_scs, pilot_symbols]

        Returns:
            Real-valued channel estimate [batch, ofdm_scs, ofdm_symbols]
        """
        batch_size = x.shape[0]

        # Flatten spatial dimensions for linear upsampling
        if x.dim() > 2:
            x = x.view(batch_size, -1)

        # Stage 1: Upsample from pilot grid to OFDM grid
        upsampled = self.pilot_upsampler(x)

        # Reshape for convolutional processing
        upsampled_2d = upsampled.view(batch_size, 1, *self.ofdm_size)

        # Stage 2: Initial convolutional enhancement
        conv_enhanced = torch.squeeze(self.initial_enhancer(upsampled_2d), dim=1)

        # Stage 3: Convert to patch embeddings
        patch_embeddings = self.patch_embedder(conv_enhanced)

        # Stage 4: Transformer processing for long-range dependencies
        transformer_output = self.transformer_encoder(patch_embeddings)

        # Stage 5: Reconstruct spatial representation
        reconstructed = self.patch_reconstructor(transformer_output)

        # Stage 6: Apply residual connection
        residual_combined = conv_enhanced + reconstructed

        # Stage 7: Final convolutional refinement
        refined_output = torch.squeeze(self.final_refiner(torch.unsqueeze(residual_combined, dim=1)), dim=1)

        return refined_output

    def get_model_info(self) -> dict:
        """Return model configuration and statistics."""
        return {
            'model_name': self.__class__.__name__,
            'ofdm_size': self.ofdm_size,
            'pilot_size': self.pilot_size,
            'patch_size': self.model_config.patch_size,
            'patch_length': self.patch_length,
            'model_dim': self.model_config.model_dim,
            'num_layers': self.model_config.num_layers,
            'device': str(self.device),
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
