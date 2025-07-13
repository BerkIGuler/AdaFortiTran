from torch import nn
import torch

from .positional_encodings import LearnablePositionalEncoding, SinusoidalPositionalEncoding


class TransformerEncoderForChannels(nn.Module):
    """Transformer encoder for channels"""

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            model_dim: int = 128,
            num_head: int = 4,
            activation: str = "gelu",
            dropout: float = 0.1,
            num_layers: int = 3,
            max_len: int = 512,
            pos_encoding_type: str = "learnable"
    ):
        """Initialize the encoder.

        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            model_dim: Model dimension
            num_head: Number of attention heads
            activation: Activation function name
            dropout: Dropout rate
            num_layers: Number of transformer layers
            max_len: Maximum sequence length
            pos_encoding_type: Type of positional encoding
        """
        super().__init__()
        self.linear_1 = nn.Linear(input_dim, model_dim)
        if pos_encoding_type == "learnable":
            self.positional_encoding = LearnablePositionalEncoding(max_len, model_dim)
        elif pos_encoding_type == "sinusoidal":
            self.positional_encoding = SinusoidalPositionalEncoding(max_len, model_dim)
        else:
            raise ValueError("pos_encoding_type must be 'learnable' or 'sinusoidal'")

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_head,
            dim_feedforward=2 * model_dim,
            activation=activation,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            transformer_layer,
            num_layers=num_layers
        )
        self.linear_2 = nn.Linear(model_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input through the encoder.

        Args:
            x: Input tensor (batch_size, seq_length, input_dim)

        Returns:
            Processed tensor (batch_size, seq_length, output_dim)
        """
        x = self.linear_1(x)
        x = self.positional_encoding(x)
        x = self.transformer(x)
        return self.linear_2(x)
