from torch import nn
import torch


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding"""

    def __init__(self, max_len: int, d_model: int):
        """Initialize the positional encoding.

        Args:
            max_len: Maximum sequence length
            d_model: Model dimension
        """
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) *
            (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor.

        Args:
            x: Input tensor (batch_size, seq_length, d_model)

        Returns:
            Tensor with added positional encodings (batch_size, seq_length, d_model)
        """
        return x + self.pe[:, :x.size(1), :]


class LearnablePositionalEncoding(nn.Module):
    """Learnable positional encoding for transformers."""

    def __init__(self, max_len: int, d_model: int):
        """Initialize the learnable encoding.

        Args:
            max_len: Maximum sequence length
            d_model: Model dimension
        """
        super().__init__()
        self.position_embeddings = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.position_embeddings, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add learnable positional encoding to input tensor.

        Args:
            x: Input tensor (batch_size, seq_length, d_model)

        Returns:
            Tensor with added positional encodings (batch_size, seq_length, d_model)
        """
        return x + self.position_embeddings[:, :x.size(1), :]
