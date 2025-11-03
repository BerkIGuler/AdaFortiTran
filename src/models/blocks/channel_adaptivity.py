from torch import nn
from typing import Tuple
import torch


class ChannelAdapter(nn.Module):
    """Nonlinear encoder for channel condition tokens.

    Creates token embeddings for SNR, delay spread, and Doppler shift parameters.
    Each embedding is conditioned on a single real value and is the output of an MLP
    """

    def __init__(self, hidden_sizes: Tuple[int, int, int]) -> None:
        """Initialize the token encoder modules.

        Args:
            hidden_sizes: Tuple of hidden layer dimensions (h1, h2, h3) for the MLP encoders
        """
        super().__init__()
        self.snr_encoder = self._create_mlp(hidden_sizes)
        self.ds_encoder = self._create_mlp(hidden_sizes)
        self.dop_encoder = self._create_mlp(hidden_sizes)

    @staticmethod
    def _create_mlp(hidden_sizes: Tuple[int, int, int]) -> nn.Sequential:
        """Create a multi-layer perceptron with specified dimensions.

        Args:
            hidden_sizes: Tuple of hidden layer dimensions (h1, h2, h3)

        Returns:
            Sequential MLP model with ReLU activations between linear layers
        """
        return nn.Sequential(
            nn.Linear(1, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], hidden_sizes[2])
        )

    def forward(
            self,
            snr: torch.Tensor,
            delay_spread: torch.Tensor,
            doppler_shift: torch.Tensor
    ) -> torch.Tensor:
        """Create token embeddings from channel conditions.

        Args:
            snr: Signal-to-Noise Ratio tensor of shape (batch_size, 1)
            delay_spread: Delay spread tensor of shape (batch_size, 1)
            doppler_shift: Doppler shift tensor of shape (batch_size, 1)

        Returns:
            Concatenated token embeddings of shape (batch_size, h3/2, 6)
            where h3 is the third element of hidden_sizes
        """
        batch_size = snr.shape[0]
        snr_emb = torch.reshape(self.snr_encoder(snr), (batch_size, -1, 2))
        ds_emb = torch.reshape(self.ds_encoder(delay_spread), (batch_size, -1, 2))
        dop_emb = torch.reshape(self.dop_encoder(doppler_shift), (batch_size, -1, 2))
        return torch.cat((snr_emb, ds_emb, dop_emb), dim=2)
