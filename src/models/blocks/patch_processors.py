from torch import nn
from typing import Tuple
import torch


class PatchEmbedding(nn.Module):
    """Transform channel matrix into sequence

    Extracts non-overlapping 2D regions from the matrix, flattens them
    and outputs a sequence of flattened vectors in row-major order.

    """

    def __init__(self, patch_size: Tuple[int, int] = (10, 4)):
        """Initialize the PatchEmbedding layer.

        Args:
            patch_size: Size of patches to extract (subcarriers_per_patch, symbols_per_patch)
        """
        super().__init__()
        self.patch_size = patch_size
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform input tensor into patch embeddings.

        Args:
            x: Input tensor of shape (batch_size, num_subcarriers, num_symbols)

        Returns:
            Tensor of shape (batch_size, num_patches, patch_size[0]*patch_size[1])
            where num_patches = (num_subcarriers // patch_size[0]) * (num_symbols // patch_size[1])
        """
        x = self.unfold(torch.unsqueeze(x, dim=1))
        return torch.permute(x, dims=(0, 2, 1))


class InversePatchEmbedding(nn.Module):
    """Transform patch embeddings back to original matrix format."""

    def __init__(
            self,
            output_size: Tuple[int, int] = (120, 14),
            patch_size: Tuple[int, int] = (3, 2)
    ):
        """Initialize the InversePatchEmbedding layer.

        Args:
            output_size: Size of output matrix (num_subcarriers, num_symbols)
            patch_size: Size of input patches (subcarriers_per_patch, symbols_per_patch)
        """
        super().__init__()
        self.fold = nn.Fold(
            output_size=output_size,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform patch embeddings back to matrix format.

        Args:
            x: Input tensor of shape (batch_size, num_patches, patch_size[0]*patch_size[1])
              where num_patches = (output_size[0] // patch_size[0]) * (output_size[1] // patch_size[1])

        Returns:
            Tensor of shape (batch_size, num_subcarriers, num_symbols)
        """
        x = torch.permute(x, dims=(0, 2, 1))
        x = self.fold(x)
        return torch.squeeze(x, dim=1)