import torch
import torch.nn as nn


class ConvEnhancer(nn.Module):
    """Convolutional enhancement network with 1->8->32->8->1 channel pattern."""

    def __init__(self):
        """Initialize the ConvEnhancer with convolutional blocks."""
        super(ConvEnhancer, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        """Forward pass through the convolutional enhancement network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, height, width)

        Returns:
            torch.Tensor: Enhanced tensor of shape (batch_size, 1, height, width)
        """
        return self.conv_block(x)