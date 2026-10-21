
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """A small residual block for learning image corrections."""

    def __init__(self, channels: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activation(x + self.block(x))


class UnderwaterRestorationNet(nn.Module):
    """
    Input:  degraded RGB image, shape [B, 3, H, W], values in [0, 1]
    Output: restored RGB image, same shape, values in [0, 1]
    """

    def __init__(self, channels: int = 32, num_blocks: int = 5):
        super().__init__()

        self.input_conv = nn.Conv2d(
            3, channels, kernel_size=3, padding=1
        )

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_blocks)]
        )

        self.output_conv = nn.Conv2d(
            channels, 3, kernel_size=3, padding=1
        )

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        features = self.activation(self.input_conv(x))
        features = self.residual_blocks(features)

        # Predict a correction to the original image.
        correction = self.output_conv(features)
        restored = x + correction

        return torch.clamp(restored, 0.0, 1.0)
