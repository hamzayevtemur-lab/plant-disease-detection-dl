from dataclasses import dataclass
from typing import Tuple
import torch
import torch.nn as nn


@dataclass
class CNNConfig:
    name: str = "baseline"
    num_classes: int = 38
    in_channels: int = 3
    channels: Tuple[int, ...] = (32, 64, 128, 256)
    kernel_size: int = 3
    pool_size: int = 2
    dropout: float = 0.5


class PlantDiseaseCNN(nn.Module):
    def __init__(self, config: CNNConfig):
        super().__init__()
        self.config = config

        layers = []
        in_channels = config.in_channels

        for out_channels in config.channels:
            layers.extend([
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=config.kernel_size,
                    padding=config.kernel_size // 2
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=config.pool_size)
            ])
            in_channels = out_channels

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.channels[-1], config.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W)
        x = self.features(x)
        # (B, final_channels, 1, 1)
        x = self.pool(x)
        # (B, final_channels)
        x = x.flatten(1)
        x = self.dropout(x)
        # (B, num_classes)
        x = self.fc(x)
        return x

    def count_parameters(self) -> int:
        """Returns total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self) -> str:
        """Returns a string summary of the model architecture."""
        lines = [
            f"Model: {self.__class__.__name__} ({self.config.name})",
            f"Input channels: {self.config.in_channels}",
            f"Channel sequence: {self.config.channels}",
            f"Kernel size: {self.config.kernel_size}x{self.config.kernel_size}, Pool: {self.config.pool_size}x{self.config.pool_size}",
            f"Dropout: {self.config.dropout}",
            f"Output classes: {self.config.num_classes}",
            f"Trainable parameters: {self.count_parameters():,}"
        ]
        return "\n".join(lines)