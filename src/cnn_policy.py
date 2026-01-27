from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallGridCNN(nn.Module):
    """Tiny CNN for (B, 2, 4, 7) grid inputs.

    Channel 0: obstacles (0/1)
    Channel 1: ego one-hot

    If wall_padding=True, the obstacle channel is padded with 1s (walls),
    and ego channel with 0s. In that case, conv1 uses padding=0 to avoid
    double padding.
    """

    def __init__(
        self,
        n_actions: int = 4,
        features_dim: int = 128,
        kernel_size: int = 5,
        stride: int = 1,
        padding: int = 2,
        padding_mode: str = "zeros",
        wall_padding: bool = False,
    ) -> None:
        super().__init__()

        self.wall_padding = bool(wall_padding)
        self.pad = int(padding)

        conv1_padding = 0 if self.wall_padding else self.pad

        self.conv1 = nn.Conv2d(
            in_channels=2,
            out_channels=32,
            kernel_size=int(kernel_size),
            stride=int(stride),
            padding=conv1_padding,
            padding_mode=str(padding_mode),
        )
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 7, int(features_dim)),
            nn.ReLU(),
            nn.Linear(int(features_dim), int(n_actions)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.wall_padding and self.pad > 0:
            p = self.pad
            obs = x[:, 0:1]
            ego = x[:, 1:2]
            obs = F.pad(obs, (p, p, p, p), value=1.0)
            ego = F.pad(ego, (p, p, p, p), value=0.0)
            x = torch.cat([obs, ego], dim=1)

        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        return self.head(x)
