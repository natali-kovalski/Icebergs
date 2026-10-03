"""Small CNN for 75x75 SAR chips, with optional incidence angle as a scalar input."""

from __future__ import annotations

import torch
from torch import nn


def _block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class IcebergCNN(nn.Module):
    """4 conv blocks (75 -> 37 -> 18 -> 9 -> 4 px), global average pool, small head -> 1 logit."""

    def __init__(self, in_channels: int, use_inc: bool, width: int = 16, dropout: float = 0.3):
        super().__init__()
        w = width
        self.use_inc = use_inc
        self.features = nn.Sequential(
            _block(in_channels, w), _block(w, 2 * w), _block(2 * w, 4 * w), _block(4 * w, 8 * w)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(8 * w + int(use_inc), 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor, inc: torch.Tensor | None = None) -> torch.Tensor:
        z = self.pool(self.features(x)).flatten(1)
        if self.use_inc:
            if inc is None:
                raise ValueError("Model was built with use_inc=True; pass incidence")
            z = torch.cat([z, inc.view(-1, 1)], dim=1)
        return self.head(z).squeeze(1)
