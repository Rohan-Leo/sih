"""SpeedNet: forward speed (+ its uncertainty) and a stationary flag from an IMU window."""
from __future__ import annotations

import torch
from torch import nn

from .features import CHANNELS, WINDOW


class SpeedNet(nn.Module):
    def __init__(self, mean: torch.Tensor | None = None, std: torch.Tensor | None = None, width: int = 48):
        super().__init__()
        c = len(CHANNELS)
        # normalisation baked into the model so the exported graph takes raw features
        self.register_buffer("mean", mean if mean is not None else torch.zeros(c))
        self.register_buffer("std", std if std is not None else torch.ones(c))
        # post-hoc variance calibration factor (fitted on validation data)
        self.register_buffer("var_scale", torch.ones(1))
        self.body = nn.Sequential(
            nn.Conv1d(c, width, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(width, width, 5, padding=4, dilation=2),
            nn.GELU(),
            nn.Conv1d(width, width, 5, padding=8, dilation=4),
            nn.GELU(),
            nn.Conv1d(width, width, 5, padding=16, dilation=8),
            nn.GELU(),
            nn.Conv1d(width, width, 5, padding=32, dilation=16),
            nn.GELU(),
        )
        self.head = nn.Sequential(nn.Linear(2 * width, 32), nn.GELU(), nn.Dropout(0.2), nn.Linear(32, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, C, WINDOW) → (N, 3): [speed m/s, log-variance, stationary logit]"""
        x = (x - self.mean[None, :, None]) / self.std[None, :, None]
        h = self.body(x)
        z = torch.cat([h.mean(-1), h[..., -1]], dim=1)
        o = self.head(z)
        speed = nn.functional.softplus(o[:, 0])
        logvar = o[:, 1].clamp(-4, 6) + self.var_scale.log()
        return torch.stack([speed, logvar, o[:, 2]], 1)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())



