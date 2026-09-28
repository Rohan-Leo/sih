"""
Small networks shared by Clew's models.

Every model is an *encoder* plus a task head, and all of them follow the same
layout, so one dependency-free TypeScript runner (src/ml/tinynet.ts) can
execute any of them in the browser:

  mean, std        input normalisation baked into the graph (raw features in)
  body.{0,2,4..}   Conv1d layers, dilation 1, 2, 4, …, each followed by GELU
                   (TCN encoder; absent for tabular models)
  head.{i}         Linear layers, GELU between them (Dropout is a no-op at
                   inference and has no weights)

TCN encoder output is [mean over time ‖ last time step] of the final conv.
"""
from __future__ import annotations

import torch
from torch import nn


class TCN(nn.Module):
    """Dilated causal-window 1D CNN encoder + MLP head, for (N, C, T) IMU windows."""

    def __init__(self, in_ch: int, n_out: int, mean: torch.Tensor | None = None, std: torch.Tensor | None = None,
                 width: int = 32, layers: int = 5, hidden: int = 32, k: int = 5):
        super().__init__()
        self.register_buffer("mean", mean if mean is not None else torch.zeros(in_ch))
        self.register_buffer("std", std if std is not None else torch.ones(in_ch))
        body: list[nn.Module] = []
        c = in_ch
        for i in range(layers):
            d = 2**i
            body += [nn.Conv1d(c, width, k, padding=(k - 1) // 2 * d, dilation=d), nn.GELU()]
            c = width
        self.body = nn.Sequential(*body)
        self.head = nn.Sequential(nn.Linear(2 * width, hidden), nn.GELU(), nn.Dropout(0.2), nn.Linear(hidden, n_out))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean[None, :, None]) / self.std[None, :, None]
        h = self.body(x)
        return torch.cat([h.mean(-1), h[..., -1]], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x))


class MLP(nn.Module):
    """Tabular encoder + head for per-fix / per-second feature vectors (N, F)."""

    def __init__(self, in_dim: int, n_out: int, mean: torch.Tensor | None = None, std: torch.Tensor | None = None,
                 hidden: int = 64):
        super().__init__()
        self.register_buffer("mean", mean if mean is not None else torch.zeros(in_dim))
        self.register_buffer("std", std if std is not None else torch.ones(in_dim))
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, n_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head((x - self.mean) / self.std)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def fit(model: nn.Module, Xtr: torch.Tensor, ytr: torch.Tensor, Xva: torch.Tensor, yva: torch.Tensor, loss_fn,
        score_fn, epochs: int = 15, bs: int = 256, lr: float = 3e-3, augment=None, log=print) -> float:
    """AdamW + one-cycle; keeps the state with the best validation score (lower is better)."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * (len(Xtr) // bs + 1))
    best, best_state = float("inf"), None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), bs):
            b = perm[i : i + bs]
            xb = Xtr[b]
            if augment is not None:
                xb = augment(xb)
            opt.zero_grad()
            loss = loss_fn(model(xb), ytr[b])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            o = model(Xva)
            vl, sc = float(loss_fn(o, yva)), float(score_fn(o, yva))
        log(f"  epoch {ep + 1:2d}  val loss {vl:.4f}  val score {sc:.4f}")
        if sc < best:
            best, best_state = sc, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return best
