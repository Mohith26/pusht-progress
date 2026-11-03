"""Small progress head with either an MSE or a K-bin distributional loss."""

import numpy as np
import torch
from torch import nn


def bin_centers(k: int) -> torch.Tensor:
    return (torch.arange(k, dtype=torch.float32) + 0.5) / k


def progress_to_bin(p: torch.Tensor, k: int) -> torch.Tensor:
    """Map progress in [0, 1] to a bin id in [0, k-1]."""
    return torch.clamp((p * k).long(), 0, k - 1)


class ProgressHead(nn.Module):
    def __init__(self, in_dim: int, loss: str = "dist", k: int = 20, hidden: int = 256):
        super().__init__()
        if loss not in ("mse", "dist"):
            raise ValueError(loss)
        self.loss_kind, self.k = loss, k
        out = k if loss == "dist" else 1
        # per-feature standardization, fitted on the training split in train_head
        self.register_buffer("mu", torch.zeros(in_dim))
        self.register_buffer("sd", torch.ones(in_dim))
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden, out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - self.mu) / self.sd)

    def predict(self, out: torch.Tensor) -> torch.Tensor:
        """Point estimate of progress in [0, 1]."""
        if self.loss_kind == "mse":
            return torch.sigmoid(out.squeeze(-1))
        return torch.softmax(out, dim=-1) @ bin_centers(self.k)

    def loss(self, out: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_kind == "mse":
            return nn.functional.mse_loss(torch.sigmoid(out.squeeze(-1)), target)
        return nn.functional.cross_entropy(out, progress_to_bin(target, self.k))


def train_head(x, y, loss="dist", k=20, epochs=15, lr=1e-3, batch=512, seed=0) -> ProgressHead:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    xt, yt = torch.from_numpy(x), torch.from_numpy(y.astype(np.float32))
    head = ProgressHead(x.shape[1], loss=loss, k=k)
    head.mu.copy_(xt.mean(0))
    head.sd.copy_(xt.std(0) + 1e-6)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        head.train()
        for idx in np.array_split(rng.permutation(len(x)), max(1, len(x) // batch)):
            opt.zero_grad()
            head.loss(head(xt[idx]), yt[idx]).backward()
            opt.step()
    head.eval()
    return head


def predict(head: ProgressHead, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return head.predict(head(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)))).numpy()
