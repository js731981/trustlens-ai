"""Feedforward model: four scalar signals -> trust_score in [0, 1]."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Feature order: ranking consistency, sentiment, feature coverage, LLM confidence
INPUT_FEATURE_NAMES = (
    "ranking_consistency",
    "sentiment",
    "feature_coverage",
    "llm_confidence",
)
INPUT_DIM = len(INPUT_FEATURE_NAMES)


class TrustScoreMLP(nn.Module):
    """Simple MLP mapping 4 inputs to a single trust score in (0, 1)."""

    def __init__(
        self,
        hidden_dims: tuple[int, ...] = (32, 16),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        dims = (INPUT_DIM, *hidden_dims, 1)
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU(inplace=True))
                if dropout > 0:
                    layers.append(nn.Dropout(p=dropout))
        self.net = nn.Sequential(*layers)
        self._final_act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return self._final_act(logits)


class DummyTrustDataset(Dataset):
    """Synthetic samples in [0,1]^4 with a noisy linear target (for smoke training only)."""

    def __init__(self, num_samples: int, seed: int = 42) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.features = torch.rand(num_samples, INPUT_DIM, generator=g)
        # Ground truth: weighted sum + noise (replace with real labels later)
        w = torch.tensor([0.35, 0.25, 0.20, 0.20])
        noise = 0.08 * torch.randn(num_samples, generator=g)
        raw = (self.features * w).sum(dim=1) + noise
        self.targets = raw.clamp(0.0, 1.0).unsqueeze(1)

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 50
    batch_size: int = 32
    lr: float = 1e-2
    weight_decay: float = 1e-4
    device: str | None = None  # default: cuda if available else cpu
    seed: int = 42


def train_trust_model(
    model: TrustScoreMLP,
    train_loader: DataLoader,
    config: TrainConfig | None = None,
) -> list[float]:
    """Train with MSE on trust targets; returns per-epoch mean loss."""
    cfg = config or TrainConfig()
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device_str = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    history: list[float] = []
    model.train()
    for _ in range(cfg.epochs):
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.detach())
            n_batches += 1
        history.append(epoch_loss / max(n_batches, 1))
    return history


def _demo() -> None:
    ds = DummyTrustDataset(num_samples=512, seed=0)
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    net = TrustScoreMLP(hidden_dims=(32, 16), dropout=0.05)
    hist = train_trust_model(net, loader, TrainConfig(epochs=30, batch_size=32, lr=1e-2))
    net.eval()
    with torch.no_grad():
        sample = ds.features[:5]
        out = net(sample)
    print("First epoch loss:", f"{hist[0]:.6f}", "| Last:", f"{hist[-1]:.6f}")
    print("Sample preds:", out.squeeze(-1).tolist())


if __name__ == "__main__":
    _demo()
