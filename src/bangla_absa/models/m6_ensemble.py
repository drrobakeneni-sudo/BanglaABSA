import torch
from torch import nn


def weighted_features(p4, p5, weights):
    """Branch-major: M4(Product,Shop,Transport), then M5; each N,Neu,P."""
    if p4.shape != p5.shape or p4.ndim != 3 or p4.shape[1:] != (3, 3):
        raise ValueError("Branch probabilities must have shape [B,3,3]")
    if weights.shape != (3, 2) or not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Weights must be finite nonnegative [3,2]")
    if not torch.allclose(weights.sum(-1), torch.ones(3, device=weights.device)):
        raise ValueError("Weights must sum to one per aspect")
    for p in (p4, p5):
        if (
            not torch.isfinite(p).all()
            or (p < 0).any()
            or not torch.allclose(p.sum(-1), torch.ones_like(p[..., 0]), atol=1e-5)
        ):
            raise ValueError("Invalid branch probabilities")
    return torch.cat(
        [(p4 * weights[:, 0, None]).flatten(1), (p5 * weights[:, 1, None]).flatten(1)], -1
    )


class Stacker(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(18, 768),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(768, 576),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(576, 9),
        )
        assert sum(p.numel() for p in self.parameters()) == 462729

    def forward(self, features):
        if isinstance(features, dict):
            features = features["features"]
        if features.ndim != 2 or features.shape[1] != 18:
            raise ValueError("Stacker requires exactly 18 features")
        return self.mlp(features).reshape(-1, 3, 3)


class EnsembleNet(nn.Module):
    def __init__(self, branch4, branch5, weights=None, mode="stacking"):
        super().__init__()
        if mode not in {"stacking", "equal_vote", "weighted_vote", "unweighted_stacking"}:
            raise ValueError("Unknown fusion mode")
        self.branches = nn.ModuleList([branch4, branch5])
        for branch in self.branches:
            branch.requires_grad_(False).eval()
        self.mode = mode
        self.register_buffer("weights", torch.full((3, 2), 0.5) if weights is None else weights)
        self.stacker = Stacker() if "stacking" in mode else None

    def train(self, mode=True):
        super().train(mode)
        for branch in self.branches:
            branch.eval()
        return self

    def forward(self, batch):
        with torch.no_grad():
            p4, p5 = [branch(batch).softmax(-1) for branch in self.branches]
        weights = (
            torch.full_like(self.weights, 0.5)
            if self.mode in {"equal_vote", "unweighted_stacking"}
            else self.weights
        )
        if self.stacker is not None:
            return self.stacker(weighted_features(p4, p5, weights))
        return (p4 * weights[:, 0, None] + p5 * weights[:, 1, None]).clamp_min(1e-12).log()
