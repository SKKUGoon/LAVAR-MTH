import torch
from typing import Optional


# Simple scalers (fit on train only)
class StandardScalerTorch:
    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None

    def fit(self, x: torch.Tensor) -> "StandardScalerTorch":
        self.mean = x.mean(dim=0, keepdim=True)
        self.std = x.std(dim=0, keepdim=True).clamp_min(self.eps)
        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        assert self.mean is not None and self.std is not None
        return (x - self.mean) / self.std

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        assert self.mean is not None and self.std is not None
        return x * self.std + self.mean