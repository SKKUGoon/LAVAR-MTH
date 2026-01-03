# LAVAR: Latent Autoencoder VAR
# Combination of Encoder and Decoder with VAR as latent factor dynamics (nonlinear + VAR)
# - The model is non linear in the observation space, but linear in the latent state dynamics
# Intuition
# Mathmatics
# - $LatentDynamics(linear): z_t = f(z_{t-1}, \epsilon_t)$
# - $ObservationModel(nonlinear): x_t = g(z_t, \eta_t)$
# - z_t is a low dimentional and structured
# - g is a neural network that maps non linear observation space to latent space
# - f is a learnable VAR model with matrix parameters
#   - Latent dynamics are often approximately linear
#   - Non linearity is pushed into the observation model
#   - Linear latent dynamics will give us the interpretability

import torch
import torch.nn as nn
from typing import List, Optional

# Encoder and Decoder object
class MLP(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 hidden_dims: List[int], 
                 output_dim: int, 
                 activation: nn.Module = nn.ReLU(),
                 dropout_rate: float = 0.0):
        super().__init__()
        layers: List[nn.Module] = []
        dims = [input_dim] + hidden_dims

        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(d_in, d_out))
            layers.append(activation)
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))

        layers.append(nn.Linear(dims[-1], output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class SupplyHeadNB(nn.Module):
    """
    Negative Binomial head: predicts (mu, theta) per supply target.

    - mu: mean parameter, constrained to be positive via Softplus
    - theta: dispersion / total_count, constrained to be positive via Softplus
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Optional[List[int]] = [64, 64],
        activation: nn.Module = nn.ReLU(),
        eps: float = 1e-8,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.eps = eps

        layers: List[nn.Module] = []
        dims = [input_dim] + (hidden_dims or [])
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(d_in, d_out))
            layers.append(activation)

        self.trunk = nn.Sequential(*layers) if len(layers) > 0 else nn.Identity()
        trunk_out_dim = dims[-1]

        self.mu_head = nn.Linear(trunk_out_dim, output_dim)
        self.theta_head = nn.Linear(trunk_out_dim, output_dim)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        x: (B, input_dim)
        returns:
          - mu:    (B, Dy)  mean
          - theta: (B, Dy)  dispersion
        """
        h = self.trunk(x)
        mu = self.softplus(self.mu_head(h)) + self.eps
        theta = self.softplus(self.theta_head(h)) + self.eps
        return {"mu": mu, "theta": theta}


class SupplyHeadZINB(nn.Module):
    """
    Zero-Inflated Negative Binomial head: predicts (pi, mu, theta) per supply target.

    - pi: probability of a structural zero, constrained to (0,1) via Sigmoid
    - mu, theta: as in SupplyHeadNB
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Optional[List[int]] = [64, 64],
        activation: nn.Module = nn.ReLU(),
        eps: float = 1e-8,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.eps = eps

        layers: List[nn.Module] = []
        dims = [input_dim] + (hidden_dims or [])
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(d_in, d_out))
            layers.append(activation)

        self.trunk = nn.Sequential(*layers) if len(layers) > 0 else nn.Identity()
        trunk_out_dim = dims[-1]

        self.pi_head = nn.Linear(trunk_out_dim, output_dim)
        self.mu_head = nn.Linear(trunk_out_dim, output_dim)
        self.theta_head = nn.Linear(trunk_out_dim, output_dim)

        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        x: (B, input_dim)
        returns:
          - pi:    (B, Dy)
          - mu:    (B, Dy)
          - theta: (B, Dy)
        """
        h = self.trunk(x)
        pi = self.sigmoid(self.pi_head(h)).clamp(min=self.eps, max=1.0 - self.eps)
        mu = self.softplus(self.mu_head(h)) + self.eps
        theta = self.softplus(self.theta_head(h)) + self.eps
        return {"pi": pi, "mu": mu, "theta": theta}


class SupplyHeadMSE(nn.Module):
    """
    Deterministic regression head for MSE training.

    Used for the dense bucket "increment" model: predict raw Δy (same dimensionality as y),
    and train with MSE on Δy.
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Optional[List[int]] = [64, 64],
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        dims = [input_dim] + (hidden_dims or []) + [output_dim]
        layers: List[nn.Module] = []
        for d_in, d_out in zip(dims[:-2], dims[1:-1]):
            layers.append(nn.Linear(d_in, d_out))
            layers.append(activation)
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers) if len(layers) > 1 else nn.Linear(dims[0], dims[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)