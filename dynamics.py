import torch
import torch.nn as nn

# VAR(p) latent dynamics
class VARDynamics(nn.Module):
    """
    Linear VAR(p) latent dynamics:

      z(t) = c + sum_{i=1}^{p} A_i z(t-i)

    - A_i is a matrix of size k x k
    - c is an intercept (bias) vector of size k
    - k is the dimension of the latent state
    - p is the order of the VAR model
    - z(t) is the latent state at time t
    - z(t-i) is the latent state at time t-i

    Notes:
    - This module is deterministic; any stochasticity/noise is not explicitly modeled here.
      If you want a stochastic VAR, you would add a noise model on top of the mean prediction.
    """
    def __init__(self, latent_dim: int, order: int, use_intercept: bool = True):
        super().__init__()
        self.latent_dim = latent_dim
        self.order = order
        self.use_intercept = use_intercept
        
        A = torch.zeros(order, latent_dim, latent_dim)
        A[0] = torch.eye(latent_dim) + 0.01 * torch.randn(latent_dim, latent_dim)  # Initialize close to identity
        for i in range(1, order):
            A[i] = 0.05 * torch.randn(latent_dim, latent_dim)
        self.A = nn.Parameter(A)  # (order, latent_dim, latent_dim). Register them as parameters making them learnable.

        # Intercept term c in standard VAR(p): z_t = c + sum_i A_i z_{t-i}
        if use_intercept:
            self.c = nn.Parameter(torch.zeros(latent_dim))  # (k,)
        else:
            self.register_parameter("c", None)
    
    def forward(self, z_history: torch.Tensor) -> torch.Tensor:
        """
        z_history: (B, p, k)
        A:         (p, k, k)
        return:    (B, k)
        """
        # z_t = sum_{lag=1..p} z_{t-lag} @ A_lag^T
        # (B,p,k) x (p,k,k) -> (B,k)
        z = torch.einsum("bpk,pkh->bh", z_history, self.A)
        if self.c is not None:
            z = z + self.c  # broadcast (k,) -> (B,k)
        return z