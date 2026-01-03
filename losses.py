import torch

def negative_binomial_nll(
    mu: torch.Tensor,
    theta: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Negative Binomial NLL with mean/dispersion parameterization.

    We use the common (mu, theta) form where:
      - mu > 0 is the mean
      - theta > 0 is the "total count" / inverse-dispersion (larger => closer to Poisson)

    Log PMF:
      log p(y) =
        lgamma(y + theta) - lgamma(theta) - lgamma(y + 1)
        + theta * (log theta - log(theta + mu))
        + y * (log mu - log(theta + mu))

    Shapes:
      - mu, theta, y: (..., Dy) broadcastable
    """
    mu = mu.clamp_min(eps)
    theta = theta.clamp_min(eps)
    y = y.clamp_min(0.0)

    log_theta_mu = torch.log(theta + mu)
    log_prob = (
        torch.lgamma(y + theta)
        - torch.lgamma(theta)
        - torch.lgamma(y + 1.0)
        + theta * (torch.log(theta) - log_theta_mu)
        + y * (torch.log(mu) - log_theta_mu)
    )
    return (-log_prob).mean()


def zinb_nll(
    pi: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Zero-Inflated Negative Binomial NLL.

    Parameters:
      - pi in (0,1): probability of a structural zero (extra mass at zero)
      - mu > 0, theta > 0: NB parameters as in negative_binomial_nll

    For y == 0:
      p(y=0) = pi + (1-pi) * NB(y=0)
    For y > 0:
      p(y) = (1-pi) * NB(y)
    """
    mu = mu.clamp_min(eps)
    theta = theta.clamp_min(eps)
    y = y.clamp_min(0.0)
    pi = pi.clamp(min=eps, max=1.0 - eps)

    # NB log-prob for all y
    log_theta_mu = torch.log(theta + mu)
    nb_log_prob = (
        torch.lgamma(y + theta)
        - torch.lgamma(theta)
        - torch.lgamma(y + 1.0)
        + theta * (torch.log(theta) - log_theta_mu)
        + y * (torch.log(mu) - log_theta_mu)
    )

    # NB probability of zero: (theta / (theta + mu)) ** theta
    nb_p0 = torch.exp(theta * (torch.log(theta) - log_theta_mu))

    is_zero = (y == 0)
    log_prob_zero = torch.log(pi + (1.0 - pi) * nb_p0 + eps)
    log_prob_nonzero = torch.log(1.0 - pi + eps) + nb_log_prob
    log_prob = torch.where(is_zero, log_prob_zero, log_prob_nonzero)
    return (-log_prob).mean()