# ACADEMIC NOTES (implementation-faithful)

This document describes how the **LAVAR** model in this repository works, *as implemented* in `models.py`, `dynamics.py`, `network.py`, `train_stage1.py`, and `train_stage2.py`.

## 1) Model: what is learned and how it is used

We model a multivariate time series of:

- non-supply covariates $x_t \in \mathbb{R}^{D_x}$
- supply targets $y_t \in \mathbb{R}^{D_y}$

LAVAR introduces a low-dimensional latent state:

$
z_t \in \mathbb{R}^{k}
$

The implementation decomposes the problem into:

- **Nonlinear observation mapping** (autoencoder):

$
z_t = e_\theta(x_t), \qquad \hat{x}_t = d_\phi(z_t)
$

- **Linear latent dynamics** (VAR($p$)):

$
\hat{z}_t
  = c + \sum_{i=1}^{p} A_i z_{t-i}
$

where $A_i \in \mathbb{R}^{k \times k}$ and $c \in \mathbb{R}^k$.

### Mapping to code

- Encoder/decoder are MLPs: `network.MLP`
- VAR dynamics are deterministic: `dynamics.VARDynamics`
- Full stage-1 model: `models.LAVAR`

## 2) “How do we find the latent variable?”

This codebase does **not** implement a variational posterior or stochastic latent sampling.

The latent state is computed **deterministically** by an encoder network:

$
z_t = e_\theta(x_t)
$

During training (stage 1), each time step in a window is encoded **independently**, and temporal structure is enforced via explicit latent-dynamics losses (see §3).

Implementation detail: the encoder first replaces exact zeros in $x$ with a small $\varepsilon$ (to avoid dead gradients in downstream operations):

$
x \leftarrow \begin{cases}
\varepsilon & \text{if } x = 0 \\
x & \text{otherwise}
\end{cases}
$

See `LAVAR.replace_zeros()` and `LAVAR.encode()` in `models.py`.

## 3) Stage 1 training objective (learn $e_\theta, d_\phi, A_{1:p}, c$)

### Windowing used by the trainers

The dataset provides, for each index $t$:

$
x_{\text{past}} = [x_{t-p}, \dots, x_t] \in \mathbb{R}^{(p+1)\times D_x}
$
$
x_{\text{future}} = [x_{t+1}, \dots, x_{t+H}] \in \mathbb{R}^{H\times D_x}
$
$
y_{\text{future}} = [y_{t+1}, \dots, y_{t+H}] \in \mathbb{R}^{H\times D_y}
$

See `dataset.RollingXYDataset`.

### Per-window computations

Encode each step in $x_{\text{past}}$:

$
z_{t-i} = e_\theta(x_{t-i}) \quad (i = 0,\dots,p)
$

Latent VAR prediction uses the history $z_{t-p:t-1}$ to predict $z_t$:

$
\hat{z}_t = c + \sum_{i=1}^{p} A_i z_{t-i}
$

Reconstruction uses $z_t$:

$
\hat{x}_t = d_\phi(z_t)
$

### Loss terms in `train_stage1.py`

The stage-1 objective is a weighted sum of mean-squared errors:

- **Reconstruction loss**:

$
\mathcal{L}_{\text{recon}}
  = \| \hat{x}_t - x_t \|_2^2
$

- **One-step latent dynamics loss**:

$
\mathcal{L}_{\text{dyn}}
  = \| \hat{z}_t - z_t \|_2^2
$

- **Optional multi-step latent supervision** (`cfg.multi_step_latent_supervision=True`):
  - Roll out the latent dynamics $H$ steps ahead from the past latents:

$
\hat{z}_{t+1:t+H} = \text{rollout}([z_{t-p+1},\dots,z_t], H)
$

  - Encode the true future observations to obtain target future latents:

$
z_{t+1:t+H} = e_\theta(x_{t+1:t+H})
$

  - Penalize the difference:

$
\mathcal{L}_{\text{ms}}
  = \| \hat{z}_{t+1:t+H} - z_{t+1:t+H} \|_2^2
$

Overall (exact weighting matches `train_stage1.py`):

$
\mathcal{L}_{\text{stage1}}
  = \lambda_{\text{recon}}\mathcal{L}_{\text{recon}}
  + \lambda_{\text{dyn}}\mathcal{L}_{\text{dyn}}
  + \lambda_{\text{dyn}}\mathcal{L}_{\text{ms}} \;\; (\text{if enabled})
$

## 4) Stage 2: “How do we find the supplies?”

In stage 2, supplies are predicted by:

1. encoding the most recent $p$ covariate steps into a latent history
2. rolling out future latents with the learned VAR
3. mapping each future latent $z_{t+h}$ to $y_{t+h}$ via a small “supply head”

Concretely:

$
z_{t-p+1:t} = e_\theta(x_{t-p+1:t})
$
$
\hat{z}_{t+1:t+H} = \text{rollout}(z_{t-p+1:t}, H)
$
$
\hat{y}_{t+h} = s_\psi(\hat{z}_{t+h})
$

Implementation: `models.LAVARWithSupply.forward()`.

### Are parameters frozen or not?

In stage 2, **all of LAVAR is frozen** and only the supply head is trained.

In `train_stage2.train_supply_head_indexed()`:

- `for p in model.lavar.parameters(): p.requires_grad = False`
- optimizer is built from `model.supply_head.parameters()` only

So:

- **Stage 1**: encoder + decoder + VAR parameters learn
- **Stage 2**: encoder + decoder + VAR parameters are frozen; only the supply head learns

## 5) “How did I make multiple supply heads?”

The repo implements “multiple heads” via **density splitting** of the supply dimensions $j \in \{1,\dots,D_y\}$.

### Density metric

Compute a per-target nonzero rate on the *training windows*:

$
r_j = \mathbb{E}[\mathbf{1}\{y_{t,j} > 0\}]
$

Implementation: `train_stage2.compute_nonzero_rate_from_loader()`.

### Buckets

Using thresholds `dense_nonzero_rate_thr` and `ultra_nonzero_rate_thr`:

- dense: $r_j \ge \tau_{\text{dense}}$
- ultra-sparse: $r_j \le \tau_{\text{ultra}}$
- sparse: otherwise

Implementation: `train_stage2.split_indices_by_density()`.

### How this becomes “multiple heads” in code

The training code then trains **three separate `LAVARWithSupply` models**, each configured with a different head type and trained only on its subset of dimensions:

- **dense bucket**: `head_type="delta_mse"`
- **sparse bucket**: `head_type="delta_mse"`
- **ultra bucket**: `head_type="zinb"`

Each call is done by `train_stage2.train_supply_head_indexed(...)`.

Predictions can be recombined into the original $D_y$ ordering using `train_stage2.stitch_bucket_predictions(...)`.

## 6) What probability distribution did I use?

The latent model (encoder/decoder + VAR) is **deterministic** (no explicit latent noise distribution).

For supplies, the repo supports:

- **Negative Binomial (NB)**: `SupplyHeadNB` + `losses.negative_binomial_nll`
- **Zero-Inflated Negative Binomial (ZINB)**: `SupplyHeadZINB` + `losses.zinb_nll`

It also supports a deterministic **increment regression** head:

- **Δy + MSE**: `SupplyHeadMSE` with an integration step (cumulative sum)

### NB parameterization (mean/dispersion)

The NB head predicts:

- $\mu > 0$ (mean)
- $\theta > 0$ (dispersion / “total_count”)

and the negative log-likelihood uses:

$
\log p(y \mid \mu, \theta) := \log \Gamma(y+\theta) - \log \Gamma(\theta) - \log \Gamma(y+1) + \theta \big(\log \theta - \log(\theta+\mu)\big) + y \big(\log \mu - \log(\theta+\mu)\big)
$

The loss is $-\mathbb{E}[\log p(\cdot)]$ averaged over batch/time/targets.

### ZINB: mixture of a point-mass at zero and NB

The ZINB head predicts:

- $\pi \in (0,1)$: probability of a *structural* zero
- $\mu>0, \theta>0$: NB parameters

For $y=0$:

$
p(y=0) = \pi + (1-\pi)\,p_{\text{NB}}(0\mid \mu,\theta)
$

For $y>0$:

$
p(y) = (1-\pi)\,p_{\text{NB}}(y\mid \mu,\theta)
$

The implementation uses:

$
p_{\text{NB}}(0\mid \mu,\theta) = \left(\frac{\theta}{\theta+\mu}\right)^{\theta}
$

See `losses.zinb_nll()`.

### Δy (increment) head used for dense/sparse buckets

For dense/sparse targets, the repo uses a deterministic head predicting increments:

$
\Delta y_{t+1} = y_{t+1} - y_t,\quad
\Delta y_{t+h} = y_{t+h} - y_{t+h-1}
$

It trains with MSE:

$
\mathcal{L}_{\Delta}
  = \| \widehat{\Delta y}_{t+1:t+H} - \Delta y_{t+1:t+H} \|_2^2
$

and converts increments to level forecasts by cumulative summation:

$
\hat{y}_{t+h} = \max\left(0,\; y_t + \sum_{i=1}^{h} \widehat{\Delta y}_{t+i}\right)
$

Implementation: integration/clamp in `LAVARWithSupply.forward()` and delta target construction in `train_stage2.train_supply_head_indexed()`.

## 7) Code pointers (quick)

- Stage 1 core forward pass: `models.LAVAR.forward()`
- Latent rollout: `models.LAVAR.rollout_latent()`
- VAR definition: `dynamics.VARDynamics`
- Supply heads: `network.SupplyHeadMSE`, `network.SupplyHeadNB`, `network.SupplyHeadZINB`
- Supply NLLs: `losses.negative_binomial_nll`, `losses.zinb_nll`
- Stage 1 trainer: `train_stage1.stage1_train_lavar`
- Stage 2 density split + freezing: `train_stage2.stage2_train_supply_density_split`


