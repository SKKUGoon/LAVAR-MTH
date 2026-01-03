from pydantic import BaseModel, Field
from typing import List, Literal

class LAVARConfig(BaseModel):
    device: Literal["cpu", "mps", "cuda"] = Field(default="cpu", description="Device to use for training")
    num_workers: int = 0

    # Window
    dyn_p: int = 7  # History length used as VAR order input

    # training data input
    batch_size: int = 64
    train_days: int = 365 * 3  # ~3 years for training.
    horizon: int = 14  # 14 days

    # stage 1 (LAVAR + VAR Dynamics)
    latent_dim: int = 8
    encoder_hidden: List[int] = [32, 16]
    decoder_hidden: List[int] = [16, 32]
    lr_lavar: float = 1e-3
    epochs_lavar: int = 100
    lambda_dyn: float = 1.0  # weight for latent dynamics loss
    lambda_recon: float = 1.0  # weight for reconstruction loss
    multi_step_latent_supervision: bool = True  # z rollout against future encoded z

    # stage 2 (Supply Model)
    # Trained via *density split* only:
    #   - dense targets: NB head
    #   - sparse targets: ZINB head
    #   - ultra-sparse targets: ZINB head (separate head)
    #
    # Buckets are computed on TRAIN windows only and are defined by nonzero rate:
    #   nonzero_rate(j) = mean_t[ y_tj > 0 ]
    #   - dense:        nonzero_rate >= dense_nonzero_rate_thr
    #   - ultra_sparse: nonzero_rate <= ultra_nonzero_rate_thr
    #   - sparse:       otherwise
    supply_hidden: List[int] = []
    lr_supply: float = 1e-3
    epochs_supply: int = 100
    dense_nonzero_rate_thr: float = 0.70
    ultra_nonzero_rate_thr: float = 0.005

    # Split