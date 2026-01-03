from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Sequence, Tuple

from config import LAVARConfig
from models import LAVAR, LAVARWithSupply
from losses import negative_binomial_nll, zinb_nll


@torch.no_grad()
def compute_nonzero_rate_from_loader(train_loader: DataLoader) -> torch.Tensor:
    """
    Compute per-target nonzero rate from a train DataLoader.

    Expected batch: (x_past, x_future, y_future) where y_future has shape (B, H, Dy).
    Returns: nonzero_rate with shape (Dy,), on CPU.
    """
    nonzero_sum: Optional[torch.Tensor] = None
    count_sum: int = 0

    for batch in train_loader:
        if len(batch) == 3:
            _x_past, _x_future, y_future = batch
        elif len(batch) == 4:
            _x_past, _x_future, _y0, y_future = batch
        else:
            raise ValueError(
                "Density split training expects batches of (x_past, x_future, y_future). "
                f"Got batch size {len(batch)}."
            )
        # y_future: (B, H, Dy)
        y_flat = y_future.reshape(-1, y_future.shape[-1])
        nz = (y_flat > 0).to(dtype=torch.float32).sum(dim=0)  # (Dy,)
        if nonzero_sum is None:
            nonzero_sum = nz.detach().cpu()
        else:
            nonzero_sum += nz.detach().cpu()
        count_sum += int(y_flat.shape[0])

    if nonzero_sum is None or count_sum == 0:
        raise RuntimeError("train_loader yielded no batches; cannot compute nonzero rates.")

    return nonzero_sum / float(count_sum)


def split_indices_by_density(
    nonzero_rate: torch.Tensor,
    dense_thr: float,
    ultra_thr: float,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Split Dy indices into dense/sparse/ultra-sparse based on nonzero_rate thresholds.
    """
    if nonzero_rate.dim() != 1:
        raise ValueError(f"nonzero_rate must have shape (Dy,). Got {tuple(nonzero_rate.shape)}")
    if not (0.0 <= ultra_thr <= dense_thr <= 1.0):
        raise ValueError(f"Expected 0 <= ultra_thr <= dense_thr <= 1. Got ultra={ultra_thr}, dense={dense_thr}")

    dense_mask = nonzero_rate >= dense_thr
    ultra_mask = nonzero_rate <= ultra_thr
    sparse_mask = (~dense_mask) & (~ultra_mask)

    dense_idx = dense_mask.nonzero(as_tuple=False).squeeze(-1).tolist()
    sparse_idx = sparse_mask.nonzero(as_tuple=False).squeeze(-1).tolist()
    ultra_idx = ultra_mask.nonzero(as_tuple=False).squeeze(-1).tolist()
    return dense_idx, sparse_idx, ultra_idx


def _slice_targets(
    y: torch.Tensor,
    idx: torch.Tensor,
) -> torch.Tensor:
    # y: (B, H, Dy) -> (B, H, Dy_sel)
    return y.index_select(dim=-1, index=idx)


def train_supply_head_indexed(
    lavar: LAVAR,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: LAVARConfig,
    y_indices: Sequence[int],
    head_type: str,
    save_path: str,
) -> Optional[LAVARWithSupply]:
    """
    Train one supply head on a subset of targets.
    """
    y_indices = list(y_indices)
    if len(y_indices) == 0:
        return None

    if head_type not in {"nb", "zinb", "delta_mse"}:
        raise ValueError(f"head_type must be 'nb', 'zinb', or 'delta_mse'. Got {head_type!r}")

    device = torch.device(cfg.device)
    idx = torch.as_tensor(y_indices, dtype=torch.long, device=device)

    model = LAVARWithSupply(
        lavar=lavar,
        supply_dim=len(y_indices),
        horizon=cfg.horizon,
        supply_hidden=cfg.supply_hidden,
        supply_head_type=head_type,  # type: ignore[arg-type]
    ).to(device)

    # Freeze LAVAR weights; train only supply head.
    for p in model.lavar.parameters():
        p.requires_grad = False
    model.lavar.eval()

    opt = torch.optim.Adam(model.supply_head.parameters(), lr=cfg.lr_supply)

    best_val = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None

    for _epoch in range(1, cfg.epochs_supply + 1):
        model.train()
        for batch in train_loader:
            if len(batch) == 3:
                x_past, _x_future, y_future = batch
                y0 = None
            elif len(batch) == 4:
                x_past, _x_future, y0, y_future = batch
            else:
                raise ValueError(
                    "Density split training expects batches of (x_past, x_future, y_future). "
                    f"Got batch size {len(batch)}."
                )
            x_past = x_past.to(device)
            y_future = y_future.to(device)
            y_sel = _slice_targets(y_future, idx)

            if head_type == "delta_mse":
                if y0 is None:
                    raise ValueError(
                        "Delta-MSE training requires y0. Use RollingXYDatasetWithY0 for stage2 loaders."
                    )
                y0 = y0.to(device)
                y0_sel = y0.index_select(dim=-1, index=idx)  # (B, Dy_sel)

                out = model(x_past, y0=y0_sel, return_delta=True)
                assert isinstance(out, dict)
                delta_hat = out["delta"]  # (B, H, Dy_sel)

                # Raw delta-y targets:
                #   Δy_1 = y_{t+1} - y_t; Δy_h = y_{t+h} - y_{t+h-1}
                y_prev = torch.cat([y0_sel.unsqueeze(1), y_sel[:, :-1, :]], dim=1)
                delta_true = y_sel - y_prev
                loss = torch.mean((delta_hat - delta_true) ** 2)
            else:
                params = model(x_past, return_params=True)
                assert isinstance(params, dict)
                if head_type == "nb":
                    loss = negative_binomial_nll(params["mu"], params["theta"], y_sel)
                else:
                    loss = zinb_nll(params["pi"], params["mu"], params["theta"], y_sel)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.supply_head.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            va_loss = 0.0
            vn = 0
            for batch in val_loader:
                if len(batch) == 3:
                    x_past, _x_future, y_future = batch
                    y0 = None
                elif len(batch) == 4:
                    x_past, _x_future, y0, y_future = batch
                else:
                    raise ValueError(
                        "Density split training expects batches of (x_past, x_future, y_future). "
                        f"Got batch size {len(batch)}."
                    )
                x_past = x_past.to(device)
                y_future = y_future.to(device)
                y_sel = _slice_targets(y_future, idx)

                if head_type == "delta_mse":
                    if y0 is None:
                        raise ValueError(
                            "Delta-MSE training requires y0. Use RollingXYDatasetWithY0 for stage2 loaders."
                        )
                    y0 = y0.to(device)
                    y0_sel = y0.index_select(dim=-1, index=idx)

                    out = model(x_past, y0=y0_sel, return_delta=True)
                    assert isinstance(out, dict)
                    delta_hat = out["delta"]

                    y_prev = torch.cat([y0_sel.unsqueeze(1), y_sel[:, :-1, :]], dim=1)
                    delta_true = y_sel - y_prev
                    loss = torch.mean((delta_hat - delta_true) ** 2)
                else:
                    params = model(x_past, return_params=True)
                    assert isinstance(params, dict)
                    if head_type == "nb":
                        loss = negative_binomial_nll(params["mu"], params["theta"], y_sel)
                    else:
                        loss = zinb_nll(params["pi"], params["mu"], params["theta"], y_sel)

                va_loss += float(loss.item()) * x_past.size(0)
                vn += int(x_past.size(0))

            va_loss /= max(vn, 1)

        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, save_path)

        if best_state is not None:
            model.load_state_dict(best_state)

    return model


def stitch_bucket_predictions(
    Dy: int,
    horizon: int,
    batch_size: int,
    dense_indices: Sequence[int],
    sparse_indices: Sequence[int],
    ultra_indices: Sequence[int],
    dense_pred: Optional[torch.Tensor],
    sparse_pred: Optional[torch.Tensor],
    ultra_pred: Optional[torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """
    Stitch per-bucket predictions back into the original Dy ordering.
    """
    y_hat = torch.zeros(batch_size, horizon, Dy, device=device)
    if dense_pred is not None and len(dense_indices) > 0:
        y_hat.index_copy_(dim=2, index=torch.as_tensor(dense_indices, device=device), source=dense_pred)
    if sparse_pred is not None and len(sparse_indices) > 0:
        y_hat.index_copy_(dim=2, index=torch.as_tensor(sparse_indices, device=device), source=sparse_pred)
    if ultra_pred is not None and len(ultra_indices) > 0:
        y_hat.index_copy_(dim=2, index=torch.as_tensor(ultra_indices, device=device), source=ultra_pred)
    return y_hat


def stage2_train_supply_density_split(
    lavar: LAVAR,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: LAVARConfig,
    save_dense: str = "lavar_supply_dense_best.pth",
    save_sparse: str = "lavar_supply_sparse_best.pth",
    save_ultra: str = "lavar_supply_ultra_best.pth",
) -> Dict[str, object]:
    """
    Stage 2: train three supply heads split by value density (nonzero rate).

      - dense bucket -> increment head (raw Δy trained with MSE)
      - sparse bucket -> increment head (raw Δy trained with MSE)
      - ultra-sparse bucket -> ZINB head (separate)
    """
    nonzero_rate = compute_nonzero_rate_from_loader(train_loader)
    dense_thr = float(cfg.dense_nonzero_rate_thr)
    ultra_thr = float(cfg.ultra_nonzero_rate_thr)
    dense_idx, sparse_idx, ultra_idx = split_indices_by_density(nonzero_rate, dense_thr=dense_thr, ultra_thr=ultra_thr)

    out: Dict[str, object] = {
        "dense_indices": dense_idx,
        "sparse_indices": sparse_idx,
        "ultra_indices": ultra_idx,
        "dense_model": None,
        "sparse_model": None,
        "ultra_model": None,
    }

    out["dense_model"] = train_supply_head_indexed(
        lavar=lavar,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        y_indices=dense_idx,
        head_type="delta_mse",
        save_path=save_dense,
    )
    out["sparse_model"] = train_supply_head_indexed(
        lavar=lavar,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        y_indices=sparse_idx,
        head_type="delta_mse",
        save_path=save_sparse,
    )
    out["ultra_model"] = train_supply_head_indexed(
        lavar=lavar,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        y_indices=ultra_idx,
        head_type="zinb",
        save_path=save_ultra,
    )

    return out


