# Stage 1: Train LAVAR
# This is the core. Trains only the latent state model.
# No supplies are involved

# Its goal is to learn
# Meaningful latent state z_t
# Linear latent dynamics z_t = A z_{t-1:t-p}. p is the latent history length
# Decoder that prevents latent collapse


import torch
from torch.utils.data import DataLoader
from config import LAVARConfig
from models import LAVAR
from dynamics import VARDynamics


def stage1_train_lavar(model: LAVAR, train_loader: DataLoader, val_loader: DataLoader, cfg: LAVARConfig) -> None:
    device = torch.device(cfg.device)
    model.to(device)

    # Ensure the model's VAR order matches the dataset/config windowing.
    # (Common footgun: constructing LAVAR without transition_order=cfg.p.)
    if model.transition_order != cfg.dyn_p:
        print(
            f"[stage1] Adjusting LAVAR.transition_order from {model.transition_order} to cfg.p={cfg.dyn_p} "
            f"(re-initializing VAR dynamics parameters)."
        )
        model.transition_order = cfg.dyn_p
        model.dynamics = VARDynamics(model.latent_dim, cfg.dyn_p).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr_lavar)

    # Training start
    best_val = float("inf")
    best_state = None
    for epoch in range(1, cfg.epochs_lavar + 1):
        model.train()
        tr_loss, n = 0.0, 0

        for x_past, x_future, _y_future in train_loader:
            # x_past: (B, p+1, Dx) where the last step is "current" and previous p are history
            x_past = x_past.to(device)
            x_future = x_future.to(device)  # (B, horizon, Dx)

            x_seq = x_past
            out = model(x_seq)
            x_hat = out["x_hat"]  # (B, Dx)
            z_pred = out["z_pred"]  # (B, k latent dimension)
            z_true = out["z_true"]  # (B, k latent dimension)

            reconstruct = torch.mean((x_hat - x_seq[:, -1, :]) ** 2)  # Autoencoder reconstruction loss
            dynamics = torch.mean((z_pred - z_true) ** 2)  # Latent dynamic loss

            loss = cfg.lambda_recon * reconstruct + cfg.lambda_dyn * dynamics

            # TODO: Multi step latent supervision
            if cfg.multi_step_latent_supervision:
                B, _, Dx = x_seq.shape
                z_hist = model.encode(x_seq[:, :-1, :].reshape(-1, Dx)).reshape(B, cfg.dyn_p, -1)   # (B, p, k)
                z_roll = model.rollout_latent(z_hist, cfg.horizon)                              # (B, H, k)
                z_fut_true = model.encode(x_future.reshape(-1, Dx)).reshape(B, cfg.horizon, -1) # (B, H, k)
                loss += cfg.lambda_dyn * torch.mean((z_roll - z_fut_true) ** 2)

                # Optional: also supervise in observation space by decoding rolled-out latents.
                # This helps when latent matching alone is not enough to keep rollouts meaningful.
                if getattr(cfg, "multi_step_x_reconstruction", False):
                    B, H, k = z_roll.shape
                    x_roll = model.decode(z_roll.reshape(B * H, k)).reshape(B, H, -1)  # (B, H, Dx)
                    loss += float(getattr(cfg, "lambda_future_recon", 1.0)) * torch.mean((x_roll - x_future) ** 2)

            opt.zero_grad()  # What does this do?
            loss.backward()  # Backpropagate the loss
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Clip gradients to prevent exploding gradients
            opt.step()

            tr_loss += loss.item() * x_past.size(0)
            n += x_past.size(0)
        
        tr_loss /= max(n, 1)

        model.eval()
        with torch.no_grad():
            va_loss = 0.0
            vn = 0
            for x_past, x_future, _y_future in val_loader:
                x_past = x_past.to(device)
                x_future = x_future.to(device)

                out = model(x_past)
                reconstruct = torch.mean((out["x_hat"] - x_past[:, -1, :]) ** 2)
                dynamics = torch.mean((out["z_pred"] - out["z_true"]) ** 2)
                loss = cfg.lambda_recon * reconstruct + cfg.lambda_dyn * dynamics

                if cfg.multi_step_latent_supervision:
                    B, _, Dx = x_past.shape
                    z_hist = model.encode(x_past[:, :-1, :].reshape(-1, Dx)).reshape(B, cfg.dyn_p, -1)
                    z_roll = model.rollout_latent(z_hist, cfg.horizon)
                    z_fut_true = model.encode(x_future.reshape(-1, Dx)).reshape(B, cfg.horizon, -1)
                    loss += cfg.lambda_dyn * torch.mean((z_roll - z_fut_true) ** 2)

                    if getattr(cfg, "multi_step_x_reconstruction", False):
                        B, H, k = z_roll.shape
                        x_roll = model.decode(z_roll.reshape(B * H, k)).reshape(B, H, -1)
                        loss += float(getattr(cfg, "lambda_future_recon", 1.0)) * torch.mean((x_roll - x_future) ** 2)

                va_loss += loss.item() * x_past.size(0)
                vn += x_past.size(0)
            
            va_loss /= max(vn, 1)
        
        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, "lavar_best.pth")
            # print(f"Epoch {epoch}: New best validation loss {best_val:.6f}")
        # else:
        #     print(f"Epoch {epoch}: Validation loss {va_loss:.6f} (no improvement)")
        
        # if epoch % 10 == 0:
        #     print(f"Epoch {epoch}: Training loss {tr_loss:.6f}, Validation loss {va_loss:.6f}")
        
        model.load_state_dict(best_state)
        # print(f"Loaded best model from epoch {epoch}, best validation loss {best_val:.6f}")