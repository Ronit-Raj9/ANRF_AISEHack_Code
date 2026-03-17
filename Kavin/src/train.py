"""
Training loop with:
  - Normalized-space RMSE loss (targets and preds in [0, 1])
  - Persistence gate: val RMSE must beat 0.0208 (EDA baseline) to be useful
  - Early stopping on val RMSE with configurable patience
  - Physical RMSE estimate printed each epoch for interpretability
"""

import os
import numpy as np
import torch
from .data import get_hotspot_maps

# Persistence RMSE baseline in normalized [0,1] cpm25 space (from EDA, t+16 avg)
PERSISTENCE_RMSE_NORM = 0.0208
PERSISTENCE_RMSE_PHYS = 30.83  # µg/m³


# ─────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────

def _horizon_weights(cfg, target: torch.Tensor) -> torch.Tensor:
    """Linearly increasing horizon weights to emphasize longer leads."""
    t_out = target.shape[-1]
    lo = cfg.get('loss', {}).get('horizon_weight_min', 0.8)
    hi = cfg.get('loss', {}).get('horizon_weight_max', 1.4)
    return torch.linspace(lo, hi, t_out, device=target.device, dtype=target.dtype)


def rmse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    cfg: dict | None = None,
    spatial_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Spatial RMSE averaged over batch and time steps.

    Parameters
    ----------
    pred, target : (B, H, W, T)  — values in normalized [0, 1] space
    """
    sq_err = (pred - target) ** 2
    if spatial_weights is not None:
        sq_err = sq_err * spatial_weights[None, :, :, None]
    spatial_mse = torch.mean(sq_err, dim=(1, 2))  # (B, T)
    if cfg is not None:
        weights = _horizon_weights(cfg, target)[None]
        spatial_mse = spatial_mse * weights
    return torch.mean(torch.sqrt(spatial_mse + 1e-8))


def rmse_loss_plain(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Unweighted spatial RMSE averaged over batch and horizon."""
    spatial_mse = torch.mean((pred - target) ** 2, dim=(1, 2))  # (B, T)
    return torch.mean(torch.sqrt(spatial_mse + 1e-8))


def mae_loss(pred: torch.Tensor, target: torch.Tensor, cfg: dict | None = None) -> torch.Tensor:
    """Weighted MAE over batch and horizon."""
    spatial_mae = torch.mean(torch.abs(pred - target), dim=(1, 2))
    if cfg is not None:
        weights = _horizon_weights(cfg, target)[None]
        spatial_mae = spatial_mae * weights
    return torch.mean(spatial_mae)


def objective_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    cfg: dict,
    spatial_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Main optimization objective: weighted RMSE + light MAE regularization."""
    wrmse = rmse_loss(pred, target, cfg, spatial_weights=spatial_weights)
    wmae = mae_loss(pred, target, cfg)
    a = cfg.get('loss', {}).get('rmse_weight', 0.8)
    b = cfg.get('loss', {}).get('mae_weight', 0.2)
    return a * wrmse + b * wmae


def _align_pred_target_layout(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert model/target tensors into common (B, H, W, T_out) layout.

    Supported:
    - Existing models: pred/target already (B, H, W, T)
    - SimVP: (B, T, 1, H, W) or (B, T, H, W)
    """
    if pred.ndim == 4 and target.ndim == 4:
        return pred, target

    if pred.ndim == 5:
        if pred.shape[2] == 1:  # (B, T, 1, H, W)
            pred = pred.squeeze(2)
        pred = pred.permute(0, 2, 3, 1)  # -> (B, H, W, T)

    if target.ndim == 5:
        if target.shape[2] == 1:  # (B, T, 1, H, W)
            target = target.squeeze(2)
        target = target.permute(0, 2, 3, 1)  # -> (B, H, W, T)

    if pred.ndim != 4 or target.ndim != 4:
        raise ValueError(f"Unsupported pred/target shapes after alignment: pred={pred.shape}, target={target.shape}")

    return pred, target


def get_spatial_weight_tensor(cfg: dict, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
    """Return hotspot-based spatial weight tensor (H, W) or None if disabled."""
    if not cfg.get('loss', {}).get('use_spatial_hotspot_weight', True):
        return None

    cached = cfg.get('_hotspot_weight_map', None)
    if cached is None:
        _, weight_map = get_hotspot_maps(cfg)
        cfg['_hotspot_weight_map'] = weight_map
        cached = weight_map
    return torch.as_tensor(cached, device=device, dtype=dtype)


# ─────────────────────────────────────────────
# Optimizer & Scheduler
# ─────────────────────────────────────────────

def get_optimizer(cfg, model, steps_per_epoch):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = cfg['training']['lr'],
        weight_decay = cfg['training']['weight_decay'],
    )
    sched_type = cfg['training'].get('scheduler', 'coswr')
    if sched_type == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr           = cfg['training']['lr'],
            steps_per_epoch  = steps_per_epoch,
            epochs           = cfg['training']['epochs'],
            pct_start        = cfg['training']['pct_start'],
        )
    else:  # 'coswr' — Cosine Annealing with Warm Restarts
        T_0    = max(1, steps_per_epoch * cfg['training'].get('t0_epochs', 10))
        T_mult = cfg['training'].get('t_mult', 2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0    = T_0,
            T_mult = T_mult,
            eta_min = cfg['training']['lr'] * 1e-2,  # floor = 1% of initial LR
        )
    return optimizer, scheduler

# ─────────────────────────────────────────────
# PINO Physics-Informed Loss
# ─────────────────────────────────────────────
def compute_physics_loss(pred, xb, yb, cfg):
    """
    Compute Advection-Diffusion residual for PINO:
    R = dC/dt + u · grad(C) - kappa * laplacian(C) - S
    Penalize squared residual.
    """
    import torch

    # Convert SimVP layout to legacy layout used below.
    # pred: (B,T,1,H,W) -> (B,H,W,T)
    # xb  : (B,T,C,H,W) -> (B,C,T,H,W)
    if pred.ndim == 5:
        if pred.shape[2] == 1:
            pred = pred.squeeze(2)
        pred = pred.permute(0, 2, 3, 1)
    if xb.ndim == 5 and xb.shape[1] != xb.shape[2]:
        xb = xb.permute(0, 2, 1, 3, 4)

    if pred.ndim != 4 or xb.ndim != 5:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)

    # Temporal derivative against last observed cpm25 (channel 0 at last input step)
    dt = 1.0
    last_c = xb[:, 0, -1, :, :].unsqueeze(-1)  # (B, H, W, 1)
    dC_dt = (pred - last_c) / dt               # (B, H, W, T)

    # Spatial gradients on H and W axes
    grad_h = torch.gradient(pred, dim=1)[0]
    grad_w = torch.gradient(pred, dim=2)[0]

    # 'features.all' does not exist in config — use 'features.base' which is the
    # actual list of base feature names used at training time.
    feature_names = cfg.get('features', {}).get('base', [])

    def _feat_idx(name: str):
        if name in feature_names:
            idx = feature_names.index(name)
            if idx < xb.shape[1]:
                return idx
        return None

    # Wind vectors at the latest input step, then broadcast across forecast horizon
    u_idx = _feat_idx('u10')
    v_idx = _feat_idx('v10')
    if u_idx is None or v_idx is None:
        advection = torch.zeros_like(pred)
    else:
        u = xb[:, u_idx, -1, :, :].unsqueeze(-1)
        v = xb[:, v_idx, -1, :, :].unsqueeze(-1)
        advection = u * grad_h + v * grad_w

    # Diffusion (Laplacian on spatial axes)
    kappa = cfg.get('physics', {}).get('diffusivity', 1.0)
    laplacian = torch.gradient(grad_h, dim=1)[0] + torch.gradient(grad_w, dim=2)[0]
    diffusion = kappa * laplacian

    # Optional source term from available emission-like channels
    S = torch.zeros_like(pred)
    for name in ('SO2', 'NOx', 'PM25', 'cpm25'):
        idx = _feat_idx(name)
        if idx is not None:
            S = S + xb[:, idx, -1, :, :].unsqueeze(-1)

    R = dC_dt + advection - diffusion - S
    return torch.mean(R ** 2)

# ─────────────────────────────────────────────
# Persistence Gate
# ─────────────────────────────────────────────

def check_persistence_gate(val_rmse: float, epoch: int) -> None:
    """
    Print a warning if the model is not yet beating persistence at t+16.
    Called after each epoch so the user knows whether to abort early.
    """
    gap = val_rmse - PERSISTENCE_RMSE_NORM
    if gap > 0:
        print(
            f"  ⚠  Persistence gate NOT met at epoch {epoch}: "
            f"val_rmse={val_rmse:.4f}  >  persistence={PERSISTENCE_RMSE_NORM:.4f} "
            f"(gap={gap:+.4f})"
        )
    else:
        print(
            f"  ✓  Persistence gate MET: "
            f"val_rmse={val_rmse:.4f}  <  persistence={PERSISTENCE_RMSE_NORM:.4f} "
            f"(gap={gap:+.4f})"
        )


def persistence_rmse_from_batch(xb: torch.Tensor, yb: torch.Tensor, t_in_cpm: int) -> torch.Tensor:
    """Compute persistence baseline RMSE on the current batch in normalized space."""
    # Legacy layout: xb=(B,C,T,H,W)
    # SimVP layout : xb=(B,T,C,H,W)
    if xb.ndim == 5 and xb.shape[1] > 1 and xb.shape[2] > 1 and xb.shape[2] < xb.shape[1]:
        # likely legacy (C,T)
        last_obs = xb[:, 0, t_in_cpm - 1]
    elif xb.ndim == 5:
        # likely simvp (T,C)
        last_obs = xb[:, t_in_cpm - 1, 0]
    else:
        raise ValueError(f"Unsupported xb shape for persistence baseline: {xb.shape}")

    if yb.ndim == 5:
        if yb.shape[2] == 1:
            yb = yb.squeeze(2)
        yb = yb.permute(0, 2, 3, 1)

    persist = last_obs[..., None].repeat(1, 1, 1, yb.shape[-1])
    return rmse_loss_plain(persist, yb)


# ─────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────

def train(cfg, model, train_dl, val_dl, bounds: dict = None):
    """
    Full training loop.

    Parameters
    ----------
    bounds : optional dict {feat: (fmin, fmax)} — if provided, physical RMSE
             is estimated each epoch using cpm25 normalization range.

    Returns
    -------
    history : dict with keys 'train_loss', 'val_loss' (normalized RMSE per epoch)
    """
    device    = cfg['device']
    epochs    = cfg['training']['epochs']
    patience  = cfg['training'].get('patience', 8)
    grad_clip = cfg['training']['grad_clip']
    save_path = cfg['paths']['model_save']
    t_in_cpm  = cfg['time']['t_in_cpm']

    # cpm25 range for physical RMSE estimation
    cpm_range = None
    if bounds is not None and 'cpm25' in bounds:
        fmin, fmax = bounds['cpm25']
        cpm_range  = fmax - fmin   # 1464.25 µg/m³

    optimizer, scheduler = get_optimizer(cfg, model, len(train_dl))

    best_val_loss  = float('inf')
    patience_count = 0
    history        = {'train_loss': [], 'val_loss': [], 'train_objective': [], 'val_persistence': []}
    spatial_weight_map = get_spatial_weight_tensor(cfg, device, torch.float32)

    print(f"\n{'─'*60}")
    print(f"  Persistence gate  (normalized RMSE): {PERSISTENCE_RMSE_NORM:.4f}")
    if spatial_weight_map is not None:
        print("  Training loss uses hotspot-weighted spatial RMSE (val remains unweighted)")
    if cpm_range:
        print(f"  Persistence gate  (physical RMSE) : {PERSISTENCE_RMSE_PHYS:.2f} µg/m³")
    print(f"{'─'*60}\n")

    for epoch in range(epochs):
        # ── Train ──
        model.train()
        epoch_losses = []
        epoch_rmse = []
        lambda_d = cfg['training'].get('lambda_d', 1.0)
        lambda_p_target = cfg['training'].get('lambda_p', 0.1)
        physics_warmup_epochs = max(0, int(cfg['training'].get('physics_warmup_epochs', 0)))
        if physics_warmup_epochs > 0:
            warmup_scale = min(1.0, float(epoch + 1) / float(physics_warmup_epochs))
            lambda_p = lambda_p_target * warmup_scale
        else:
            lambda_p = lambda_p_target

        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred_raw = model(xb)
            pred, yb_aligned = _align_pred_target_layout(pred_raw, yb)
            sw = spatial_weight_map.to(pred.dtype) if spatial_weight_map is not None else None
            data_loss = objective_loss(pred, yb_aligned, cfg, spatial_weights=sw)
            physics_loss = compute_physics_loss(pred_raw, xb, yb, cfg)
            loss = lambda_d * data_loss + lambda_p * physics_loss
            # Skip batch if loss is NaN/Inf (bad inputs slipping through)
            if not torch.isfinite(loss):
                optimizer.zero_grad()
                continue
            rmse_metric = rmse_loss_plain(pred, yb_aligned)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            epoch_losses.append(loss.item())
            epoch_rmse.append(rmse_metric.item())

        # ── Validate ──
        model.eval()
        val_losses = []
        val_persist = []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred_raw = model(xb)
                pred, yb_aligned = _align_pred_target_layout(pred_raw, yb)
                val_losses.append(rmse_loss_plain(pred, yb_aligned).item())
                val_persist.append(persistence_rmse_from_batch(xb, yb, t_in_cpm).item())

        train_objective = float(np.mean(epoch_losses))
        train_loss = float(np.mean(epoch_rmse))
        val_loss   = float(np.mean(val_losses))
        persist_loss = float(np.mean(val_persist))
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_objective'].append(train_objective)
        history['val_persistence'].append(persist_loss)

        # Physical RMSE estimate
        phys_str = ""
        if cpm_range is not None:
            phys_rmse = val_loss * cpm_range
            phys_str  = f"  |  ~{phys_rmse:.1f} µg/m³"

        # Checkpoint
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss  = val_loss
            patience_count = 0
            torch.save(model.state_dict(), save_path)
            tag = "  ← saved"
        else:
            patience_count += 1
            tag = f"  (no improvement {patience_count}/{patience})"

        print(
            f"Epoch {epoch+1:3d}/{epochs} | "
            f"Train: {train_loss:.4f} | "
            f"Val: {val_loss:.4f}{phys_str} | "
            f"ValPersist: {persist_loss:.4f} | "
            f"Best: {best_val_loss:.4f}{tag}"
        )
        if physics_warmup_epochs > 0:
            print(f"           lambda_p_eff={lambda_p:.4f} (target={lambda_p_target:.4f})")

        # Persistence gate check every 5 epochs
        if (epoch + 1) % 5 == 0:
            check_persistence_gate(val_loss, epoch + 1)

        # Early stopping
        if patience_count >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs "
                  f"(no improvement for {patience} epochs).")
            break

    # Final gate check
    print(f"\n{'─'*60}")
    print(f"Training complete.  Best val RMSE (normalized): {best_val_loss:.4f}")
    if cpm_range is not None:
        print(f"Best val RMSE (physical estimate): {best_val_loss * cpm_range:.2f} µg/m³")
    check_persistence_gate(best_val_loss, epoch + 1)
    print(f"{'─'*60}")

    return history


# ─────────────────────────────────────────────
# Legacy alias
# ─────────────────────────────────────────────

# Keep old 'metric_loss' name accessible in case any caller uses it
metric_loss = rmse_loss

