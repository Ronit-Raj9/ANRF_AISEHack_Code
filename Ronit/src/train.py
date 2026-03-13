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
from contextlib import nullcontext

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


def rmse_loss(pred: torch.Tensor, target: torch.Tensor, cfg: dict | None = None) -> torch.Tensor:
    """
    Spatial RMSE averaged over batch and time steps.

    Parameters
    ----------
    pred, target : (B, H, W, T)  — values in normalized [0, 1] space
    """
    spatial_mse = torch.mean((pred - target) ** 2, dim=(1, 2))  # (B, T)
    if cfg is not None:
        weights = _horizon_weights(cfg, target)[None]
        spatial_mse = spatial_mse * weights
    return torch.mean(torch.sqrt(spatial_mse + 1e-8))


def mae_loss(pred: torch.Tensor, target: torch.Tensor, cfg: dict | None = None) -> torch.Tensor:
    """Weighted MAE over batch and horizon."""
    spatial_mae = torch.mean(torch.abs(pred - target), dim=(1, 2))
    if cfg is not None:
        weights = _horizon_weights(cfg, target)[None]
        spatial_mae = spatial_mae * weights
    return torch.mean(spatial_mae)


def _normalized_to_physical_domain(x: torch.Tensor, cfg: dict, bounds: dict) -> torch.Tensor:
    """Convert normalized cpm25 tensor into physical µg/m³ domain."""
    log_domain = _normalized_to_log1p_domain(x, cfg, bounds)
    if bool(cfg.get('preprocessing', {}).get('cpm25_log1p', False)):
        return torch.clamp(torch.expm1(log_domain), min=0.0)
    return torch.clamp(log_domain, min=0.0)


def _intensity_weights(target_phys: torch.Tensor, cfg: dict) -> torch.Tensor:
    """Per-pixel intensity weights derived from physical target values."""
    lc = cfg.get('loss', {})
    alpha = float(lc.get('intensity_alpha', 1.5))
    ref = float(lc.get('intensity_ref', 59.0))
    cap = float(lc.get('intensity_cap', 3.0))
    factor = torch.clamp(target_phys / max(ref, 1e-6), min=0.0, max=cap)
    return 1.0 + alpha * factor


def weighted_mae_phys_loss(pred: torch.Tensor, target: torch.Tensor, cfg: dict, bounds: dict | None) -> torch.Tensor:
    """Intensity-weighted MAE in physical cpm25 space."""
    if bounds is None or 'cpm25' not in bounds:
        return mae_loss(pred, target, cfg)
    pred_phys = _normalized_to_physical_domain(pred, cfg, bounds)
    target_phys = _normalized_to_physical_domain(target, cfg, bounds)
    weights = _intensity_weights(target_phys, cfg)
    mae = torch.abs(pred_phys - target_phys)
    weighted = mae * weights
    spatial = torch.mean(weighted, dim=(1, 2))
    horizon_w = _horizon_weights(cfg, target)[None]
    return torch.mean(spatial * horizon_w)


def _normalized_to_log1p_domain(x: torch.Tensor, cfg: dict, bounds: dict) -> torch.Tensor:
    """
    Convert normalized cpm25 tensors (optionally z-scored) into log1p-physical domain.

    Input shape: (B, H, W, T)
    Output shape: (B, H, W, T)
    """
    y = x

    if bool(cfg.get('preprocessing', {}).get('cpm25_grid_zscore', False)):
        runtime = cfg.get('_runtime', {})
        mean = runtime.get('cpm25_grid_mean')
        std = runtime.get('cpm25_grid_std')
        if mean is None or std is None:
            raise RuntimeError(
                "cpm25 grid z-score enabled but runtime stats are missing in cfg['_runtime']."
            )
        mean_t = torch.as_tensor(mean, device=x.device, dtype=x.dtype)[None, :, :, None]
        std_t = torch.as_tensor(std, device=x.device, dtype=x.dtype)[None, :, :, None]
        y = y * std_t + mean_t

    fmin, fmax = bounds['cpm25']
    if bool(cfg.get('preprocessing', {}).get('cpm25_log1p', False)):
        lo = float(np.log1p(max(fmin, 0.0)))
        hi = float(np.log1p(max(fmax, 0.0)))
    else:
        lo = float(fmin)
        hi = float(fmax)
    return y * (hi - lo) + lo


def log_rmse_loss(pred: torch.Tensor, target: torch.Tensor, cfg: dict, bounds: dict | None) -> torch.Tensor:
    """RMSE in log1p-physical domain (after inverse normalization transform)."""
    if bounds is None or 'cpm25' not in bounds:
        return rmse_loss(pred, target, cfg)
    pred_log = _normalized_to_log1p_domain(pred, cfg, bounds)
    target_log = _normalized_to_log1p_domain(target, cfg, bounds)
    spatial_mse = torch.mean((pred_log - target_log) ** 2, dim=(1, 2))
    weights = _horizon_weights(cfg, target)[None]
    spatial_mse = spatial_mse * weights
    return torch.mean(torch.sqrt(spatial_mse + 1e-8))


def objective_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    cfg: dict,
    bounds: dict | None = None,
    epoch_idx: int | None = None,
) -> torch.Tensor:
    """Main optimization objective: configurable (log-RMSE or weighted RMSE+MAE)."""
    loss_type = str(cfg.get('loss', {}).get('type', 'rmse_mae')).lower()
    if loss_type == 'log_rmse':
        base = log_rmse_loss(pred, target, cfg, bounds)
        if bool(cfg.get('loss', {}).get('intensity_weighted', False)):
            phase_switch = int(cfg.get('loss', {}).get('phase_switch_epoch', 20))
            current_epoch = 0 if epoch_idx is None else epoch_idx
            if current_epoch >= phase_switch:
                phase2_mode = str(cfg.get('loss', {}).get('phase2_mode', 'weighted_mae_phys')).lower()
                if phase2_mode == 'weighted_mae_phys':
                    return weighted_mae_phys_loss(pred, target, cfg, bounds)
            aux = weighted_mae_phys_loss(pred, target, cfg, bounds)
            return 0.8 * base + 0.2 * aux
        return base

    wrmse = rmse_loss(pred, target, cfg)
    wmae = mae_loss(pred, target, cfg)
    a = cfg.get('loss', {}).get('rmse_weight', 0.8)
    b = cfg.get('loss', {}).get('mae_weight', 0.2)
    return a * wrmse + b * wmae


# ─────────────────────────────────────────────
# Optimizer & Scheduler
# ─────────────────────────────────────────────

def get_optimizer(cfg, model, steps_per_epoch):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = cfg['training']['lr'],
        weight_decay = cfg['training']['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr           = cfg['training']['lr'],
        steps_per_epoch  = steps_per_epoch,
        epochs           = cfg['training']['epochs'],
        pct_start        = cfg['training']['pct_start'],
    )
    return optimizer, scheduler


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
    last_obs = xb[:, 0, t_in_cpm - 1]                     # (B, H, W)
    persist = last_obs[..., None].repeat(1, 1, 1, yb.shape[-1])
    return rmse_loss(persist, yb)


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
    use_amp   = bool(cfg['training'].get('use_amp', True) and device.type == 'cuda')

    # cpm25 range for physical RMSE estimation
    cpm_range = None
    if bounds is not None and 'cpm25' in bounds:
        fmin, fmax = bounds['cpm25']
        cpm_range  = fmax - fmin   # 1464.25 µg/m³

    optimizer, scheduler = get_optimizer(cfg, model, len(train_dl))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_loss  = float('inf')
    patience_count = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_objective': [],
        'val_objective': [],
        'val_persistence': [],
        'selection_metric': [],
    }

    print(f"\n{'─'*60}")
    print(f"  Persistence gate  (normalized RMSE): {PERSISTENCE_RMSE_NORM:.4f}")
    if cpm_range:
        print(f"  Persistence gate  (physical RMSE) : {PERSISTENCE_RMSE_PHYS:.2f} µg/m³")
    print(f"{'─'*60}\n")

    for epoch in range(epochs):
        # ── Train ──
        model.train()
        epoch_losses = []
        epoch_rmse = []
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)

            amp_ctx = torch.autocast(device_type='cuda', dtype=torch.float16) if use_amp else nullcontext()
            with amp_ctx:
                pred = model(xb)
                loss = objective_loss(pred, yb, cfg, bounds, epoch_idx=epoch)
            rmse_metric = rmse_loss(pred.detach(), yb, cfg)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            epoch_losses.append(loss.item())
            epoch_rmse.append(rmse_metric.item())

        # ── Validate ──
        model.eval()
        val_losses = []
        val_objectives = []
        val_persist = []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred   = model(xb)
                val_losses.append(rmse_loss(pred, yb, cfg).item())
                val_objectives.append(objective_loss(pred, yb, cfg, bounds, epoch_idx=epoch).item())
                val_persist.append(persistence_rmse_from_batch(xb, yb, t_in_cpm).item())

        train_objective = float(np.mean(epoch_losses))
        train_loss = float(np.mean(epoch_rmse))
        val_loss   = float(np.mean(val_losses))
        val_objective = float(np.mean(val_objectives))
        persist_loss = float(np.mean(val_persist))
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_objective'].append(train_objective)
        history['val_objective'].append(val_objective)
        history['val_persistence'].append(persist_loss)

        metric_mode = str(cfg['training'].get('checkpoint_metric', 'val_rmse')).lower()
        if metric_mode == 'val_objective':
            selection_metric = val_objective
        elif metric_mode == 'mixed':
            alpha = float(cfg['training'].get('checkpoint_mixed_alpha', 0.7))
            selection_metric = alpha * val_objective + (1.0 - alpha) * val_loss
        else:
            selection_metric = val_loss
        history['selection_metric'].append(selection_metric)

        # Physical RMSE estimate
        phys_str = ""
        if cpm_range is not None:
            phys_rmse = val_loss * cpm_range
            phys_str  = f"  |  ~{phys_rmse:.1f} µg/m³"

        # Checkpoint
        improved = selection_metric < best_val_loss
        if improved:
            best_val_loss  = selection_metric
            patience_count = 0
            torch.save(model.state_dict(), save_path)
            tag = "  ← saved"
        else:
            patience_count += 1
            tag = f"  (no improvement {patience_count}/{patience})"

        print(
            f"Epoch {epoch+1:3d}/{epochs} | "
            f"Train: {train_loss:.4f} | "
            f"ValRMSE: {val_loss:.4f}{phys_str} | "
            f"ValObj: {val_objective:.4f} | "
            f"ValPersist: {persist_loss:.4f} | "
            f"Sel: {selection_metric:.4f} | "
            f"BestSel: {best_val_loss:.4f}{tag}"
        )

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
    best_idx = int(np.argmin(np.asarray(history['selection_metric'])))
    best_rmse = history['val_loss'][best_idx]
    print(f"Training complete. Best checkpoint by selection metric: {best_val_loss:.4f}")
    print(f"Best val RMSE (normalized): {best_rmse:.4f}")
    if cpm_range is not None:
        print(f"Best val RMSE (physical estimate): {best_rmse * cpm_range:.2f} µg/m³")
    check_persistence_gate(best_rmse, epoch + 1)
    print(f"{'─'*60}")

    return history


# ─────────────────────────────────────────────
# Legacy alias
# ─────────────────────────────────────────────

# Keep old 'metric_loss' name accessible in case any caller uses it
metric_loss = rmse_loss

