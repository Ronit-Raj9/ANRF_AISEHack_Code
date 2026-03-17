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

def get_optimizer(
    cfg,
    model,
    steps_per_epoch,
    params=None,
    epochs_override: int | None = None,
    lr_override: float | None = None,
):
    train_params = list(params) if params is not None else [p for p in model.parameters() if p.requires_grad]
    if len(train_params) == 0:
        raise RuntimeError("No trainable parameters passed to optimizer.")

    lr_value = float(lr_override if lr_override is not None else cfg['training']['lr'])
    total_epochs = int(epochs_override if epochs_override is not None else cfg['training']['epochs'])

    optimizer = torch.optim.AdamW(
        train_params,
        lr           = lr_value,
        weight_decay = cfg['training']['weight_decay'],
    )
    sched_type = cfg['training'].get('scheduler', 'coswr')
    if sched_type == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr           = lr_value,
            steps_per_epoch  = steps_per_epoch,
            epochs           = total_epochs,
            pct_start        = cfg['training']['pct_start'],
        )
    else:  # 'coswr' — Cosine Annealing with Warm Restarts
        T_0    = max(1, steps_per_epoch * cfg['training'].get('t0_epochs', 10))
        T_mult = cfg['training'].get('t_mult', 2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0    = T_0,
            T_mult = T_mult,
            eta_min = lr_value * 1e-2,  # floor = 1% of initial LR
        )
    return optimizer, scheduler

# ─────────────────────────────────────────────
# PINO Physics-Informed Loss
# ─────────────────────────────────────────────
def _spectral_spatial_derivatives(
    field: torch.Tensor,
    dx: float = 1.0,
    dy: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute exact spatial derivatives in Fourier space.

    Parameters
    ----------
    field : (B, H, W, T)
    dx, dy : grid spacing along H/W axes

    Returns
    -------
    dfdh, dfdw, laplacian : each (B, H, W, T)
    """
    if field.ndim != 4:
        raise ValueError(f"Expected field shape (B,H,W,T), got {field.shape}")

    # FFT on spatial axes after moving T next to batch for efficient batched fft2.
    x = field.permute(0, 3, 1, 2)  # (B, T, H, W)
    B, T, H, W = x.shape
    x_hat = torch.fft.fft2(x, dim=(-2, -1), norm='ortho')

    k_h = (2.0 * np.pi) * torch.fft.fftfreq(H, d=float(dx), device=field.device, dtype=field.dtype)
    k_w = (2.0 * np.pi) * torch.fft.fftfreq(W, d=float(dy), device=field.device, dtype=field.dtype)
    k_h = k_h.view(1, 1, H, 1)
    k_w = k_w.view(1, 1, 1, W)

    ik_h = (1j * k_h).to(x_hat.dtype)
    ik_w = (1j * k_w).to(x_hat.dtype)
    k2 = (k_h ** 2 + k_w ** 2).to(x_hat.dtype)

    dfdh = torch.fft.ifft2(ik_h * x_hat, dim=(-2, -1), norm='ortho').real
    dfdw = torch.fft.ifft2(ik_w * x_hat, dim=(-2, -1), norm='ortho').real
    lap = torch.fft.ifft2(-k2 * x_hat, dim=(-2, -1), norm='ortho').real

    return dfdh.permute(0, 2, 3, 1), dfdw.permute(0, 2, 3, 1), lap.permute(0, 2, 3, 1)


def compute_physics_loss(pred, xb, yb, cfg, residual_mode: bool = False):
    """
    Compute Advection-Diffusion residual for PINO:
    R = dC/dt + u · grad(C) - kappa * laplacian(C) - S
    Penalize squared residual.

    When residual_mode=True (Topic 8), ``pred`` is a delta (Δ concentration),
    so we reconstruct the absolute prediction before computing gradients.
    """
    import torch

    # Expected layout for PINO forward output: (B, H, W, T_out)
    if pred.ndim != 4:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)

    # When model predicts residuals, reconstruct absolute concentration map
    if residual_mode:
        t_in_cpm = cfg['time']['t_in_cpm']
        last_obs = xb[:, 0, t_in_cpm - 1, :, :].unsqueeze(-1)  # (B, H, W, 1)
        pred = last_obs + pred  # convert to absolute normalized space

    # Temporal derivative against last observed cpm25 (channel 0 at last observed step)
    dt = 1.0
    t_in_cpm = cfg['time']['t_in_cpm']
    last_c = xb[:, 0, t_in_cpm - 1, :, :].unsqueeze(-1)  # (B, H, W, 1)
    dC_dt = (pred - last_c) / dt               # (B, H, W, T)

    # Spatial derivatives in Fourier domain: exact and lower-noise than finite diff.
    dx = cfg.get('physics', {}).get('dx', 1.0)
    dy = cfg.get('physics', {}).get('dy', 1.0)
    grad_h, grad_w, laplacian = _spectral_spatial_derivatives(pred, dx=dx, dy=dy)

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

    # Diffusion (spectral Laplacian)
    kappa = cfg.get('physics', {}).get('diffusivity', 1.0)
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
    last_obs = xb[:, 0, t_in_cpm - 1]                     # (B, H, W)
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
    residual_mode = cfg['training'].get('residual_target', False)

    # Physics-Informed Fine-Tuning (PFT) schedule
    use_pft = bool(cfg['training'].get('use_pft', True))
    pft_ratio = float(cfg['training'].get('pft_phase1_ratio', 0.7))
    pft_ratio = min(max(pft_ratio, 0.05), 0.95)
    phase1_epochs = max(1, int(round(epochs * pft_ratio))) if use_pft else epochs
    phase2_start_epoch = phase1_epochs
    phase2_epochs = max(0, epochs - phase1_epochs)
    pft_lr = float(cfg['training'].get('pft_lr', cfg['training']['lr'] * 0.35))

    # cpm25 range for physical RMSE estimation
    cpm_range = None
    if bounds is not None and 'cpm25' in bounds:
        fmin, fmax = bounds['cpm25']
        cpm_range  = fmax - fmin   # 1464.25 µg/m³

    optimizer, scheduler = get_optimizer(cfg, model, len(train_dl), epochs_override=max(1, phase1_epochs))

    best_val_loss  = float('inf')
    patience_count = 0
    history        = {'train_loss': [], 'val_loss': [], 'train_objective': [], 'val_persistence': []}
    spatial_weight_map = get_spatial_weight_tensor(cfg, device, torch.float32)

    print(f"\n{'─'*60}")
    print(f"  Persistence gate  (normalized RMSE): {PERSISTENCE_RMSE_NORM:.4f}")
    if residual_mode:
        print("  Residual-target mode ON (Topic 8): model predicts delta from last obs")
    if use_pft:
        print(
            f"  PFT enabled: phase-1 data-only epochs=1..{phase1_epochs}, "
            f"phase-2 spectral-physics epochs={phase1_epochs+1}..{epochs}"
        )
    if spatial_weight_map is not None:
        print("  Training loss uses hotspot-weighted spatial RMSE (val remains unweighted)")
    if cpm_range:
        print(f"  Persistence gate  (physical RMSE) : {PERSISTENCE_RMSE_PHYS:.2f} µg/m³")
    print(f"{'─'*60}\n")

    for epoch in range(epochs):
        in_phase2 = use_pft and (epoch >= phase2_start_epoch)

        if use_pft and epoch == phase2_start_epoch:
            print(
                f"\n→ Entering PFT phase-2 at epoch {epoch+1}: "
                "freezing non-spectral weights, optimizing PDE residual only."
            )
            if hasattr(model, 'freeze_non_spectral'):
                model.freeze_non_spectral()
            else:
                for name, p in model.named_parameters():
                    keep = ('spectral' in name) or ('weight_pos' in name) or ('weight_neg' in name)
                    p.requires_grad = bool(keep)

            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer, scheduler = get_optimizer(
                cfg,
                model,
                len(train_dl),
                params=trainable_params,
                epochs_override=max(1, phase2_epochs),
                lr_override=pft_lr,
            )

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
            pred   = model(xb)
            sw = spatial_weight_map.to(pred.dtype) if spatial_weight_map is not None else None

            if residual_mode:
                # Topic 8: train on delta from last observed cpm25 (z-score space)
                last_obs = xb[:, 0, t_in_cpm - 1, :, :]          # (B, H, W)
                target_for_loss = yb - last_obs.unsqueeze(-1)      # (B, H, W, T_out) residual
                data_loss = objective_loss(pred, target_for_loss, cfg, spatial_weights=sw)
                # For plain RMSE tracking, reconstruct absolute pred to report
                pred_abs = last_obs.unsqueeze(-1) + pred
                rmse_metric = rmse_loss_plain(pred_abs, yb)
            else:
                data_loss   = objective_loss(pred, yb, cfg, spatial_weights=sw)
                rmse_metric = rmse_loss_plain(pred, yb)

            physics_loss = compute_physics_loss(pred, xb, yb, cfg, residual_mode=residual_mode)
            if in_phase2:
                # Phase 2: spectral-only physics fine-tune (decoupled from data loss)
                loss = physics_loss
            elif use_pft:
                # Phase 1: data-only training (no physics gradient conflict)
                loss = data_loss
            else:
                # Backward-compatible joint objective when PFT disabled
                loss = lambda_d * data_loss + lambda_p * physics_loss
            # Skip batch if loss is NaN/Inf (bad inputs slipping through)
            if not torch.isfinite(loss):
                optimizer.zero_grad()
                continue

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
                pred   = model(xb)
                if residual_mode:
                    last_obs = xb[:, 0, t_in_cpm - 1, :, :]
                    pred_abs = last_obs.unsqueeze(-1) + pred
                else:
                    pred_abs = pred
                val_losses.append(rmse_loss_plain(pred_abs, yb).item())
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
        if use_pft:
            phase_tag = 'P2-physics-only' if in_phase2 else 'P1-data-only'
            print(f"           phase={phase_tag}")
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

