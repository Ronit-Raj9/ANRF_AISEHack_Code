"""
Training loop with:
  - Normalized-space RMSE loss (targets and preds in [0, 1])
  - Persistence gate: val RMSE must beat 0.0208 (EDA baseline) to be useful
  - Early stopping on val RMSE with configurable patience
  - Physical RMSE estimate printed each epoch for interpretability
  - Weights & Biases integration: per-epoch metrics, per-horizon breakdown,
    LR tracking, gradient norms, model artifacts, and leaderboard proxy score
"""

import os
import numpy as np
import torch
from contextlib import nullcontext

# ── Optional wandb import (graceful degradation if not installed) ──────────────
try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _wandb = None
    _WANDB_AVAILABLE = False

# Persistence RMSE baseline in normalized [0,1] cpm25 space (from EDA, t+16 avg)
PERSISTENCE_RMSE_NORM = 0.0208
PERSISTENCE_RMSE_PHYS = 30.83  # µg/m³


# ─────────────────────────────────────────────
# Weights & Biases helpers
# ─────────────────────────────────────────────

def init_wandb(cfg: dict):
    """
    Initialize a W&B run from the 'wandb' section of cfg.
    Returns the run object (or None if wandb disabled / not installed).
    """
    if not _WANDB_AVAILABLE:
        print("wandb not installed — skipping W&B logging.")
        return None
    wcfg = cfg.get('wandb', {})
    if not bool(wcfg.get('enabled', False)):
        return None

    flat_cfg = {
        # Training
        'lr':               cfg['training']['lr'],
        'weight_decay':     cfg['training']['weight_decay'],
        'epochs':           cfg['training']['epochs'],
        'patience':         cfg['training'].get('patience', 8),
        'batch_size_train': cfg['training']['batch_size_train'],
        'grad_clip':        cfg['training']['grad_clip'],
        'use_amp':          cfg['training'].get('use_amp', True),
        'checkpoint_metric':cfg['training'].get('checkpoint_metric', 'val_rmse'),
        # Model
        'model_type':       cfg['model']['type'],
        # Loss
        'loss_type':        cfg['loss']['type'],
        'horizon_w_min':    cfg['loss']['horizon_weight_min'],
        'horizon_w_max':    cfg['loss']['horizon_weight_max'],
        'intensity_weighted': cfg['loss'].get('intensity_weighted', False),
        'intensity_alpha':  cfg['loss'].get('intensity_alpha', 1.5),
        # Data
        'val_month':        cfg['data']['val_month'],
        't_in_cpm':         cfg['time']['t_in_cpm'],
        't_in_met':         cfg['time']['t_in_met'],
        't_out':            cfg['time']['t_out'],
        # Baselines
        'persistence_rmse_norm': PERSISTENCE_RMSE_NORM,
        'persistence_rmse_phys': PERSISTENCE_RMSE_PHYS,
    }

    run = _wandb.init(
        entity=wcfg.get('entity', None),
        project=wcfg.get('project', 'aisehack'),
        name=wcfg.get('run_name') or None,
        tags=wcfg.get('tags', []),
        config=flat_cfg,
        reinit=True,
    )

    # Custom chart definitions for layout
    _wandb.define_metric('epoch')
    # All per-epoch metrics use epoch as x-axis
    for metric in [
        'train/rmse_norm', 'train/objective',
        'val/rmse_norm', 'val/rmse_phys', 'val/objective',
        'val/persistence_rmse', 'val/persistence_ratio',
        'val/gap_vs_persistence',
        'lr', 'grad_norm',
        'best/val_rmse_norm', 'best/val_rmse_phys',
    ]:
        _wandb.define_metric(metric, step_metric='epoch')
    # Per-horizon metrics
    for t in range(1, 17):
        _wandb.define_metric(f'horizon/val_rmse_h{t:02d}', step_metric='epoch')

    print(f"W&B run: {run.url}")
    return run


def finish_wandb(run, history: dict, bounds: dict | None, cfg: dict):
    """
    Log final summary tables + charts and finish the run.
    """
    if run is None or not _WANDB_AVAILABLE:
        return

    cpm_range = None
    if bounds is not None and 'cpm25' in bounds:
        fmin, fmax = bounds['cpm25']
        cpm_range = fmax - fmin

    epochs_ran = len(history['val_loss'])
    if epochs_ran == 0:
        run.finish()
        return

    # ── Final summary epoch-level table  ──────────────────────────────────────
    columns = ['epoch', 'train_rmse', 'val_rmse', 'val_phys_rmse',
               'val_objective', 'persistence_rmse', 'persistence_ratio',
               'selection_metric']
    rows = []
    for i in range(epochs_ran):
        vl = history['val_loss'][i]
        vp = history.get('val_persistence', [None] * epochs_ran)[i]
        pr = (vl / vp) if (vp and vp > 0) else None
        rows.append([
            i + 1,
            round(history['train_loss'][i], 5),
            round(vl, 5),
            round(vl * cpm_range, 3) if cpm_range else None,
            round(history.get('val_objective', [None] * epochs_ran)[i] or 0, 5),
            round(vp, 5) if vp is not None else None,
            round(pr, 4) if pr is not None else None,
            round(history.get('selection_metric', [None] * epochs_ran)[i] or 0, 5),
        ])
    run.log({'training_history': _wandb.Table(columns=columns, data=rows)})

    # ── Best run summary ──────────────────────────────────────────────────────
    best_idx  = int(np.argmin(np.asarray(history['selection_metric'])))
    best_vrmse = history['val_loss'][best_idx]
    run.summary['best_epoch']      = best_idx + 1
    run.summary['best_val_rmse']   = best_vrmse
    if cpm_range:
        run.summary['best_val_phys_rmse'] = best_vrmse * cpm_range
    run.summary['beats_persistence'] = int(best_vrmse < PERSISTENCE_RMSE_NORM)
    run.summary['persistence_ratio'] = best_vrmse / PERSISTENCE_RMSE_NORM

    run.finish()
    print(f"W&B run finished: {run.url}")


def _log_horizon_rmse(run, pred: torch.Tensor, target: torch.Tensor, epoch: int):
    """
    Compute and log per-horizon RMSE to wandb.

    pred / target : (B, H, W, T)
    """
    if run is None:
        return
    with torch.no_grad():
        spatial_mse = torch.mean((pred - target) ** 2, dim=(1, 2))  # (B, T)
        per_horizon = torch.sqrt(spatial_mse.mean(0) + 1e-8)          # (T,)
    log_dict = {'epoch': epoch}
    for t_idx, rmse_val in enumerate(per_horizon.cpu().tolist()):
        log_dict[f'horizon/val_rmse_h{t_idx + 1:02d}'] = rmse_val
    run.log(log_dict)


def rmse_per_horizon(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return per-horizon RMSE vector of shape (T,) in normalized space."""
    spatial_mse = torch.mean((pred - target) ** 2, dim=(1, 2))  # (B, T)
    return torch.sqrt(spatial_mse.mean(0) + 1e-8)               # (T,)


# ─────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────
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

def train(cfg, model, train_dl, val_dl, bounds: dict = None, wandb_run=None):
    """
    Full training loop.

    Parameters
    ----------
    bounds : optional dict {feat: (fmin, fmax)} — if provided, physical RMSE
             is estimated each epoch using cpm25 normalization range.
    wandb_run : optional W&B run object returned by init_wandb(cfg).

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

    # W&B: watch model for gradient/param histograms
    wcfg = cfg.get('wandb', {})
    if wandb_run is not None and _WANDB_AVAILABLE:
        _wandb.watch(model, log='gradients', log_freq=max(1, len(train_dl) * 2))

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
        total_grad_norm = 0.0
        num_batches = 0
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
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            total_grad_norm += float(grad_norm)
            num_batches += 1
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
        # Accumulate per-horizon sums for wandb
        horizon_rmse_accum: list[torch.Tensor] | None = None
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred   = model(xb)
                val_losses.append(rmse_loss(pred, yb, cfg).item())
                val_objectives.append(objective_loss(pred, yb, cfg, bounds, epoch_idx=epoch).item())
                val_persist.append(persistence_rmse_from_batch(xb, yb, t_in_cpm).item())
                # Per-horizon RMSE (accumulated over batches)
                if wandb_run is not None:
                    h_rmse = rmse_per_horizon(pred, yb)   # (T,)
                    if horizon_rmse_accum is None:
                        horizon_rmse_accum = [h_rmse]
                    else:
                        horizon_rmse_accum.append(h_rmse)

        train_objective = float(np.mean(epoch_losses))
        train_loss = float(np.mean(epoch_rmse))
        val_loss   = float(np.mean(val_losses))
        val_objective = float(np.mean(val_objectives))
        persist_loss = float(np.mean(val_persist))
        avg_grad_norm = total_grad_norm / max(num_batches, 1)
        current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr']
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
        phys_rmse = None
        if cpm_range is not None:
            phys_rmse = val_loss * cpm_range
            phys_str  = f"  |  ~{phys_rmse:.1f} µg/m³"

        # Persistence ratio: < 1 means beating the baseline → closer to winning
        persistence_ratio = val_loss / persist_loss if persist_loss > 0 else None
        gap_vs_persistence = val_loss - PERSISTENCE_RMSE_NORM

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

        # ── W&B per-epoch logging ──────────────────────────────────────────────
        if wandb_run is not None and _WANDB_AVAILABLE:
            log_dict = {
                'epoch': epoch + 1,
                # Core losses
                'train/rmse_norm':         train_loss,
                'train/objective':         train_objective,
                'val/rmse_norm':           val_loss,
                'val/objective':           val_objective,
                'val/persistence_rmse':    persist_loss,
                # Diagnostic scalars
                'val/gap_vs_persistence':  gap_vs_persistence,
                'lr':                      current_lr,
                'grad_norm':               avg_grad_norm,
                # Best-so-far
                'best/selection_metric':   best_val_loss,
                # Leaderboard proxy: < 1 → beating local persistence
                'leaderboard/persistence_ratio': val_loss / PERSISTENCE_RMSE_NORM,
            }
            if phys_rmse is not None:
                log_dict['val/rmse_phys'] = phys_rmse
                log_dict['best/val_rmse_phys'] = history['val_loss'][int(np.argmin(history['selection_metric']))] * cpm_range
            if persistence_ratio is not None:
                log_dict['val/persistence_ratio'] = persistence_ratio
            # Best val RMSE so far (for reference line in charts)
            best_norm_so_far = min(history['val_loss'])
            log_dict['best/val_rmse_norm'] = best_norm_so_far

            # Per-horizon RMSE breakdown
            if horizon_rmse_accum:
                avg_horizon = torch.stack(horizon_rmse_accum, dim=0).mean(0)  # (T,)
                for t_idx, v in enumerate(avg_horizon.cpu().tolist()):
                    log_dict[f'horizon/val_rmse_h{t_idx + 1:02d}'] = v
                # Also log as a bar-chart-friendly table every N epochs
                hz_log_every = int(wcfg.get('horizon_log_every', 1))
                if (epoch + 1) % hz_log_every == 0:
                    hz_table = _wandb.Table(
                        columns=['horizon', 'val_rmse_norm', 'val_rmse_phys'],
                        data=[
                            [
                                f'H+{t_idx + 1}',
                                round(float(v), 5),
                                round(float(v) * cpm_range, 3) if cpm_range else None,
                            ]
                            for t_idx, v in enumerate(avg_horizon.cpu().tolist())
                        ]
                    )
                    log_dict['charts/horizon_rmse_bar'] = _wandb.plot.bar(
                        hz_table, 'horizon', 'val_rmse_norm',
                        title=f'Val RMSE per Horizon (epoch {epoch + 1})'
                    )

            # Save checkpoint artifact on improvement
            if improved and bool(wcfg.get('log_model', True)):
                artifact = _wandb.Artifact(
                    name=f'best_model-{wandb_run.id}',
                    type='model',
                    description=f'Best checkpoint — epoch {epoch + 1}, val_rmse={val_loss:.5f}',
                    metadata={
                        'epoch': epoch + 1,
                        'val_rmse_norm': val_loss,
                        'val_rmse_phys': phys_rmse,
                        'selection_metric': selection_metric,
                    }
                )
                artifact.add_file(save_path)
                wandb_run.log_artifact(artifact)

            wandb_run.log(log_dict)

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

