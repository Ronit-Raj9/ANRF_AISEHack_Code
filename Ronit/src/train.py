import os
import numpy as np
import torch
from contextlib import nullcontext

try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _wandb = None
    _WANDB_AVAILABLE = False

PERSISTENCE_RMSE_NORM = 0.0208
PERSISTENCE_RMSE_PHYS = 30.83  # µg/m³

def _checkpoint_enabled(cfg: dict) -> bool:
    return bool(cfg.get('training', {}).get('enable_checkpointing', True))

def _checkpoint_dir(cfg: dict) -> str:
    return cfg.get('paths', {}).get('checkpoint_dir', os.path.join(cfg['paths']['temp'], 'checkpoints'))

def _checkpoint_last_path(cfg: dict) -> str:
    return cfg.get('paths', {}).get('checkpoint_last', os.path.join(_checkpoint_dir(cfg), 'last_checkpoint.pt'))

def _save_training_checkpoint(cfg, epoch_1based, model, optimizer, scheduler, scaler, history, best_metric, patience_count, is_best=False):
    if not _checkpoint_enabled(cfg): return
    os.makedirs(_checkpoint_dir(cfg), exist_ok=True)
    payload = {
        'epoch': epoch_1based, 'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'history': history, 'best_metric': float(best_metric), 'patience_count': int(patience_count),
    }
    # Always save/overwrite the last checkpoint
    torch.save(payload, _checkpoint_last_path(cfg))
    # When a new best metric is achieved, also save/overwrite the best checkpoint
    if is_best:
        torch.save(payload, cfg['paths']['model_save'])

def _resolve_resume_checkpoint(cfg: dict) -> str | None:
    explicit = cfg.get('training', {}).get('resume_checkpoint_path')
    if explicit and os.path.exists(explicit): return explicit
    last_path = _checkpoint_last_path(cfg)
    if os.path.exists(last_path): return last_path
    return None

def _try_resume_training_state(cfg, model, optimizer, scheduler, scaler):
    if not bool(cfg.get('training', {}).get('resume_if_available', True)):
        return 0, float('inf'), 0, None
    path = _resolve_resume_checkpoint(cfg)
    if not path: return 0, float('inf'), 0, None
    payload = torch.load(path, map_location=cfg['device'])
    model.load_state_dict(payload['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in payload: optimizer.load_state_dict(payload['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in payload: scheduler.load_state_dict(payload['scheduler_state_dict'])
    if scaler and payload.get('scaler_state_dict'): scaler.load_state_dict(payload['scaler_state_dict'])
    print(f"Resumed from {path}")
    return int(payload.get('epoch', 0)), float(payload.get('best_metric', float('inf'))), int(payload.get('patience_count', 0)), payload.get('history')

def init_wandb(cfg: dict):
    if not _WANDB_AVAILABLE or not bool(cfg.get('wandb', {}).get('enabled', False)): return None
    run = _wandb.init(
        entity=cfg['wandb'].get('entity', None), project=cfg['wandb'].get('project', 'aisehack'),
        name=cfg['wandb'].get('run_name'), tags=cfg['wandb'].get('tags', []), reinit='finish_previous'
    )
    return run

def finish_wandb(run, history: dict, bounds: dict, cfg: dict):
    if run is None or not _WANDB_AVAILABLE: return
    if history.get('val_phys_rmse'):
        best_idx = int(np.argmin(history.get('selection_metric', history['val_phys_rmse'])))
        run.summary['best_val_rmse_phys'] = history['val_phys_rmse'][best_idx]
        run.summary['best_val_rmse_std'] = history['val_loss'][best_idx]
    run.finish()

def _horizon_weights(cfg, target: torch.Tensor) -> torch.Tensor:
    t_out = target.shape[-1]
    lo = cfg.get('loss', {}).get('horizon_weight_min', 0.8)
    hi = cfg.get('loss', {}).get('horizon_weight_max', 1.4)
    return torch.linspace(lo, hi, t_out, device=target.device, dtype=target.dtype)


def rmse_loss(pred, target, cfg: dict | None = None):
    spatial_mse = torch.mean((pred - target) ** 2, dim=(1, 2))
    if cfg is not None:
        weights = _horizon_weights(cfg, target)[None]
        spatial_mse = spatial_mse * weights
    return torch.mean(torch.sqrt(spatial_mse + 1e-8))


def rmse_loss_plain(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    spatial_mse = torch.mean((pred - target) ** 2, dim=(1, 2))
    return torch.mean(torch.sqrt(spatial_mse + 1e-8))


def mae_loss(pred: torch.Tensor, target: torch.Tensor, cfg: dict | None = None) -> torch.Tensor:
    spatial_mae = torch.mean(torch.abs(pred - target), dim=(1, 2))
    if cfg is not None:
        weights = _horizon_weights(cfg, target)[None]
        spatial_mae = spatial_mae * weights
    return torch.mean(spatial_mae)


def objective_loss(pred: torch.Tensor, target: torch.Tensor, cfg: dict) -> torch.Tensor:
    wrmse = rmse_loss(pred, target, cfg)
    wmae = mae_loss(pred, target, cfg)
    a = cfg.get('loss', {}).get('rmse_weight', 0.8)
    b = cfg.get('loss', {}).get('mae_weight', 0.2)
    return a * wrmse + b * wmae


def compute_physics_loss(pred, xb, yb, cfg):
    if pred.ndim != 4:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)

    dt = 1.0
    last_c = xb[:, 0, -1, :, :].unsqueeze(-1)
    dC_dt = (pred - last_c) / dt

    grad_h = torch.gradient(pred, dim=1)[0]
    grad_w = torch.gradient(pred, dim=2)[0]

    feature_names = cfg.get('features', {}).get('all', [])

    def _feat_idx(name: str):
        if name in feature_names:
            idx = feature_names.index(name)
            if idx < xb.shape[1]:
                return idx
        return None

    u_idx = _feat_idx('u10')
    v_idx = _feat_idx('v10')
    if u_idx is None or v_idx is None:
        advection = torch.zeros_like(pred)
    else:
        u = xb[:, u_idx, -1, :, :].unsqueeze(-1)
        v = xb[:, v_idx, -1, :, :].unsqueeze(-1)
        advection = u * grad_h + v * grad_w

    kappa = cfg.get('physics', {}).get('diffusivity', 1.0)
    laplacian = torch.gradient(grad_h, dim=1)[0] + torch.gradient(grad_w, dim=2)[0]
    diffusion = kappa * laplacian

    S = torch.zeros_like(pred)
    for name in ('SO2', 'NOx', 'PM25', 'cpm25'):
        idx = _feat_idx(name)
        if idx is not None:
            S = S + xb[:, idx, -1, :, :].unsqueeze(-1)

    R = dC_dt + advection - diffusion - S
    return torch.mean(R ** 2)


def check_persistence_gate(val_rmse: float, epoch: int) -> None:
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

def _grid_stats_tensors(cfg: dict, device, dtype):
    stats = cfg.get('_runtime', {}).get('grid_stats')
    mean, std = stats['cpm25']
    return torch.as_tensor(mean, device=device, dtype=dtype)[None, :, :, None], torch.as_tensor(std, device=device, dtype=dtype)[None, :, :, None]

def _normalized_to_log1p_domain(x, cfg, bounds):
    if str(cfg.get('preprocessing', {}).get('normalization', '')).lower() == 'grid_log_standardize':
        mean_t, std_t = _grid_stats_tensors(cfg, x.device, x.dtype)
        return x * std_t + mean_t
    fmin, fmax = bounds['cpm25']
    if bool(cfg.get('preprocessing', {}).get('cpm25_log1p', False)):
        lo = float(np.log1p(max(fmin, 0.0)))
        hi = float(np.log1p(max(fmax, 0.0)))
        return x * (hi - lo) + lo
    return x * (fmax - fmin) + fmin

def _normalized_to_physical_domain(x, cfg, bounds):
    log_domain = _normalized_to_log1p_domain(x, cfg, bounds)
    norm = str(cfg.get('preprocessing', {}).get('normalization', '')).lower()
    if 'grid_log' in norm:
        log_domain = torch.clamp(log_domain, min=-20.0, max=20.0)
        return torch.clamp(torch.expm1(log_domain), min=0.0)

    is_log1p = bool(cfg.get('preprocessing', {}).get('cpm25_log1p', False))
    if is_log1p:
        log_domain = torch.clamp(log_domain, min=-20.0, max=20.0)
        return torch.clamp(torch.expm1(log_domain), min=0.0)
    return torch.clamp(log_domain, min=0.0)

def physical_rmse_metric(pred, target, cfg, bounds):
    if bounds is None or 'cpm25' not in bounds: return rmse_loss(pred, target)
    pred_phys = _normalized_to_physical_domain(pred, cfg, bounds)
    target_phys = _normalized_to_physical_domain(target, cfg, bounds)
    return torch.mean(torch.sqrt(torch.mean((pred_phys - target_phys) ** 2, dim=(1, 2)) + 1e-8))

def persistence_rmse_from_batch(xb, yb, cfg, bounds):
    last_obs = xb[:, 0, cfg['time']['t_in_cpm'] - 1]
    persist = last_obs[..., None].repeat(1, 1, 1, yb.shape[-1])
    return physical_rmse_metric(persist, yb, cfg, bounds)

def get_optimizer(cfg, model, steps_per_epoch):
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['training']['lr'], weight_decay=cfg['training']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg['training']['lr'], steps_per_epoch=steps_per_epoch, epochs=cfg['training']['epochs'], pct_start=cfg['training'].get('pct_start', 0.1))
    return optimizer, scheduler

def train(cfg, model, train_dl, val_dl, bounds: dict = None, wandb_run=None):
    device = cfg['device']
    epochs = cfg['training']['epochs']
    patience = cfg['training'].get('patience', 12)
    grad_clip = cfg['training'].get('grad_clip', 1.0)

    requested_amp = bool(cfg['training'].get('use_amp', False) and device.type == 'cuda')
    has_complex_params = any(p.is_complex() for p in model.parameters())
    use_amp = requested_amp and not has_complex_params
    if requested_amp and has_complex_params:
        print("AMP disabled automatically: model has complex-valued parameters (TFNO spectral weights).")

    optimizer, scheduler = get_optimizer(cfg, model, len(train_dl))
    
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    feat_to_idx = {name: i for i, name in enumerate(cfg.get('features', {}).get('input', []))}

    def diffusion_loss(pred, xb=None, eps=1e-8):
        # pred: (B, H, W, T) — Laplacian over spatial dims
        p = pred.permute(0, 3, 1, 2)  # (B, T, H, W)
        lap = (p[:, :, 2:, 1:-1] + p[:, :, :-2, 1:-1] +
               p[:, :, 1:-1, 2:] + p[:, :, 1:-1, :-2] -
               4 * p[:, :, 1:-1, 1:-1])

        if xb is None or ('u10' not in feat_to_idx) or ('v10' not in feat_to_idx):
            return torch.mean(lap ** 2)

        u_idx = feat_to_idx['u10']
        v_idx = feat_to_idx['v10']
        u = xb[:, u_idx, -1, :, :]
        v = xb[:, v_idx, -1, :, :]
        wind_speed = torch.sqrt(torch.clamp(u * u + v * v, min=0.0))
        calm_threshold = float(cfg.get('physics', {}).get('calm_wind_threshold', 0.5))
        calm_mask = (wind_speed <= calm_threshold).float()[:, None, 1:-1, 1:-1]
        weighted = (lap ** 2) * calm_mask
        denom = torch.clamp(calm_mask.mean(), min=eps)
        return weighted.mean() / denom

    lambda_p = cfg['training'].get('lambda_p', 0.0)

    best_metric = float('inf')
    patience_count = 0
    start_epoch = 0

    history = {
        'train_loss': [], 'val_loss': [], 'train_objective': [],
        'val_objective': [], 'val_phys_rmse': [], 'val_persistence_phys': [],
        'selection_metric': []
    }

    resumed_history = None
    if _checkpoint_enabled(cfg):
        start_epoch, best_metric, patience_count, resumed_history = _try_resume_training_state(cfg, model, optimizer, scheduler, scaler)
        if isinstance(resumed_history, dict): history = resumed_history

    print(f"\n{'─'*70}")
    print("Starting Simple 1-Phase Training Baseline")
    print(f"Global Persistence (Physical): {PERSISTENCE_RMSE_PHYS:.2f} µg/m³")
    print(f"{'─'*70}\n")

    for epoch in range(start_epoch, epochs):
        model.train()
        train_losses, train_rmse_std = [], []

        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            if use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    pred = model(xb)
                    pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
                    loss = rmse_loss(pred, yb)
                    if lambda_p > 0:
                        loss = loss + lambda_p * diffusion_loss(pred, xb)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(xb)
                pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
                loss = rmse_loss(pred, yb)
                if lambda_p > 0:
                    loss = loss + lambda_p * diffusion_loss(pred, xb)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            scheduler.step()

            train_losses.append(loss.item())
            train_rmse_std.append(loss.item())

        model.eval()
        val_losses, val_phys, val_persist = [], [], []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
                
                val_losses.append(rmse_loss(pred, yb).item())
                val_phys.append(physical_rmse_metric(pred, yb, cfg, bounds).item())
                val_persist.append(persistence_rmse_from_batch(xb, yb, cfg, bounds).item())

        t_loss = float(np.mean(train_losses))
        v_loss = float(np.mean(val_losses))
        v_phys = float(np.mean(val_phys))
        v_pers = float(np.mean(val_persist))

        history['train_loss'].append(float(np.mean(train_rmse_std)))
        history['train_objective'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['val_objective'].append(v_loss)
        history['val_phys_rmse'].append(v_phys)
        history['val_persistence_phys'].append(v_pers)

        selection_metric = v_loss if cfg['training'].get('checkpoint_metric', 'val_rmse_std') == 'val_rmse_std' else v_phys
        history['selection_metric'].append(selection_metric)

        improved = selection_metric < best_metric
        if improved:
            best_metric = selection_metric
            patience_count = 0
            tag = '  ← saved'
        else:
            patience_count += 1
            tag = f'  (no improvement {patience_count}/{patience})'

        print(
            f"Epoch {epoch+1:3d}/{epochs} | "
            f"Train RMSE(std): {t_loss:.4f} | "
            f"Val RMSE(std): {v_loss:.4f} | "
            f"Val RMSE(phys): {v_phys:.3f} µg/m³ | "
            f"Sel: {selection_metric:.4f}{tag}"
        )

        if wandb_run and _WANDB_AVAILABLE:
            wandb_run.log({
                'epoch': epoch + 1,
                'train/rmse_std': t_loss,
                'val/rmse_std': v_loss,
                'val/rmse_phys': v_phys,
                'val/persistence_phys': v_pers,
                'best/selection_metric': best_metric,
            })

        # Always save last checkpoint; also save best checkpoint when improved
        _save_training_checkpoint(cfg, epoch + 1, model, optimizer, scheduler, scaler, history, best_metric, patience_count, is_best=improved)

        if patience_count >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"\n{'─'*70}")
    best_idx = int(np.argmin(history['selection_metric']))
    print(f"Training complete. Best epoch: {best_idx + 1}")
    print(f"Best Val RMSE (std):  {history['val_loss'][best_idx]:.4f}")
    print(f"Best Val RMSE (phys): {history['val_phys_rmse'][best_idx]:.3f} µg/m³")
    print(f"{'─'*70}")
    return history

metric_loss = rmse_loss
