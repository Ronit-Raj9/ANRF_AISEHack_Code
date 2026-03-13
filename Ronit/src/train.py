"""Three-phase training loop for the physics-informed FrNO pipeline."""

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

PERSISTENCE_RMSE_PHYS = 30.83


def _checkpoint_enabled(cfg: dict) -> bool:
    return bool(cfg.get('training', {}).get('enable_checkpointing', True))


def _checkpoint_every(cfg: dict) -> int:
    return int(cfg.get('training', {}).get('checkpoint_every_epochs', 5))


def _checkpoint_dir(cfg: dict) -> str:
    cdir = cfg.get('paths', {}).get('checkpoint_dir')
    if cdir:
        return cdir
    return os.path.join(cfg['paths']['temp'], 'checkpoints')


def _checkpoint_last_path(cfg: dict) -> str:
    last = cfg.get('paths', {}).get('checkpoint_last')
    if last:
        return last
    return os.path.join(_checkpoint_dir(cfg), 'last_checkpoint.pt')


def _checkpoint_epoch_path(cfg: dict, epoch_1based: int) -> str:
    return os.path.join(_checkpoint_dir(cfg), f'checkpoint_epoch_{epoch_1based:04d}.pt')


def _save_training_checkpoint(
    cfg: dict,
    epoch_1based: int,
    model,
    optimizer,
    scheduler,
    scaler,
    history: dict,
    best_metric: float,
    patience_count: int,
) -> None:
    if not _checkpoint_enabled(cfg):
        return
    os.makedirs(_checkpoint_dir(cfg), exist_ok=True)
    payload = {
        'epoch': epoch_1based,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
        'history': history,
        'best_metric': float(best_metric),
        'patience_count': int(patience_count),
    }
    epoch_path = _checkpoint_epoch_path(cfg, epoch_1based)
    last_path = _checkpoint_last_path(cfg)
    torch.save(payload, epoch_path)
    torch.save(payload, last_path)
    print(f"Checkpoint saved: {epoch_path}")


def _resolve_resume_checkpoint(cfg: dict) -> str | None:
    explicit = cfg.get('training', {}).get('resume_checkpoint_path')
    if isinstance(explicit, str) and explicit and os.path.exists(explicit):
        return explicit
    last_path = _checkpoint_last_path(cfg)
    if os.path.exists(last_path):
        return last_path
    cdir = _checkpoint_dir(cfg)
    if not os.path.isdir(cdir):
        return None
    candidates = [
        os.path.join(cdir, name)
        for name in os.listdir(cdir)
        if name.startswith('checkpoint_epoch_') and name.endswith('.pt')
    ]
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1]


def _try_resume_training_state(cfg: dict, model, optimizer, scheduler, scaler):
    if not bool(cfg.get('training', {}).get('resume_if_available', True)):
        return 0, float('inf'), 0, None
    path = _resolve_resume_checkpoint(cfg)
    if not path:
        return 0, float('inf'), 0, None

    payload = torch.load(path, map_location=cfg['device'])
    model.load_state_dict(payload['model_state_dict'])
    if 'optimizer_state_dict' in payload:
        optimizer.load_state_dict(payload['optimizer_state_dict'])
    if 'scheduler_state_dict' in payload:
        scheduler.load_state_dict(payload['scheduler_state_dict'])
    if scaler is not None and payload.get('scaler_state_dict') is not None:
        scaler.load_state_dict(payload['scaler_state_dict'])

    start_epoch = int(payload.get('epoch', 0))
    best_metric = float(payload.get('best_metric', float('inf')))
    patience_count = int(payload.get('patience_count', 0))
    history = payload.get('history')
    print(f"Resumed from checkpoint: {path} (next epoch: {start_epoch + 1})")
    return start_epoch, best_metric, patience_count, history


def init_wandb(cfg: dict):
    if not _WANDB_AVAILABLE:
        print('wandb not installed — skipping W&B logging.')
        return None
    wcfg = cfg.get('wandb', {})
    if not bool(wcfg.get('enabled', False)):
        return None

    run = _wandb.init(
        entity=wcfg.get('entity', None),
        project=wcfg.get('project', 'aisehack'),
        name=wcfg.get('run_name') or None,
        tags=wcfg.get('tags', []),
        config={
            'model_type': cfg['model']['type'],
            'val_month': cfg['data']['val_month'],
            'epochs': cfg['training']['epochs'],
            'phase1_epochs': cfg['training'].get('phase1_epochs', 20),
            'phase2_epochs': cfg['training'].get('phase2_epochs', 20),
            'phase3_epochs': cfg['training'].get('phase3_epochs', 10),
            'batch_size_train': cfg['training']['batch_size_train'],
            'lr': cfg['training']['lr'],
            'weight_decay': cfg['training']['weight_decay'],
            'rank_ratio': cfg['model'].get('rank_ratio', 0.4),
            'alpha_init': cfg['model'].get('alpha_init', 0.35),
            'persistence_rmse_phys': PERSISTENCE_RMSE_PHYS,
        },
        reinit=True,
    )

    _wandb.define_metric('epoch')
    for metric in [
        'train/objective', 'train/rmse_std', 'train/physics',
        'val/objective', 'val/rmse_std', 'val/rmse_phys',
        'val/persistence_phys', 'val/persistence_ratio',
        'lr', 'grad_norm', 'best/selection_metric', 'best/val_rmse_phys',
    ]:
        _wandb.define_metric(metric, step_metric='epoch')
    return run


def finish_wandb(run, history: dict, bounds: dict | None, cfg: dict):
    if run is None or not _WANDB_AVAILABLE:
        return
    epochs_ran = len(history.get('val_phys_rmse', []))
    if epochs_ran == 0:
        run.finish()
        return

    best_idx = int(np.argmin(np.asarray(history['selection_metric'])))
    run.summary['best_epoch'] = best_idx + 1
    run.summary['best_phase'] = history['phase'][best_idx]
    run.summary['best_val_rmse_phys'] = history['val_phys_rmse'][best_idx]
    run.summary['best_val_rmse_std'] = history['val_loss'][best_idx]
    run.summary['beats_global_persistence'] = int(history['val_phys_rmse'][best_idx] < PERSISTENCE_RMSE_PHYS)
    run.summary['global_persistence_ratio'] = history['val_phys_rmse'][best_idx] / PERSISTENCE_RMSE_PHYS
    run.finish()


def rmse_per_horizon(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    spatial_mse = torch.mean((pred - target) ** 2, dim=(1, 2))
    return torch.sqrt(spatial_mse.mean(0) + 1e-8)


def rmse_loss(pred: torch.Tensor, target: torch.Tensor, cfg: dict | None = None) -> torch.Tensor:
    spatial_mse = torch.mean((pred - target) ** 2, dim=(1, 2))
    return torch.mean(torch.sqrt(spatial_mse + 1e-8))


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)


def _grid_stats_tensors(cfg: dict, device, dtype):
    stats = cfg.get('_runtime', {}).get('grid_stats')
    if stats is None or 'cpm25' not in stats:
        raise RuntimeError('Grid scaler statistics for cpm25 are missing.')
    mean, std = stats['cpm25']
    mean_t = torch.as_tensor(mean, device=device, dtype=dtype)[None, :, :, None]
    std_t = torch.as_tensor(std, device=device, dtype=dtype)[None, :, :, None]
    return mean_t, std_t


def _normalized_to_log1p_domain(x: torch.Tensor, cfg: dict, bounds: dict) -> torch.Tensor:
    if str(cfg.get('preprocessing', {}).get('normalization', '')).lower() == 'grid_log_standardize':
        mean_t, std_t = _grid_stats_tensors(cfg, x.device, x.dtype)
        return x * std_t + mean_t
    fmin, fmax = bounds['cpm25']
    if bool(cfg.get('preprocessing', {}).get('cpm25_log1p', False)):
        lo = float(np.log1p(max(fmin, 0.0)))
        hi = float(np.log1p(max(fmax, 0.0)))
        return x * (hi - lo) + lo
    return x * (fmax - fmin) + fmin


def _normalized_to_physical_domain(x: torch.Tensor, cfg: dict, bounds: dict) -> torch.Tensor:
    log_domain = _normalized_to_log1p_domain(x, cfg, bounds)
    return torch.clamp(torch.expm1(log_domain), min=0.0)


def physical_rmse_metric(pred: torch.Tensor, target: torch.Tensor, cfg: dict, bounds: dict | None) -> torch.Tensor:
    if bounds is None or 'cpm25' not in bounds:
        return rmse_loss(pred, target)
    pred_phys = _normalized_to_physical_domain(pred, cfg, bounds)
    target_phys = _normalized_to_physical_domain(target, cfg, bounds)
    mse = torch.mean((pred_phys - target_phys) ** 2, dim=(1, 2))
    return torch.mean(torch.sqrt(mse + 1e-8))


def _intensity_weights(target_phys: torch.Tensor, cfg: dict) -> torch.Tensor:
    ref = float(cfg.get('loss', {}).get('intensity_ref', 59.1))
    return 1.0 + torch.clamp(target_phys / max(ref, 1e-6), min=0.0)


def weighted_mae_phys_loss(pred: torch.Tensor, target: torch.Tensor, cfg: dict, bounds: dict | None) -> torch.Tensor:
    if bounds is None or 'cpm25' not in bounds:
        return torch.mean(torch.abs(pred - target))
    pred_phys = _normalized_to_physical_domain(pred, cfg, bounds)
    target_phys = _normalized_to_physical_domain(target, cfg, bounds)
    weights = _intensity_weights(target_phys, cfg)
    return torch.mean(torch.abs(pred_phys - target_phys) * weights)


def current_phase(epoch_idx: int, cfg: dict) -> str:
    p1 = int(cfg['training'].get('phase1_epochs', 20))
    p2 = int(cfg['training'].get('phase2_epochs', 20))
    if epoch_idx < p1:
        return 'phase1_global_stability'
    if epoch_idx < p1 + p2:
        return 'phase2_tail_sharpening'
    return 'phase3_rno_finetune'


def objective_loss(pred: torch.Tensor, target: torch.Tensor, cfg: dict, bounds: dict | None, epoch_idx: int) -> torch.Tensor:
    phase = current_phase(epoch_idx, cfg)
    if phase == 'phase1_global_stability':
        return mse_loss(pred, target)
    return weighted_mae_phys_loss(pred, target, cfg, bounds)


def get_optimizer(cfg, model, steps_per_epoch):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['training']['lr'],
        weight_decay=cfg['training']['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg['training']['lr'],
        steps_per_epoch=steps_per_epoch,
        epochs=cfg['training']['epochs'],
        pct_start=cfg['training']['pct_start'],
    )
    return optimizer, scheduler


def _physics_feature_index(cfg: dict, xb: torch.Tensor, name: str) -> int | None:
    feature_names = cfg.get('features', {}).get('all', [])
    if name not in feature_names:
        return None
    idx = feature_names.index(name)
    if idx >= xb.shape[1]:
        return None
    return idx


def compute_physics_loss(pred: torch.Tensor, xb: torch.Tensor, cfg: dict) -> torch.Tensor:
    physics_cfg = cfg.get('physics', {})
    if not bool(physics_cfg.get('enabled', False)):
        return torch.zeros((), device=pred.device, dtype=pred.dtype)

    kappa = float(physics_cfg.get('diffusivity', 0.05))
    source_weight = float(physics_cfg.get('source_weight', 0.0))
    eps = float(physics_cfg.get('epsilon', 1e-8))

    grad_h = torch.gradient(pred, dim=1)[0]
    grad_w = torch.gradient(pred, dim=2)[0]
    laplacian = torch.gradient(grad_h, dim=1)[0] + torch.gradient(grad_w, dim=2)[0]

    cpm_idx = _physics_feature_index(cfg, xb, 'cpm25')
    last_c = xb[:, cpm_idx, cfg['time']['t_in_cpm'] - 1, :, :].unsqueeze(-1) if cpm_idx is not None else torch.zeros_like(pred[..., :1])
    dC_dt = torch.empty_like(pred)
    dC_dt[..., :1] = pred[..., :1] - last_c
    dC_dt[..., 1:] = pred[..., 1:] - pred[..., :-1]

    u_idx = _physics_feature_index(cfg, xb, 'u10')
    v_idx = _physics_feature_index(cfg, xb, 'v10')
    if u_idx is None or v_idx is None:
        advection = torch.zeros_like(pred)
    else:
        u = xb[:, u_idx, -1, :, :].unsqueeze(-1)
        v = xb[:, v_idx, -1, :, :].unsqueeze(-1)
        advection = u * grad_h + v * grad_w

    source = torch.zeros_like(pred)
    if source_weight > 0:
        pm_idx = _physics_feature_index(cfg, xb, 'PM25')
        if pm_idx is not None:
            source = source_weight * xb[:, pm_idx, -1, :, :].unsqueeze(-1)

    residual = dC_dt + advection - (kappa * laplacian) - source
    return torch.mean(residual * residual + eps)


def check_persistence_gate(val_rmse_phys: float, epoch: int) -> None:
    gap = val_rmse_phys - PERSISTENCE_RMSE_PHYS
    if gap > 0:
        print(f'  ⚠  Persistence gate NOT met at epoch {epoch}: val_rmse_phys={val_rmse_phys:.3f} > {PERSISTENCE_RMSE_PHYS:.2f} (gap={gap:+.3f})')
    else:
        print(f'  ✓  Persistence gate MET at epoch {epoch}: val_rmse_phys={val_rmse_phys:.3f} < {PERSISTENCE_RMSE_PHYS:.2f} (gap={gap:+.3f})')


def persistence_rmse_from_batch(xb: torch.Tensor, yb: torch.Tensor, cfg: dict, bounds: dict | None) -> torch.Tensor:
    last_obs = xb[:, 0, cfg['time']['t_in_cpm'] - 1]
    persist = last_obs[..., None].repeat(1, 1, 1, yb.shape[-1])
    return physical_rmse_metric(persist, yb, cfg, bounds)


def _forward_model(model, xb: torch.Tensor, autoregressive: bool, detach_feedback: bool = False) -> torch.Tensor:
    if autoregressive and hasattr(model, 'rollout'):
        return model.rollout(xb, detach_feedback=detach_feedback)
    if not autoregressive and hasattr(model, 'forward_parallel'):
        return model.forward_parallel(xb)
    return model(xb)


def train(cfg, model, train_dl, val_dl, bounds: dict = None, wandb_run=None):
    device = cfg['device']
    epochs = cfg['training']['epochs']
    patience = cfg['training'].get('patience', 8)
    grad_clip = cfg['training']['grad_clip']
    save_path = cfg['paths']['model_save']
    use_amp = bool(cfg['training'].get('use_amp', True) and device.type == 'cuda')
    lambda_d = float(cfg['training'].get('lambda_d', 1.0))
    lambda_p = float(cfg['training'].get('lambda_p', 0.0))
    detach_feedback = bool(cfg['training'].get('phase3_detach_feedback', False))

    optimizer, scheduler = get_optimizer(cfg, model, len(train_dl))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    history = {
        'phase': [],
        'train_loss': [],
        'val_loss': [],
        'train_objective': [],
        'train_physics': [],
        'val_objective': [],
        'val_physics': [],
        'val_persistence_phys': [],
        'val_phys_rmse': [],
        'selection_metric': [],
    }
    best_metric = float('inf')
    patience_count = 0
    start_epoch = 0

    resumed_history = None
    if _checkpoint_enabled(cfg):
        start_epoch, best_metric, patience_count, resumed_history = _try_resume_training_state(
            cfg, model, optimizer, scheduler, scaler
        )
        if isinstance(resumed_history, dict):
            history = resumed_history

    if wandb_run is not None and _WANDB_AVAILABLE:
        _wandb.watch(model, log='gradients', log_freq=max(1, len(train_dl) * 2))

    print(f"\n{'─' * 72}")
    print(f"3-phase schedule: {cfg['training'].get('phase1_epochs', 20)} + {cfg['training'].get('phase2_epochs', 20)} + {cfg['training'].get('phase3_epochs', 10)} epochs")
    print(f"Global persistence reference: {PERSISTENCE_RMSE_PHYS:.2f} µg/m³")
    if lambda_p > 0 and bool(cfg.get('physics', {}).get('enabled', False)):
        print(f"Physics loss enabled: λ_d={lambda_d:.3f}, λ_p={lambda_p:.3f}")
    print(f"{'─' * 72}\n")

    for epoch in range(start_epoch, epochs):
        phase = current_phase(epoch, cfg)
        use_autoregressive = phase == 'phase3_rno_finetune'

        model.train()
        train_obj_vals = []
        train_rmse_vals = []
        train_phys_loss_vals = []
        total_grad_norm = 0.0
        num_batches = 0

        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            amp_ctx = torch.autocast(device_type='cuda', dtype=torch.float16) if use_amp else nullcontext()

            with amp_ctx:
                pred = _forward_model(model, xb, autoregressive=use_autoregressive, detach_feedback=detach_feedback)
                data_loss = objective_loss(pred, yb, cfg, bounds, epoch_idx=epoch)
                physics_loss = compute_physics_loss(pred, xb, cfg)
                loss = (lambda_d * data_loss) + (lambda_p * physics_loss)

            rmse_std = rmse_loss(pred.detach(), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            total_grad_norm += float(grad_norm)
            num_batches += 1
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_obj_vals.append(float(loss.item()))
            train_rmse_vals.append(float(rmse_std.item()))
            train_phys_loss_vals.append(float(physics_loss.detach().item()))

        model.eval()
        val_rmse_std_vals = []
        val_obj_vals = []
        val_phys_vals = []
        val_phys_rmse_vals = []
        val_persist_vals = []
        horizon_rmse_accum = []

        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred = _forward_model(model, xb, autoregressive=use_autoregressive, detach_feedback=False)
                val_rmse_std_vals.append(float(rmse_loss(pred, yb).item()))
                val_obj_vals.append(float(objective_loss(pred, yb, cfg, bounds, epoch_idx=epoch).item()))
                val_phys_vals.append(float(compute_physics_loss(pred, xb, cfg).item()))
                val_phys_rmse_vals.append(float(physical_rmse_metric(pred, yb, cfg, bounds).item()))
                val_persist_vals.append(float(persistence_rmse_from_batch(xb, yb, cfg, bounds).item()))
                horizon_rmse_accum.append(rmse_per_horizon(pred, yb))

        train_loss = float(np.mean(train_rmse_vals))
        train_objective = float(np.mean(train_obj_vals))
        train_physics = float(np.mean(train_phys_loss_vals)) if train_phys_loss_vals else 0.0
        val_loss = float(np.mean(val_rmse_std_vals))
        val_objective = float(np.mean(val_obj_vals))
        val_physics = float(np.mean(val_phys_vals)) if val_phys_vals else 0.0
        val_phys_rmse = float(np.mean(val_phys_rmse_vals))
        val_persistence_phys = float(np.mean(val_persist_vals))
        persistence_ratio = val_phys_rmse / max(val_persistence_phys, 1e-8)
        current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr']
        avg_grad_norm = total_grad_norm / max(num_batches, 1)

        metric_mode = str(cfg['training'].get('checkpoint_metric', 'val_rmse_phys')).lower()
        if metric_mode == 'val_objective':
            selection_metric = val_objective
        elif metric_mode == 'mixed':
            alpha = float(cfg['training'].get('checkpoint_mixed_alpha', 0.5))
            selection_metric = alpha * val_objective + (1.0 - alpha) * val_phys_rmse
        else:
            selection_metric = val_phys_rmse

        improved = selection_metric < best_metric
        if improved:
            best_metric = selection_metric
            patience_count = 0
            torch.save(model.state_dict(), save_path)
            tag = '  ← saved'
        else:
            patience_count += 1
            tag = f'  (no improvement {patience_count}/{patience})'

        history['phase'].append(phase)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_objective'].append(train_objective)
        history['train_physics'].append(train_physics)
        history['val_objective'].append(val_objective)
        history['val_physics'].append(val_physics)
        history['val_persistence_phys'].append(val_persistence_phys)
        history['val_phys_rmse'].append(val_phys_rmse)
        history['selection_metric'].append(selection_metric)

        print(
            f"Epoch {epoch + 1:3d}/{epochs} | {phase} | "
            f"TrainObj: {train_objective:.4f} | TrainRMSE(std): {train_loss:.4f} | "
            f"ValRMSE(std): {val_loss:.4f} | ValRMSE(phys): {val_phys_rmse:.3f} | "
            f"ValPersist: {val_persistence_phys:.3f} | Ratio: {persistence_ratio:.3f} | "
            f"Sel: {selection_metric:.4f} | Best: {best_metric:.4f}{tag}"
        )

        if wandb_run is not None and _WANDB_AVAILABLE:
            log_dict = {
                'epoch': epoch + 1,
                'phase/id': 1 if phase.startswith('phase1') else 2 if phase.startswith('phase2') else 3,
                'train/objective': train_objective,
                'train/rmse_std': train_loss,
                'train/physics': train_physics,
                'val/objective': val_objective,
                'val/rmse_std': val_loss,
                'val/rmse_phys': val_phys_rmse,
                'val/physics': val_physics,
                'val/persistence_phys': val_persistence_phys,
                'val/persistence_ratio': persistence_ratio,
                'lr': current_lr,
                'grad_norm': avg_grad_norm,
                'best/selection_metric': best_metric,
                'best/val_rmse_phys': min(history['val_phys_rmse']),
            }
            if horizon_rmse_accum:
                avg_horizon = torch.stack(horizon_rmse_accum, dim=0).mean(0)
                for t_idx, v in enumerate(avg_horizon.cpu().tolist()):
                    log_dict[f'horizon/val_rmse_h{t_idx + 1:02d}'] = v
            if improved and bool(cfg.get('wandb', {}).get('log_model', True)):
                artifact = _wandb.Artifact(
                    name=f'best_model-{wandb_run.id}',
                    type='model',
                    description=f'Best checkpoint — epoch {epoch + 1}, val_rmse_phys={val_phys_rmse:.4f}',
                    metadata={
                        'epoch': epoch + 1,
                        'phase': phase,
                        'val_rmse_phys': val_phys_rmse,
                        'selection_metric': selection_metric,
                    },
                )
                artifact.add_file(save_path)
                wandb_run.log_artifact(artifact)
            wandb_run.log(log_dict)

        if (epoch + 1) % 5 == 0:
            check_persistence_gate(val_phys_rmse, epoch + 1)

        ckpt_every = _checkpoint_every(cfg)
        if ckpt_every > 0 and ((epoch + 1) % ckpt_every == 0):
            _save_training_checkpoint(
                cfg=cfg,
                epoch_1based=epoch + 1,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                history=history,
                best_metric=best_metric,
                patience_count=patience_count,
            )

        if patience_count >= patience:
            print(f'\nEarly stopping triggered after {epoch + 1} epochs (no improvement for {patience} epochs).')
            break

    print(f"\n{'─' * 72}")
    best_idx = int(np.argmin(np.asarray(history['selection_metric'])))
    print(f"Training complete. Best epoch: {best_idx + 1} ({history['phase'][best_idx]})")
    print(f"Best val RMSE (std-space): {history['val_loss'][best_idx]:.4f}")
    print(f"Best val RMSE (physical):  {history['val_phys_rmse'][best_idx]:.3f} µg/m³")
    check_persistence_gate(history['val_phys_rmse'][best_idx], best_idx + 1)

    if _checkpoint_enabled(cfg):
        _save_training_checkpoint(
            cfg=cfg,
            epoch_1based=len(history['val_loss']),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            history=history,
            best_metric=best_metric,
            patience_count=patience_count,
        )

    print(f"{'─' * 72}")
    return history


metric_loss = rmse_loss

