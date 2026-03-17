"""
Test-time inference and prediction saving.

Expected test_in layout (per hackathon spec):
    test_in/cpm25.npy  → (996, 10, 140, 124)   — 10 known cpm25 hours
    test_in/{feat}.npy → (996, 26, 140, 124)    — 26-hour met/emis window
                                                  (10 past + 16 NWP forecast)

Normalization: official min-max from feat_min_max.mat (same as training).
Output:        preds.npy of shape (996, 140, 124, 16) in physical µg/m³ units.
"""

import os
import numpy as np
import torch

from .data import build_test_input, denormalize_cpm25


def _add_engineered_test_features(x_batch: np.ndarray, cfg: dict) -> np.ndarray:
    """Baseline-compatible engineered PINO test channels helper."""
    x = np.asarray(x_batch, dtype=np.float32)
    B, C, T, H, W = x.shape

    base_feats = cfg.get('features', {}).get('base', [])
    feat_idx = {name: i for i, name in enumerate(base_feats)}

    required = ['u10', 'v10', 't2', 'q2', 'cpm25']
    missing = [name for name in required if name not in feat_idx]
    if missing:
        raise KeyError(f"Missing base features for engineered inference channels: {missing}")

    u10 = x[:, feat_idx['u10']]
    v10 = x[:, feat_idx['v10']]
    t2 = x[:, feat_idx['t2']]
    q2 = x[:, feat_idx['q2']]
    cpm25 = x[:, feat_idx['cpm25']]

    wind_speed = np.sqrt(u10 ** 2 + v10 ** 2)
    wind_dir = np.arctan2(v10, u10)

    denom_safe = np.where(
        np.abs(t2 - np.float32(29.65)) < np.float32(1e-3),
        np.sign(t2 - np.float32(29.65) + np.float32(1e-3)) * np.float32(1e-3),
        t2 - np.float32(29.65),
    )
    exponent = np.clip(
        np.float32(17.67) * (t2 - np.float32(273.15)) / denom_safe,
        np.float32(-87.0),
        np.float32(87.0),
    )
    with np.errstate(over='ignore', invalid='ignore'):
        rh = q2 / (np.float32(0.622) + np.float32(0.378) * q2 + np.float32(1e-8)) * np.exp(exponent)
    rh = np.clip(rh, np.float32(0.0), np.float32(1.5))

    hour = (np.arange(T, dtype=np.float32) % 24)
    hour_sin = np.sin(np.float32(2 * np.pi) * hour / np.float32(24.0)).reshape(1, T, 1, 1)
    hour_cos = np.cos(np.float32(2 * np.pi) * hour / np.float32(24.0)).reshape(1, T, 1, 1)
    month_sin = np.full((1, T, 1, 1), np.sin(np.float32(2 * np.pi / 12.0)), dtype=np.float32)
    month_cos = np.full((1, T, 1, 1), np.cos(np.float32(2 * np.pi / 12.0)), dtype=np.float32)
    hour_sin = np.broadcast_to(hour_sin, (B, T, H, W))
    hour_cos = np.broadcast_to(hour_cos, (B, T, H, W))
    month_sin = np.broadcast_to(month_sin, (B, T, H, W))
    month_cos = np.broadcast_to(month_cos, (B, T, H, W))

    lat_long_path = os.path.join(cfg['paths']['data'], 'raw', 'lat_long.npy')
    lat_long = np.load(lat_long_path).astype(np.float32)
    lat = lat_long[:, :, 0]
    lon = lat_long[:, :, 1]
    lat = (lat - lat.min()) / (lat.max() - lat.min() + np.float32(1e-8))
    lon = (lon - lon.min()) / (lon.max() - lon.min() + np.float32(1e-8))
    lat = np.broadcast_to(lat.reshape(1, 1, H, W), (B, T, H, W))
    lon = np.broadcast_to(lon.reshape(1, 1, H, W), (B, T, H, W))

    lag_1 = np.roll(cpm25, 1, axis=1)
    lag_3 = np.roll(cpm25, 3, axis=1)
    lag_6 = np.roll(cpm25, 6, axis=1)

    engineered = np.stack([
        wind_speed, wind_dir, rh,
        hour_sin, hour_cos, month_sin, month_cos,
        lat, lon, lag_1, lag_3, lag_6,
    ], axis=1).astype(np.float32)

    out = np.concatenate([x, engineered], axis=1).astype(np.float32)
    np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def _tta_modes(cfg: dict) -> list[str]:
    inf = cfg.get('inference', {})
    if not bool(inf.get('use_tta', False)):
        return ['identity']
    modes = inf.get('tta_modes', ['identity', 'hflip', 'vflip'])
    valid = {'identity', 'hflip', 'vflip', 'hvflip'}
    modes = [m for m in modes if m in valid]
    return modes or ['identity']


def _apply_tta_input(x: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == 'hflip':
        return x.flip(-1)
    if mode == 'vflip':
        return x.flip(-2)
    if mode == 'hvflip':
        return x.flip(-2).flip(-1)
    return x


def _invert_tta_output(y: torch.Tensor, mode: str) -> torch.Tensor:
    # y shape: (B, H, W, T)
    if mode == 'hflip':
        return y.flip(2)
    if mode == 'vflip':
        return y.flip(1)
    if mode == 'hvflip':
        return y.flip(1).flip(2)
    return y


def _predict_with_tta(model, batch: torch.Tensor, cfg: dict) -> torch.Tensor:
    modes = _tta_modes(cfg)
    preds = []
    use_autoregressive = bool(cfg.get('inference', {}).get('use_autoregressive', True))
    for mode in modes:
        x_aug = _apply_tta_input(batch, mode)
        if use_autoregressive and hasattr(model, 'rollout'):
            y_aug = model.rollout(x_aug, detach_feedback=False)
        elif hasattr(model, 'forward_parallel'):
            y_aug = model.forward_parallel(x_aug)
        else:
            y_aug = model(x_aug)
        preds.append(_invert_tta_output(y_aug, mode))
    return torch.stack(preds, dim=0).mean(dim=0)


def run_inference(cfg, model, bounds: dict) -> np.ndarray:
    """
    Load test inputs, run model, denormalize cpm25 predictions, save preds.npy.

    Parameters
    ----------
    bounds : dict returned by load_minmax_bounds(cfg)

    Returns
    -------
    preds : np.ndarray (996, 140, 124, 16)  — physical PM2.5 µg/m³
    """
    device      = cfg['device']
    n_test      = cfg['data']['test_samples']
    batch_size  = cfg['training']['batch_size_test']
    output_path = cfg['paths']['output']
    ckpt_paths = [p for p in cfg.get('inference', {}).get('model_paths', []) if isinstance(p, str) and p]
    state_backup = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()} if ckpt_paths else None

    # ── Batched inference ──
    all_preds = []

    model.eval()
    with torch.no_grad():
        for i in range(0, n_test, batch_size):
            j = min(i + batch_size, n_test)
            x_batch = build_test_input(cfg, bounds, start=i, end=j)
            if i == 0:
                print(f"Test batch shape: {x_batch.shape} (chunked mode)")
            batch = torch.from_numpy(x_batch).to(device)
            # batch: (B, C, T, H, W)
            if ckpt_paths:
                member_preds = []
                for ckpt in ckpt_paths:
                    state = torch.load(ckpt, map_location=device)
                    model.load_state_dict(state)
                    member_preds.append(_predict_with_tta(model, batch, cfg))
                pred_norm = torch.stack(member_preds, dim=0).mean(dim=0)
            else:
                pred_norm = _predict_with_tta(model, batch, cfg)

            pred_phys = denormalize_cpm25(pred_norm.cpu().numpy(), bounds, cfg)
            all_preds.append(pred_phys)

    if state_backup is not None:
        model.load_state_dict(state_backup)

    preds = np.maximum(np.concatenate(all_preds, axis=0).astype(np.float32), 0.0)
    print(f"Output shape: {preds.shape} | "
          f"range: [{preds.min():.1f}, {preds.max():.1f}] µg/m³")

    assert preds.shape == (n_test, 140, 124, 16), f"Unexpected shape: {preds.shape}"
    assert np.isfinite(preds).all(), "Predictions contain NaN/Inf!"

    np.save(output_path, preds)
    print(f"preds.npy saved → {output_path}")

    return preds

