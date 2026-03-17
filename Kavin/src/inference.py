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

from .data import build_test_input, denormalize, get_hotspot_maps


def _add_engineered_test_features(x_batch: np.ndarray, cfg: dict) -> np.ndarray:
    """
    Append engineered PINO channels to a test batch.

    Parameters
    ----------
    x_batch : np.ndarray
        Shape (B, C, T, H, W), where C corresponds to cfg['features']['base'].

    Returns
    -------
    np.ndarray
        Shape (B, C+11, T, H, W), matching training-time engineered channels.
    """
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

    hotspot_prior, _ = get_hotspot_maps(cfg)
    hotspot = np.broadcast_to(hotspot_prior.reshape(1, 1, H, W), (B, T, H, W)).astype(np.float32)

    lag_1 = np.roll(cpm25, 1, axis=1)
    lag_3 = np.roll(cpm25, 3, axis=1)
    lag_6 = np.roll(cpm25, 6, axis=1)

    engineered = np.stack([
        wind_speed, wind_dir, rh,
        hour_sin, hour_cos, month_sin, month_cos,
        hotspot, lag_1, lag_3, lag_6,
    ], axis=1).astype(np.float32)

    out = np.concatenate([x, engineered], axis=1).astype(np.float32)
    np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return out


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

    # ── Batched inference ──
    fmin_cpm, fmax_cpm = bounds['cpm25']
    all_preds = []

    model.eval()
    with torch.no_grad():
        for i in range(0, n_test, batch_size):
            j = min(i + batch_size, n_test)
            x_batch = build_test_input(cfg, bounds, start=i, end=j)

            # If model expects engineered channels (e.g., 28 x 16 = 448 lift input),
            # augment test inputs to match training-time channel construction.
            t_in = x_batch.shape[2]
            expected_flat = cfg.get('tensor_channels', x_batch.shape[1] * t_in)
            expected_c = expected_flat // t_in
            if expected_c > x_batch.shape[1]:
                x_batch = _add_engineered_test_features(x_batch, cfg)

            if i == 0:
                print(f"Test batch shape: {x_batch.shape} (chunked mode)")
            batch = torch.from_numpy(x_batch).to(device)
            # batch: (B, C, T, H, W)
            pred_norm = model(batch)          # (B, H, W, T_out) — normalized
            pred_phys = denormalize(pred_norm.cpu().numpy(), fmin_cpm, fmax_cpm)
            pred_phys = np.clip(pred_phys, 0.0, None)   # PM2.5 cannot be negative
            all_preds.append(pred_phys)

    preds = np.concatenate(all_preds, axis=0).astype(np.float32)
    print(f"Output shape: {preds.shape} | "
          f"range: [{preds.min():.1f}, {preds.max():.1f}] µg/m³")

    assert preds.shape == (n_test, 140, 124, 16), f"Unexpected shape: {preds.shape}"
    assert np.isfinite(preds).all(), "Predictions contain NaN/Inf!"

    np.save(output_path, preds)
    print(f"preds.npy saved → {output_path}")

    return preds

