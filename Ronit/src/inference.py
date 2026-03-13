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
            pred_norm = model(batch)          # (B, H, W, T_out) — normalized
            pred_phys = denormalize_cpm25(pred_norm.cpu().numpy(), bounds, cfg)
            all_preds.append(pred_phys)

    preds = np.concatenate(all_preds, axis=0).astype(np.float32)
    print(f"Output shape: {preds.shape} | "
          f"range: [{preds.min():.1f}, {preds.max():.1f}] µg/m³")

    assert preds.shape == (n_test, 140, 124, 16), f"Unexpected shape: {preds.shape}"
    assert np.isfinite(preds).all(), "Predictions contain NaN/Inf!"

    np.save(output_path, preds)
    print(f"preds.npy saved → {output_path}")

    return preds

