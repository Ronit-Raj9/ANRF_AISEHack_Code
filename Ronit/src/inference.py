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

