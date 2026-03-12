"""
Test-time inference and prediction saving.
"""

import os
import numpy as np
import torch


def run_inference(cfg, model, stats):
    """
    Load test inputs, run model inference, denormalize, and save preds.npy.

    Returns:
        preds: np.ndarray of shape (996, 140, 124, 16)
    """
    device     = cfg['device']
    features   = cfg['features']['all']
    data_dir   = cfg['paths']['data']
    t_in_cpm   = cfg['time']['t_in_cpm']
    t_in_met   = cfg['time']['t_in_met']
    n_test     = cfg['data']['test_samples']
    batch_size = cfg['training']['batch_size_test']
    output_path = cfg['paths']['output']

    # ── Load & normalize test inputs ──
    test_inputs = []
    for feat in features:
        path = os.path.join(data_dir, 'test_in', f'{feat}.npy')
        arr = np.load(path).astype(np.float32)  # (996, T_feat, 140, 124)
        mean = stats[feat]['mean'][None, None]
        std  = stats[feat]['std'][None, None]
        arr_norm = (arr - mean) / std

        # Pad cpm25 to t_in_met timesteps (zero-pad future)
        if feat == 'cpm25':
            pad = np.zeros(
                (n_test, t_in_met - t_in_cpm, 140, 124), dtype=np.float32
            )
            arr_norm = np.concatenate([arr_norm, pad], axis=1)

        test_inputs.append(arr_norm)

    X_test = np.stack(test_inputs, axis=-1)  # (996, 26, 140, 124, 10)
    print(f"Test input shape: {X_test.shape}")

    # ── Batched inference ──
    all_preds = []
    test_tensor = torch.from_numpy(X_test)
    cpm_mean = torch.from_numpy(stats['cpm25']['mean']).to(device)
    cpm_std  = torch.from_numpy(stats['cpm25']['std']).to(device)

    model.eval()
    with torch.no_grad():
        for i in range(0, len(test_tensor), batch_size):
            batch = test_tensor[i : i + batch_size]
            batch = batch.permute(0, 4, 1, 2, 3).to(device)  # (B, C, T, H, W)
            pred = model(batch)  # (B, H, W, T_OUT)

            # Denormalize
            pred = pred * cpm_std[None, :, :, None] + cpm_mean[None, :, :, None]
            pred = torch.clamp(pred, min=0.0)
            all_preds.append(pred.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0).astype(np.float32)
    print(f"Output shape: {preds.shape}")

    assert preds.shape == (n_test, 140, 124, 16), f"Unexpected shape: {preds.shape}"
    assert np.isfinite(preds).all(), "Predictions contain NaN/Inf!"

    np.save(output_path, preds)
    print(f"preds.npy saved to {output_path}")

    return preds
