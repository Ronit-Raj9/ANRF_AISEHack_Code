"""
Data loading, normalization, sample construction, and PyTorch Dataset.
"""

import os
import gc
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────────────────────────
# Normalization Statistics
# ─────────────────────────────────────────────
def compute_stats(cfg, months):
    """Compute per-gridpoint mean/std across given months."""
    features = cfg['features']['all']
    data_dir = cfg['paths']['data']
    stats = {}
    for feat in features:
        arrays = []
        for m in months:
            path = os.path.join(data_dir, 'raw', m, f'{feat}.npy')
            arr = np.load(path).astype(np.float32)  # (T, 140, 124)
            arrays.append(arr)
        concat = np.concatenate(arrays, axis=0)
        stats[feat] = {
            'mean': concat.mean(axis=0),   # (140, 124)
            'std':  concat.std(axis=0) + 1e-8,
        }
        del concat
        gc.collect()
    return stats


def save_stats(stats, path):
    np.save(path, stats)


def load_stats(path):
    return np.load(path, allow_pickle=True).item()


# ─────────────────────────────────────────────
# Sample Construction
# ─────────────────────────────────────────────
def build_samples(cfg, months, stride, stats):
    """
    Build input/output samples for training or validation.

    Returns:
        X: (N, T_IN_MET, H, W, C)  — all features; cpm25 zero-padded after t=10
        y: (N, H, W, T_OUT)        — future cpm25 in physical units
    """
    features = cfg['features']['all']
    data_dir = cfg['paths']['data']
    t_in_cpm = cfg['time']['t_in_cpm']
    t_in_met = cfg['time']['t_in_met']
    t_out    = cfg['time']['t_out']

    X_list, y_list = [], []
    for m in months:
        # Load & normalize all features (once per month)
        raw = {}
        for feat in features:
            arr = np.load(os.path.join(data_dir, 'raw', m, f'{feat}.npy')).astype(np.float32)
            raw[feat] = (arr - stats[feat]['mean']) / stats[feat]['std']

        # Load raw (physical) cpm25 once per month for targets
        cpm_raw = np.load(
            os.path.join(data_dir, 'raw', m, 'cpm25.npy')
        ).astype(np.float32)

        T = raw['cpm25'].shape[0]

        for i in range(0, T - t_in_met - t_out + 1, stride):
            # Input: t_in_met hours of all features
            inp = np.stack(
                [raw[feat][i : i + t_in_met] for feat in features], axis=-1
            )  # (26, 140, 124, C)

            # Mask cpm25 to only first t_in_cpm hours (simulate test conditions)
            inp[t_in_cpm:, :, :, 0] = 0.0

            # Output: next t_out hours of cpm25 in physical units
            out_raw = cpm_raw[i + t_in_met : i + t_in_met + t_out]  # (16, 140, 124)
            out = out_raw.transpose(1, 2, 0)  # (140, 124, 16)

            X_list.append(inp)
            y_list.append(out)

        del raw, cpm_raw
        gc.collect()

    return np.stack(X_list), np.stack(y_list)


# ─────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────
class PM25Dataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)  # (N, T, H, W, C)
        self.y = torch.from_numpy(y)  # (N, H, W, T_OUT)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].permute(3, 0, 1, 2)  # (C, T, H, W)
        return x, self.y[idx]


def get_dataloaders(cfg, X_train, y_train, X_val, y_val):
    """Create train and validation DataLoaders."""
    train_ds = PM25Dataset(X_train, y_train)
    val_ds   = PM25Dataset(X_val, y_val)

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg['training']['batch_size_train'],
        shuffle=True,
        num_workers=cfg['training']['num_workers'],
        pin_memory=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg['training']['batch_size_val'],
        shuffle=False,
        num_workers=cfg['training']['num_workers'],
        pin_memory=True,
    )
    return train_dl, val_dl
