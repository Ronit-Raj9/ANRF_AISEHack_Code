#!/usr/bin/env python
# coding: utf-8

# # PINO Baseline for PM2.5 Nowcasting (Murtuza Folder Only)
# 
# This notebook refactors the FNO2D baseline to a Physics-Informed Neural Operator (PINO) for PM2.5 forecasting, using engineered physical features, multi-objective loss, and baseline-compatible prediction export. All code and paths are strictly under the Murtuza folder for Kaggle compatibility.

# In[ ]:


# Section 1: Kaggle Runtime Bootstrap (Ronit Paths)
import os, sys

# Kaggle input mounting
KAGGLE_SRC_DATASET = "ronit-pm25-src"
KAGGLE_DATA_ROOT = "/kaggle/input/datasets/khushisingh942004/aisehack"
RONIT_CANDIDATES = [
    f"/kaggle/input/{KAGGLE_SRC_DATASET}",
    f"/kaggle/input/datasets/ronitraj1/{KAGGLE_SRC_DATASET}",
]
DATA_DIR = KAGGLE_DATA_ROOT
CKPT_DIR = "/kaggle/temp"
OUT_DIR = "/kaggle/working"

if os.path.exists('/kaggle'):
    os.environ['AISEHACK_DATA'] = DATA_DIR
    SRC_ROOT = next((p for p in RONIT_CANDIDATES if os.path.exists(os.path.join(p, 'src'))), RONIT_CANDIDATES[0])
else:
    SRC_ROOT = os.path.abspath('../Ronit')
    DATA_DIR = os.path.abspath('../aisehack-theme-2')
    CKPT_DIR = os.path.abspath('./temp')
    OUT_DIR = os.path.abspath('./working')

if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

print(f"SRC_ROOT: {SRC_ROOT}")
print(f"AISEHACK_DATA: {os.environ.get('AISEHACK_DATA', 'not set')}")
print(f"DATA_DIR: {DATA_DIR}")
print(f"CKPT_DIR: {CKPT_DIR}")
print(f"OUT_DIR: {OUT_DIR}")
assert os.path.exists(os.path.join(SRC_ROOT, 'src')), "Could not locate src/ under SRC_ROOT."


# In[ ]:


# Section 2: Repository Import, Seed, and Config Load
import random
import numpy as np
import torch
from src.config import load_config
from src.utils import seed_everything, print_device_info

cfg = load_config()
seed_everything(cfg['training']['seed'])
print_device_info(cfg['device'])

# Ensure data path is correct for loader
cfg['paths']['data'] = DATA_DIR

print(f"Batch size: {cfg['training']['batch_size_train']}")
print(f"Learning rate: {cfg['training']['lr']}")
print(f"Scheduler: {cfg['training'].get('scheduler', 'coswr')} (T0={cfg['training'].get('t0_epochs', 10)} epochs, T_mult={cfg['training'].get('t_mult', 2)})")
print(f"Epochs: {cfg['training']['epochs']}  |  Patience: {cfg['training']['patience']}")
print(f"Forecast horizon: {cfg['time']['t_out']}")
print(f"lambda_d: {cfg['training']['lambda_d']}")
print(f"lambda_p: {cfg['training']['lambda_p']}")


# In[ ]:


# Section 3: Load ALL months, stack, add physical features
from src.data import load_all_months, load_minmax_bounds
import numpy as np
import gc
import os

def add_physical_features(tensor, features_dict=None, lat_long_path='lat_long.npy'):
    import numpy as np
    import gc
    tensor = np.asarray(tensor, dtype=np.float32)
    N, C, T, H, W = tensor.shape
    N_ENG = 12
    out = np.empty((N, C + N_ENG, T, H, W), dtype=np.float32)
    out[:, :C] = tensor
    ch = C
    u10 = tensor[:, features_dict['u10'], :, :, :]
    v10 = tensor[:, features_dict['v10'], :, :, :]
    out[:, ch] = np.sqrt(u10 ** 2 + v10 ** 2)
    ch += 1
    out[:, ch] = np.arctan2(v10, u10)
    ch += 1
    del u10, v10
    gc.collect()
    t2 = tensor[:, features_dict['t2'], :, :, :].copy().astype(np.float32)
    q2 = tensor[:, features_dict['q2'], :, :, :].copy().astype(np.float32)
    denom_safe = np.where(np.abs(t2 - 29.65) < 1e-3,
                          np.sign(t2 - 29.65 + np.float32(1e-3)) * np.float32(1e-3),
                          t2 - np.float32(29.65)).astype(np.float32)
    exponent = np.clip(np.float32(17.67) * (t2 - np.float32(273.15)) / denom_safe,
                       np.float32(-100.0), np.float32(100.0))
    rh = q2 / (np.float32(0.622) + np.float32(0.378) * q2 + np.float32(1e-8)) * np.exp(exponent)
    out[:, ch] = np.clip(rh, np.float32(0.0), np.float32(1.5))
    ch += 1
    del t2, q2, denom_safe, exponent, rh
    gc.collect()
    hour = (np.arange(T, dtype=np.float32) % 24)
    for vals in [
        np.sin(np.float32(2 * np.pi) * hour / np.float32(24)),
        np.cos(np.float32(2 * np.pi) * hour / np.float32(24)),
        np.full(T, np.sin(np.float32(2 * np.pi / 12)), dtype=np.float32),
        np.full(T, np.cos(np.float32(2 * np.pi / 12)), dtype=np.float32),
    ]:
        out[:, ch] = vals.reshape(1, T, 1, 1)
        ch += 1
    del hour
    gc.collect()
    ll = np.load(lat_long_path).astype(np.float32)
    lat = ll[:, :, 0];  lon = ll[:, :, 1]
    lat = (lat - lat.min()) / (lat.max() - lat.min() + np.float32(1e-8))
    lon = (lon - lon.min()) / (lon.max() - lon.min() + np.float32(1e-8))
    out[:, ch] = lat.reshape(1, 1, H, W)
    ch += 1
    out[:, ch] = lon.reshape(1, 1, H, W)
    ch += 1
    del ll, lat, lon
    gc.collect()
    cpm25 = tensor[:, features_dict['cpm25'], :, :, :].copy()
    out[:, ch] = np.roll(cpm25, 1, axis=1);  ch += 1
    out[:, ch] = np.roll(cpm25, 3, axis=1);  ch += 1
    out[:, ch] = np.roll(cpm25, 6, axis=1);  ch += 1
    del cpm25
    gc.collect()
    n_bad = int(np.sum(~np.isfinite(out)))
    if n_bad > 0:
        import warnings
        warnings.warn(f"add_physical_features: replacing {n_bad} NaN/Inf values with 0")
        np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return out

def stack_months(month_dicts, feature_list):
    """Stack list of month dicts into (N, C, T, H, W) float32 tensor."""
    T_min = min(d[feature_list[0]].shape[0] for d in month_dicts)
    return np.stack([
        np.stack([d[feat][:T_min] for feat in feature_list], axis=0)
        for d in month_dicts
    ], axis=0).astype(np.float32)

bounds = load_minmax_bounds(cfg)
all_months = cfg['data']['months']
print('Using months:', all_months)
train_data = load_all_months(cfg, all_months, bounds)
feature_list = cfg['features']['base']
if 'base' not in cfg['features']:
    # Fallback to recreate base features if not strictly set
    cfg['features']['base'] = ['cpm25'] + cfg['features']['met'] + cfg['features']['emis']
    feature_list = cfg['features']['base']

tensor = stack_months(train_data, feature_list)

del train_data
gc.collect()
print(f"Base tensor shape (float32): {tensor.shape}  ({tensor.nbytes / 1e9:.2f} GB)")

features_dict = {name: idx for idx, name in enumerate(feature_list)}
lat_long_path = os.path.join(cfg['paths']['data'], 'raw', 'lat_long.npy')
tensor = add_physical_features(tensor, features_dict, lat_long_path=lat_long_path)
gc.collect()

print(f"Tensor after physical features: {tensor.shape}  ({tensor.nbytes / 1e9:.2f} GB)")
print(f"Window-level validation split: {cfg['data'].get('val_split', 0.1):.2f}")


# In[ ]:


# Section 4: (no-op) Lags and lat/lon are already built inside add_physical_features.
# Keeping this cell to avoid re-numbering but NOT allocating any extra arrays.
print("Engineered channels already in tensor — skipping duplicate computation.")
import gc; gc.collect()


# In[ ]:


# Section 6: Patch Murtuza/src/model.py: FNO2D Input Expansion for PINO
from src.model import build_model

# Keep full time history in tensor for sliding-window supervision
print("Tensor shape before DataLoader windows:", tensor.shape)

# Model lift sees flattened (C * T_in) channels after reshape in forward
T_model = cfg['time'].get('t_in', 16)
input_channels = tensor.shape[1] * T_model
cfg['tensor_channels'] = input_channels
model = build_model(cfg)
print(f"Configured T_model: {T_model}")
print(f"Model input channels (C*T_model): {input_channels}")
print(f"Model type: {cfg['model']['type']}")


# In[ ]:


# Section 7: Patch Murtuza/src/model.py: Circular/Reflect Padding Update
# Confirm model uses circular padding for Conv2d and spectral blocks
print('Model padding mode:', cfg.get('model', {}).get('padding_mode', 'circular'))


# In[ ]:


# Section 8: Local definition for compute_physics_loss
import torch

def compute_physics_loss(pred, xb, yb, cfg):
    if pred.ndim != 4:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)

    dt = 1.0
    last_c = xb[:, 0, -1, :, :].unsqueeze(-1)
    dC_dt = (pred - last_c) / dt

    grad_h = torch.gradient(pred, dim=1)[0]
    grad_w = torch.gradient(pred, dim=2)[0]

    feature_names = cfg.get('features', {}).get('all', [])
    if not feature_names:
        feature_names = cfg['features']['base'] + ['wind_speed', 'wind_dir', 'RH', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'lat', 'lon', 'lag_1', 'lag_3', 'lag_6']

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

print('compute_physics_loss function ready for PINO training.')


# In[ ]:


# Section 8: Local definition for compute_physics_loss
import torch

def compute_physics_loss(pred, xb, yb, cfg):
    if pred.ndim != 4:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)

    dt = 1.0
    last_c = xb[:, 0, -1, :, :].unsqueeze(-1)
    dC_dt = (pred - last_c) / dt

    grad_h = torch.gradient(pred, dim=1)[0]
    grad_w = torch.gradient(pred, dim=2)[0]

    feature_names = cfg.get('features', {}).get('all', [])
    if not feature_names:
        feature_names = cfg['features']['base'] + ['wind_speed', 'wind_dir', 'RH', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'lat', 'lon', 'lag_1', 'lag_3', 'lag_6']

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

print('compute_physics_loss function ready for PINO training.')


# In[ ]:


# Section 10: Patch Murtuza/config.yaml: Add lambda_p, lambda_d, and PINO Hyperparameters
print(f"lambda_d: {cfg['training']['lambda_d']}")
print(f"lambda_p: {cfg['training']['lambda_p']}")
print('PINO hyperparameters loaded from config.yaml.')


# In[ ]:


# Section 10b: Local Dataloader Creation
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class SlidingWindowTensorDataset(Dataset):
    def __init__(self, tensor, window_size, t_out, target_channel=0):
        self.tensor = tensor
        self.window_size = window_size
        self.t_out = t_out
        self.target_channel = target_channel
        self.N, self.C, self.T, self.H, self.W = tensor.shape
        self.num_windows = self.T - window_size - t_out + 1
        if self.num_windows <= 0:
            raise ValueError(f"Not enough time steps")
        self.total = self.N * self.num_windows

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        n = idx // self.num_windows
        w = idx % self.num_windows
        xb = self.tensor[n, :, w:w+self.window_size, :, :]
        y_start = w + self.window_size
        y_end = y_start + self.t_out
        yb = self.tensor[n, self.target_channel, y_start:y_end, :, :]
        xb = torch.tensor(xb, dtype=torch.float32)
        yb = torch.tensor(yb, dtype=torch.float32).permute(1, 2, 0)
        return xb, yb

def make_dataloaders(cfg, tensor, bounds):
    batch_size = cfg['training']['batch_size_train']
    val_batch_size = cfg['training'].get('batch_size_val', batch_size)
    window_size = cfg['time']['t_in']
    t_out = cfg['time']['t_out']
    val_split = cfg['data'].get('val_split', 0.1)

    full_ds = SlidingWindowTensorDataset(tensor, window_size, t_out=t_out, target_channel=0)
    n_total = len(full_ds)
    n_val = max(1, int(n_total * val_split))
    n_train = n_total - n_val

    seed = cfg.get('training', {}).get('seed', 42)
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=generator)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=val_batch_size, shuffle=False)
    return train_dl, val_dl

train_dl, val_dl = make_dataloaders(cfg, tensor, bounds)
print('DataLoaders created for training and validation.')


# In[ ]:


print("tensor.shape:", tensor.shape)


# In[ ]:


# Optional diagnostics (do not mutate tensor before DataLoader creation)
print("Current tensor shape:", tensor.shape)
print("Expected training window (t_in):", cfg['time'].get('t_in', 16))
print("Expected forecast horizon (t_out):", cfg['time'].get('t_out', 16))


# In[ ]:


# Section 11: Train Loop Execution with Logging (Murtuza Baseline Logic)
import numpy as np

PERSISTENCE_RMSE_NORM = 0.0208
PERSISTENCE_RMSE_PHYS = 30.83

def _horizon_weights(cfg, target: torch.Tensor) -> torch.Tensor:
    t_out = target.shape[-1]
    lo = cfg.get('loss', {}).get('horizon_weight_min', 0.8)
    hi = cfg.get('loss', {}).get('horizon_weight_max', 1.4)
    return torch.linspace(lo, hi, t_out, device=target.device, dtype=target.dtype)

def rmse_loss(pred, target, cfg=None):
    spatial_mse = torch.mean((pred - target) ** 2, dim=(1, 2))
    if cfg is not None:
        weights = _horizon_weights(cfg, target)[None]
        spatial_mse = spatial_mse * weights
    return torch.mean(torch.sqrt(spatial_mse + 1e-8))

def mae_loss(pred, target, cfg=None):
    spatial_mae = torch.mean(torch.abs(pred - target), dim=(1, 2))
    if cfg is not None:
        weights = _horizon_weights(cfg, target)[None]
        spatial_mae = spatial_mae * weights
    return torch.mean(spatial_mae)

def objective_loss(pred, target, cfg):
    wrmse = rmse_loss(pred, target, cfg)
    wmae = mae_loss(pred, target, cfg)
    a = cfg.get('loss', {}).get('rmse_weight', 0.8)
    b = cfg.get('loss', {}).get('mae_weight', 0.2)
    return a * wrmse + b * wmae

def rmse_loss_plain(pred, target):
    spatial_mse = torch.mean((pred - target) ** 2, dim=(1, 2))
    return torch.mean(torch.sqrt(spatial_mse + 1e-8))

def get_optimizer(cfg, model, steps_per_epoch):
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['training']['lr'], weight_decay=cfg['training']['weight_decay'])
    T_0 = max(1, steps_per_epoch * cfg['training'].get('t0_epochs', 10))
    T_mult = cfg['training'].get('t_mult', 2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=cfg['training']['lr'] * 1e-2)
    return optimizer, scheduler

def train_pino(cfg, model, train_dl, val_dl, bounds=None):
    device = cfg['device']
    epochs = cfg['training']['epochs']
    patience = cfg['training'].get('patience', 8)
    grad_clip = cfg['training']['grad_clip']
    save_path = cfg['paths']['model_save']
    t_in_cpm = cfg['time']['t_in_cpm']

    cpm_range = None
    if bounds is not None and 'cpm25' in bounds:
        fmin, fmax = bounds['cpm25']
        cpm_range = fmax - fmin

    optimizer, scheduler = get_optimizer(cfg, model, len(train_dl))

    best_val_loss = float('inf')
    patience_count = 0
    history = {'train_loss': [], 'val_loss': [], 'train_objective': [], 'val_persistence': []}

    print(f"\n{'─'*60}")
    print(f"  Persistence gate  (normalized RMSE): {PERSISTENCE_RMSE_NORM:.4f}")
    print(f"{'─'*60}\n")

    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        epoch_rmse = []
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            data_loss = objective_loss(pred, yb, cfg)
            physics_loss = compute_physics_loss(pred, xb, yb, cfg)
            lambda_d = cfg['training'].get('lambda_d', 1.0)
            lambda_p = cfg['training'].get('lambda_p', 0.1)
            loss = lambda_d * data_loss + lambda_p * physics_loss
            if not torch.isfinite(loss):
                optimizer.zero_grad()
                continue
            rmse_metric = rmse_loss_plain(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            epoch_losses.append(loss.item())
            epoch_rmse.append(rmse_metric.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_losses.append(rmse_loss_plain(pred, yb).item())

        train_loss = float(np.mean(epoch_rmse))
        val_loss = float(np.mean(val_losses))
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_count += 1

        print(f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Best: {best_val_loss:.4f}")
        if patience_count >= patience:
            break

    return history

history = train_pino(cfg, model, train_dl, val_dl, bounds=bounds)

import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
ax = axes[0]
ax.plot(history['train_loss'], label='Train RMSE (norm)', alpha=0.8)
ax.plot(history['val_loss'],   label='Val RMSE (norm)',   alpha=0.9)
ax.set_xlabel('Epoch');  ax.set_ylabel('Normalized RMSE')
ax.set_title('Training Curves — Normalized Space')
ax.legend();  ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# In[ ]:


# Section 12: Notebook Cell: Checkpoint Save/Load Compatible with exp_01_baseline.ipynb
import torch

# Save model checkpoint
torch.save(model.state_dict(), os.path.join(CKPT_DIR, 'best_model.pt'))
print('Model checkpoint saved.')

# Load and sanity check
model.load_state_dict(torch.load(os.path.join(CKPT_DIR, 'best_model.pt'), map_location=cfg['device']))
print('Model checkpoint loaded and verified.')


# In[ ]:


# Section 13: Reliable checkpoint load + notebook-only inference
import os
import shutil
import numpy as np
import torch
from src.data import build_test_input

def denormalize(arr: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
    return arr.astype(np.float32) * (fmax - fmin) + fmin

os.makedirs(OUT_DIR, exist_ok=True)

ckpt_temp = os.path.join(CKPT_DIR, 'best_model.pt')
ckpt_work = os.path.join(OUT_DIR, 'best_model.pt')

if os.path.exists(ckpt_temp):
    shutil.copy2(ckpt_temp, ckpt_work)
    ckpt_to_load = ckpt_work
    print(f"Checkpoint mirrored to working: {ckpt_work}")
elif os.path.exists(ckpt_work):
    ckpt_to_load = ckpt_work
    print(f"Using existing checkpoint in working: {ckpt_work}")
else:
    torch.save(model.state_dict(), ckpt_work)
    ckpt_to_load = ckpt_work
    print(f"No checkpoint file found; saved current model weights to: {ckpt_work}")

state = torch.load(ckpt_to_load, map_location=cfg['device'])
model.load_state_dict(state)
model.eval()
print(f"Checkpoint loaded: {ckpt_to_load}")


# In[ ]:


from src.data import build_test_input, denormalize

def _add_engineered_test_features(x_batch, cfg):
    x = np.asarray(x_batch, dtype=np.float32)
    B, C, T, H, W = x.shape
    feat_idx = {name: i for i, name in enumerate(cfg['features']['base'])}

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
    hour_sin = np.broadcast_to(np.sin(np.float32(2 * np.pi) * hour / np.float32(24.0)).reshape(1, T, 1, 1), (B, T, H, W))
    hour_cos = np.broadcast_to(np.cos(np.float32(2 * np.pi) * hour / np.float32(24.0)).reshape(1, T, 1, 1), (B, T, H, W))
    month_sin = np.broadcast_to(np.full((1, T, 1, 1), np.sin(np.float32(2 * np.pi / 12.0)), dtype=np.float32), (B, T, H, W))
    month_cos = np.broadcast_to(np.full((1, T, 1, 1), np.cos(np.float32(2 * np.pi / 12.0)), dtype=np.float32), (B, T, H, W))

    lat_long = np.load(os.path.join(cfg['paths']['data'], 'raw', 'lat_long.npy')).astype(np.float32)
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


# Notebook-only inference
fmin_cpm, fmax_cpm = bounds['cpm25']
device = cfg['device']
n_test = cfg['data']['test_samples']
batch_size = cfg['training']['batch_size_test']
all_preds = []

t_model = cfg['time'].get('t_in', 16)

# Expected flattened channel count from trained model
if hasattr(model, 'lift') and hasattr(model.lift, 'in_channels'):
    expected_flat = model.lift.in_channels
else:
    expected_flat = cfg.get('tensor_channels', None)

if expected_flat is None:
    raise RuntimeError("Cannot infer expected model input channels.")

expected_c = expected_flat // t_model

with torch.no_grad():
    for i in range(0, n_test, batch_size):
        j = min(i + batch_size, n_test)
        x_batch = build_test_input(cfg, bounds, start=i, end=j)  # (B, C, T, H, W)

        # Match training-time temporal window (t_in=16)
        if x_batch.shape[2] < t_model:
            raise RuntimeError(f"Test batch time dim {x_batch.shape[2]} < t_model {t_model}")
        x_batch = x_batch[:, :, :t_model, :, :]

        # Match training-time feature channels (e.g., 28)
        if x_batch.shape[1] < expected_c:
            x_batch = _add_engineered_test_features(x_batch, cfg)
        elif x_batch.shape[1] > expected_c:
            raise RuntimeError(f"Test batch has {x_batch.shape[1]} channels, expected {expected_c}")

        if i == 0:
            print(f"Test batch shape: {x_batch.shape} (chunked mode)")

        batch = torch.from_numpy(x_batch).to(device)
        pred_norm = model(batch)
        pred_phys = denormalize(pred_norm.cpu().numpy(), fmin_cpm, fmax_cpm)
        pred_phys = np.clip(pred_phys, 0.0, None)
        all_preds.append(pred_phys)

preds = np.concatenate(all_preds, axis=0).astype(np.float32)
print(f"Output shape: {preds.shape} | range: [{preds.min():.1f}, {preds.max():.1f}] µg/m³")
assert preds.shape == (n_test, 140, 124, 16), f"Unexpected shape: {preds.shape}"
assert np.isfinite(preds).all(), "Predictions contain NaN/Inf!"

print('Autoregressive 16-hour forecasting complete.')


# In[ ]:


# Section 14: Reliable prediction export with verification (atomic save)
import os
import time
import numpy as np

os.makedirs(OUT_DIR, exist_ok=True)

if 'preds' not in globals() or preds is None:
    raise RuntimeError('`preds` is missing. Run Cell 13 first.')

preds = np.asarray(preds, dtype=np.float32)
if not np.isfinite(preds).all():
    raise RuntimeError('`preds` contains NaN/Inf; aborting save.')

final_path = os.path.join(OUT_DIR, 'preds.npy')
tmp_path = final_path + '.tmp'
backup_path = os.path.join(OUT_DIR, f"preds_backup_{int(time.time())}.npy")

# Save temp -> fsync -> atomic replace
with open(tmp_path, 'wb') as f:
    np.save(f, preds)
    f.flush()
    os.fsync(f.fileno())
os.replace(tmp_path, final_path)

# Optional backup copy for extra safety
np.save(backup_path, preds)

# Read-back verification
loaded = np.load(final_path, mmap_mode='r')
assert loaded.shape == preds.shape, f"Saved shape mismatch: {loaded.shape} vs {preds.shape}"

print(f'Predictions saved to {final_path}')
print(f'Backup copy saved to {backup_path}')
print(f'Final file size: {os.path.getsize(final_path) / (1024**2):.2f} MB')

