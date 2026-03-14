---
jupyter:
  kernelspec:
    display_name: angentgate
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.12.12
  nbformat: 4
  nbformat_minor: 5
---

::: {#975c37c5 .cell .markdown}
# Experiment 01 --- Baseline TFNO2D (Kaggle First Run) {#experiment-01--baseline-tfno2d-kaggle-first-run}

This notebook is the **first official baseline run** using the
competition\'s 16 input features only. All reusable code lives in
`src/`.

**Notes:**

-   Uses official min-max normalization (`feat_min_max.mat`)
-   Uses strict temporal blocking (`OCT_16` validation)
-   Uses baseline `tfno2d` model (no extra auxiliary channels)
-   Uses persistence gate from EDA to verify learning quality

**Pipeline:**

1.  Load config\
2.  Load official normalization bounds\
3.  Build train/validation loaders\
4.  Train baseline TFNO2D and compare against persistence\
5.  Run inference and save `preds.npy`
:::

::: {#f6c64039 .cell .code}
``` python
import sys, os

# ── Path resolution: works both locally and on Kaggle ──
# On Kaggle: attach your src dataset as input, set KAGGLE_SRC_DATASET below.
# Locally:   runs from notebooks/ so '../' resolves to Ronit/

KAGGLE_SRC_DATASET = "ronit-pm25-src"
KAGGLE_DATA_ROOT = "/kaggle/input/datasets/khushisingh942004/aisehack"

if os.path.exists('/kaggle'):
    os.environ['AISEHACK_DATA'] = KAGGLE_DATA_ROOT

LOCAL_SRC  = os.path.abspath('../')
KAGGLE_SRC = f'/kaggle/input/{KAGGLE_SRC_DATASET}'

if os.path.exists('/kaggle'):
    SRC_ROOT = KAGGLE_SRC
else:
    SRC_ROOT = LOCAL_SRC

if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

print(f"SRC_ROOT: {SRC_ROOT}")
print(f"AISEHACK_DATA: {os.environ.get('AISEHACK_DATA', 'not set')}")
from src.config import load_config
from src.utils import seed_everything, print_device_info, count_parameters, sanity_check_bounds
from src.data import load_minmax_bounds, load_all_months, get_dataloaders
from src.model import build_model
from src.train import train
from src.inference import run_inference
```
:::

::: {#34e5a124 .cell .markdown}
## 1. Configuration {#1-configuration}
:::

::: {#23207574 .cell .code}
``` python
cfg = load_config()

seed_everything(cfg['training']['seed'])
print_device_info(cfg['device'])

print(f"Base features ({cfg['features']['n_features']}): {cfg['features']['base']}")
print(f"Aux features  ({len(cfg['features']['aux'])}): {cfg['features']['aux']}")
print(f"Input channels: {cfg['features']['input_channels']}")
print(f"Train months: {cfg['data']['train_months']}")
print(f"Val month:    {cfg['data']['val_month']}")
print(f"Model type:   {cfg['model']['type']}")
print(f"Data root:    {cfg['paths']['data']}")
```
:::

::: {#b8eda0c8 .cell .markdown}
## 2. Load Normalization Bounds (feat_min_max.mat) {#2-load-normalization-bounds-feat_min_maxmat}
:::

::: {#53baffb3 .cell .code}
``` python
# Load official min-max bounds from feat_min_max.mat
bounds = load_minmax_bounds(cfg)
print(f"Loaded bounds for {len(bounds)} features")
print(f"Source: {cfg['paths']['min_max']}")

# Sanity check: print ranges, flag any zero-range features
sanity_check_bounds(bounds, cfg['features']['all'])
```
:::

::: {#379daf90 .cell .markdown}
## 3. Load & Preprocess Training / Validation Data {#3-load--preprocess-training--validation-data}

Baseline preprocessing used here:

-   official `feat_min_max.mat` normalization for all 16 competition
    features
-   strict temporal blocking (`OCT_16` held out entirely)
-   lazy sliding-window dataset to avoid materializing massive tensors
:::

::: {#a7821120 .cell .code}
``` python
print("Loading + normalizing training months ...")
train_data = load_all_months(cfg, cfg['data']['train_months'], bounds)

print("\nLoading + normalizing validation month ...")
val_data = load_all_months(cfg, [cfg['data']['val_month']], bounds)
```
:::

::: {#e15d22f0 .cell .code}
``` python
train_dl, val_dl = get_dataloaders(cfg, train_data, val_data, bounds)
print("Batch shape check ...")
xb, yb = next(iter(train_dl))
print(f"  x: {tuple(xb.shape)}  (B, C={xb.shape[1]}, T={xb.shape[2]}, H={xb.shape[3]}, W={xb.shape[4]})")
print(f"  y: {tuple(yb.shape)}  (B, H, W, T_out={yb.shape[3]})")
print(f"  x range: [{xb.min():.3f}, {xb.max():.3f}]")
print(f"  y range: [{yb.min():.3f}, {yb.max():.3f}]")
print("  Baseline mode: only 16 official features (no auxiliary channels)")
```
:::

::: {#56edecea .cell .markdown}
## 4. Build & Train Baseline TFNO2D {#4-build--train-baseline-tfno2d}
:::

::: {#6f642b80 .cell .code}
``` python
model = build_model(cfg)
print(f"Parameters: {count_parameters(model):,}")
print(f"Using model: {cfg['model']['type']}")
print("Expected first target: beat persistence gate before trying larger architectures")
```
:::

::: {#b71e1ffa .cell .code}
``` python
history = train(cfg, model, train_dl, val_dl, bounds=bounds)
```
:::

::: {#dc990cc8 .cell .markdown}
### Training Curves
:::

::: {#31535ceb .cell .code}
``` python
import matplotlib.pyplot as plt
from src.train import PERSISTENCE_RMSE_NORM

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# ── Left: normalized RMSE ──
ax = axes[0]
ax.plot(history['train_loss'], label='Train RMSE (norm)', alpha=0.8)
ax.plot(history['val_loss'],   label='Val RMSE (norm)',   alpha=0.9)
if 'val_persistence' in history:
    ax.plot(history['val_persistence'], label='Val persistence RMSE', alpha=0.8)
ax.axhline(PERSISTENCE_RMSE_NORM, color='red', linestyle='--',
           label=f'Global persistence gate ({PERSISTENCE_RMSE_NORM})')
ax.set_xlabel('Epoch');  ax.set_ylabel('Normalized RMSE')
ax.set_title('Training Curves — Normalized Space')
ax.legend();  ax.grid(True, alpha=0.3)

# ── Right: physical RMSE estimate ──
ax = axes[1]
fmin, fmax = bounds['cpm25']
cpm_range  = fmax - fmin
ax.plot([v * cpm_range for v in history['train_loss']], label='Train RMSE (µg/m³)', alpha=0.8)
ax.plot([v * cpm_range for v in history['val_loss']],   label='Val RMSE (µg/m³)',   alpha=0.9)
if 'val_persistence' in history:
    ax.plot([v * cpm_range for v in history['val_persistence']], label='Val persistence (µg/m³)', alpha=0.8)
ax.axhline(30.83, color='red', linestyle='--', label='Global persistence baseline (30.83 µg/m³)')
ax.set_xlabel('Epoch');  ax.set_ylabel('RMSE (µg/m³)')
ax.set_title('Training Curves — Physical Units (estimate)')
ax.legend();  ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```
:::

::: {#f65b0be4 .cell .markdown}
## 5. Inference & Submit {#5-inference--submit}
:::

::: {#ac137e9d .cell .code}
``` python
import torch

model.load_state_dict(torch.load(cfg['paths']['model_save'], map_location=cfg['device']))
preds = run_inference(cfg, model, bounds)
print('Done!')
```
:::

::: {#db48f7c1 .cell .markdown}
## 6. Score `preds.npy` using provided test data {#6-score-predsnpy-using-provided-test-data}

The official competition score cannot be computed locally because test
future ground truth is hidden. This section computes **proxy scores**
against available `test_in/cpm25.npy` history:

-   **H+1 vs last observed RMSE/MAE**
-   **All-horizon distance from persistence baseline**
-   **Temporal smoothness (mean absolute step change)**
:::

::: {#abd98894 .cell .code execution_count="5"}
``` python
import os
import numpy as np

# ------------------------------
# Environment-aware path helpers
# ------------------------------
def first_existing(paths):
    return next((p for p in paths if p and os.path.exists(p)), None)

on_kaggle = os.path.exists('/kaggle')

# Resolve source/project root
if 'SRC_ROOT' in globals() and isinstance(SRC_ROOT, str):
    src_root = SRC_ROOT
else:
    cwd = os.getcwd()
    candidate_roots = [
        os.path.abspath(os.path.join(cwd, '..')),
        os.path.abspath(os.path.join(cwd, '../..')),
        os.path.abspath(cwd),
        os.path.abspath('/home/raj/Documents/CODING/Hackathon/ANRF_AISEHack_Code/Ronit'),
        '/kaggle/input/ronit-pm25-src',
    ]
    src_root = first_existing([r for r in candidate_roots if os.path.exists(os.path.join(r, 'outputs'))]) or candidate_roots[0]

# Resolve data root (works for local + Kaggle competition/input variants)
if 'cfg' in globals() and isinstance(cfg, dict) and 'paths' in cfg:
    cfg_output = cfg['paths'].get('output')
    cfg_model_save = cfg['paths'].get('model_save', '')
    cfg_temp = cfg['paths'].get('temp', '')
    data_root = cfg['paths'].get('data')
else:
    cfg_output = None
    cfg_model_save = ''
    cfg_temp = ''
    data_root = None

data_candidates = [
    data_root,
    os.environ.get('AISEHACK_DATA'),
    '/kaggle/input/aisehack-theme-2',
    '/kaggle/input/competitions/aisehack-theme-2',
    '/kaggle/input/datasets/khushisingh942004/aisehack',
    os.path.abspath(os.path.join(src_root, '..', 'aisehack-theme-2')),
    os.path.abspath(os.path.join(src_root, 'aisehack-theme-2')),
]
data_root = first_existing([p for p in data_candidates if p and os.path.exists(os.path.join(p, 'test_in'))])
if data_root is None:
    raise FileNotFoundError('Could not locate data root containing test_in/. Set AISEHACK_DATA or attach competition data.')

# ------------------------------
# Locate preds.npy
# ------------------------------
pred_candidates = [
    '/kaggle/working/preds.npy' if on_kaggle else None,
    cfg_output,
    os.path.join(os.path.dirname(cfg_model_save), 'preds.npy') if cfg_model_save else None,
    os.path.join(os.path.dirname(cfg_temp), 'preds.npy') if cfg_temp else None,
    os.path.abspath(os.path.join(src_root, 'outputs', 'models', 'preds.npy')),
    os.path.abspath(os.path.join(src_root, 'outputs', 'submissions', 'preds.npy')),
]
preds_path = first_existing(pred_candidates)
if preds_path is None:
    raise FileNotFoundError('Could not find preds.npy in expected locations (including /kaggle/working).')

preds = np.load(preds_path)
print(f"Using preds: {preds_path}")
print(f"preds shape (raw): {preds.shape}")

# ------------------------------
# Load available test history
# ------------------------------
test_cpm25_path = os.path.join(data_root, 'test_in', 'cpm25.npy')
test_cpm25_hist = np.load(test_cpm25_path)
print(f"Using test history: {test_cpm25_path}")
print(f"test_in/cpm25 shape: {test_cpm25_hist.shape}")

# Expected:
# test_cpm25_hist: (N, 10, H, W)
# preds:           (N, H, W, 16)

# Handle common prediction layout variant: (N, 16, H, W)
if preds.ndim == 4 and preds.shape[1] == 16 and preds.shape[-1] != 16:
    preds = np.transpose(preds, (0, 2, 3, 1))
    print(f"preds shape (transposed to NHWT): {preds.shape}")

if preds.ndim != 4 or preds.shape[-1] != 16:
    raise ValueError(f"Expected preds shape like (N, H, W, 16), got {preds.shape}")

n, h, w, t_out = preds.shape
if t_out != 16:
    raise ValueError(f"Expected 16 forecast steps, got {t_out}")
if test_cpm25_hist.shape[0] != n or test_cpm25_hist.shape[2] != h or test_cpm25_hist.shape[3] != w:
    raise ValueError(
        'Spatial/sample mismatch between preds and test history: '
        f'preds={preds.shape}, test_hist={test_cpm25_hist.shape}'
    )

last_obs = test_cpm25_hist[:, -1, :, :]  # (N, H, W)

# 1) Horizon-1 consistency with latest observation
h1_pred = preds[..., 0]
rmse_h1_vs_last = float(np.sqrt(np.mean((h1_pred - last_obs) ** 2)))
mae_h1_vs_last = float(np.mean(np.abs(h1_pred - last_obs)))

# 2) Distance from persistence baseline over all forecast horizons
persistence = np.repeat(last_obs[..., None], preds.shape[-1], axis=-1)
rmse_all_vs_persistence = float(np.sqrt(np.mean((preds - persistence) ** 2)))
mae_all_vs_persistence = float(np.mean(np.abs(preds - persistence)))

# 3) Temporal smoothness inside forecast trajectory
step_delta = np.diff(preds, axis=-1)
mean_abs_step_change = float(np.mean(np.abs(step_delta)))

print('\nProxy scores (lower is generally better for RMSE/MAE):')
print(f"- RMSE(H+1 vs last observed):        {rmse_h1_vs_last:.4f}")
print(f"- MAE(H+1 vs last observed):         {mae_h1_vs_last:.4f}")
print(f"- RMSE(all horizons vs persistence): {rmse_all_vs_persistence:.4f}")
print(f"- MAE(all horizons vs persistence):  {mae_all_vs_persistence:.4f}")
print(f"- Mean abs step change (H to H+1):   {mean_abs_step_change:.4f}")

print('\nNote: Official competition score is computed on hidden future targets on Kaggle.')
```

::: {.output .stream .stdout}
    Using preds: /home/raj/Documents/CODING/Hackathon/ANRF_AISEHack_Code/Ronit/outputs/models/preds.npy
    preds shape (raw): (996, 140, 124, 16)
    Using test history: /home/raj/Documents/CODING/Hackathon/ANRF_AISEHack_Code/aisehack-theme-2/test_in/cpm25.npy
    test_in/cpm25 shape: (996, 10, 140, 124)

    Proxy scores (lower is generally better for RMSE/MAE):
    - RMSE(H+1 vs last observed):        13.5603
    - MAE(H+1 vs last observed):         8.3809
    - RMSE(all horizons vs persistence): 27.3255
    - MAE(all horizons vs persistence):  13.9827
    - Mean abs step change (H to H+1):   2.3469

    Note: Official competition score is computed on hidden future targets on Kaggle.
:::
:::
