# Experiment 04 — TFNO2D Pipeline: Detailed Report

> **Notebook**: `exp_04_fno.ipynb`  
> **Task**: PM2.5 spatiotemporal forecasting on a 140×124 gridded domain  
> **Objective**: Given 10 hours of past PM2.5 concentrations (cpm25) and 26 hours of meteorological/emission forcing, forecast the next **16 hours** of PM2.5 concentration at every grid cell.

---

## Table of Contents

1. [Problem Overview](#1-problem-overview)
2. [Data Description](#2-data-description)
3. [Preprocessing Pipeline](#3-preprocessing-pipeline)
4. [Data Loading & Windowing](#4-data-loading--windowing)
5. [Model Architecture: TFNO2D](#5-model-architecture-tfno2d)
6. [Training Procedure](#6-training-procedure)
7. [Inference & Post-processing](#7-inference--post-processing)
8. [Results](#8-results)
9. [Complete Hyperparameter Reference](#9-complete-hyperparameter-reference)

---

## 1. Problem Overview

The experiment trains a **Tucker-Factorized Fourier Neural Operator (TFNO2D)** to forecast PM2.5 air pollution concentrations across a 140×124 spatial grid. The input tensor is `(B, C=9, T=26, H=140, W=124)` and the output is `(B, H=140, W=124, T_out=16)`, predicting 16 future hourly PM2.5 fields.

The pipeline runs on Kaggle with a **Tesla P100-PCIE-16GB GPU** and integrates Weights & Biases (W&B) for experiment tracking.

---

## 2. Data Description

### 2.1 Raw Data Layout

Raw data is stored as `.npy` files under `raw/<MONTH>/<feature>.npy`, each shaped `(T, 140, 124)` where `T` varies by month.

### 2.2 Months Used

| Month      | Time Steps | Role        | Sampling Weight |
|------------|-----------|-------------|-----------------|
| APRIL_16   | 715       | Training    | 0.5             |
| JULY_16    | 739       | Training    | 0.5             |
| DEC_16     | 739       | Training    | 3.0 (upweighted for winter PM2.5 patterns) |
| OCT_16     | 739       | Validation  | — (held out entirely) |

### 2.3 Input Features (9 channels)

| # | Feature | Category | Description |
|---|---------|----------|-------------|
| 1 | `cpm25`  | Target/Input | PM2.5 concentration (µg/m³) — available for first 10 hours only |
| 2 | `pblh`   | Meteorological | Planetary Boundary Layer Height |
| 3 | `t2`     | Meteorological | 2-meter Temperature (K) |
| 4 | `u10`    | Meteorological | 10-meter U-component of Wind |
| 5 | `v10`    | Meteorological | 10-meter V-component of Wind |
| 6 | `q2`     | Meteorological | 2-meter Specific Humidity |
| 7 | `rain`   | Meteorological | Rainfall accumulation |
| 8 | `PM25`   | Emission | PM2.5 emission inventory |
| 9 | `NOx`    | Emission | NOx emission inventory |

**Auxiliary features** (hour encodings, ventilation index, wind convergence, land mask, sparse masks) are **disabled** in this experiment (`use_aux = False`).

### 2.4 Temporal Structure

| Parameter | Value | Description |
|-----------|-------|-------------|
| `t_in_cpm` | 10 | Hours of past cpm25 history available at test time |
| `t_in_met` | 26 | Total input window length (10 past + 16 NWP forecast hours) |
| `t_out` | 16 | Forecast horizon (hours ahead) |

**Critical**: cpm25 is only available for the first 10 time steps. For time steps 11–26, the cpm25 channel is zero-padded (masked). Other meteorological/emission features span all 26 time steps.

---

## 3. Preprocessing Pipeline

The preprocessing strategy is **`grid_log_standardize`** — a three-stage pipeline applied independently per grid cell:

### 3.1 Stage 1: Feature-Specific Log Transforms

Different log transforms are applied depending on the feature type:

| Transform Type | Features | Formula | Purpose |
|---------------|----------|---------|---------|
| **`log1p`** | `cpm25`, `rain` | `log(1 + max(x, 0))` | Compresses right-skewed positive distributions |
| **`log(x + eps)`** | `PM25`, `NOx` | `log(max(x, 0) + ε)` where `ε = 1e-12` | Handles near-zero emission values safely |
| **`signed_log`** | `u10`, `v10` | `sign(x) · log(1 + |x|)` | Preserves sign for wind vectors while compressing magnitude |
| **None** | `pblh`, `t2`, `q2` | Identity (no log transform) | Already reasonably distributed |

### 3.2 Stage 2: Grid-Wise (Per-Cell) Standardization

For each feature, per-grid-cell mean and standard deviation maps `(140 × 124)` are computed **from training months only** (APRIL_16, JULY_16, DEC_16). The statistics are computed in the **log-transformed domain** using chunked streaming (chunk size = 48 time steps) for memory efficiency.

**Formula**: `x_normalized = (x_log_transformed - μ_cell) / max(σ_cell, ε)`

where `ε = 1e-6` (grid_stats_eps) prevents division by zero.

These statistics are saved to `grid_log_scaler_2016.npz` with metadata including the scaler version, months used, and train-only flag. On subsequent runs, the scaler is loaded from disk. An automatic rebuild is triggered if the stored months don't match the expected training months.

### 3.3 Stage 3: NaN/Inf Cleanup

After each transform, `nan_to_num` is applied to replace any NaN/Inf values with 0.0.

### 3.4 Min-Max Bounds

Official physical bounds are loaded from `feat_min_max.mat` (or fallback constants). These are used for diagnostics and inverse transforms, **not** for the primary normalization:

| Feature | Min | Max |
|---------|-----|-----|
| cpm25 | 0.994 | 1465.25 |
| pblh | 52.12 | 6271.37 |
| t2 | 223.53 | 324.77 |
| u10 | -26.83 | 29.03 |
| v10 | -29.22 | 31.93 |
| q2 | 0.0 | 0.0459 |
| rain | 0.0 | 96.63 |
| PM25 | 0.0 | 1.427e-07 |
| NOx | 0.0 | 7.977e-08 |

### 3.5 Post-Preprocessing Statistics

After preprocessing, features have near-zero mean and unit-scale standard deviation (target range roughly [-5, +5] for most features), confirming that the grid-wise standardization is effective.

---

## 4. Data Loading & Windowing

### 4.1 Sliding Window Dataset

The `PM25Dataset` class creates input-output pairs using a **sliding window** approach:

- **Window size**: `t_in_cpm + t_out = 10 + 16 = 26` time steps
- **Training stride**: `stride_train = 2` (step every 2 time steps → denser sampling)
- **Validation stride**: `stride_val = 4` (step every 4 time steps → sparser, unbiased)

### 4.2 Temporal Firewall

A **12-hour temporal gap** (`temporal_gap_hours = 12`) is enforced between training and validation data to prevent information leakage at month boundaries:

| Gap | Details |
|-----|---------|
| Train skip-start for DEC_16 | First 12 hours skipped (DEC follows OCT in month order) |
| Train skip-end for JULY_16 | Last 12 hours skipped (JULY precedes OCT) |
| Val skip-start for OCT_16 | First 12 hours skipped |
| Val skip-end for OCT_16 | Last 12 hours skipped |

### 4.3 Dataset Sizes

- **Training samples**: 1,047
- **Validation samples**: 173

### 4.4 Weighted Random Sampling

A `WeightedRandomSampler` is used to oversample winter months (DEC_16 has weight 3.0 vs. 0.5 for spring/summer), reflecting the higher PM2.5 concentrations and forecasting difficulty in winter.

### 4.5 DataLoader Configuration

| Parameter | Training | Validation |
|-----------|----------|------------|
| Batch size | 4 | 8 |
| Shuffle | Via WeightedRandomSampler | False |
| num_workers | 2 | 2 |
| pin_memory | True | True |
| drop_last | True | False |

### 4.6 Batch Shape

- **Input `x`**: `(B=4, C=9, T=26, H=140, W=124)`
- **Target `y`**: `(B=4, H=140, W=124, T_out=16)`

The target is the **absolute** cpm25 value (in standardized log-space), not a residual — the residual-over-persistence is handled within certain model variants but **not** in the TFNO2D used in this experiment.

---

## 5. Model Architecture: TFNO2D

### 5.1 Architecture Overview

The **TFNO2D** (Tucker-Factorized Fourier Neural Operator, 2D variant) learns **global spectral patterns** via FFT-based spectral convolutions. It is well-suited for periodic/wave-like features common in atmospheric data such as wind and pressure fields.

### 5.2 Architecture Diagram

```
Input: (B, C=9, T=26, H=140, W=124)
         │
         ▼ Reshape → (B, C*T=234, H=140, W=124)
         │
         ▼ Lift: Conv2d(234 → 64, kernel=1)
         │
         ▼ 6× FNO Blocks (width=64, modes=20)
         │    ┌─────────────────────────────────────────┐
         │    │  SpectralConv2d(64→64, modes=20×20)     │
         │    │       +                                 │
         │    │  Conv2d(64→64, kernel=1)  (bypass path) │
         │    │       → GroupNorm(8 groups) → GELU      │
         │    └─────────────────────────────────────────┘
         │
         ▼ Projection Head:
         │    Conv2d(64 → 128, kernel=1) → GELU
         │    Conv2d(128 → 16, kernel=1)
         │
         ▼ Permute → (B, H=140, W=124, T_out=16)
```

### 5.3 Spectral Convolution (`_SpectralConv2d`)

The core of each FNO block:

1. **Forward FFT**: `rfft2(x, norm='ortho')` converts the spatial field to Fourier space
2. **Mode Selection**: Retains `modes = 20` frequency modes along each spatial dimension (both positive and negative frequencies separately)
3. **Spectral Multiplication**: Two separate complex-valued weight matrices (`weight_pos`, `weight_neg`) are multiplied with spectral coefficients via `einsum('bixy,ioxy->boxy')`
4. **Inverse FFT**: `irfft2(out, norm='ortho')` converts back to physical space

The scale initialization for weights is `1 / max(1, in_ch * out_ch)`.

### 5.4 FNO Block (`_FNOBlock`)

Each block combines:
- **Spectral path**: `SpectralConv2d` for global frequency interactions
- **Bypass path**: `Conv2d(1×1)` for local residual connection
- **Normalization**: `GroupNorm` with 8 groups
- **Activation**: `GELU`

Output = `GELU(GroupNorm(SpectralConv(x) + Conv1x1(x)))`

### 5.5 TFNO2D Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `width` | 64 | Hidden channel dimension throughout the FNO blocks |
| `modes` | 20 | Number of Fourier modes retained per spatial dimension |
| `depth` | 6 | Number of stacked FNO blocks |
| `in_channels` | 234 (= 9 × 26) | Channels after time-flattening (C×T) |
| `out_steps` | 16 | Number of forecast time steps |
| `t_in_cpm` | 10 | Used for optional residual anchor (disabled here) |
| `use_residual_anchor` | False | No persistence residual connection |
| `use_reflect_pad` | False | No reflect padding |
| **Total parameters** | **19,711,952** | Trainable parameters |

---

## 6. Training Procedure

### 6.1 Training Protocol

This experiment uses a **simple one-phase training** approach (not the multi-phase approach available in the codebase):

- **Loss function**: RMSE in standardized space throughout all epochs
- **No phase switching** (phase_switch_epoch = 20 is configured but the actual training uses `rmse_loss` only)
- **No physics loss** (`physics.enabled = False`, `lambda_p = 0.0`)

### 6.2 Loss Function: RMSE

```python
def rmse_loss(pred, target):
    return torch.mean(torch.sqrt(torch.mean((pred - target) ** 2, dim=(1, 2)) + 1e-8))
```

This computes the RMSE across spatial dimensions `(H, W)` for each sample, then averages over the batch. The `1e-8` epsilon prevents numerical instabilities in the square root.

### 6.3 Optimizer: AdamW

| Parameter | Value |
|-----------|-------|
| Optimizer | `AdamW` |
| Learning rate | `1e-3` (0.001) |
| Weight decay | `1e-4` (0.0001) |

### 6.4 Learning Rate Scheduler: OneCycleLR

| Parameter | Value | Description |
|-----------|-------|-------------|
| `max_lr` | 1e-3 | Peak learning rate |
| `pct_start` | 0.1 | Warmup fraction (first 10% of training steps) |
| `epochs` | 60 (max) | Total epochs for scheduler |
| Steps per epoch | len(train_dl) | Determined by dataset size / batch size |

The scheduler steps **per batch** (not per epoch), providing smooth annealing.

### 6.5 Regularization

| Technique | Value |
|-----------|-------|
| Gradient clipping | `grad_clip = 1.0` (max norm) |
| Gradient clipping method | `torch.nn.utils.clip_grad_norm_` |
| AMP (mixed precision) | **Disabled** (`use_amp = False`) — complex-valued spectral weights are incompatible with AMP |
| NaN guard | `torch.nan_to_num(pred)` applied after every forward pass |

### 6.6 Early Stopping

| Parameter | Value |
|-----------|-------|
| Patience | 8 epochs |
| Checkpoint metric | `val_rmse_std` (validation RMSE in standardized space) |

The best model (lowest `val_rmse_std`) is saved to disk. Training stops if no improvement for 8 consecutive epochs.

### 6.7 Checkpointing

| Parameter | Value |
|-----------|-------|
| `enable_checkpointing` | True |
| `checkpoint_every_epochs` | 5 |
| `resume_if_available` | **False** (disabled to prevent architecture mismatch crashes) |

Full training state (model, optimizer, scheduler, scaler, history) is saved every 5 epochs.

### 6.8 Reproducibility

| Parameter | Value |
|-----------|-------|
| `seed` | 42 |
| `torch.backends.cudnn.deterministic` | True |
| `torch.backends.cudnn.benchmark` | False |

Seeds set for: Python `random`, NumPy, PyTorch (CPU + CUDA), and `PYTHONHASHSEED`.

### 6.9 Validation Metrics

During validation, three metrics are computed:

1. **`val_rmse_std`** — RMSE in standardized (grid-log) space (used for checkpoint selection)
2. **`val_rmse_phys`** — RMSE in physical space (µg/m³), computed by inverse-transforming both prediction and target to physical domain with clipping:
   - Clamp log-domain values to `[-20, 20]` to prevent `expm1` explosion
   - `physical = clamp(expm1(log_domain), min=0)`
3. **`val_persistence_phys`** — Persistence baseline RMSE: last observed cpm25 repeated for all 16 forecast steps

### 6.10 Loss Configuration (Designed but Not Fully Applied in Training)

The following loss parameters are configured in the hyperparameter dictionary for potential multi-phase training, but in **this experiment**, only standard RMSE is used:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `loss_type` | `log_rmse` | Intended loss type |
| `phase_switch_epoch` | 20 | Epoch to switch from log_rmse to weighted_mae |
| `intensity_alpha` | 1.5 | Intensity weighting alpha |
| `intensity_ref` | 59.0 µg/m³ | Reference concentration for intensity weighting |
| `intensity_cap` | 3.0 | Maximum weight multiplier |
| `horizon_weight_min` | 0.8 | Weight for h+1 (near horizon) |
| `horizon_weight_max` | 1.4 | Weight for h+16 (far horizon) |
| `checkpoint_mixed_alpha` | 0.3 | Alpha for mixed checkpoint metric |

---

## 7. Inference & Post-processing

### 7.1 Test-Time Augmentation (TTA)

TTA is **enabled** with 4 geometric augmentation modes:

| Mode | Input Transform | Output Inverse |
|------|----------------|----------------|
| `identity` | None | None |
| `hflip` | Horizontal flip (`flip(-1)`) | Horizontal flip (`flip(2)`) |
| `vflip` | Vertical flip (`flip(-2)`) | Vertical flip (`flip(1)`) |
| `hvflip` | Both flips | Both flips (reversed) |

**Final prediction**: Average of all 4 TTA outputs → more robust predictions.

### 7.2 Inference Flow

1. Load best model checkpoint from `model_save` path
2. Build test input from `test_in/` directory:
   - cpm25: `(N, 10, H, W)` → padded to `(N, 26, H, W)` (zero-fill for hours 11–26)
   - Met/emission features: `(N, 26, H, W)` (full 26-hour window)
3. Stack into tensor: `(N, C=9, T=26, H=140, W=124)`
4. Run batched inference (batch_size_test = 16) with TTA
5. **Denormalize** predictions back to physical space:
   - Clip standardized output to `[-10, 10]`
   - Un-standardize: `x = x_std × σ_cell + μ_cell`
   - Clip log-domain to `[-20, 20]`
   - Inverse log1p: `expm1(x)` then clamp ≥ 0
6. Final clamp: all predictions ≥ 0 (PM2.5 cannot be negative)
7. Assert output shape: `(996, 140, 124, 16)` and all finite

### 7.3 Test Configuration

| Parameter | Value |
|-----------|-------|
| Test samples | 996 |
| Batch size (test) | 16 |
| TTA modes | identity, hflip, vflip, hvflip |
| Output format | `preds.npy` — shape `(996, 140, 124, 16)` in µg/m³ |

---

## 8. Results

### 8.1 Training Summary

- **Training ran for 19 epochs** before early stopping (patience = 8)
- **Best epoch**: 11
- **Best Val RMSE (standardized)**: 0.5318
- **Best Val RMSE (physical)**: 0.444 µg/m³
- **Persistence baseline**: 30.83 µg/m³ (the threshold to beat)
- **Training time**: ~21 minutes (on Tesla P100-PCIE-16GB)

### 8.2 Training Curve Highlights

| Epoch | Train RMSE (std) | Val RMSE (std) | Val RMSE (phys) µg/m³ | Status |
|-------|-----------------|----------------|----------------------|--------|
| 1     | 0.7406          | 0.6142         | 0.512                | ← saved |
| 5     | 0.2028          | 0.5405         | 0.448                | ← saved |
| 8     | 0.1560          | 0.5339         | 0.443                | ← saved |
| **11** | **0.1351** | **0.5318** | **0.444** | **← best** |
| 19    | 0.1062          | 0.5440         | 0.453                | Early stop |

The model converges quickly (major improvement in first 5 epochs), then continues to overfit the training set while val metrics plateau, indicating the early stopping was appropriate.

---

## 9. Complete Hyperparameter Reference

### 9.1 Model Architecture

| Parameter | Value |
|-----------|-------|
| `model_type` | `tfno2d` |
| `width` | 64 |
| `modes` | 20 |
| `depth` | 6 |
| `use_aux` | False |
| Total parameters | 19,711,952 |

### 9.2 Training Configuration

| Parameter | Value |
|-----------|-------|
| `epochs` | 60 (max, actual: 19 with early stopping) |
| `batch_size_train` | 4 |
| `batch_size_val` | 8 |
| `lr` | 0.001 |
| `weight_decay` | 0.0001 |
| `patience` | 8 |
| `grad_clip` | 1.0 |
| `pct_start` | 0.1 |
| `seed` | 42 |
| `use_amp` | False |
| `resume_if_available` | False |
| `num_workers` | 2 |
| `pin_memory` | True |

### 9.3 Data Configuration

| Parameter | Value |
|-----------|-------|
| `stride_train` | 2 |
| `stride_val` | 4 |
| `use_weighted_sampler` | True |
| `month_weights.APRIL_16` | 0.5 |
| `month_weights.JULY_16` | 0.5 |
| `month_weights.DEC_16` | 3.0 |
| `temporal_gap_hours` | 12 |
| `t_in_cpm` | 10 |
| `t_in_met` | 26 |
| `t_out` | 16 |

### 9.4 Preprocessing Configuration

| Parameter | Value |
|-----------|-------|
| `normalization` | `grid_log_standardize` |
| `use_mmap` | True |
| `grid_chunk_size` | 48 |
| `grid_stats_eps` | 1e-6 |
| `log_eps` | 1e-12 |
| `auto_build_grid_stats` | True |
| `grid_stats_train_only` | True |
| `enforce_grid_stats_train_months` | True |
| `signed_log_for_negative` | True |
| `log1p_features` | cpm25, rain, ventilation_index |
| `log_eps_features` | PM25, SO2, NOx, NH3, NMVOC_e, NMVOC_finn, bio |
| `signed_log_features` | u10, v10, wind_convergence |
| `feature_time_limits.cpm25` | 10 |

### 9.5 Loss Configuration

| Parameter | Value |
|-----------|-------|
| `loss_type` | `log_rmse` |
| `phase_switch_epoch` | 20 |
| `intensity_alpha` | 1.5 |
| `intensity_ref` | 59.0 µg/m³ |
| `intensity_cap` | 3.0 |
| `horizon_weight_min` | 0.8 |
| `horizon_weight_max` | 1.4 |

### 9.6 Checkpoint Configuration

| Parameter | Value |
|-----------|-------|
| `checkpoint_metric` | `val_rmse_std` |
| `checkpoint_mixed_alpha` | 0.3 |
| `enable_checkpointing` | True |
| `checkpoint_every_epochs` | 5 |

### 9.7 Inference Configuration

| Parameter | Value |
|-----------|-------|
| `use_tta` | True |
| `tta_modes` | identity, hflip, vflip, hvflip |
| `batch_size_test` | 16 |
| `test_samples` | 996 |

### 9.8 Physics Configuration (Disabled)

| Parameter | Value |
|-----------|-------|
| `physics.enabled` | False |
| `lambda_p` | 0.0 |

### 9.9 W&B Configuration

| Parameter | Value |
|-----------|-------|
| `wandb.enabled` | True |
| `wandb.entity` | ronitraj |
| `wandb.project` | aisehack |
| `wandb.run_name` | exp04-tfno2d-tuned |

### 9.10 Persistence Baseline

| Parameter | Value |
|-----------|-------|
| `persistence_rmse_phys` | 30.83 µg/m³ |

---

## Key Techniques Summary

| Technique | Details |
|-----------|---------|
| **Feature-specific log transforms** | `log1p` for positive-skewed vars, `log(x+eps)` for near-zero emissions, `signed_log` for wind |
| **Grid-wise standardization** | Per-cell (140×124) mean/std maps computed from training months only |
| **Temporal masking** | cpm25 zero-padded beyond t=10 (future PM2.5 unavailable at test time) |
| **Temporal firewall** | 12-hour gap between train/val months to prevent leakage |
| **Weighted sampling** | Winter months upweighted 3× for PM2.5 patterns |
| **Fourier Neural Operator** | Global spectral convolutions capturing periodic atmospheric patterns |
| **OneCycleLR** | Cosine annealing with 10% linear warmup |
| **Gradient clipping** | Max-norm = 1.0 for training stability |
| **NaN guarding** | Post-forward nan_to_num replacement |
| **Test-Time Augmentation** | 4-fold geometric averaging (identity + flips) |
| **Robust denormalization** | Double clipping in standardized and log-domain before expm1 |
| **Early stopping** | Patience=8 on val_rmse_std |
