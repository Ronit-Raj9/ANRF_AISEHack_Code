# PINO Baseline Pipeline: Detailed Report

> **Notebook**: `pino_baseline.ipynb`  
> **Task**: PM2.5 spatiotemporal nowcasting on a 140×124 gridded domain  
> **Objective**: Given a sliding window of 16 hours of past PM2.5 concentrations, meteorological, and emission forcing, forecast the next **16 hours** of PM2.5 concentration at every grid cell.

---

## Table of Contents

1. [Problem Overview](#1-problem-overview)
2. [Data Description](#2-data-description)
3. [Preprocessing Pipeline](#3-preprocessing-pipeline)
4. [Data Loading & Windowing](#4-data-loading--windowing)
5. [Model Architecture: TFNO2D (PINO)](#5-model-architecture-tfno2d-pino)
6. [Training Procedure](#6-training-procedure)
7. [Inference & Post-processing](#7-inference--post-processing)
8. [Complete Hyperparameter Reference](#8-complete-hyperparameter-reference)

---

## 1. Problem Overview

This experiment establishes a **Physics-Informed Neural Operator (PINO)** baseline. It relies on a **Tucker-Factorized Fourier Neural Operator (TFNO2D)** architecture, but upgrades the previous baseline with:
1. Engineered physical features (e.g., wind speed, relative humidity, lagged variables).
2. A multi-objective loss combining RMSE and MAE, emphasizing longer lead times via horizon weights.
3. An Advection-Diffusion residual physics loss.

The final model consumes a single concatenated input tensor of shape `(B, C=28, T=16, H=140, W=124)` and outputs forecasts of shape `(B, H=140, W=124, T_out=16)`.

---

## 2. Data Description

### 2.1 Time and Split

Rather than holding out a full month to validate, this approach loads all standard months (`APRIL_16`, `JULY_16`, `OCT_16`, `DEC_16`) and uses a random **15% window-level validation split** (`val_split: 0.15`) pulled globally from the sliding-window dataset.

### 2.2 Input Features

A total of **28 features** are provided to the model at every time step, comprising 16 base channels and 12 engineered physical channels.

**Base Features (16 channels):**
- **Target:** `cpm25` (PM2.5 concentration; available only for the first 10 hours of the input window in practice, but padded/masked properly)
- **Meteorological:** `u10`, `v10`, `pblh`, `rain`, `t2`, `q2`, `swdown`, `psfc`
- **Emissions:** `PM25`, `SO2`, `NOx`, `NH3`, `NMVOC_e`, `NMVOC_finn`, `bio`

**Engineered Physical Features (12 channels):**
Constructed in an optimized float32 memory pipeline inside `add_physical_features()`:
1. `wind_speed`
2. `wind_dir`
3. `RH` (Relative Humidity)
4. `hour_sin`, `hour_cos` (Diurnal cycle)
5. `month_sin`, `month_cos` (Seasonal placeholder cycles)
6. `lat`, `lon` (Static geographical maps)
7. `lag_1`, `lag_3`, `lag_6` (PM2.5 history lags)

### 2.3 Temporal Structure

- **Model Input Window (`t_in`)**: 16 hours
- **Input CPM History (`t_in_cpm`)**: 10 hours
- **Meteorological Forecast History (`t_in_met`)**: 26 hours
- **Forecast Horizon (`t_out`)**: 16 hours

---

## 3. Preprocessing Pipeline

This experiment transitions away from `grid_log_standardize` and uses straightforward canonical scaling limits to map tensors strictly into the `[0, 1]` envelope.

### 3.1 Official Min-Max Normalization

- Pre-loaded bounds act as the single source of truth (`feat_min_max.mat`).
- Scaler logic: `x_normalized = clip((x - fmin) / (fmax - fmin), 0.0, 1.0)`.

### 3.2 Robust Feature Engineering

Static maps (`lat`, `lon`) and cyclic properties (`hour_sin`, `hour_cos`, etc.) are organically bounded within `[-1, 1]` or `[0, 1]`. They're broadcasted implicitly onto the batch without requiring heavy external matrix normalization pipelines. All downstream arrays get checked for `NaN/Inf` prior to network ingestion.

---

## 4. Data Loading & Windowing

### 4.1 Lazy Sliding Windows

An intelligent `SlidingWindowTensorDataset` operates natively over the fully-merged `(N, C, T, H, W)` master array. Instead of cropping entirely at the month-loader level, sliding window sequences (`w : w + t_in`) dynamically feed PyTorch batches without pre-materializing out-of-core redundancies. 

### 4.2 DataLoaders

- **Batch Size (Train)**: 4
- **Batch Size (Val)**: 8
- **Workers**: 2 with `pin_memory=True`
- **Reproducibility**: `seed: 42` for random state control.

---

## 5. Model Architecture: TFNO2D (PINO)

### 5.1 Tucker-Factorized Fourier Neural Operator

The **TFNO2D** is an efficient spatial modeling tool relying on spectral convolution operators in the Fourier domain. The architecture processes an expanded channel field representing depth (`width = 32`) and resolves global feature interaction across (`modes = 12`) harmonics per dimension, layered across `depth = 3`. 

The total raw temporal window collapses via reshaping into uniform input channels prior to lift convolutions: `C × T_in` = `28 × 16` = **`448 channels`**.

---

## 6. Training Procedure

### 6.1 Multi-Objective Data Loss (`objective_loss`)

Combines Structural Similarity + L1 Penalty, scaled against timeline distance mapping variables:

**Formula**: 
`loss.data = (rmse_weight * RMSE_H) + (mae_weight * MAE_H)`
where the horizon `H` linearly scales up from 0.8 at `t=1` to 1.4 at `t=16`.
- `rmse_weight = 0.8`
- `mae_weight = 0.2`

### 6.2 Physics Advection-Diffusion Penalty

The Physics loss formulates an approximate partial differential residual:
`R = dC/dt + u · grad(C) - kappa * laplacian(C) - S`

- `dC/dt` compares step sequence outputs.
- Advective gradients (`grad(C)`) use spatial tensors merged functionally alongside `u10`, `v10` wind indices.
- Laplacians mimic diffusion (`kappa = 1.0`).
- Sources `S` represent localized forcing constants (`SO2`, `NOx`, `cpm25`).
- The final metric is the Mean-Squared Residual `mean(R^2)`.

Combined: `Loss = (lambda_d × Data Loss) + (lambda_p × Physics Loss)` (`1.0` and `0.1` respectively).

### 6.3 Optimization & Scheduling

- **Optimizer**: `AdamW`
- **Learning Rate**: `6.0e-4`
- **Weight Decay**: `5.0e-3`
- **Scheduler**: **`coswr`** (Cosine Annealing with Warm Restarts). `T_0 = 10` epochs, bounded with doubling factor `T_mult = 2`.
- **Epochs**: Max `80`. Early stopping terminates execution if the `val_loss` fails to improve for `12` contiguous validation sweeps.

### 6.4 The Persistence Gate

An explicit historical baseline filter prevents saving models that are practically useless:
- **Baseline target**: `0.0208` (normalized) or `30.83` (µg/m³ physical).
- The training loop dynamically rejects/flags models tracking above this threshold via `check_persistence_gate`.

---

## 7. Inference & Post-processing

The inference pipeline remains encapsulated exclusively within `pino_baseline.ipynb`. 

1. Tests iterate over predefined Kaggle ranges pulling `test_samples = 996`.
2. Tensors are fed through `build_test_input` pulling dynamic engineered variables implicitly.
3. Outputs are converted from normalized `[0, 1]` ranges using `denormalize(pred_norm, cpm_min, cpm_max)`.
4. Outputs are strictly clamped to zero as physical PM2.5 counts strictly remain non-negative (`np.clip(p, 0.0, None)`).
5. Resulting dimensions match `(996, 140, 124, 16) µg/m³`. 
6. `preds.npy` is dumped iteratively using a temporary overwrite pattern to satisfy safety protocols (`os.fsync`). 

---

## 8. Complete Hyperparameter Reference

### 8.1 Model Architecture
| Parameter | Value |
|-----------|-------|
| `model_type` | `tfno2d` |
| `width` | 32 |
| `modes` | 12 |
| `depth` | 3 |
| `t_in` (time window) | 16 |
| Input Channels | 448 (`28 features × 16` time steps) |

### 8.2 Training Configuration
| Parameter | Value |
|-----------|-------|
| `epochs` | 80 |
| `batch_size_train` | 4 |
| `batch_size_val` | 8 |
| `lr` | 6.0e-4 |
| `weight_decay` | 5.0e-3 |
| `patience` | 12 |
| `scheduler` | `coswr` |
| `t0_epochs` | 10 |
| `t_mult` | 2 |
| `val_split` | 0.15 (Window-Level) |

### 8.3 Feature Space
| Parameter | Value |
|-----------|-------|
| `met` | u10, v10, pblh, rain, t2, q2, swdown, psfc |
| `emis` | PM25, SO2, NOx, NH3, NMVOC_e, NMVOC_finn, bio |
| `engineered` | wind_speed, wind_dir, RH, hour_sin, hour_cos, month_sin, month_cos, lat, lon, lag_1, lag_3, lag_6 |

### 8.4 Loss & Physics Configuration
| Parameter | Value |
|-----------|-------|
| `lambda_d` | 1.0 (Data Loss Priority) |
| `lambda_p` | 0.1 (Physics Loss Priority) |
| `rmse_weight` | 0.8 |
| `mae_weight` | 0.2 |
| `horizon_weight_min` | 0.8 |
| `horizon_weight_max` | 1.4 |
