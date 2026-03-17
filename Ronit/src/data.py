"""
Comprehensive preprocessing/data pipeline for PM2.5 forecasting.

Implements:
1) Feature-specific log protocol (log1p, log(x+eps), signed-log)
2) Grid-wise per-feature normalization (140x124 mean/std maps)
3) Cyclic temporal encodings (sin/cos hour/day)
4) Feature sparsity masks (rain/NMVOC_finn style triggers)
5) Dynamic feature toggling across input window
6) Derived physical indices (ventilation index, wind convergence)
7) Boundary-aware model compatibility via optional land mask channel
8) Residual-target-ready dataloader (absolute target; residual handled in model)
"""

from __future__ import annotations

import gc
import os
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, random_split


FALLBACK_OFFICIAL_BOUNDS = {
    'cpm25': (0.9940, 1465.25),
    'u10': (-26.829, 29.026),
    'v10': (-29.216, 31.930),
    'pblh': (52.115, 6271.4),
    'rain': (0.0, 96.627),
    't2': (223.53, 324.77),
    'q2': (0.0, 0.045855),
    'swdown': (0.0, 1320.3),
    'psfc': (47353.0, 102290.0),
    'PM25': (0.0, 1.4269e-07),
    'SO2': (0.0, 5.4354e-08),
    'NOx': (0.0, 7.9771e-08),
    'NH3': (0.0, 2.0868e-08),
    'NMVOC_e': (0.0, 1.0691e-08),
    'NMVOC_finn': (0.0, 7.0491e-06),
    'bio': (0.0, 8.2258e-09),
}

DEFAULT_LOG1P_FEATURES = {'cpm25', 'rain', 'ventilation_index'}
DEFAULT_LOG_EPS_FEATURES = {'PM25', 'SO2', 'NOx', 'NH3', 'NMVOC_e', 'NMVOC_finn', 'bio'}
DEFAULT_SIGNED_LOG_FEATURES = {'u10', 'v10', 'wind_convergence'}
DEFAULT_SPARSE_MASK_FEATURES = {'rain', 'NMVOC_finn'}
DEFAULT_AUX_STANDARDIZE = {'ventilation_index', 'wind_convergence'}

_SCALER_META_VERSION = 2


class SlidingWindowTensorDataset(Dataset):
    """Baseline-compatible sliding-window dataset over `(N, C, T, H, W)` tensors."""

    def __init__(self, tensor, window_size, t_out, target_channel=0):
        self.tensor = tensor
        self.window_size = window_size
        self.t_out = t_out
        self.target_channel = target_channel
        self.N, self.C, self.T, self.H, self.W = tensor.shape
        self.num_windows = self.T - window_size - t_out + 1
        if self.num_windows <= 0:
            raise ValueError(
                f"Not enough time steps for sliding windows: T={self.T}, "
                f"t_in={window_size}, t_out={t_out}."
            )
        self.total = self.N * self.num_windows

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        n = idx // self.num_windows
        w = idx % self.num_windows
        xb = self.tensor[n, :, w:w + self.window_size, :, :]
        y_start = w + self.window_size
        y_end = y_start + self.t_out
        yb = self.tensor[n, self.target_channel, y_start:y_end, :, :]
        xb = torch.tensor(xb, dtype=torch.float32)
        yb = torch.tensor(yb, dtype=torch.float32).permute(1, 2, 0)
        return xb, yb


def add_physical_features(tensor, features_dict=None, lat_long_path='lat_long.npy'):
    """
    Add 12 engineered PINO channels:
    wind_speed, wind_dir, RH, hour_sin, hour_cos, month_sin, month_cos,
    lat, lon, lag_1, lag_3, lag_6
    """
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
    denom_safe = np.where(
        np.abs(t2 - 29.65) < 1e-3,
        np.sign(t2 - 29.65 + np.float32(1e-3)) * np.float32(1e-3),
        t2 - np.float32(29.65),
    ).astype(np.float32)
    exponent = np.clip(
        np.float32(17.67) * (t2 - np.float32(273.15)) / denom_safe,
        np.float32(-100.0), np.float32(100.0)
    )
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
    lat = ll[:, :, 0]
    lon = ll[:, :, 1]
    lat = (lat - lat.min()) / (lat.max() - lat.min() + np.float32(1e-8))
    lon = (lon - lon.min()) / (lon.max() - lon.min() + np.float32(1e-8))
    out[:, ch] = lat.reshape(1, 1, H, W)
    ch += 1
    out[:, ch] = lon.reshape(1, 1, H, W)
    ch += 1
    del ll, lat, lon
    gc.collect()

    cpm25 = tensor[:, features_dict['cpm25'], :, :, :].copy()
    out[:, ch] = np.roll(cpm25, 1, axis=1)
    ch += 1
    out[:, ch] = np.roll(cpm25, 3, axis=1)
    ch += 1
    out[:, ch] = np.roll(cpm25, 6, axis=1)
    ch += 1
    del cpm25
    gc.collect()

    np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def _prep_cfg(cfg: dict) -> dict:
    prep = cfg.setdefault('preprocessing', {})
    prep.setdefault('normalization', 'grid_log_standardize')
    prep.setdefault('use_mmap', True)
    prep.setdefault('grid_chunk_size', 48)
    prep.setdefault('grid_stats_eps', 1e-6)
    prep.setdefault('temporal_gap_hours', 12)
    prep.setdefault('grid_stats_train_only', True)
    prep.setdefault('enforce_grid_stats_train_months', True)
    prep.setdefault('log_eps', 1e-12)
    prep.setdefault('log1p_features', sorted(DEFAULT_LOG1P_FEATURES))
    prep.setdefault('log_eps_features', sorted(DEFAULT_LOG_EPS_FEATURES))
    prep.setdefault('signed_log_features', sorted(DEFAULT_SIGNED_LOG_FEATURES))
    prep.setdefault('sparse_mask_features', sorted(DEFAULT_SPARSE_MASK_FEATURES))
    prep.setdefault('sparse_mask_threshold', 0.0)
    prep.setdefault('aux_standardize_features', sorted(DEFAULT_AUX_STANDARDIZE))
    prep.setdefault('feature_time_limits', {'cpm25': int(cfg['time']['t_in_cpm'])})
    prep.setdefault('derived_features', ['ventilation_index', 'wind_convergence'])
    prep.setdefault('land_mask_file', '')
    prep.setdefault('add_land_mask', 'land_mask' in cfg.get('features', {}).get('aux', []))
    _validate_preprocessing_contract(cfg)
    return prep


def _validate_preprocessing_contract(cfg: dict) -> None:
    prep = cfg.setdefault('preprocessing', {})
    log_eps = float(prep.get('log_eps', 1e-12))
    if not np.isfinite(log_eps) or log_eps <= 0:
        raise ValueError(f"preprocessing.log_eps must be a finite positive value, got {log_eps}")

    log1p_feats = _feature_set(cfg, 'log1p_features', DEFAULT_LOG1P_FEATURES)
    log_eps_feats = _feature_set(cfg, 'log_eps_features', DEFAULT_LOG_EPS_FEATURES)
    signed_feats = _feature_set(cfg, 'signed_log_features', DEFAULT_SIGNED_LOG_FEATURES)

    tiny_emissions = set(DEFAULT_LOG_EPS_FEATURES)
    present = set(cfg.get('features', {}).get('base', [])) | set(cfg.get('features', {}).get('all', []))
    required_emis = sorted(tiny_emissions & present)
    missing = [feat for feat in required_emis if feat not in log_eps_feats]
    if missing:
        raise ValueError(
            f"Emission/tiny-scale features must use log(x+eps). Missing in preprocessing.log_eps_features: {missing}"
        )

    overlap_bad = (log1p_feats & tiny_emissions & present)
    if overlap_bad:
        print(
            f"Warning: tiny emission features present in log1p_features {sorted(overlap_bad)}. "
            "log(x+eps) path takes precedence, but remove overlap for clarity."
        )

    if log_eps_feats & signed_feats:
        raise ValueError(
            f"Features cannot be both signed-log and log-eps transformed: {sorted(log_eps_feats & signed_feats)}"
        )


def _grid_scaler_enabled(cfg: dict) -> bool:
    return str(cfg.get('preprocessing', {}).get('normalization', 'grid_log_standardize')).lower() == 'grid_log_standardize'


def _use_mmap(cfg: dict) -> bool:
    return bool(cfg.get('preprocessing', {}).get('use_mmap', True))


def _stats_key(feat: str, suffix: str) -> str:
    return f'{feat}__{suffix}'


def _feature_set(cfg: dict, key: str, default: set[str]) -> set[str]:
    raw = cfg.get('preprocessing', {}).get(key)
    if raw is None:
        return set(default)
    return {str(x) for x in raw}


def _log_eps(cfg: dict) -> float:
    return float(cfg.get('preprocessing', {}).get('log_eps', 1e-12))


def _signed_log1p(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def _inverse_signed_log1p(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.expm1(np.abs(x))


def _derived_feature_names(cfg: dict) -> set[str]:
    return {str(x) for x in cfg.get('preprocessing', {}).get('derived_features', [])}


def _sparse_mask_names(cfg: dict) -> set[str]:
    return {f'{name}_mask' for name in _feature_set(cfg, 'sparse_mask_features', DEFAULT_SPARSE_MASK_FEATURES)}


def _aux_standardized_names(cfg: dict) -> set[str]:
    return set(cfg.get('preprocessing', {}).get('aux_standardize_features', []))


def _is_train_standardized_feature(cfg: dict, feat: str) -> bool:
    return feat in set(cfg['features']['base']) or feat in _aux_standardized_names(cfg)


def _log_transform_feature(arr: np.ndarray, feat: str, cfg: dict) -> np.ndarray:
    prep = _prep_cfg(cfg)
    x = arr.astype(np.float32)

    signed = _feature_set(cfg, 'signed_log_features', DEFAULT_SIGNED_LOG_FEATURES)
    log1p_feats = _feature_set(cfg, 'log1p_features', DEFAULT_LOG1P_FEATURES)
    log_eps_feats = _feature_set(cfg, 'log_eps_features', DEFAULT_LOG_EPS_FEATURES)

    if feat in signed:
        out = _signed_log1p(x)
    elif feat in log_eps_feats:
        out = np.log(np.maximum(x, 0.0) + _log_eps(cfg))
    elif feat in log1p_feats:
        out = np.log1p(np.maximum(x, 0.0))
    else:
        out = x
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _inverse_log_transform_feature(arr: np.ndarray, feat: str, cfg: dict) -> np.ndarray:
    x = arr.astype(np.float32)
    signed = _feature_set(cfg, 'signed_log_features', DEFAULT_SIGNED_LOG_FEATURES)
    log1p_feats = _feature_set(cfg, 'log1p_features', DEFAULT_LOG1P_FEATURES)
    log_eps_feats = _feature_set(cfg, 'log_eps_features', DEFAULT_LOG_EPS_FEATURES)

    if feat in signed:
        out = _inverse_signed_log1p(x)
    elif feat in log_eps_feats:
        out = np.maximum(np.exp(x) - _log_eps(cfg), 0.0)
    elif feat in log1p_feats:
        out = np.maximum(np.expm1(x), 0.0)
    else:
        out = x
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def load_minmax_bounds(cfg) -> dict:
    path = cfg['paths']['min_max']
    features = cfg['features']['all']
    if not cfg.get('runtime', {}).get('on_kaggle', False):
        return {feat: FALLBACK_OFFICIAL_BOUNDS[feat] for feat in features}

    if os.path.exists(path):
        try:
            import scipy.io as sio
            mat = sio.loadmat(path)
            return {
                feat: (
                    float(mat[f'{feat}_min'].squeeze()),
                    float(mat[f'{feat}_max'].squeeze()),
                )
                for feat in features
            }
        except Exception as exc:
            print(f"Warning: failed to read feat_min_max.mat via scipy ({exc}). Using fallback bounds.")

    missing = [feat for feat in features if feat not in FALLBACK_OFFICIAL_BOUNDS]
    if missing:
        raise KeyError(f"Missing fallback bounds for features: {missing}")
    return {feat: FALLBACK_OFFICIAL_BOUNDS[feat] for feat in features}


def normalize(arr: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
    rng = fmax - fmin
    if rng == 0:
        return np.zeros_like(arr, dtype=np.float32)
    normed = (arr.astype(np.float32) - fmin) / rng
    normed = np.nan_to_num(normed, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(normed, 0.0, 1.0)


def _expected_grid_stats_months(cfg: dict) -> list[str]:
    prep = _prep_cfg(cfg)
    if bool(prep.get('grid_stats_train_only', True)):
        return list(cfg.get('data', {}).get('train_months', []))
    return list(cfg.get('data', {}).get('months', []))


def denormalize(arr: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
    return arr.astype(np.float32) * (fmax - fmin) + fmin


def _broadcast_scalar_series(values: np.ndarray, H: int, W: int) -> np.ndarray:
    return np.broadcast_to(values[:, None, None].astype(np.float32), (len(values), H, W)).copy()


def _parse_time_strings(time_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ts = np.asarray(time_arr).astype(str)
    hours = np.array([int(t[11:13]) for t in ts], dtype=np.int32)
    dt = ts.astype('datetime64[h]')
    year_start = dt.astype('datetime64[Y]')
    doys = (dt.astype('datetime64[D]') - year_start).astype(np.int32) + 1
    return hours, doys


def load_static_maps(cfg) -> dict:
    ll_path = os.path.join(cfg['paths']['data'], 'raw', 'lat_long.npy')
    ll = np.load(ll_path).astype(np.float32)
    lat = ll[:, :, 0]
    lon = ll[:, :, 1]
    lat = (lat - lat.min()) / (lat.max() - lat.min() + 1e-8)
    lon = (lon - lon.min()) / (lon.max() - lon.min() + 1e-8)
    return {'lat': lat.astype(np.float32), 'lon': lon.astype(np.float32)}


def build_aux_feature_maps(cfg, time_arr: np.ndarray | None, T: int, H: int, W: int) -> dict:
    """Baseline-compatible helper to build calendar/static aux maps."""
    aux_names = cfg.get('features', {}).get('aux', [])
    static = load_static_maps(cfg)
    aux = {}

    if 'lat' in aux_names:
        aux['lat'] = np.broadcast_to(static['lat'][None], (T, H, W)).copy()
    if 'lon' in aux_names:
        aux['lon'] = np.broadcast_to(static['lon'][None], (T, H, W)).copy()

    if time_arr is None:
        hours = np.zeros(T, dtype=np.float32)
        doys = np.ones(T, dtype=np.float32)
    else:
        hours, doys = _parse_time_strings(time_arr)

    if 'hour_sin' in aux_names:
        aux['hour_sin'] = _broadcast_scalar_series(np.sin(2 * np.pi * hours / 24.0), H, W)
    if 'hour_cos' in aux_names:
        aux['hour_cos'] = _broadcast_scalar_series(np.cos(2 * np.pi * hours / 24.0), H, W)
    if 'doy_sin' in aux_names:
        aux['doy_sin'] = _broadcast_scalar_series(np.sin(2 * np.pi * doys / 366.0), H, W)
    if 'doy_cos' in aux_names:
        aux['doy_cos'] = _broadcast_scalar_series(np.cos(2 * np.pi * doys / 366.0), H, W)

    return aux


def load_land_mask(cfg, H: int, W: int) -> np.ndarray:
    prep = _prep_cfg(cfg)
    mask_file = prep.get('land_mask_file', '')
    candidates = [
        mask_file,
        os.path.join(cfg['paths']['data'], 'stats', 'land_mask.npy'),
        os.path.join(cfg['paths']['data'], 'raw', 'land_mask.npy'),
    ]
    for path in candidates:
        if path and os.path.exists(path):
            arr = np.load(path).astype(np.float32)
            if arr.shape == (H, W):
                return np.clip(arr, 0.0, 1.0)
    return np.ones((H, W), dtype=np.float32)


def compute_derived_features(raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if all(key in raw for key in ('pblh', 'u10', 'v10')):
        pblh = np.asarray(raw['pblh'], dtype=np.float32)
        u10 = np.asarray(raw['u10'], dtype=np.float32)
        v10 = np.asarray(raw['v10'], dtype=np.float32)
        wind_speed = np.sqrt(np.maximum(u10 * u10 + v10 * v10, 0.0)).astype(np.float32)
        out['ventilation_index'] = (pblh * wind_speed).astype(np.float32)

        du_dx = np.gradient(u10, axis=2)
        dv_dy = np.gradient(v10, axis=1)
        out['wind_convergence'] = (-(du_dx + dv_dy)).astype(np.float32)
    return out


def compute_sparse_masks(
    raw: dict[str, np.ndarray],
    cfg: dict,
    default_shape: tuple[int, int, int] | None = None,
) -> dict[str, np.ndarray]:
    threshold = float(cfg.get('preprocessing', {}).get('sparse_mask_threshold', 0.0))
    out: dict[str, np.ndarray] = {}
    for feat in _feature_set(cfg, 'sparse_mask_features', DEFAULT_SPARSE_MASK_FEATURES):
        if feat in raw:
            out[f'{feat}_mask'] = (np.asarray(raw[feat], dtype=np.float32) > threshold).astype(np.float32)
        elif default_shape is not None:
            out[f'{feat}_mask'] = np.zeros(default_shape, dtype=np.float32)
    return out


def _build_aux_from_raw(
    cfg: dict,
    raw: dict[str, np.ndarray],
    time_arr: np.ndarray | None,
    T: int,
    H: int,
    W: int,
) -> dict[str, np.ndarray]:
    aux_names = set(cfg.get('features', {}).get('aux', []))
    aux: dict[str, np.ndarray] = {}

    if any(name in aux_names for name in ('lat', 'lon')):
        static = load_static_maps(cfg)
        if 'lat' in aux_names:
            aux['lat'] = np.broadcast_to(static['lat'][None], (T, H, W)).copy()
        if 'lon' in aux_names:
            aux['lon'] = np.broadcast_to(static['lon'][None], (T, H, W)).copy()

    if 'land_mask' in aux_names:
        land = load_land_mask(cfg, H, W)
        aux['land_mask'] = np.broadcast_to(land[None], (T, H, W)).copy()

    if time_arr is None:
        hours = np.zeros(T, dtype=np.float32)
        doys = np.ones(T, dtype=np.float32)
    else:
        hours, doys = _parse_time_strings(time_arr)

    if 'hour_sin' in aux_names:
        aux['hour_sin'] = _broadcast_scalar_series(np.sin(2 * np.pi * hours / 24.0), H, W)
    if 'hour_cos' in aux_names:
        aux['hour_cos'] = _broadcast_scalar_series(np.cos(2 * np.pi * hours / 24.0), H, W)
    if 'doy_sin' in aux_names:
        aux['doy_sin'] = _broadcast_scalar_series(np.sin(2 * np.pi * doys / 366.0), H, W)
    if 'doy_cos' in aux_names:
        aux['doy_cos'] = _broadcast_scalar_series(np.cos(2 * np.pi * doys / 366.0), H, W)

    derived = compute_derived_features(raw)
    for feat in _derived_feature_names(cfg):
        if feat in derived and feat in aux_names:
            aux[feat] = derived[feat]

    masks = compute_sparse_masks(raw, cfg, default_shape=(T, H, W))
    for name, arr in masks.items():
        if name in aux_names:
            aux[name] = arr

    return aux


def build_grid_stats(cfg: dict, bounds: dict, months: list[str] | None = None, force: bool = False) -> dict:
    _prep_cfg(cfg)
    path = cfg['paths']['grid_stats']

    # Kaggle input mounts are read-only. Force writable target for newly built scaler files.
    if os.path.exists('/kaggle') and str(path).startswith('/kaggle/input/'):
        path = os.path.join('/kaggle/working', os.path.basename(path))
        cfg['paths']['grid_stats'] = path

    if os.path.exists(path) and not force:
        return load_grid_stats(cfg)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    data_root = cfg['paths']['data']
    months = list(months) if months is not None else _expected_grid_stats_months(cfg)
    features_for_stats = [
        feat for feat in cfg['features']['input']
        if _is_train_standardized_feature(cfg, feat)
    ]

    eps = float(cfg.get('preprocessing', {}).get('grid_stats_eps', 1e-6))
    chunk_size = int(cfg.get('preprocessing', {}).get('grid_chunk_size', 48))
    save_dict: dict[str, np.ndarray] = {}
    stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    print(f"Building grid scaler → {path}")
    for feat in features_for_stats:
        running_sum = None
        running_sumsq = None
        count = 0

        for month in months:
            month_arrays: dict[str, np.ndarray] = {}
            # load base once for month
            for bfeat in cfg['features']['base']:
                month_arrays[bfeat] = np.load(
                    os.path.join(data_root, 'raw', month, f'{bfeat}.npy'),
                    mmap_mode='r' if _use_mmap(cfg) else None,
                )

            if feat in cfg['features']['base']:
                arr = month_arrays[feat]
            else:
                # derived features are computed from month arrays; masks are not standardized
                derived = compute_derived_features(month_arrays)
                if feat not in derived:
                    continue
                arr = derived[feat]

            t_size = arr.shape[0]
            for start in range(0, t_size, chunk_size):
                chunk = np.asarray(arr[start:start + chunk_size], dtype=np.float32)
                chunk = _log_transform_feature(chunk, feat, cfg)
                chunk = np.nan_to_num(chunk, nan=0.0, posinf=0.0, neginf=0.0)
                if running_sum is None:
                    running_sum = np.zeros(chunk.shape[1:], dtype=np.float64)
                    running_sumsq = np.zeros(chunk.shape[1:], dtype=np.float64)
                running_sum += chunk.sum(axis=0, dtype=np.float64)
                running_sumsq += np.square(chunk, dtype=np.float32).sum(axis=0, dtype=np.float64)
                count += chunk.shape[0]

        if running_sum is None or count == 0:
            continue

        mean = np.nan_to_num((running_sum / count).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        var = (running_sumsq / count) - np.square(mean, dtype=np.float32)
        var = np.nan_to_num(var.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        std = np.sqrt(np.maximum(var, eps)).astype(np.float32)
        std = np.maximum(np.nan_to_num(std, nan=eps, posinf=eps, neginf=eps), eps).astype(np.float32)
        save_dict[_stats_key(feat, 'mean')] = mean
        save_dict[_stats_key(feat, 'std')] = std
        stats[feat] = (mean, std)
        print(f"  {feat:18s} mean∈[{mean.min():.4f}, {mean.max():.4f}] std∈[{std.min():.4f}, {std.max():.4f}]")

    save_dict['__meta_version'] = np.asarray([_SCALER_META_VERSION], dtype=np.int32)
    save_dict['__meta_months'] = np.asarray(months, dtype=np.str_)
    save_dict['__meta_train_only'] = np.asarray([1 if bool(cfg.get('preprocessing', {}).get('grid_stats_train_only', True)) else 0], dtype=np.int8)
    save_dict['__meta_val_month'] = np.asarray([str(cfg.get('data', {}).get('val_month', ''))], dtype=np.str_)

    try:
        np.savez_compressed(path, **save_dict)
    except OSError as exc:
        if getattr(exc, 'errno', None) == 30 or 'Read-only file system' in str(exc):
            fallback = os.path.join('/kaggle/working', os.path.basename(path)) if os.path.exists('/kaggle') else path
            print(f"Read-only path for grid scaler. Retrying at writable path: {fallback}")
            np.savez_compressed(fallback, **save_dict)
            cfg['paths']['grid_stats'] = fallback
        else:
            raise
    return stats


def load_grid_stats(cfg: dict) -> dict:
    _prep_cfg(cfg)
    path = cfg['paths']['grid_stats']
    expected = [
        feat for feat in cfg['features']['input']
        if _is_train_standardized_feature(cfg, feat)
    ]

    if not os.path.exists(path):
        if bool(cfg.get('preprocessing', {}).get('auto_build_grid_stats', True)):
            return build_grid_stats(cfg, bounds=load_minmax_bounds(cfg), force=False)
        raise FileNotFoundError(f"Grid scaler file not found: {path}")

    npz = np.load(path)

    enforce_train_months = bool(cfg.get('preprocessing', {}).get('enforce_grid_stats_train_months', True))
    expected_months = _expected_grid_stats_months(cfg)
    stored_months = [str(x) for x in npz['__meta_months'].tolist()] if '__meta_months' in npz else None
    if enforce_train_months and expected_months:
        if stored_months is None:
            if bool(cfg.get('preprocessing', {}).get('auto_build_grid_stats', True)):
                print('Grid scaler metadata missing; rebuilding with train-only month audit.')
                return build_grid_stats(cfg, bounds=load_minmax_bounds(cfg), force=True)
            raise RuntimeError('Grid scaler metadata missing and enforce_grid_stats_train_months=True.')
        if list(stored_months) != list(expected_months):
            if bool(cfg.get('preprocessing', {}).get('auto_build_grid_stats', True)):
                print(f'Grid scaler month mismatch. expected={expected_months} stored={stored_months}. Rebuilding.')
                return build_grid_stats(cfg, bounds=load_minmax_bounds(cfg), force=True)
            raise RuntimeError(
                f'Grid scaler months mismatch. expected={expected_months} stored={stored_months}'
            )
    missing = [feat for feat in expected if _stats_key(feat, 'mean') not in npz or _stats_key(feat, 'std') not in npz]
    if missing and bool(cfg.get('preprocessing', {}).get('auto_build_grid_stats', True)):
        print(f"Grid stats missing {missing}; rebuilding scaler file.")
        return build_grid_stats(cfg, bounds=load_minmax_bounds(cfg), force=True)
    elif missing:
        raise KeyError(f"Grid scaler missing features: {missing}")

    stats = {
        feat: (
            np.nan_to_num(np.asarray(npz[_stats_key(feat, 'mean')], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0),
            np.maximum(
                np.nan_to_num(
                    np.asarray(npz[_stats_key(feat, 'std')], dtype=np.float32),
                    nan=float(cfg.get('preprocessing', {}).get('grid_stats_eps', 1e-6)),
                    posinf=float(cfg.get('preprocessing', {}).get('grid_stats_eps', 1e-6)),
                    neginf=float(cfg.get('preprocessing', {}).get('grid_stats_eps', 1e-6)),
                ),
                float(cfg.get('preprocessing', {}).get('grid_stats_eps', 1e-6)),
            ).astype(np.float32),
        )
        for feat in expected
    }
    return stats


def describe_grid_stats(grid_stats: dict, features: list[str] | None = None) -> None:
    features = features or list(grid_stats)
    print(f"\n{'Feature':20s} {'mean[min,max]':>30s} {'std[min,max]':>30s}")
    print('─' * 86)
    for feat in features:
        if feat not in grid_stats:
            continue
        mean, std = grid_stats[feat]
        print(
            f"{feat:20s} [{mean.min():9.4f}, {mean.max():9.4f}] "
            f"[{std.min():9.4f}, {std.max():9.4f}]"
        )
    print()


def normalize_feature(arr: np.ndarray, feat: str, bounds: dict, cfg: dict, grid_stats: dict | None = None) -> np.ndarray:
    if _grid_scaler_enabled(cfg):
        stats = grid_stats or cfg.get('_runtime', {}).get('grid_stats')
        if _is_train_standardized_feature(cfg, feat):
            if stats is None or feat not in stats:
                raise RuntimeError(f"Grid scaler statistics missing for feature: {feat}")
            mean, std = stats[feat]
            eps = float(cfg.get('preprocessing', {}).get('grid_stats_eps', 1e-6))
            std = np.maximum(np.asarray(std, dtype=np.float32), eps)
            x = _log_transform_feature(arr, feat, cfg)
            out = ((x - mean) / std).astype(np.float32)
            return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return np.nan_to_num(np.asarray(arr, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    if feat == 'cpm25' and bool(cfg.get('preprocessing', {}).get('cpm25_log1p', False)):
        fmin, fmax = bounds['cpm25']
        x = np.log1p(np.maximum(arr.astype(np.float32), 0.0))
        return normalize(x, np.log1p(max(fmin, 0.0)), np.log1p(max(fmax, 0.0)))
    if feat in bounds:
        return normalize(arr, *bounds[feat])
    return np.asarray(arr, dtype=np.float32)


def denormalize_cpm25(arr: np.ndarray, bounds: dict, cfg: dict) -> np.ndarray:
    x = arr.astype(np.float32)
    if _grid_scaler_enabled(cfg):
        stats = cfg.get('_runtime', {}).get('grid_stats')
        if stats is None or 'cpm25' not in stats:
            raise RuntimeError('Grid scaler statistics for cpm25 are missing in cfg["_runtime"].')
        mean, std = stats['cpm25']
        eps = float(cfg.get('preprocessing', {}).get('grid_stats_eps', 1e-6))
        std = np.maximum(np.asarray(std, dtype=np.float32), eps)
        
        # Robust clipping in standard space against highly unconstrained raw outputs
        x = np.clip(x, -10.0, 10.0) 
        
        x = x * std[None, :, :, None] + mean[None, :, :, None]
        
        # Clip log-domain to explicitly prevent exp() blowing up to inf
        x = np.clip(x, -20.0, 20.0)
        
        x = _inverse_log_transform_feature(x, 'cpm25', cfg)
        return np.maximum(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), 0.0).astype(np.float32)

    fmin, fmax = bounds['cpm25']
    if bool(cfg.get('preprocessing', {}).get('cpm25_log1p', False)):
        x = denormalize(x, np.log1p(max(fmin, 0.0)), np.log1p(max(fmax, 0.0)))
        x = np.expm1(x)
        return np.maximum(x, 0.0).astype(np.float32)
    return np.maximum(denormalize(x, fmin, fmax), 0.0).astype(np.float32)


def _load_month(cfg, month: str, bounds: dict) -> dict:
    data_dir = cfg['paths']['data']
    mmap_mode = 'r' if _use_mmap(cfg) else None

    raw = {
        feat: np.load(os.path.join(data_dir, 'raw', month, f'{feat}.npy'), mmap_mode=mmap_mode)
        for feat in cfg['features']['base']
    }

    T, H, W = raw['cpm25'].shape
    time_path = os.path.join(data_dir, 'raw', month, 'time.npy')
    time_arr = np.load(time_path) if os.path.exists(time_path) else None

    aux = _build_aux_from_raw(cfg, raw=raw, time_arr=time_arr, T=T, H=H, W=W)
    return {
        'raw': raw,
        'aux': aux,
        'name': month,
        'shape': (T, H, W),
    }


def load_all_months(cfg, months: list, bounds: dict) -> list:
    _prep_cfg(cfg)
    all_data = []
    for month in months:
        print(f"  Loading {month} ...", end=' ', flush=True)
        all_data.append(_load_month(cfg, month, bounds))
        print('OK')
        gc.collect()
    return all_data


class PM25Dataset(Dataset):
    def __init__(
        self,
        months_data: list,
        cfg: dict,
        bounds: dict,
        grid_stats: dict,
        stride: int,
        month_names: list[str] | None = None,
        month_skip_start: dict[str, int] | None = None,
        month_skip_end: dict[str, int] | None = None,
    ):
        self.data = months_data
        self.cfg = cfg
        self.bounds = bounds
        self.grid_stats = grid_stats
        self.feats = cfg['features']['input']
        self.base_feats = set(cfg['features']['base'])
        self.time_limits = {
            str(k): int(v)
            for k, v in cfg.get('preprocessing', {}).get('feature_time_limits', {}).items()
        }
        self.t_in = int(cfg['time']['t_in_met'])
        self.t_out = int(cfg['time']['t_out'])
        self.t_cpm = int(cfg['time']['t_in_cpm'])

        self.month_names = month_names or [f'month_{i}' for i in range(len(months_data))]
        self.month_skip_start = {str(k): int(v) for k, v in (month_skip_start or {}).items()}
        self.month_skip_end = {str(k): int(v) for k, v in (month_skip_end or {}).items()}
        self.sample_months: list[str] = []
        self.index: list[tuple[int, int]] = []

        window = self.t_cpm + self.t_out
        for m_idx, mdata in enumerate(months_data):
            T = int(mdata['shape'][0])
            month_name = self.month_names[m_idx]
            skip_start = max(0, int(self.month_skip_start.get(month_name, 0)))
            skip_end = max(0, int(self.month_skip_end.get(month_name, 0)))
            start_t = skip_start
            stop_t_exclusive = (T - window + 1) - skip_end
            if stop_t_exclusive <= start_t:
                continue
            for t in range(start_t, stop_t_exclusive, stride):
                self.index.append((m_idx, t))
                self.sample_months.append(month_name)

    def __len__(self):
        return len(self.index)

    def sample_target_intensity(self, idx: int) -> float:
        m_idx, t = self.index[idx]
        mdata = self.data[m_idx]
        target_raw = np.asarray(
            mdata['raw']['cpm25'][t + self.t_cpm:t + self.t_cpm + self.t_out],
            dtype=np.float32,
        )
        return float(np.mean(target_raw))

    def _slice_feature(self, mdata: dict, feat: str, start: int, stop: int) -> np.ndarray:
        if feat in self.base_feats:
            raw = np.asarray(mdata['raw'][feat][start:stop], dtype=np.float32)
            return normalize_feature(raw, feat, self.bounds, self.cfg, self.grid_stats)

        if feat in mdata['aux']:
            aux_arr = np.asarray(mdata['aux'][feat][start:stop], dtype=np.float32)
            return normalize_feature(aux_arr, feat, self.bounds, self.cfg, self.grid_stats)

        raise KeyError(f"Feature '{feat}' not found in base/aux for month {mdata.get('name', '?')}")

    def __getitem__(self, idx):
        m_idx, t = self.index[idx]
        mdata = self.data[m_idx]

        channels = []
        for feat in self.feats:
            chunk = self._slice_feature(mdata, feat, t, t + self.t_in)
            limit = self.time_limits.get(feat)
            if limit is not None and 0 <= limit < self.t_in:
                chunk[limit:] = 0.0
            channels.append(chunk)

        x = torch.from_numpy(np.stack(channels, axis=0).astype(np.float32))

        target_raw = np.asarray(
            mdata['raw']['cpm25'][t + self.t_cpm:t + self.t_cpm + self.t_out],
            dtype=np.float32,
        )
        target = normalize_feature(target_raw, 'cpm25', self.bounds, self.cfg, self.grid_stats)
        y = torch.from_numpy(target).permute(1, 2, 0)
        return x, y


def get_dataloaders(cfg, train_data: list, val_data: list, bounds: dict):
    _prep_cfg(cfg)
    grid_stats = load_grid_stats(cfg)
    cfg.setdefault('_runtime', {})['grid_stats'] = grid_stats

    train_month_names = list(cfg['data']['train_months'])
    train_data_effective = list(train_data)
    val_month_names = [cfg['data']['val_month']] * len(val_data)

    temporal_gap = int(cfg.get('preprocessing', {}).get('temporal_gap_hours', 0))
    train_skip_start: dict[str, int] = {}
    train_skip_end: dict[str, int] = {}
    val_skip_start: dict[str, int] = {}
    val_skip_end: dict[str, int] = {}
    month_order = list(cfg.get('data', {}).get('months', []))
    val_month = str(cfg.get('data', {}).get('val_month', ''))
    train_month_set = {str(m) for m in train_month_names}

    if temporal_gap > 0 and val_month in month_order:
        v_idx = month_order.index(val_month)
        prev_month = month_order[v_idx - 1] if v_idx > 0 else None
        next_month = month_order[v_idx + 1] if v_idx < (len(month_order) - 1) else None

        if prev_month is not None and prev_month in train_month_set:
            train_skip_end[prev_month] = max(train_skip_end.get(prev_month, 0), temporal_gap)
            val_skip_start[val_month] = max(val_skip_start.get(val_month, 0), temporal_gap)
        if next_month is not None and next_month in train_month_set:
            train_skip_start[next_month] = max(train_skip_start.get(next_month, 0), temporal_gap)
            val_skip_end[val_month] = max(val_skip_end.get(val_month, 0), temporal_gap)

    if temporal_gap > 0:
        print(f"  Temporal firewall (hours): {temporal_gap}")
        if train_skip_start or train_skip_end or val_skip_start or val_skip_end:
            print(f"  Train skip-start: {train_skip_start}")
            print(f"  Train skip-end:   {train_skip_end}")
            print(f"  Val skip-start:   {val_skip_start}")
            print(f"  Val skip-end:     {val_skip_end}")
        else:
            print("  Temporal firewall active, but no adjacent train/val month boundary found in month order.")

    val_tail_fraction = float(cfg.get('data', {}).get('val_tail_fraction', 0.0))
    if 0.0 < val_tail_fraction < 1.0 and len(val_data) > 0:
        split_month_name = str(cfg['data']['val_month'])
        split_month_data = val_data[0]
        t_total = int(split_month_data['shape'][0])
        t_in_cpm = int(cfg['time']['t_in_cpm'])
        t_out = int(cfg['time']['t_out'])
        window = t_in_cpm + t_out
        total_starts = max(0, t_total - window + 1)

        split_start = int(np.floor((1.0 - val_tail_fraction) * t_total))
        split_start = min(max(split_start, 0), t_total)

        train_stop_exclusive = max(0, split_start - temporal_gap)
        val_start_index = min(t_total, split_start + temporal_gap)

        split_skip_end = max(0, total_starts - train_stop_exclusive)
        train_skip_end[split_month_name] = max(train_skip_end.get(split_month_name, 0), split_skip_end)
        val_skip_start[split_month_name] = max(val_skip_start.get(split_month_name, 0), val_start_index)

        if split_month_name not in train_month_names:
            train_month_names.append(split_month_name)
            train_data_effective.append(split_month_data)

        print(
            f"  Intra-month split on {split_month_name}: "
            f"train_head≈{int((1.0 - val_tail_fraction) * 100)}% | "
            f"val_tail≈{int(val_tail_fraction * 100)}% | "
            f"split_hour={split_start}"
        )

    train_ds = PM25Dataset(
        train_data_effective,
        cfg,
        bounds,
        grid_stats,
        stride=cfg['time']['stride_train'],
        month_names=train_month_names,
        month_skip_start=train_skip_start,
        month_skip_end=train_skip_end,
    )
    val_ds = PM25Dataset(
        val_data,
        cfg,
        bounds,
        grid_stats,
        stride=cfg['time']['stride_val'],
        month_names=val_month_names,
        month_skip_start=val_skip_start,
        month_skip_end=val_skip_end,
    )

    print(f"  Train samples: {len(train_ds):,}  |  Val samples: {len(val_ds):,}")

    sampler = None
    if bool(cfg['training'].get('use_weighted_sampler', False)):
        print("  Computing intensity-based weights for train_ds...")
        target_means = np.asarray([train_ds.sample_target_intensity(i) for i in range(len(train_ds))], dtype=np.float32)
        global_mean = float(target_means.mean()) + 1e-8
        weights = np.array([(tm / global_mean) ** 2 for tm in target_means], dtype=np.float32)
        weights = np.clip(weights, 0.1, 10.0)  # cap extremes

        sampler = WeightedRandomSampler(
            torch.from_numpy(weights).float(),
            num_samples=len(weights),
            replacement=True
        )

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg['training']['batch_size_train'],
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=cfg['training']['num_workers'],
        pin_memory=cfg['training'].get('pin_memory', True),
        drop_last=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg['training']['batch_size_val'],
        shuffle=False,
        num_workers=cfg['training']['num_workers'],
        pin_memory=cfg['training'].get('pin_memory', True),
    )
    return train_dl, val_dl


def _read_test_base_feature(cfg: dict, feat: str, start: int, end: int, t_in_cpm: int, t_in_met: int) -> np.ndarray:
    arr = np.load(os.path.join(cfg['paths']['data'], 'test_in', f'{feat}.npy'), mmap_mode='r')[start:end]
    arr = np.asarray(arr, dtype=np.float32)
    if feat == 'cpm25':
        pad = np.zeros((end - start, t_in_met - t_in_cpm, arr.shape[-2], arr.shape[-1]), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=1)
    return arr


def build_test_input(cfg, bounds: dict, start: int = 0, end: int | None = None) -> np.ndarray:
    _prep_cfg(cfg)

    features = cfg['features']['base']
    input_feats = cfg['features']['input']
    t_in_cpm = cfg['time']['t_in_cpm']
    t_in_met = cfg['time']['t_in_met']
    n_test = cfg['data']['test_samples']

    if end is None:
        end = n_test
    if not (0 <= start < end <= n_test):
        raise ValueError(f"Invalid test slice [{start}, {end}) for n_test={n_test}")

    bs = end - start
    grid_stats = cfg.get('_runtime', {}).get('grid_stats') or load_grid_stats(cfg)
    cfg.setdefault('_runtime', {})['grid_stats'] = grid_stats

    base_raw = {
        feat: _read_test_base_feature(cfg, feat, start, end, t_in_cpm=t_in_cpm, t_in_met=t_in_met)
        for feat in features
    }
    base = {
        feat: normalize_feature(arr, feat, bounds, cfg, grid_stats)
        for feat, arr in base_raw.items()
    }

    _, T, H, W = next(iter(base_raw.values())).shape

    # time support: if present and has same chunked temporal shape, use it; else neutral defaults
    time_path = os.path.join(cfg['paths']['data'], 'test_in', 'time.npy')
    time_arr = np.load(time_path) if os.path.exists(time_path) else None

    # Derived and masks from base raw
    derived = compute_derived_features({k: v for k, v in base_raw.items() if k in ('pblh', 'u10', 'v10')})
    masks = compute_sparse_masks(base_raw, cfg, default_shape=(bs, T, H, W))

    aux = {}
    aux_names = set(cfg.get('features', {}).get('aux', []))

    if any(name in aux_names for name in ('lat', 'lon')):
        static = load_static_maps(cfg)
        if 'lat' in aux_names:
            aux['lat'] = np.broadcast_to(static['lat'][None, None], (bs, T, H, W)).copy()
        if 'lon' in aux_names:
            aux['lon'] = np.broadcast_to(static['lon'][None, None], (bs, T, H, W)).copy()

    if 'land_mask' in aux_names:
        land = load_land_mask(cfg, H, W)
        aux['land_mask'] = np.broadcast_to(land[None, None], (bs, T, H, W)).copy()

    # cyclic clocks
    if time_arr is None:
        hours = np.zeros(T, dtype=np.float32)
        doys = np.ones(T, dtype=np.float32)
        hour_sin = _broadcast_scalar_series(np.sin(2 * np.pi * hours / 24.0), H, W)
        hour_cos = _broadcast_scalar_series(np.cos(2 * np.pi * hours / 24.0), H, W)
        doy_sin = _broadcast_scalar_series(np.sin(2 * np.pi * doys / 366.0), H, W)
        doy_cos = _broadcast_scalar_series(np.cos(2 * np.pi * doys / 366.0), H, W)
        if 'hour_sin' in aux_names:
            aux['hour_sin'] = np.broadcast_to(hour_sin[None], (bs, T, H, W)).copy()
        if 'hour_cos' in aux_names:
            aux['hour_cos'] = np.broadcast_to(hour_cos[None], (bs, T, H, W)).copy()
        if 'doy_sin' in aux_names:
            aux['doy_sin'] = np.broadcast_to(doy_sin[None], (bs, T, H, W)).copy()
        if 'doy_cos' in aux_names:
            aux['doy_cos'] = np.broadcast_to(doy_cos[None], (bs, T, H, W)).copy()
    else:
        # best-effort parse for common formats
        if time_arr.ndim == 1 and len(time_arr) == T:
            h, d = _parse_time_strings(time_arr)
            if 'hour_sin' in aux_names:
                aux['hour_sin'] = np.broadcast_to(_broadcast_scalar_series(np.sin(2 * np.pi * h / 24.0), H, W)[None], (bs, T, H, W)).copy()
            if 'hour_cos' in aux_names:
                aux['hour_cos'] = np.broadcast_to(_broadcast_scalar_series(np.cos(2 * np.pi * h / 24.0), H, W)[None], (bs, T, H, W)).copy()
            if 'doy_sin' in aux_names:
                aux['doy_sin'] = np.broadcast_to(_broadcast_scalar_series(np.sin(2 * np.pi * d / 366.0), H, W)[None], (bs, T, H, W)).copy()
            if 'doy_cos' in aux_names:
                aux['doy_cos'] = np.broadcast_to(_broadcast_scalar_series(np.cos(2 * np.pi * d / 366.0), H, W)[None], (bs, T, H, W)).copy()
        else:
            # fallback to neutral clocks if shape is unknown
            if 'hour_sin' in aux_names:
                aux['hour_sin'] = np.zeros((bs, T, H, W), dtype=np.float32)
            if 'hour_cos' in aux_names:
                aux['hour_cos'] = np.ones((bs, T, H, W), dtype=np.float32)
            if 'doy_sin' in aux_names:
                aux['doy_sin'] = np.zeros((bs, T, H, W), dtype=np.float32)
            if 'doy_cos' in aux_names:
                aux['doy_cos'] = np.ones((bs, T, H, W), dtype=np.float32)

    for name in _derived_feature_names(cfg):
        if name in aux_names and name in derived:
            aux[name] = normalize_feature(derived[name], name, bounds, cfg, grid_stats)

    for name, arr in masks.items():
        if name in aux_names:
            aux[name] = arr.astype(np.float32)

    arrays = []
    for feat in input_feats:
        if feat in base:
            arrays.append(base[feat])
        elif feat in aux:
            arrays.append(aux[feat])
        else:
            raise KeyError(f"Input feature '{feat}' missing in both base and aux at test time")

    stacked = np.stack(arrays, axis=1).astype(np.float32)

    # Dynamic feature toggling
    time_limits = {
        str(k): int(v)
        for k, v in cfg.get('preprocessing', {}).get('feature_time_limits', {}).items()
    }
    feat_to_idx = {name: idx for idx, name in enumerate(input_feats)}
    for feat_name, limit in time_limits.items():
        idx = feat_to_idx.get(feat_name)
        if idx is not None and 0 <= limit < stacked.shape[2]:
            stacked[:, idx, limit:] = 0.0

    return stacked


def compute_stats(cfg, months):
    raise DeprecationWarning('compute_stats is removed. Use build_grid_stats(...) or load_minmax_bounds(cfg).')


def save_stats(stats, path):
    np.save(path, stats)


def load_stats(path):
    return np.load(path, allow_pickle=True).item()


def make_dataloaders(cfg, tensor, bounds):
    """
    Baseline-compatible window-level random split dataloaders for PINO notebooks.
    This complements `get_dataloaders` and keeps Ronit's advanced pipeline intact.
    """
    batch_size = cfg['training']['batch_size_train']
    val_batch_size = cfg['training'].get('batch_size_val', batch_size)
    window_size = cfg['time']['t_in']
    t_out = cfg['time']['t_out']
    val_split = cfg['data'].get('val_split', 0.1)

    full_ds = SlidingWindowTensorDataset(tensor, window_size, t_out=t_out, target_channel=0)
    n_total = len(full_ds)
    n_val = max(1, int(n_total * val_split))
    n_train = n_total - n_val
    if n_train <= 0:
        raise ValueError(f"Invalid val_split={val_split}; not enough windows for training.")

    seed = cfg.get('training', {}).get('seed', 42)
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=generator)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=val_batch_size, shuffle=False)
    return train_dl, val_dl
