import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split

# Sliding window dataset for time series
class SlidingWindowTensorDataset(Dataset):
    def __init__(self, tensor, window_size, t_out, target_channel=0):
        # tensor: (N, C, T, H, W)
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
        xb = self.tensor[n, :, w:w+self.window_size, :, :]  # (C, window_size, H, W)
        y_start = w + self.window_size
        y_end = y_start + self.t_out
        yb = self.tensor[n, self.target_channel, y_start:y_end, :, :]  # (T_out, H, W)
        xb = torch.tensor(xb, dtype=torch.float32)
        yb = torch.tensor(yb, dtype=torch.float32).permute(1, 2, 0)  # (H, W, T_out)
        return xb, yb

"""
Data loading, normalization, sample construction, and PyTorch Dataset.
 
Design notes:
- Normalization: official min-max from feat_min_max.mat → clip to [0, 1].
    This ensures train/test consistency and makes the persistence RMSE gate
    directly interpretable (threshold = 0.0208 in normalized space).
- Preprocessing: append static geography (`lat`, `lon`) and calendar signals
        (`hour_sin`, `hour_cos`, `doy_sin`, `doy_cos`) because EDA showed strong
        spatial stationarity, diurnal solar forcing, and seasonal shift.
- Lazy Dataset: month arrays are pre-normalized and kept in RAM; windows are
    sliced on-the-fly in __getitem__.  This avoids building a ~27 GB materialized
    sample array while still allowing random-access DataLoader batching.
- Temporal blocking: entire val_month is held out; no window straddles the
    train/val boundary because they are loaded from separate .npy files.

import torch
from torch.utils.data import TensorDataset, DataLoader
"""

import os
import gc
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

def add_physical_features(tensor, features_dict=None, lat_long_path='lat_long.npy', cfg=None):
    """
        Add engineered physics channels for PINO in a memory-efficient way:
      wind_speed, wind_dir, RH, hour_sin, hour_cos, month_sin, month_cos,
            hotspot_clim, lag_1, lag_3, lag_6

    Memory strategy: pre-allocate a single float32 output array and fill each
    engineered channel one at a time (rather than materialising a large stack).
    This cuts peak RAM from ~3x tensor size down to ~1.75x tensor size.
    """
    import numpy as np
    import gc

    # ── Normalise dtype to float32 immediately ───────────────────────────────
    tensor = np.asarray(tensor, dtype=np.float32)
    N, C, T, H, W = tensor.shape
    N_ENG = 11

    # Pre-allocate output: (N, C+N_ENG, T, H, W) float32
    out = np.empty((N, C + N_ENG, T, H, W), dtype=np.float32)
    out[:, :C] = tensor          # copy base channels (no extra alloc beyond out)
    ch = C                       # next free channel index

    # ── Wind speed ──────────────────────────────────────────────────────────
    u10 = tensor[:, features_dict['u10'], :, :, :]   # (N,T,H,W) view, float32
    v10 = tensor[:, features_dict['v10'], :, :, :]
    out[:, ch] = np.sqrt(u10 ** 2 + v10 ** 2)
    ch += 1

    # ── Wind direction ───────────────────────────────────────────────────────
    out[:, ch] = np.arctan2(v10, u10)
    ch += 1
    del u10, v10
    gc.collect()

    # ── Relative humidity ────────────────────────────────────────────────────
    t2 = tensor[:, features_dict['t2'], :, :, :].copy().astype(np.float32)
    q2 = tensor[:, features_dict['q2'], :, :, :].copy().astype(np.float32)
    denom_safe = np.where(np.abs(t2 - 29.65) < 1e-3,
                          np.sign(t2 - 29.65 + np.float32(1e-3)) * np.float32(1e-3),
                          t2 - np.float32(29.65)).astype(np.float32)
    exponent = np.clip(
        np.float32(17.67) * (t2 - np.float32(273.15)) / denom_safe,
        np.float32(-87.0),
        np.float32(87.0),
    )
    with np.errstate(over='ignore', invalid='ignore'):
        rh = q2 / (np.float32(0.622) + np.float32(0.378) * q2 + np.float32(1e-8)) * np.exp(exponent)
    np.nan_to_num(rh, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    out[:, ch] = np.clip(rh, np.float32(0.0), np.float32(1.5))
    ch += 1
    del t2, q2, denom_safe, exponent, rh
    gc.collect()

    # ── Temporal cycles (float32, broadcast-on-assign, no materialisation) ──
    hour = (np.arange(T, dtype=np.float32) % 24)
    for vals in [
        np.sin(np.float32(2 * np.pi) * hour / np.float32(24)),
        np.cos(np.float32(2 * np.pi) * hour / np.float32(24)),
        np.full(T, np.sin(np.float32(2 * np.pi / 12)), dtype=np.float32),  # placeholder month
        np.full(T, np.cos(np.float32(2 * np.pi / 12)), dtype=np.float32),
    ]:
        # Assign (T,) → (N, T, H, W) via numpy broadcasting during assignment
        out[:, ch] = vals.reshape(1, T, 1, 1)
        ch += 1
    del hour
    gc.collect()

    # ── Static hotspot climatology prior ────────────────────────────────────
    if cfg is not None and cfg.get('data', {}).get('use_hotspot_climatology_channel', True):
        prior_map, _ = get_hotspot_maps(cfg)
    else:
        # fallback: derive prior from already-loaded cpm25 channel
        cpm = tensor[:, features_dict['cpm25'], :, :, :]
        prior_map = np.log1p(np.clip(np.mean(cpm, axis=(0, 1)), 0.0, None)).astype(np.float32)
        prior_map = (prior_map - prior_map.mean()) / (prior_map.std() + np.float32(1e-6))
    out[:, ch] = prior_map.reshape(1, 1, H, W)
    ch += 1
    del prior_map
    gc.collect()

    # ── PM2.5 lags ───────────────────────────────────────────────────────────
    cpm25 = tensor[:, features_dict['cpm25'], :, :, :].copy()
    out[:, ch] = np.roll(cpm25, 1, axis=1);  ch += 1   # axis=1 is T in (N,T,H,W)
    out[:, ch] = np.roll(cpm25, 3, axis=1);  ch += 1
    out[:, ch] = np.roll(cpm25, 6, axis=1);  ch += 1
    del cpm25
    gc.collect()

    assert ch == C + N_ENG, f"Channel count mismatch: expected {C + N_ENG}, got {ch}"

    # ── NaN/Inf guard (in-place, no extra copy) ──────────────────────────────
    n_bad = int(np.sum(~np.isfinite(out)))
    if n_bad > 0:
        import warnings
        warnings.warn(f"add_physical_features: replacing {n_bad} NaN/Inf values with 0")
        np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    return out

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


# ─────────────────────────────────────────────
# Official Min-Max Normalization Bounds
# ─────────────────────────────────────────────

def load_minmax_bounds(cfg) -> dict:
    """
    Load per-feature normalization bounds from the official feat_min_max.mat.

    Returns
    -------
    bounds : dict
        {feat: (fmin: float, fmax: float)}
        The mat file stores scalars under keys ``{feat}_min`` / ``{feat}_max``.
    """
    path = cfg['paths']['min_max']
    features = cfg['features']['all']

    # Local machines can have binary mismatches between NumPy and SciPy.
    # The official bounds were already verified in EDA, so use the exact
    # fallback values locally for clean, dependency-light execution.
    if not cfg.get('runtime', {}).get('on_kaggle', False):
        return {feat: FALLBACK_OFFICIAL_BOUNDS[feat] for feat in features}

    sio = None
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
    """Min-max normalize to [0, 1], clip any out-of-range values."""
    rng = fmax - fmin
    if rng == 0:
        return np.zeros_like(arr, dtype=np.float32)
    normed = (arr.astype(np.float32) - fmin) / rng
    return np.clip(normed, 0.0, 1.0)


def denormalize(arr: np.ndarray, fmin: float, fmax: float, feat: str = 'cpm25') -> np.ndarray:
    """Inverse of preprocessing-aware normalization for a feature (default: cpm25)."""
    lo_t, hi_t = _transform_bounds(fmin, fmax, feat)
    x = arr.astype(np.float32) * (hi_t - lo_t) + lo_t

    if feat in MET_LOG1P_FEATURES:
        return np.expm1(x).astype(np.float32)

    if feat in WIND_SIGNED_LOG1P_FEATURES:
        return (np.sign(x) * np.expm1(np.abs(x))).astype(np.float32)

    if feat in EMIS_LOG_EPS_FEATURES:
        return (np.exp(x) - LOG_EPS).astype(np.float32)

    if feat in MASK_ONLY_FEATURES or feat.endswith('_mask'):
        return (x > 0.5).astype(np.float32)

    return x.astype(np.float32)


MET_LOG1P_FEATURES = {'cpm25', 't2', 'pblh', 'q2', 'swdown', 'psfc', 'rain'}
WIND_SIGNED_LOG1P_FEATURES = {'u10', 'v10'}
EMIS_LOG_EPS_FEATURES = {'PM25', 'NOx', 'NH3', 'NMVOC_e', 'SO2', 'bio'}
MASK_ONLY_FEATURES = {'NMVOC_finn'}
LOG_EPS = np.float32(1e-12)
DEC_STD_EPS = np.float32(1e-6)
_DECEMBER_STATS_CACHE: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
_HOTSPOT_MAP_CACHE: dict[tuple[str, tuple[str, ...], int], tuple[np.ndarray, np.ndarray]] = {}


def get_hotspot_maps(cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Build hotspot climatology prior and spatial loss-weight map from raw cpm25.

    Returns
    -------
    prior_z : (H, W) float32
        log1p(climatology) standardized to zero mean / unit std.
    weights : (H, W) float32
        log1p(climatology) normalized so global mean weight = 1.0.
    """
    data_dir = cfg['paths']['data']
    months = tuple(cfg.get('data', {}).get('train_months', cfg['data']['months']))
    t_in_cpm = int(cfg.get('time', {}).get('t_in_cpm', 10))
    cache_key = (data_dir, months, t_in_cpm)
    if cache_key in _HOTSPOT_MAP_CACHE:
        return _HOTSPOT_MAP_CACHE[cache_key]

    sum_map = None
    count = 0
    for month in months:
        cpm_path = os.path.join(data_dir, 'raw', month, 'cpm25.npy')
        if not os.path.exists(cpm_path):
            raise FileNotFoundError(f"Missing cpm25 file for hotspot climatology: {cpm_path}")
        arr = np.load(cpm_path).astype(np.float32)  # (T,H,W)
        if arr.ndim != 3:
            raise ValueError(f"Unexpected cpm25 shape in {cpm_path}: {arr.shape}")
        month_sum = np.sum(arr, axis=0)
        if sum_map is None:
            sum_map = month_sum
        else:
            sum_map += month_sum
        count += arr.shape[0]

    if sum_map is None or count <= 0:
        raise RuntimeError("Failed to build hotspot climatology map (empty training cpm25).")

    clim_raw = (sum_map / np.float32(count)).astype(np.float32)
    clim_log = np.log1p(np.clip(clim_raw, 0.0, None)).astype(np.float32)

    prior_mean = float(np.mean(clim_log))
    prior_std = float(np.std(clim_log))
    prior_z = ((clim_log - np.float32(prior_mean)) / np.float32(prior_std + 1e-6)).astype(np.float32)

    weights = (clim_log / np.float32(np.mean(clim_log) + 1e-6)).astype(np.float32)
    _HOTSPOT_MAP_CACHE[cache_key] = (prior_z, weights)
    return prior_z, weights


def _transform_feature_values(arr: np.ndarray, feat: str) -> np.ndarray:
    """Apply feature-specific numeric transforms before normalization."""
    x = np.asarray(arr, dtype=np.float32)

    if feat in WIND_SIGNED_LOG1P_FEATURES:
        return np.sign(x) * np.log1p(np.abs(x))

    if feat in MET_LOG1P_FEATURES:
        return np.log1p(np.clip(x, 0.0, None))

    if feat in EMIS_LOG_EPS_FEATURES:
        return np.log(np.clip(x, 0.0, None) + LOG_EPS)

    if feat in MASK_ONLY_FEATURES:
        return (x > 0).astype(np.float32)

    return x


def _transform_bounds(fmin: float, fmax: float, feat: str) -> tuple[float, float]:
    """Transform min-max bounds into the same numeric domain as feature values."""
    lo = np.float32(fmin)
    hi = np.float32(fmax)

    if feat in WIND_SIGNED_LOG1P_FEATURES:
        lo_t = np.sign(lo) * np.log1p(np.abs(lo))
        hi_t = np.sign(hi) * np.log1p(np.abs(hi))
    elif feat in MET_LOG1P_FEATURES:
        lo_t = np.log1p(np.clip(lo, 0.0, None))
        hi_t = np.log1p(np.clip(hi, 0.0, None))
    elif feat in EMIS_LOG_EPS_FEATURES:
        lo_t = np.log(np.clip(lo, 0.0, None) + LOG_EPS)
        hi_t = np.log(np.clip(hi, 0.0, None) + LOG_EPS)
    elif feat in MASK_ONLY_FEATURES or feat.endswith('_mask'):
        return 0.0, 1.0
    else:
        lo_t, hi_t = lo, hi

    return float(min(lo_t, hi_t)), float(max(lo_t, hi_t))


def preprocess_feature(arr: np.ndarray, feat: str, bounds: dict, month: str | None = None) -> np.ndarray:
    """
    Feature-specific preprocessing + normalization.

    - Meteo/cpm25 scalar fields: log1p domain (or signed-log1p for winds)
    - Emissions: log(x+1e-12) in float32-safe range
    - NMVOC_finn: binary mask only
    - PBLH: season-aware scaling via month-local min-max after transform
    """
    x = _transform_feature_values(arr, feat)

    if feat == 'pblh' and month is not None:
        lo = float(np.nanmin(x))
        hi = float(np.nanmax(x))
        if np.isclose(hi - lo, 0.0):
            return np.zeros_like(x, dtype=np.float32)
        return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)

    fmin, fmax = _transform_bounds(*bounds[feat], feat)
    return normalize(x, fmin, fmax)


def _get_december_grid_stats(cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (mu, sigma) over time for DEC_16 cpm25 in log1p domain, cached per data root."""
    data_dir = cfg['paths']['data']
    dec_month = cfg.get('data', {}).get('december_month', 'DEC_16')
    cache_key = (data_dir, dec_month)
    if cache_key in _DECEMBER_STATS_CACHE:
        return _DECEMBER_STATS_CACHE[cache_key]

    dec_path = os.path.join(data_dir, 'raw', dec_month, 'cpm25.npy')
    if not os.path.exists(dec_path):
        raise FileNotFoundError(f"December cpm25 file not found for grid-wise sigma stats: {dec_path}")

    dec_raw = np.load(dec_path).astype(np.float32)  # (T,H,W)
    dec_log = _transform_feature_values(dec_raw, 'cpm25')
    mu = np.mean(dec_log, axis=0).astype(np.float32)
    sigma = np.std(dec_log, axis=0).astype(np.float32)
    sigma = np.maximum(sigma, DEC_STD_EPS)
    _DECEMBER_STATS_CACHE[cache_key] = (mu, sigma)
    return mu, sigma


def _broadcast_scalar_series(values: np.ndarray, H: int, W: int) -> np.ndarray:
    """Broadcast a `(T,)` time series to `(T, H, W)` float32 array."""
    return np.broadcast_to(values[:, None, None].astype(np.float32), (len(values), H, W)).copy()


def load_static_maps(cfg) -> dict:
    """Load and normalize static `lat`/`lon` maps to [0, 1]."""
    ll_path = os.path.join(cfg['paths']['data'], 'raw', 'lat_long.npy')
    ll = np.load(ll_path).astype(np.float32)
    lat = ll[:, :, 0]
    lon = ll[:, :, 1]
    lat = (lat - lat.min()) / (lat.max() - lat.min() + 1e-8)
    lon = (lon - lon.min()) / (lon.max() - lon.min() + 1e-8)
    return {'lat': lat.astype(np.float32), 'lon': lon.astype(np.float32)}


def _parse_time_strings(time_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse hackathon time strings into hour-of-day and day-of-year arrays.

    Returns
    -------
    hours : (T,) int32 in [0, 23]
    doys  : (T,) int32 in [1, 366]
    """
    ts = np.asarray(time_arr).astype(str)
    hours = np.array([int(t[11:13]) for t in ts], dtype=np.int32)

    # Use numpy datetime for day-of-year to keep leap-year handling correct.
    dt = ts.astype('datetime64[h]')
    year_start = dt.astype('datetime64[Y]')
    doys = (dt.astype('datetime64[D]') - year_start).astype(np.int32) + 1
    return hours, doys


def build_aux_feature_maps(cfg, time_arr: np.ndarray | None, T: int, H: int, W: int) -> dict:
    """
    Build auxiliary feature maps used by the research-backed model.

    Features:
    - `lat`, `lon`: static geography in [0, 1]
    - `hour_sin`, `hour_cos`: diurnal cycle
    - `doy_sin`, `doy_cos`: seasonal cycle
    """
    aux_names = cfg['features'].get('aux', [])
    static = load_static_maps(cfg)
    aux = {}

    # Static maps
    if 'lat' in aux_names:
        aux['lat'] = np.broadcast_to(static['lat'][None], (T, H, W)).copy()
    if 'lon' in aux_names:
        aux['lon'] = np.broadcast_to(static['lon'][None], (T, H, W)).copy()

    # Calendar maps
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


# ─────────────────────────────────────────────
# Month-level Data Loading
# ─────────────────────────────────────────────

def _load_month(cfg, month: str, bounds: dict) -> dict:
    """
    Load and normalize all features for one month.

    Returns
    -------
    data : dict  {feat: np.ndarray of shape (T, H, W), dtype float32}
    """
    features = cfg['features']['base']
    data_dir = cfg['paths']['data']
    data = {'__month__': month}
    raw_cache = {}
    for feat in features:
        if feat == 'rain_mask':
            if 'rain' not in raw_cache:
                rain_path = os.path.join(data_dir, 'raw', month, 'rain.npy')
                print(f"Trying to load: {rain_path}")
                if not os.path.exists(rain_path):
                    print(f"File missing: {rain_path}")
                    raise FileNotFoundError(f"Missing file: {rain_path}")
                raw_cache['rain'] = np.load(rain_path)
            data[feat] = (raw_cache['rain'] > 0).astype(np.float32)
            continue

        path = os.path.join(data_dir, 'raw', month, f'{feat}.npy')
        print(f"Trying to load: {path}")
        if not os.path.exists(path):
            print(f"File missing: {path}")
            raise FileNotFoundError(f"Missing file: {path}")
        try:
            arr = np.load(path)  # (T, H, W)
        except Exception as e:
            print(f"Failed to load {path}: {e}")
            raise
        raw_cache[feat] = arr
        data[feat] = preprocess_feature(arr, feat, bounds, month=month)
        if feat == 'cpm25':
            data['__cpm25_log1p__'] = _transform_feature_values(arr, 'cpm25').astype(np.float32)

    if month.upper().startswith('DEC') and cfg.get('data', {}).get('december_grid_sigma_norm', True):
        mu, sigma = _get_december_grid_stats(cfg)
        data['__dec_mu__'] = mu
        data['__dec_sigma__'] = sigma

    for emis_feat in ('PM25', 'NOx', 'NH3', 'NMVOC_e'):
        if emis_feat in data:
            std = float(np.std(data[emis_feat]))
            if std <= 0.0:
                raise RuntimeError(f"Emission feature {emis_feat} collapsed to zero variance after preprocessing.")

    if 'cpm25' not in data:
        print(f"cpm25 missing in month {month}, skipping.")
        return {}
    T, H, W = data['cpm25'].shape
    time_path = os.path.join(data_dir, 'raw', month, 'time.npy')
    time_arr = np.load(time_path) if os.path.exists(time_path) else None
    data.update(build_aux_feature_maps(cfg, time_arr, T, H, W))
    return data


def load_all_months(cfg, months: list, bounds: dict) -> list:
    """
    Load and normalize multiple months.  Returns a list of dicts (one per month).
    Memory: ~750 MB per month × 16 features (140×124 grid, float32).
    """
    all_data = []
    for m in months:
        print(f"  Loading {m} ...", end=" ", flush=True)
        all_data.append(_load_month(cfg, m, bounds))
        print("OK")
        gc.collect()
    return all_data


# ─────────────────────────────────────────────
# PyTorch Dataset — Lazy Sliding Windows

def make_dataloaders(cfg, tensor, bounds):
    batch_size = cfg['training']['batch_size_train']
    val_batch_size = cfg['training'].get('batch_size_val', batch_size)
    window_size = cfg['time']['t_in']  # Use model input window size from config
    t_out = cfg['time']['t_out']
    val_split = cfg['data'].get('val_split', 0.1)

    # Build windows from the full tensor first, then split window indices.
    # This uses all available months/data instead of dropping entire months
    # based on a month-level split.
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
# ─────────────────────────────────────────────

class PM25Dataset(Dataset):
    """
    Lazy sliding-window PM2.5 dataset.

    Window logic (per EDA):
    - Input  : met/emis[t : t+T_IN_MET]  — 26 hrs (10 past + 16 NWP forecast)
               cpm25  [t : t+T_IN_CPM]   — 10 hrs known; hours 10-25 → 0.0
    - Target : cpm25  [t+T_IN_CPM :
                       t+T_IN_CPM+T_OUT]  — next 16 hrs (normalized)
    Both input and target are in normalized [0, 1] space.

    Output shapes
    -------------
    x : (C=input_channels, T_in=26, H=140, W=124)  float32
    y : (H=140, W=124, T_out=16)        float32
    """

    def __init__(self, months_data: list, cfg: dict, stride: int):
        self.data     = months_data   # list of {feat: (T, H, W) arrays}
        self.feats    = cfg['features']['input']
        self.t_in     = cfg['time']['t_in_met']    # 26
        self.t_cpm    = cfg['time']['t_in_cpm']    # 10
        self.t_out    = cfg['time']['t_out']        # 16
        self.cpm_idx  = 0                           # cpm25 is always index 0
        self.december_sigma_norm = bool(cfg.get('data', {}).get('december_grid_sigma_norm', True))

        # Build (month_idx, start_t) index
        self.index = []
        window = self.t_cpm + self.t_out            # = 26 total consumed
        for m_idx, mdata in enumerate(months_data):
            T = mdata[self.feats[0]].shape[0]
            for t in range(0, T - window + 1, stride):
                self.index.append((m_idx, t))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        m_idx, t = self.index[idx]
        mdata     = self.data[m_idx]
        t_in      = self.t_in
        t_cpm     = self.t_cpm
        t_out     = self.t_out

        # ── Input tensor (C, T_in, H, W) ──
        channels = []
        for feat in self.feats:
            chunk = mdata[feat][t : t + t_in].copy()  # (T_in, H, W)
            if feat == 'cpm25':
                month_name = str(mdata.get('__month__', '')).upper()
                if self.december_sigma_norm and month_name.startswith('DEC') and '__cpm25_log1p__' in mdata:
                    mu = mdata.get('__dec_mu__', None)
                    sigma = mdata.get('__dec_sigma__', None)
                    if mu is not None and sigma is not None:
                        cpm_log_chunk = mdata['__cpm25_log1p__'][t : t + t_in].copy()
                        chunk = ((cpm_log_chunk - mu[None, :, :]) / sigma[None, :, :]).astype(np.float32)
                chunk[t_cpm:] = 0.0   # mask future cpm25 (not available at test time)
            channels.append(chunk)
        x = torch.from_numpy(np.stack(channels, axis=0))  # (C, T, H, W)

        # ── Target (H, W, T_out) — normalized cpm25 ──
        y_arr = mdata['cpm25'][t + t_cpm : t + t_cpm + t_out]  # (T_out, H, W)
        y = torch.from_numpy(y_arr).permute(1, 2, 0)            # (H, W, T_out)

        return x, y


# ─────────────────────────────────────────────
# DataLoaders
# ─────────────────────────────────────────────

def get_dataloaders(cfg, train_data: list, val_data: list, bounds: dict):
    """
    Build train + validation DataLoaders from pre-loaded month data.

    Parameters
    ----------
    train_data, val_data : list of month dicts (from load_all_months)
    bounds               : normalization bounds dict (for reference / denorm)

    Returns
    -------
    train_dl, val_dl, bounds
    """
    train_ds = PM25Dataset(train_data, cfg, stride=cfg['time']['stride_train'])
    val_ds   = PM25Dataset(val_data,   cfg, stride=cfg['time']['stride_val'])

    print(f"  Train samples: {len(train_ds):,}  |  Val samples: {len(val_ds):,}")

    train_dl = DataLoader(
        train_ds,
        batch_size  = cfg['training']['batch_size_train'],
        shuffle     = True,
        num_workers = cfg['training']['num_workers'],
        pin_memory  = cfg['training'].get('pin_memory', True),
        drop_last   = True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size  = cfg['training']['batch_size_val'],
        shuffle     = False,
        num_workers = cfg['training']['num_workers'],
        pin_memory  = cfg['training'].get('pin_memory', True),
    )
    return train_dl, val_dl


def build_test_input(cfg, bounds: dict, start: int = 0, end: int | None = None) -> np.ndarray:
    """
    Build normalized test tensor with auxiliary channels.

    Parameters
    ----------
    start, end : int
        Half-open index range [start, end). This enables memory-safe chunked
        inference. If `end` is None, it defaults to `n_test`.

    Returns
    -------
    X_test : (end-start, C=input_channels, T=26, H, W) float32
    """
    features  = cfg['features']['base']
    input_feats = cfg['features']['input']
    data_dir   = cfg['paths']['data']
    t_in_cpm   = cfg['time']['t_in_cpm']
    t_in_met   = cfg['time']['t_in_met']
    n_test     = cfg['data']['test_samples']
    if end is None:
        end = n_test
    if not (0 <= start < end <= n_test):
        raise ValueError(f"Invalid test slice [{start}, {end}) for n_test={n_test}")
    bs = end - start

    base = {}
    # NOTE: december_grid_sigma_norm is intentionally NOT applied here.
    # Training used SlidingWindowTensorDataset which stores cpm25 with plain
    # min-max log1p normalization for ALL months (including December).
    # Applying December grid-wise sigma at test time would create a train/test
    # mismatch and produce garbage predictions.

    for feat in features:
        if feat == 'rain_mask':
            rain_path = os.path.join(data_dir, 'test_in', 'rain.npy')
            rain_arr = np.load(rain_path, mmap_mode='r')[start:end]
            base[feat] = (rain_arr > 0).astype(np.float32)
            continue

        path = os.path.join(data_dir, 'test_in', f'{feat}.npy')
        arr = np.load(path, mmap_mode='r')[start:end]
        if feat == 'cpm25':
            # Standard min-max log1p normalization — matches SlidingWindowTensorDataset
            cpm_proc = preprocess_feature(arr, feat, bounds, month=None)
            # Fill unknown future cpm25 hours with persistence (repeat last known frame).
            # Training saw actual future cpm25 values; persistence is the closest
            # in-distribution approximation at test time.
            n_fill = t_in_met - t_in_cpm  # = 16
            last_known = cpm_proc[:, -1:, :, :]  # (bs, 1, H, W)
            fill = np.repeat(last_known, n_fill, axis=1)  # (bs, 16, H, W)
            base[feat] = np.concatenate([cpm_proc, fill], axis=1)  # (bs, 26, H, W)
            continue
        base[feat] = preprocess_feature(arr, feat, bounds, month=None)

    sample_shape = next(iter(base.values())).shape
    N, T, H, W = sample_shape

    time_path = os.path.join(data_dir, 'test_in', 'time.npy')
    time_arr = np.load(time_path) if os.path.exists(time_path) else None
    if time_arr is None and cfg['features'].get('aux'):
        print("Warning: test_in/time.npy not found. Calendar features fall back to neutral defaults at inference.")
    aux_single = build_aux_feature_maps(cfg, time_arr, T, H, W)
    aux = {k: np.broadcast_to(v[None], (bs, T, H, W)).copy() for k, v in aux_single.items()}

    arrays = []
    for feat in input_feats:
        arrays.append(base[feat] if feat in base else aux[feat])
    return np.stack(arrays, axis=1).astype(np.float32)


# ─────────────────────────────────────────────
# Legacy helpers (kept for compatibility)
# ─────────────────────────────────────────────

def compute_stats(cfg, months):
    """Deprecated: use load_minmax_bounds instead."""
    raise DeprecationWarning(
        "compute_stats is removed. Use load_minmax_bounds(cfg) for official "
        "min-max normalization from feat_min_max.mat."
    )


def save_stats(stats, path):
    np.save(path, stats)


def load_stats(path):
    return np.load(path, allow_pickle=True).item()

