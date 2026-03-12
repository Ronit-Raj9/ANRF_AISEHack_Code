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
"""

import os
import gc
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


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


def denormalize(arr: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
    """Inverse of min-max normalization."""
    return arr.astype(np.float32) * (fmax - fmin) + fmin


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
    data = {}
    for feat in features:
        path = os.path.join(data_dir, 'raw', month, f'{feat}.npy')
        arr  = np.load(path)                           # (T, H, W)
        data[feat] = normalize(arr, *bounds[feat])

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
    for feat in features:
        path = os.path.join(data_dir, 'test_in', f'{feat}.npy')
        arr = np.load(path, mmap_mode='r')[start:end]
        if feat == 'cpm25':
            pad = np.zeros((bs, t_in_met - t_in_cpm, 140, 124), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=1)
        base[feat] = normalize(arr, *bounds[feat])

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

