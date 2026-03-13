"""
Data loading, grid-wise preprocessing, and memory-mapped sliding-window datasets.

Primary mode for the rank-push pipeline:
- per-feature, per-grid log-standardization over 2016 data
- sign-preserving log transform for signed variables such as `u10` / `v10`
- memory-mapped month arrays to stay within Kaggle / local RAM constraints
- auxiliary diurnal channels (`hour_sin`, `hour_cos`) baked into the dataset
"""

import gc
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


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

SIGNED_FEATURES = {'u10', 'v10'}


def load_minmax_bounds(cfg) -> dict:
    """Load official fallback bounds for compatibility and inverse transforms."""
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
    return np.clip(normed, 0.0, 1.0)


def denormalize(arr: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
    return arr.astype(np.float32) * (fmax - fmin) + fmin


def _grid_scaler_enabled(cfg: dict) -> bool:
    return str(cfg.get('preprocessing', {}).get('normalization', 'grid_log_standardize')).lower() == 'grid_log_standardize'


def _use_mmap(cfg: dict) -> bool:
    return bool(cfg.get('preprocessing', {}).get('use_mmap', True))


def _stats_key(feat: str, suffix: str) -> str:
    return f'{feat}__{suffix}'


def _feature_is_signed(feat: str, bounds: dict | None = None) -> bool:
    if feat in SIGNED_FEATURES:
        return True
    if bounds is not None and feat in bounds:
        return float(bounds[feat][0]) < 0.0
    return False


def _signed_log1p(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return np.sign(x) * np.log1p(np.abs(x))


def _inverse_signed_log1p(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return np.sign(x) * np.expm1(np.abs(x))


def transform_feature(arr: np.ndarray, feat: str, cfg: dict, bounds: dict | None = None) -> np.ndarray:
    """Apply the research pipeline transform before standardization."""
    x = arr.astype(np.float32)
    if _grid_scaler_enabled(cfg):
        if _feature_is_signed(feat, bounds) and bool(cfg.get('preprocessing', {}).get('signed_log_for_negative', True)):
            return _signed_log1p(x)
        return np.log1p(np.maximum(x, 0.0))

    if feat == 'cpm25' and bool(cfg.get('preprocessing', {}).get('cpm25_log1p', False)):
        return np.log1p(np.maximum(x, 0.0))
    return x


def inverse_transform_feature(arr: np.ndarray, feat: str, cfg: dict, bounds: dict | None = None) -> np.ndarray:
    """Inverse of `transform_feature` for physical-space metrics/inference."""
    x = arr.astype(np.float32)
    if _grid_scaler_enabled(cfg):
        if _feature_is_signed(feat, bounds) and bool(cfg.get('preprocessing', {}).get('signed_log_for_negative', True)):
            return _inverse_signed_log1p(x)
        return np.expm1(x)

    if feat == 'cpm25' and bool(cfg.get('preprocessing', {}).get('cpm25_log1p', False)):
        return np.expm1(x)
    return x


def build_grid_stats(cfg: dict, bounds: dict, months: list[str] | None = None, force: bool = False) -> dict:
    """Compute and persist per-feature per-grid mean/std maps over transformed 2016 data."""
    path = cfg['paths']['grid_stats']
    if os.path.exists(path) and not force:
        return load_grid_stats(cfg)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    data_root = cfg['paths']['data']
    months = list(months or cfg['data']['months'])
    features = cfg['features']['base']
    eps = float(cfg.get('preprocessing', {}).get('grid_stats_eps', 1.0e-6))
    chunk_size = int(cfg.get('preprocessing', {}).get('grid_chunk_size', 48))
    stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    save_dict = {}

    print(f"Building grid scaler → {path}")
    for feat in features:
        running_sum = None
        running_sumsq = None
        count = 0
        for month in months:
            arr_path = os.path.join(data_root, 'raw', month, f'{feat}.npy')
            arr = np.load(arr_path, mmap_mode='r' if _use_mmap(cfg) else None)
            t_size = arr.shape[0]
            for start in range(0, t_size, chunk_size):
                chunk = np.asarray(arr[start:start + chunk_size], dtype=np.float32)
                chunk = transform_feature(chunk, feat, cfg, bounds)
                if running_sum is None:
                    running_sum = np.zeros(chunk.shape[1:], dtype=np.float64)
                    running_sumsq = np.zeros(chunk.shape[1:], dtype=np.float64)
                running_sum += chunk.sum(axis=0, dtype=np.float64)
                running_sumsq += np.square(chunk, dtype=np.float32).sum(axis=0, dtype=np.float64)
                count += chunk.shape[0]

        mean = (running_sum / max(count, 1)).astype(np.float32)
        var = (running_sumsq / max(count, 1)) - np.square(mean, dtype=np.float32)
        std = np.sqrt(np.maximum(var, eps)).astype(np.float32)
        stats[feat] = (mean, std)
        save_dict[_stats_key(feat, 'mean')] = mean
        save_dict[_stats_key(feat, 'std')] = std
        print(
            f"  {feat:8s} mean∈[{mean.min():.3f}, {mean.max():.3f}] "
            f"std∈[{std.min():.3f}, {std.max():.3f}]"
        )

    np.savez_compressed(path, **save_dict)
    return stats


def load_grid_stats(cfg: dict) -> dict:
    """Load per-feature per-grid standardization maps from disk."""
    path = cfg['paths']['grid_stats']
    if not os.path.exists(path):
        if bool(cfg.get('preprocessing', {}).get('auto_build_grid_stats', True)):
            bounds = load_minmax_bounds(cfg)
            return build_grid_stats(cfg, bounds=bounds, force=False)
        raise FileNotFoundError(f"Grid scaler file not found: {path}")

    npz = np.load(path)
    stats = {}
    for feat in cfg['features']['base']:
        stats[feat] = (
            np.asarray(npz[_stats_key(feat, 'mean')], dtype=np.float32),
            np.asarray(npz[_stats_key(feat, 'std')], dtype=np.float32),
        )
    return stats


def describe_grid_stats(grid_stats: dict, features: list[str] | None = None) -> None:
    """Print a compact scaler summary for notebooks."""
    features = features or list(grid_stats)
    print(f"\n{'Feature':10s} {'mean[min,max]':>28s} {'std[min,max]':>28s}")
    print('─' * 72)
    for feat in features:
        mean, std = grid_stats[feat]
        print(
            f"{feat:10s} "
            f"[{mean.min():8.3f}, {mean.max():8.3f}] "
            f"[{std.min():8.3f}, {std.max():8.3f}]"
        )
    print()


def normalize_feature(
    arr: np.ndarray,
    feat: str,
    bounds: dict,
    cfg: dict,
    grid_stats: dict | None = None,
) -> np.ndarray:
    """Feature-aware normalization in either grid-standardized or legacy min-max space."""
    if _grid_scaler_enabled(cfg):
        stats = grid_stats or cfg.get('_runtime', {}).get('grid_stats')
        if stats is None or feat not in stats:
            raise RuntimeError(f"Grid scaler statistics missing for feature: {feat}")
        mean, std = stats[feat]
        x = transform_feature(arr, feat, cfg, bounds)
        return ((x - mean) / std).astype(np.float32)

    if feat == 'cpm25' and bool(cfg.get('preprocessing', {}).get('cpm25_log1p', False)):
        fmin, fmax = bounds['cpm25']
        x = np.log1p(np.maximum(arr.astype(np.float32), 0.0))
        return normalize(x, np.log1p(max(fmin, 0.0)), np.log1p(max(fmax, 0.0)))
    return normalize(arr, *bounds[feat])


def denormalize_cpm25(arr: np.ndarray, bounds: dict, cfg: dict) -> np.ndarray:
    """Inverse transform for cpm25 predictions into physical µg/m³."""
    x = arr.astype(np.float32)
    if _grid_scaler_enabled(cfg):
        stats = cfg.get('_runtime', {}).get('grid_stats')
        if stats is None or 'cpm25' not in stats:
            raise RuntimeError('Grid scaler statistics for cpm25 are missing in cfg["_runtime"].')
        mean, std = stats['cpm25']
        x = x * std[None, :, :, None] + mean[None, :, :, None]
        x = inverse_transform_feature(x, 'cpm25', cfg, bounds)
        return np.maximum(x, 0.0).astype(np.float32)

    fmin, fmax = bounds['cpm25']
    if bool(cfg.get('preprocessing', {}).get('cpm25_log1p', False)):
        x = denormalize(x, np.log1p(max(fmin, 0.0)), np.log1p(max(fmax, 0.0)))
        x = np.expm1(x)
        return np.maximum(x, 0.0).astype(np.float32)
    return np.maximum(denormalize(x, fmin, fmax), 0.0).astype(np.float32)


def _broadcast_scalar_series(values: np.ndarray, H: int, W: int) -> np.ndarray:
    return np.broadcast_to(values[:, None, None].astype(np.float32), (len(values), H, W)).copy()


def load_static_maps(cfg) -> dict:
    ll_path = os.path.join(cfg['paths']['data'], 'raw', 'lat_long.npy')
    ll = np.load(ll_path).astype(np.float32)
    lat = ll[:, :, 0]
    lon = ll[:, :, 1]
    lat = (lat - lat.min()) / (lat.max() - lat.min() + 1e-8)
    lon = (lon - lon.min()) / (lon.max() - lon.min() + 1e-8)
    return {'lat': lat.astype(np.float32), 'lon': lon.astype(np.float32)}


def _parse_time_strings(time_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ts = np.asarray(time_arr).astype(str)
    hours = np.array([int(t[11:13]) for t in ts], dtype=np.int32)
    dt = ts.astype('datetime64[h]')
    year_start = dt.astype('datetime64[Y]')
    doys = (dt.astype('datetime64[D]') - year_start).astype(np.int32) + 1
    return hours, doys


def build_aux_feature_maps(cfg, time_arr: np.ndarray | None, T: int, H: int, W: int) -> dict:
    """Build static/calendar auxiliary maps requested in config."""
    aux_names = cfg['features'].get('aux', [])
    aux = {}
    static = load_static_maps(cfg) if any(a in aux_names for a in ('lat', 'lon')) else {}

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


def _load_month(cfg, month: str, bounds: dict) -> dict:
    """Load one month as memory-mapped raw arrays plus lightweight aux channels."""
    data_dir = cfg['paths']['data']
    mmap_mode = 'r' if _use_mmap(cfg) else None
    raw = {}
    for feat in cfg['features']['base']:
        raw[feat] = np.load(os.path.join(data_dir, 'raw', month, f'{feat}.npy'), mmap_mode=mmap_mode)

    T, H, W = raw['cpm25'].shape
    time_path = os.path.join(data_dir, 'raw', month, 'time.npy')
    time_arr = np.load(time_path) if os.path.exists(time_path) else None
    aux = build_aux_feature_maps(cfg, time_arr, T, H, W)
    return {
        'raw': raw,
        'aux': aux,
        'name': month,
        'shape': (T, H, W),
    }


def load_all_months(cfg, months: list, bounds: dict) -> list:
    """Load month descriptors without materializing full normalized tensors in RAM."""
    all_data = []
    for month in months:
        print(f"  Loading {month} ...", end=' ', flush=True)
        all_data.append(_load_month(cfg, month, bounds))
        print('OK')
        gc.collect()
    return all_data


class PM25Dataset(Dataset):
    """Lazy sliding-window dataset backed by memory-mapped raw arrays."""

    def __init__(
        self,
        months_data: list,
        cfg: dict,
        bounds: dict,
        grid_stats: dict,
        stride: int,
        month_names: list[str] | None = None,
    ):
        self.data = months_data
        self.cfg = cfg
        self.bounds = bounds
        self.grid_stats = grid_stats
        self.feats = cfg['features']['input']
        self.base_feats = set(cfg['features']['base'])
        self.t_in = cfg['time']['t_in_met']
        self.t_cpm = cfg['time']['t_in_cpm']
        self.t_out = cfg['time']['t_out']
        self.month_names = month_names or [f'month_{i}' for i in range(len(months_data))]
        self.sample_months = []
        self.index = []

        window = self.t_cpm + self.t_out
        for m_idx, mdata in enumerate(months_data):
            T = mdata['shape'][0]
            for t in range(0, T - window + 1, stride):
                self.index.append((m_idx, t))
                self.sample_months.append(self.month_names[m_idx])

    def __len__(self):
        return len(self.index)

    def _slice_feature(self, mdata: dict, feat: str, start: int, stop: int) -> np.ndarray:
        if feat in self.base_feats:
            raw = np.asarray(mdata['raw'][feat][start:stop], dtype=np.float32)
            return normalize_feature(raw, feat, self.bounds, self.cfg, self.grid_stats)
        return np.asarray(mdata['aux'][feat][start:stop], dtype=np.float32)

    def __getitem__(self, idx):
        m_idx, t = self.index[idx]
        mdata = self.data[m_idx]

        channels = []
        for feat in self.feats:
            chunk = self._slice_feature(mdata, feat, t, t + self.t_in)
            if feat == 'cpm25':
                chunk[self.t_cpm:] = 0.0
            channels.append(chunk)
        x = torch.from_numpy(np.stack(channels, axis=0).astype(np.float32))

        target_raw = np.asarray(mdata['raw']['cpm25'][t + self.t_cpm:t + self.t_cpm + self.t_out], dtype=np.float32)
        target = normalize_feature(target_raw, 'cpm25', self.bounds, self.cfg, self.grid_stats)
        y = torch.from_numpy(target).permute(1, 2, 0)
        return x, y


def get_dataloaders(cfg, train_data: list, val_data: list, bounds: dict):
    """Build train/validation loaders using the persisted grid scaler."""
    grid_stats = load_grid_stats(cfg)
    cfg.setdefault('_runtime', {})['grid_stats'] = grid_stats

    train_month_names = cfg['data']['train_months']
    val_month_names = [cfg['data']['val_month']] * len(val_data)
    train_ds = PM25Dataset(train_data, cfg, bounds, grid_stats, stride=cfg['time']['stride_train'], month_names=train_month_names)
    val_ds = PM25Dataset(val_data, cfg, bounds, grid_stats, stride=cfg['time']['stride_val'], month_names=val_month_names)

    print(f"  Train samples: {len(train_ds):,}  |  Val samples: {len(val_ds):,}")

    sampler = None
    if bool(cfg['training'].get('use_weighted_sampler', False)):
        month_w = cfg['training'].get('month_sampling_weights', {})
        default_w = float(cfg['training'].get('default_sampling_weight', 1.0))
        weights = [float(month_w.get(month_name, default_w)) for month_name in train_ds.sample_months]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

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


def build_test_input(cfg, bounds: dict, start: int = 0, end: int | None = None) -> np.ndarray:
    """Build chunked test tensors in the same transformed space used for training."""
    features = cfg['features']['base']
    input_feats = cfg['features']['input']
    data_dir = cfg['paths']['data']
    t_in_cpm = cfg['time']['t_in_cpm']
    t_in_met = cfg['time']['t_in_met']
    n_test = cfg['data']['test_samples']
    grid_stats = cfg.get('_runtime', {}).get('grid_stats') or load_grid_stats(cfg)
    cfg.setdefault('_runtime', {})['grid_stats'] = grid_stats

    if end is None:
        end = n_test
    if not (0 <= start < end <= n_test):
        raise ValueError(f"Invalid test slice [{start}, {end}) for n_test={n_test}")
    bs = end - start

    base = {}
    for feat in features:
        arr = np.load(os.path.join(data_dir, 'test_in', f'{feat}.npy'), mmap_mode='r')[start:end]
        arr = np.asarray(arr, dtype=np.float32)
        if feat == 'cpm25':
            pad = np.zeros((bs, t_in_met - t_in_cpm, arr.shape[-2], arr.shape[-1]), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=1)
        base[feat] = normalize_feature(arr, feat, bounds, cfg, grid_stats)

    _, T, H, W = next(iter(base.values())).shape
    time_path = os.path.join(data_dir, 'test_in', 'time.npy')
    time_arr = np.load(time_path) if os.path.exists(time_path) else None
    if time_arr is not None and len(time_arr) == n_test:
        time_arr = time_arr[start:end]
    if time_arr is None and cfg['features'].get('aux'):
        print('Warning: test_in/time.npy not found. Calendar features use neutral defaults.')

    aux_single = build_aux_feature_maps(cfg, None if time_arr is None else time_arr[0], T, H, W)
    aux = {}
    for feat, arr in aux_single.items():
        if arr.shape[0] == T:
            aux[feat] = np.broadcast_to(arr[None], (bs, T, H, W)).copy()
        else:
            aux[feat] = np.broadcast_to(arr, (bs, T, H, W)).copy()

    arrays = []
    for feat in input_feats:
        arrays.append(base[feat] if feat in base else aux[feat])
    return np.stack(arrays, axis=1).astype(np.float32)


def compute_stats(cfg, months):
    raise DeprecationWarning('compute_stats is removed. Use build_grid_stats(...) or load_minmax_bounds(cfg).')


def save_stats(stats, path):
    np.save(path, stats)


def load_stats(path):
    return np.load(path, allow_pickle=True).item()

