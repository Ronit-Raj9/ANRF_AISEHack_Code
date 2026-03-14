"""
Configuration loader — reads configs/config.yaml and exposes
a dict + derived constants used across all modules.
"""

import os
import yaml
import torch


def _resolve_local_data_path(cfg: dict) -> str:
    """Resolve the local data root from env vars or common repo-relative paths."""
    env_candidates = [
        os.environ.get('AISEHACK_DATA'),
        os.environ.get('ANRF_AISEHACK_DATA'),
        cfg.get('paths', {}).get('data_local'),
    ]
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    rel_candidates = [
        os.path.join(repo_root, '..', 'aisehack-theme-2'),
        os.path.join(repo_root, 'aisehack-theme-2'),
    ]
    for candidate in env_candidates + rel_candidates:
        if candidate and os.path.exists(candidate):
            return os.path.abspath(candidate)
    return os.path.abspath(env_candidates[-1] or rel_candidates[0])


def _resolve_kaggle_data_path(cfg: dict) -> str:
    """Resolve Kaggle data root from config/env/common competition mount paths."""
    configured = cfg.get('paths', {}).get('data')
    env_data = os.environ.get('AISEHACK_DATA')
    candidates = [
        env_data,
        configured,
        '/kaggle/input/aisehack-theme-2',
        '/kaggle/input/competitions/aisehack-theme-2',
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return os.path.abspath(candidate)
    return os.path.abspath(configured or '/kaggle/input/aisehack-theme-2')


def load_config(config_path: str = None) -> dict:
    """Load YAML config and add derived fields."""
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(__file__), '..', 'configs', 'config.yaml'
        )
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    cfg.setdefault('preprocessing', {})
    cfg.setdefault('inference', {})
    cfg.setdefault('training', {})
    cfg.setdefault('model', {})
    cfg.setdefault('paths', {})
    cfg.setdefault('features', {})
    cfg.setdefault('loss', {})

    # Modern preprocessing defaults
    cfg['preprocessing'].setdefault('normalization', 'grid_log_standardize')
    cfg['preprocessing'].setdefault('use_mmap', True)
    cfg['preprocessing'].setdefault('grid_chunk_size', 48)
    cfg['preprocessing'].setdefault('grid_stats_filename', 'grid_log_scaler_2016.npz')
    cfg['preprocessing'].setdefault('grid_stats_eps', 1.0e-6)
    cfg['preprocessing'].setdefault('auto_build_grid_stats', True)
    cfg['preprocessing'].setdefault('grid_stats_train_only', True)
    cfg['preprocessing'].setdefault('enforce_grid_stats_train_months', True)
    cfg['preprocessing'].setdefault('temporal_gap_hours', 12)
    cfg['preprocessing'].setdefault('signed_log_for_negative', True)
    cfg['preprocessing'].setdefault('log_eps', 1.0e-12)
    cfg['preprocessing'].setdefault('log1p_features', ['cpm25', 'rain', 'ventilation_index'])
    cfg['preprocessing'].setdefault('log_eps_features', ['PM25', 'SO2', 'NOx', 'NH3', 'NMVOC_e', 'NMVOC_finn', 'bio'])
    cfg['preprocessing'].setdefault('signed_log_features', ['u10', 'v10', 'wind_convergence'])
    cfg['preprocessing'].setdefault('sparse_mask_features', ['rain', 'NMVOC_finn'])
    cfg['preprocessing'].setdefault('sparse_mask_threshold', 0.0)
    cfg['preprocessing'].setdefault('aux_standardize_features', ['ventilation_index', 'wind_convergence'])
    cfg['preprocessing'].setdefault('derived_features', ['ventilation_index', 'wind_convergence'])
    cfg['preprocessing'].setdefault('add_land_mask', False)
    cfg['preprocessing'].setdefault('land_mask_file', '')

    # Derived feature lists
    cfg['features']['all'] = ['cpm25'] + cfg['features']['met'] + cfg['features']['emis']
    cfg['features']['base'] = list(cfg['features']['all'])
    use_aux = bool(cfg['features'].get('use_aux', False))
    cfg['features']['aux'] = list(cfg['features'].get('aux', [])) if use_aux else []
    if use_aux:
        if bool(cfg['preprocessing'].get('add_land_mask', False)) and 'land_mask' not in cfg['features']['aux']:
            cfg['features']['aux'].append('land_mask')
        for name in cfg['preprocessing'].get('derived_features', []):
            if name not in cfg['features']['aux']:
                cfg['features']['aux'].append(name)
        for feat in cfg['preprocessing'].get('sparse_mask_features', []):
            mask_name = f"{feat}_mask"
            if mask_name not in cfg['features']['aux']:
                cfg['features']['aux'].append(mask_name)
    cfg['features']['input'] = cfg['features']['base'] + cfg['features']['aux']
    cfg['features']['n_features'] = len(cfg['features']['base'])
    cfg['features']['input_channels'] = len(cfg['features']['input'])

    # Backward-compatible aliases
    if cfg['preprocessing'].get('cpm25_grid_zscore', False):
        cfg['preprocessing']['normalization'] = 'grid_log_standardize'
    if cfg['preprocessing'].get('cpm25_log1p', False):
        cfg['preprocessing'].setdefault('force_log_targets', True)

    # Device
    cfg['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Train months (everything except val)
    cfg['data']['train_months'] = [
        m for m in cfg['data']['months'] if m != cfg['data']['val_month']
    ]

    # Override paths when running locally (kaggle dirs don't exist)
    import os as _os
    if not _os.path.exists('/kaggle'):
        cfg['paths']['data'] = _resolve_local_data_path(cfg)

        local_out = _os.path.abspath(
            _os.path.join(_os.path.dirname(__file__), '..', 'outputs')
        )
        cfg['paths']['temp']       = _os.path.join(local_out, 'models')
        cfg['paths']['model_save'] = _os.path.join(local_out, 'models', 'best_model.pt')
        cfg['paths']['checkpoint_dir'] = _os.path.join(local_out, 'models', 'checkpoints')
        cfg['paths']['checkpoint_last'] = _os.path.join(cfg['paths']['checkpoint_dir'], 'last_checkpoint.pt')
        cfg['paths']['output']     = _os.path.join(local_out, 'submissions', 'preds.npy')
        _os.makedirs(cfg['paths']['temp'], exist_ok=True)
        _os.makedirs(cfg['paths']['checkpoint_dir'], exist_ok=True)
        _os.makedirs(_os.path.join(local_out, 'submissions'), exist_ok=True)
    else:
        cfg['paths']['data'] = _resolve_kaggle_data_path(cfg)
        cfg['paths'].setdefault('checkpoint_dir', _os.path.join(cfg['paths']['temp'], 'checkpoints'))
        cfg['paths'].setdefault('checkpoint_last', _os.path.join(cfg['paths']['checkpoint_dir'], 'last_checkpoint.pt'))
        _os.makedirs(cfg['paths']['temp'], exist_ok=True)
        _os.makedirs(cfg['paths']['checkpoint_dir'], exist_ok=True)

    # Build min_max path from data root
    cfg['paths']['min_max'] = _os.path.join(cfg['paths']['data'], 'stats', 'feat_min_max.mat')
    grid_stats_name = cfg['preprocessing']['grid_stats_filename']
    input_grid_stats = _os.path.join(cfg['paths']['data'], 'stats', grid_stats_name)

    # On Kaggle, /kaggle/input is read-only. Load from input when present;
    # otherwise save newly built scaler into /kaggle/working.
    if _os.path.exists('/kaggle'):
        working_grid_stats = _os.path.join('/kaggle/working', grid_stats_name)
        cfg['paths']['grid_stats'] = input_grid_stats if _os.path.exists(input_grid_stats) else working_grid_stats
    else:
        cfg['paths']['grid_stats'] = input_grid_stats

    # Model registry settings: allow either top-level model params or nested models.{type}
    model_type = cfg['model'].get('type', 'tfno2d').lower()
    model_block = cfg.get('models', {}).get(model_type, {})
    for key, value in model_block.items():
        cfg['model'].setdefault(key, value)

    # Training protocol defaults
    cfg['training'].setdefault('phase1_epochs', 20)
    cfg['training'].setdefault('phase2_epochs', 20)
    cfg['training'].setdefault('phase3_epochs', 10)
    cfg['training'].setdefault(
        'epochs',
        cfg['training']['phase1_epochs'] + cfg['training']['phase2_epochs'] + cfg['training']['phase3_epochs'],
    )
    cfg['training'].setdefault('checkpoint_metric', 'val_rmse_phys')
    cfg['training'].setdefault('enable_checkpointing', True)
    cfg['training'].setdefault('checkpoint_every_epochs', 5)
    cfg['training'].setdefault('resume_if_available', True)
    cfg['training'].setdefault('resume_checkpoint_path', '')
    cfg['loss'].setdefault('phase3_smooth', 'huber')
    cfg['loss'].setdefault('huber_delta', 25.0)

    cfg['preprocessing'].setdefault('feature_time_limits', {'cpm25': cfg['time']['t_in_cpm']})

    # Inference defaults
    cfg['inference'].setdefault('use_autoregressive', True)
    cfg['inference'].setdefault('use_tta', True)
    cfg['inference'].setdefault('tta_modes', ['identity', 'hflip', 'vflip'])
    cfg['inference'].setdefault('model_paths', [])

    cfg['runtime'] = {
        'on_kaggle': _os.path.exists('/kaggle'),
        'data_exists': _os.path.exists(cfg['paths']['data']),
        'min_max_exists': _os.path.exists(cfg['paths']['min_max']),
        'grid_stats_exists': _os.path.exists(cfg['paths']['grid_stats']),
    }

    if not cfg['runtime']['data_exists']:
        raise FileNotFoundError(
            f"Data root not found: {cfg['paths']['data']}. "
            "Set AISEHACK_DATA or update paths.data_local in config.yaml."
        )

    return cfg
