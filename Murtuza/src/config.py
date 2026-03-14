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

    # Derived feature lists
    cfg['features']['all'] = ['cpm25'] + cfg['features']['met'] + cfg['features']['emis']
    extra_masks = list(cfg['features'].get('add_masks', []))
    cfg['features']['base'] = list(cfg['features']['all']) + extra_masks
    use_aux = bool(cfg['features'].get('use_aux', False))
    cfg['features']['aux'] = list(cfg['features'].get('aux', [])) if use_aux else []
    cfg['features']['input'] = cfg['features']['base'] + cfg['features']['aux']
    cfg['features']['n_features'] = len(cfg['features']['base'])
    cfg['features']['input_channels'] = len(cfg['features']['input'])

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
        cfg['paths']['temp']            = _os.path.join(local_out, 'models')
        cfg['paths']['model_save']      = _os.path.join(local_out, 'models', 'best_model.pt')
        cfg['paths']['output']          = _os.path.join(local_out, 'submissions', 'preds.npy')
        # Pixel stats stored alongside model checkpoint
        cfg['paths']['pixel_stats']     = _os.path.join(local_out, 'models', 'pixel_stats.npz')
        _os.makedirs(cfg['paths']['temp'], exist_ok=True)
        _os.makedirs(_os.path.join(local_out, 'submissions'), exist_ok=True)
    else:
        cfg['paths']['data'] = _resolve_kaggle_data_path(cfg)
        _os.makedirs(cfg['paths']['temp'], exist_ok=True)
        # pixel_stats loaded from config or default /kaggle/temp path
        cfg['paths']['pixel_stats'] = cfg.get('data', {}).get(
            'pixel_stats_path', '/kaggle/temp/pixel_stats.npz'
        )

    # Build min_max path from data root
    cfg['paths']['min_max'] = _os.path.join(cfg['paths']['data'], 'stats', 'feat_min_max.mat')

    # Model registry settings: allow either top-level model params or nested models.{type}
    model_type = cfg['model'].get('type', 'tfno2d').lower()
    model_block = cfg.get('models', {}).get(model_type, {})
    for key, value in model_block.items():
        cfg['model'].setdefault(key, value)

    cfg['runtime'] = {
        'on_kaggle': _os.path.exists('/kaggle'),
        'data_exists': _os.path.exists(cfg['paths']['data']),
        'min_max_exists': _os.path.exists(cfg['paths']['min_max']),
    }

    if not cfg['runtime']['data_exists']:
        raise FileNotFoundError(
            f"Data root not found: {cfg['paths']['data']}. "
            "Set AISEHACK_DATA or update paths.data_local in config.yaml."
        )

    return cfg
