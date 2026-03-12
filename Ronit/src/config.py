"""
Configuration loader — reads configs/config.yaml and exposes
a dict + derived constants used across all modules.
"""

import os
import yaml
import torch


def load_config(config_path: str = None) -> dict:
    """Load YAML config and add derived fields."""
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(__file__), '..', 'configs', 'config.yaml'
        )
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Derived feature list
    cfg['features']['all'] = ['cpm25'] + cfg['features']['met'] + cfg['features']['emis']
    cfg['features']['n_features'] = len(cfg['features']['all'])

    # Device
    cfg['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Train months (everything except val)
    cfg['data']['train_months'] = [
        m for m in cfg['data']['months'] if m != cfg['data']['val_month']
    ]

    # Override paths when running locally (kaggle dirs don't exist)
    import os as _os
    if not _os.path.exists('/kaggle'):
        local_out = _os.path.abspath(
            _os.path.join(_os.path.dirname(__file__), '..', 'outputs')
        )
        cfg['paths']['temp']       = _os.path.join(local_out, 'models')
        cfg['paths']['model_save'] = _os.path.join(local_out, 'models', 'best_model.pt')
        cfg['paths']['output']     = _os.path.join(local_out, 'submissions', 'preds.npy')
        _os.makedirs(cfg['paths']['temp'], exist_ok=True)
        _os.makedirs(_os.path.join(local_out, 'submissions'), exist_ok=True)
    else:
        _os.makedirs(cfg['paths']['temp'], exist_ok=True)

    return cfg
