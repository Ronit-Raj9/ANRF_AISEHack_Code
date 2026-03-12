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

    return cfg
