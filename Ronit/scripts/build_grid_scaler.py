"""Build the 2016 grid-wise log-standardization maps for the physics-informed pipeline."""

from __future__ import annotations

import argparse
import os
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import load_config
from src.data import build_grid_stats, describe_grid_stats, load_grid_stats, load_minmax_bounds


def main() -> None:
    parser = argparse.ArgumentParser(description='Build per-feature per-grid log-standardization maps.')
    parser.add_argument('--config', default=None, help='Optional path to config.yaml')
    parser.add_argument('--force', action='store_true', help='Recompute even if the stats file already exists.')
    args = parser.parse_args()

    cfg = load_config(args.config)
    bounds = load_minmax_bounds(cfg)
    if args.force or not os.path.exists(cfg['paths']['grid_stats']):
        stats = build_grid_stats(cfg, bounds=bounds, force=args.force)
    else:
        stats = load_grid_stats(cfg)

    print(f"Saved grid scaler: {cfg['paths']['grid_stats']}")
    describe_grid_stats(stats, cfg['features']['base'])


if __name__ == '__main__':
    main()
