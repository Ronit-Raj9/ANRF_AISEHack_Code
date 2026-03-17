"""
General utilities — seeding, device info, parameter counting, sanity checks.
"""

import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42):
    """Set seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_device_info(device):
    """Print device and GPU memory info if available."""
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Memory: {mem:.1f} GB")


def count_parameters(model) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def sanity_check_bounds(bounds: dict, features: list) -> None:
    """
    Print a table of normalization bounds and flag any zero-range features
    (would cause divide-by-zero in normalization).
    """
    print(f"\n{'Feature':15s} {'Min':>14s} {'Max':>14s} {'Range':>14s}  {'OK?':>4s}")
    print("─" * 65)
    for feat in features:
        fmin, fmax = bounds[feat]
        rng = fmax - fmin
        ok  = "✓" if rng > 0 else "✗ ZERO RANGE!"
        print(f"{feat:15s} {fmin:14.6g} {fmax:14.6g} {rng:14.6g}  {ok}")
    print()

