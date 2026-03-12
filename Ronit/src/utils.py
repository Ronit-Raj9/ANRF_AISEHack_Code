"""
General utilities — seeding, device info, etc.
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
        mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"GPU Memory: {mem:.1f} GB")


def count_parameters(model):
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
