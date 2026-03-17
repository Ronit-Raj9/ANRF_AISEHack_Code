"""
pipeline.py — Modular PM2.5 Forecasting Pipeline
==================================================
Reusable functions for the ANRF AISEHack Theme 2 competition.
Covers: configuration, data loading, normalization, sample construction,
        dataset/dataloader, model definition, training, and inference.

Usage:
    from pipeline import *
    # or import specific sections as needed
"""

import os
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional


# ═══════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ═══════════════════════════════════════════════════════════

class Config:
    """
    Central configuration object.  Modify attributes here or override
    when instantiating to change the entire pipeline behaviour.
    """

    # ── paths ──
    DATA_DIR: str = "/kaggle/input/competitions/aisehack-theme-2"
    TEMP_DIR: str = "/kaggle/temp"
    WORK_DIR: str = "/kaggle/working"

    # ── device ──
    DEVICE: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # ── feature groups ──
    MET_FEATURES: List[str] = [
        "u10", "v10", "pblh", "rain", "t2", "q2", "swdown", "psfc",
    ]
    EMIS_FEATURES: List[str] = [
        "PM25", "SO2", "NOx", "NH3", "NMVOC_e", "NMVOC_finn", "bio",
    ]
    TARGET: str = "cpm25"

    # ── temporal structure ──
    T_IN_CPM: int = 10       # cpm25 hours available at test time
    T_IN_MET: int = 26       # met / emis hours available
    T_OUT: int = 16           # forecast horizon

    # ── months ──
    ALL_MONTHS: List[str] = ["APRIL_16", "JULY_16", "OCT_16", "DEC_16"]
    VAL_MONTH: str = "OCT_16"

    # ── sampling ──
    TRAIN_STRIDE: int = 2
    VAL_STRIDE: int = 4

    # ── training hyper-parameters ──
    BATCH_SIZE_TRAIN: int = 4
    BATCH_SIZE_VAL: int = 8
    BATCH_SIZE_TEST: int = 16
    NUM_WORKERS: int = 2
    EPOCHS: int = 30
    LR: float = 1e-3
    WEIGHT_DECAY: float = 1e-4
    GRAD_CLIP: float = 1.0
    PCT_START: float = 0.1   # OneCycleLR warmup fraction

    # ── model hyper-parameters ──
    FNO_WIDTH: int = 64
    FNO_MODES: int = 20
    FNO_DEPTH: int = 4

    # ── misc ──
    SEED: int = 42

    def __init__(self, **overrides):
        for k, v in overrides.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                raise ValueError(f"Unknown config key: {k}")

    @property
    def features(self) -> List[str]:
        """Ordered list: [target] + met + emis."""
        return [self.TARGET] + self.MET_FEATURES + self.EMIS_FEATURES

    @property
    def n_features(self) -> int:
        return len(self.features)

    @property
    def train_months(self) -> List[str]:
        return [m for m in self.ALL_MONTHS if m != self.VAL_MONTH]

    @property
    def checkpoint_path(self) -> str:
        return os.path.join(self.TEMP_DIR, "best_model.pt")

    @property
    def preds_path(self) -> str:
        return os.path.join(self.WORK_DIR, "preds.npy")

    def __repr__(self):
        attrs = {k: v for k, v in vars(Config).items()
                 if not k.startswith("_") and not callable(v)}
        attrs.update(vars(self))
        lines = [f"  {k} = {v!r}" for k, v in sorted(attrs.items())]
        return "Config(\n" + "\n".join(lines) + "\n)"


def seed_everything(seed: int = 42):
    """Reproducibility helper."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ═══════════════════════════════════════════════════════════
# 2. DATA LOADING UTILITIES
# ═══════════════════════════════════════════════════════════

def load_raw_feature(
    data_dir: str, month: str, feature: str
) -> np.ndarray:
    """
    Load a single raw feature array for a given month.

    Returns:
        np.ndarray of shape (T, 140, 124) in float32.
    """
    path = os.path.join(data_dir, "raw", month, f"{feature}.npy")
    return np.load(path).astype(np.float32)


def load_test_feature(
    data_dir: str, feature: str
) -> np.ndarray:
    """
    Load a single test-time feature array.

    Returns:
        np.ndarray of shape (N_samples, T_feat, 140, 124) in float32.
    """
    path = os.path.join(data_dir, "test_in", f"{feature}.npy")
    return np.load(path).astype(np.float32)


def load_time_stamps(data_dir: str, month: str) -> np.ndarray:
    """Load timestamp array for a given month."""
    return np.load(os.path.join(data_dir, "raw", month, "time.npy"))


def load_lat_lon(data_dir: str) -> np.ndarray:
    """Load the (140, 124) latitude/longitude grid."""
    return np.load(os.path.join(data_dir, "raw", "lat_long.npy"))


# ═══════════════════════════════════════════════════════════
# 3. NORMALIZATION (Grid-wise Mean / Std)
# ═══════════════════════════════════════════════════════════

def compute_grid_stats(
    data_dir: str,
    months: List[str],
    features: List[str],
    eps: float = 1e-8,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute per-grid-point mean and std for each feature across
    the given months.

    Returns:
        dict  { feature_name: { 'mean': (140,124), 'std': (140,124) } }
    """
    stats: Dict[str, Dict[str, np.ndarray]] = {}
    for feat in features:
        arrays = []
        for m in months:
            arr = load_raw_feature(data_dir, m, feat)
            arrays.append(arr)
        concat = np.concatenate(arrays, axis=0)           # (T_total, 140, 124)
        stats[feat] = {
            "mean": concat.mean(axis=0),                   # (140, 124)
            "std":  concat.std(axis=0) + eps,              # (140, 124)
        }
        del concat, arrays
        gc.collect()
    return stats


def normalize_array(
    arr: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """
    Grid-wise z-score normalization.

    Args:
        arr:  (..., 140, 124)
        mean: (140, 124)
        std:  (140, 124)
    """
    return (arr - mean) / std


def denormalize_array(
    arr: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Inverse of normalize_array."""
    return arr * std + mean


def save_stats(stats: Dict, path: str):
    """Persist normalization stats to disk."""
    np.save(path, stats, allow_pickle=True)
    print(f"Stats saved → {path}")


def load_stats(path: str) -> Dict:
    """Load persisted normalization stats."""
    return np.load(path, allow_pickle=True).item()


# ═══════════════════════════════════════════════════════════
# 4. SAMPLE CONSTRUCTION
# ═══════════════════════════════════════════════════════════

def build_samples(
    data_dir: str,
    months: List[str],
    features: List[str],
    stats: Dict[str, Dict[str, np.ndarray]],
    t_in_cpm: int = 10,
    t_in_met: int = 26,
    t_out: int = 16,
    stride: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create overlapping time-series samples from raw monthly data.

    For each window of length (t_in_met + t_out):
        - Input X : (t_in_met, 140, 124, C) — *normalized*
          cpm25 is zero-masked after the first t_in_cpm steps
          to match test-time availability.
        - Target y : (140, 124, t_out)       — *raw* (µg/m³)

    Returns:
        X: (N, t_in_met, 140, 124, C)
        y: (N, 140, 124, t_out)
    """
    X_list, y_list = [], []
    target_feat = features[0]  # 'cpm25' must be first

    for m in months:
        # Load and normalise all features
        normed: Dict[str, np.ndarray] = {}
        for feat in features:
            raw = load_raw_feature(data_dir, m, feat)
            normed[feat] = normalize_array(raw, stats[feat]["mean"], stats[feat]["std"])

        # Also keep un-normalised cpm25 for targets
        cpm_raw = load_raw_feature(data_dir, m, target_feat)

        T = normed[target_feat].shape[0]
        window = t_in_met + t_out  # total hours needed per sample

        for i in range(0, T - window + 1, stride):
            # --- input tensor (t_in_met, H, W, C) ---
            chans = []
            for feat in features:
                chans.append(normed[feat][i : i + t_in_met])
            inp = np.stack(chans, axis=-1)  # (t_in_met, 140, 124, C)

            # mask cpm25 beyond t_in_cpm (simulate test conditions)
            inp[t_in_cpm:, :, :, 0] = 0.0

            # --- target tensor (H, W, t_out) in physical units ---
            out = cpm_raw[i + t_in_met : i + t_in_met + t_out]  # (t_out, 140, 124)
            out = out.transpose(1, 2, 0)                         # (140, 124, t_out)

            X_list.append(inp)
            y_list.append(out)

        del normed, cpm_raw
        gc.collect()

    return np.stack(X_list), np.stack(y_list)


def build_test_inputs(
    data_dir: str,
    features: List[str],
    stats: Dict[str, Dict[str, np.ndarray]],
    t_in_cpm: int = 10,
    t_in_met: int = 26,
) -> np.ndarray:
    """
    Prepare test inputs: normalise, pad cpm25 to t_in_met, and stack.

    Returns:
        X_test: (N_samples, t_in_met, 140, 124, C)
    """
    feat_arrays = []
    for feat in features:
        arr = load_test_feature(data_dir, feat)  # (N, T_feat, 140, 124)

        # normalise grid-wise
        mean = stats[feat]["mean"][None, None]    # (1, 1, 140, 124)
        std  = stats[feat]["std"][None, None]
        arr = normalize_array(arr, mean, std)

        # cpm25 has only t_in_cpm steps → zero-pad to t_in_met
        if feat == features[0]:  # target feature
            N = arr.shape[0]
            pad = np.zeros(
                (N, t_in_met - t_in_cpm, 140, 124), dtype=np.float32
            )
            arr = np.concatenate([arr, pad], axis=1)

        feat_arrays.append(arr)

    X_test = np.stack(feat_arrays, axis=-1)  # (N, t_in_met, 140, 124, C)
    return X_test


# ═══════════════════════════════════════════════════════════
# 5. DATASET & DATALOADER
# ═══════════════════════════════════════════════════════════

class PM25Dataset(Dataset):
    """
    PyTorch Dataset wrapping pre-built numpy arrays.

    __getitem__ returns:
        x: (C, T, H, W)   — ready for 2D conv / FNO
        y: (H, W, T_OUT)   — raw µg/m³
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)  # (N, T, H, W, C)
        self.y = torch.from_numpy(y)  # (N, H, W, T_OUT)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        x = self.X[idx].permute(3, 0, 1, 2)  # (C, T, H, W)
        return x, self.y[idx]


def make_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_train: int = 4,
    batch_val: int = 8,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders."""
    train_ds = PM25Dataset(X_train, y_train)
    val_ds   = PM25Dataset(X_val, y_val)

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_train,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_val,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_dl, val_dl


# ═══════════════════════════════════════════════════════════
# 6. MODEL — Improved TFNO2D with Residual Skip Connections
# ═══════════════════════════════════════════════════════════

class SpectralConv2d(nn.Module):
    """Truncated Fourier convolution in 2-D (real FFT)."""

    def __init__(self, in_ch: int, out_ch: int, modes1: int, modes2: int):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1.0 / (in_ch * out_ch)
        self.weights = nn.Parameter(
            scale * torch.randn(in_ch, out_ch, modes1, modes2, 2)
        )

    @staticmethod
    def _complex_mul(x, w):
        """Element-wise complex multiply stored as real pairs."""
        xr, xi = x[..., 0], x[..., 1]
        wr, wi = w[..., 0], w[..., 1]
        return torch.stack([xr * wr - xi * wi, xr * wi + xi * wr], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(
            B, self.weights.shape[1], H, W // 2 + 1,
            dtype=torch.cfloat, device=x.device,
        )
        trunc = torch.view_as_real(
            x_ft[:, :, : self.modes1, : self.modes2]
        )
        prod = self._complex_mul(trunc, self.weights)
        out_ft[:, :, : self.modes1, : self.modes2] = torch.view_as_complex(prod)
        return torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")


class FNOBlock(nn.Module):
    """Single FNO layer: spectral conv + pointwise bypass + norm + GELU."""

    def __init__(self, width: int, modes1: int, modes2: int):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes1, modes2)
        self.bypass   = nn.Conv2d(width, width, kernel_size=1)
        self.norm     = nn.GroupNorm(8, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.norm(self.spectral(x) + self.bypass(x)))


class TFNO2D(nn.Module):
    """
    Improved Tucker-factorized Fourier Neural Operator (2-D).

    Improvements over vanilla baseline:
      • Residual skip connections inside the FNO stack.
      • Deeper projection head with dropout.
      • Flexible feature/time flattening.

    Input  : (B, C, T, H, W)
    Output : (B, H, W, T_out)
    """

    def __init__(
        self,
        in_channels: int,
        out_steps: int = 16,
        width: int = 64,
        modes: int = 20,
        depth: int = 4,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.lift = nn.Sequential(
            nn.Conv2d(in_channels, width, 1),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [FNOBlock(width, modes, modes) for _ in range(depth)]
        )
        self.proj = nn.Sequential(
            nn.Conv2d(width, width * 2, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(width * 2, out_steps, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        x = x.reshape(B, C * T, H, W)      # flatten channels × time
        x = self.lift(x)
        for block in self.blocks:
            x = block(x) + x                # ← residual skip connection
        x = self.proj(x)                    # (B, T_out, H, W)
        return x.permute(0, 2, 3, 1)        # (B, H, W, T_out)


def build_model(cfg: Config) -> nn.Module:
    """Instantiate the TFNO2D model from a Config object."""
    model = TFNO2D(
        in_channels=cfg.n_features * cfg.T_IN_MET,
        out_steps=cfg.T_OUT,
        width=cfg.FNO_WIDTH,
        modes=cfg.FNO_MODES,
        depth=cfg.FNO_DEPTH,
    ).to(cfg.DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    return model


# ═══════════════════════════════════════════════════════════
# 7. LOSS — Metric-Aligned (Average Domain RMSE)
# ═══════════════════════════════════════════════════════════

def metric_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Average domain RMSE — matches the competition evaluation metric.

    pred, target: (B, H, W, T)
    """
    spatial_mse = torch.mean((pred - target) ** 2, dim=(1, 2))  # (B, T)
    return torch.mean(torch.sqrt(spatial_mse + 1e-8))


# ═══════════════════════════════════════════════════════════
# 8. TRAINING LOOP
# ═══════════════════════════════════════════════════════════

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    grad_clip: float = 1.0,
) -> float:
    """Run one training epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = metric_loss(pred, yb)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Run validation. Returns average RMSE loss."""
    model.eval()
    losses = []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        losses.append(metric_loss(pred, yb).item())
    return float(np.mean(losses))


def train_model(
    model: nn.Module,
    train_dl: DataLoader,
    val_dl: DataLoader,
    cfg: Config,
) -> nn.Module:
    """
    Full training loop with OneCycleLR, gradient clipping,
    and best-checkpoint saving.

    Returns the model loaded with the best weights.
    """
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.LR,
        steps_per_epoch=len(train_dl),
        epochs=cfg.EPOCHS,
        pct_start=cfg.PCT_START,
    )

    best_val = float("inf")
    os.makedirs(cfg.TEMP_DIR, exist_ok=True)

    for epoch in range(cfg.EPOCHS):
        train_loss = train_one_epoch(
            model, train_dl, optimizer, scheduler, cfg.DEVICE, cfg.GRAD_CLIP
        )
        val_loss = validate(model, val_dl, cfg.DEVICE)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), cfg.checkpoint_path)

        if epoch % 5 == 0 or epoch == cfg.EPOCHS - 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:3d}/{cfg.EPOCHS} | "
                f"Train RMSE: {train_loss:.4f} | "
                f"Val RMSE: {val_loss:.4f} | "
                f"Best: {best_val:.4f} | "
                f"LR: {lr_now:.2e}"
            )

    # reload best checkpoint
    model.load_state_dict(torch.load(cfg.checkpoint_path, map_location=cfg.DEVICE))
    print(f"\n✅ Training complete — best val RMSE: {best_val:.4f}")
    return model


# ═══════════════════════════════════════════════════════════
# 9. INFERENCE
# ═══════════════════════════════════════════════════════════

@torch.no_grad()
def run_inference(
    model: nn.Module,
    X_test: np.ndarray,
    stats: Dict[str, Dict[str, np.ndarray]],
    target_feat: str,
    device: torch.device,
    batch_size: int = 16,
) -> np.ndarray:
    """
    Run batched inference, denormalise predictions back to µg/m³,
    and clamp negatives.

    Args:
        X_test: (N, T, H, W, C) — normalised inputs.

    Returns:
        preds: (N, 140, 124, 16) in physical units.
    """
    model.eval()
    cpm_mean = torch.from_numpy(stats[target_feat]["mean"]).to(device)
    cpm_std  = torch.from_numpy(stats[target_feat]["std"]).to(device)

    test_tensor = torch.from_numpy(X_test)
    all_preds = []

    for i in range(0, len(test_tensor), batch_size):
        batch = test_tensor[i : i + batch_size]
        batch = batch.permute(0, 4, 1, 2, 3).to(device)  # (B, C, T, H, W)

        pred = model(batch)  # (B, H, W, T_OUT) — normalised scale

        # denormalise using grid-wise stats
        pred = pred * cpm_std[None, :, :, None] + cpm_mean[None, :, :, None]
        pred = torch.clamp(pred, min=0.0)
        all_preds.append(pred.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0).astype(np.float32)
    return preds


def save_predictions(preds: np.ndarray, path: str):
    """Validate shape & finiteness, then save."""
    assert preds.shape == (996, 140, 124, 16), (
        f"Expected (996, 140, 124, 16), got {preds.shape}"
    )
    assert np.isfinite(preds).all(), "Predictions contain NaN or Inf"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, preds)
    print(f"✅ Predictions saved → {path}  |  shape={preds.shape}")


# ═══════════════════════════════════════════════════════════
# 10. EVALUATION (offline / local)
# ═══════════════════════════════════════════════════════════

def average_domain_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Competition metric: mean of per-sample, per-hour spatial RMSE.

    y_true, y_pred: (N, H, W, T)
    """
    rmse_vals = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=(1, 2)))  # (N, T)
    return float(np.mean(rmse_vals))


# ═══════════════════════════════════════════════════════════
# 11. END-TO-END PIPELINE RUNNER
# ═══════════════════════════════════════════════════════════

def run_full_pipeline(cfg: Optional[Config] = None) -> np.ndarray:
    """
    Execute the complete pipeline:
        1. Seed
        2. Compute normalisation stats
        3. Build train / val samples
        4. Create DataLoaders
        5. Build & train model
        6. Run inference on test data
        7. Save preds.npy

    Returns the predictions array.
    """
    if cfg is None:
        cfg = Config()

    print("=" * 60)
    print("  PM2.5 Forecasting — Full Pipeline")
    print("=" * 60)
    print(cfg)

    # 1. Seed
    seed_everything(cfg.SEED)
    print(f"\n[1/7] Seed set to {cfg.SEED}")

    # 2. Normalisation stats (train months only)
    print(f"\n[2/7] Computing grid-wise normalisation stats …")
    norm_stats = compute_grid_stats(
        cfg.DATA_DIR, cfg.train_months, cfg.features
    )
    stats_path = os.path.join(cfg.TEMP_DIR, "norm_stats.npy")
    os.makedirs(cfg.TEMP_DIR, exist_ok=True)
    save_stats(norm_stats, stats_path)

    # 3. Build samples
    print(f"\n[3/7] Building train samples (months={cfg.train_months}, stride={cfg.TRAIN_STRIDE}) …")
    X_train, y_train = build_samples(
        cfg.DATA_DIR, cfg.train_months, cfg.features, norm_stats,
        cfg.T_IN_CPM, cfg.T_IN_MET, cfg.T_OUT, cfg.TRAIN_STRIDE,
    )
    print(f"       X_train: {X_train.shape}  y_train: {y_train.shape}")

    print(f"       Building val samples (month={cfg.VAL_MONTH}, stride={cfg.VAL_STRIDE}) …")
    X_val, y_val = build_samples(
        cfg.DATA_DIR, [cfg.VAL_MONTH], cfg.features, norm_stats,
        cfg.T_IN_CPM, cfg.T_IN_MET, cfg.T_OUT, cfg.VAL_STRIDE,
    )
    print(f"       X_val:   {X_val.shape}    y_val:   {y_val.shape}")

    # 4. DataLoaders
    print(f"\n[4/7] Creating DataLoaders …")
    train_dl, val_dl = make_dataloaders(
        X_train, y_train, X_val, y_val,
        cfg.BATCH_SIZE_TRAIN, cfg.BATCH_SIZE_VAL, cfg.NUM_WORKERS,
    )
    del X_train, y_train, X_val, y_val
    gc.collect()

    # 5. Model
    print(f"\n[5/7] Building model on {cfg.DEVICE} …")
    model = build_model(cfg)

    # 6. Train
    print(f"\n[6/7] Training for {cfg.EPOCHS} epochs …")
    model = train_model(model, train_dl, val_dl, cfg)

    # 7. Inference
    print(f"\n[7/7] Running inference on test data …")
    X_test = build_test_inputs(
        cfg.DATA_DIR, cfg.features, norm_stats, cfg.T_IN_CPM, cfg.T_IN_MET,
    )
    print(f"       X_test: {X_test.shape}")

    preds = run_inference(
        model, X_test, norm_stats, cfg.TARGET, cfg.DEVICE, cfg.BATCH_SIZE_TEST,
    )
    save_predictions(preds, cfg.preds_path)

    return preds
