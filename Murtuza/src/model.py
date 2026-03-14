"""
Model definitions for PM2.5 spatiotemporal forecasting.

Two architectures available (select via cfg['model']['type']):

  "unet"   — Encoder-decoder U-Net that flattens the time dimension into
             channels.  Strong baseline; fast to train; handles skip
             connections cleanly.  H=140, W=124 padded to nearest multiple
             of 8 (144×128) and cropped back on output.

  "tfno2d" — Tucker-Factorized Fourier Neural Operator.  Learns global
             spectral patterns via FFT.  Well-suited to periodic/wave-like
             features (wind, pressure).

    "res_stunet" — Research-backed practical choice for this task: a residual
                         dual-branch spatiotemporal U-Net with a persistence residual
                         head. One branch encodes past state, the other future exogenous
                         forcing. Best fit for small-data PM2.5 forecasting on grids.

Both share the same interface:
    x : (B, C=16, T=26, H=140, W=124)
    → (B, H=140, W=124, T_out=16)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
#  U-Net baseline
# ─────────────────────────────────────────────────────────────────────────────

def _pad_to_multiple(x: torch.Tensor, multiple: int = 8):
    """Pad H, W up to the next multiple of `multiple`. Return tensor + (H, W)."""
    H, W = x.shape[-2], x.shape[-1]
    pH = (multiple - H % multiple) % multiple
    pW = (multiple - W % multiple) % multiple
    if pH or pW:
        x = F.pad(x, (0, pW, 0, pH))   # pad right / bottom only
    return x, H, W


class _ConvBnAct(nn.Sequential):
    """Conv2d → GroupNorm → GELU."""
    def __init__(self, in_ch: int, out_ch: int, groups: int = 8):
        g = min(groups, out_ch)
        # GroupNorm requires out_ch divisible by g
        while out_ch % g != 0 and g > 1:
            g -= 1
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(g, out_ch),
            nn.GELU(),
        )


class _DoubleConv(nn.Module):
    """Two ConvBnAct blocks."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            _ConvBnAct(in_ch, out_ch),
            _ConvBnAct(out_ch, out_ch),
        )

    def forward(self, x):
        return self.net(x)


class _Down(nn.Module):
    """MaxPool2d → DoubleConv."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            _DoubleConv(in_ch, out_ch),
        )

    def forward(self, x):
        return self.net(x)


class _Up(nn.Module):
    """Bilinear upsample to match skip shape + DoubleConv."""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = _DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class UNet(nn.Module):
    """
    4-level U-Net for PM2.5 forecasting.

    Input  : (B, C*T, H, W)  — time flattened into channels by forward()
    Output : (B, H, W, T_out)

    Encoder: [base, 2×base, 4×base, 8×base]
    Bottleneck: 16×base
    Decoder mirrors encoder with skip connections.
    """

    def __init__(self, in_channels: int, out_steps: int = 16, base_ch: int = 64):
        super().__init__()
        b = base_ch
        self.inc  = _DoubleConv(in_channels, b)
        self.d1   = _Down(b,     b * 2)
        self.d2   = _Down(b * 2, b * 4)
        self.d3   = _Down(b * 4, b * 8)
        self.bot  = _Down(b * 8, b * 16)

        self.u3   = _Up(b * 16, b * 8,  b * 8)
        self.u2   = _Up(b * 8,  b * 4,  b * 4)
        self.u1   = _Up(b * 4,  b * 2,  b * 2)
        self.u0   = _Up(b * 2,  b,      b)

        self.head = nn.Conv2d(b, out_steps, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        x = x.reshape(B, C * T, H, W)

        # Pad to multiple of 16 (4 pooling steps)
        x, H0, W0 = _pad_to_multiple(x, 16)

        e0 = self.inc(x)          # (B, b,       H,    W)
        e1 = self.d1(e0)          # (B, 2b,      H/2,  W/2)
        e2 = self.d2(e1)          # (B, 4b,      H/4,  W/4)
        e3 = self.d3(e2)          # (B, 8b,      H/8,  W/8)
        bt = self.bot(e3)         # (B, 16b,     H/16, W/16)

        d3 = self.u3(bt, e3)      # (B, 8b,      H/8,  W/8)
        d2 = self.u2(d3, e2)      # (B, 4b,      H/4,  W/4)
        d1 = self.u1(d2, e1)      # (B, 2b,      H/2,  W/2)
        d0 = self.u0(d1, e0)      # (B, b,       H,    W)

        out = self.head(d0)       # (B, T_out,   H,    W)
        out = out[..., :H0, :W0]  # crop padding
        return out.permute(0, 2, 3, 1)   # (B, H, W, T_out)


# ─────────────────────────────────────────────────────────────────────────────
#  Research-backed residual spatiotemporal U-Net
# ─────────────────────────────────────────────────────────────────────────────

class _ResBlock2d(nn.Module):
    """Residual 2D block with GroupNorm and GELU."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        g1 = min(8, out_ch)
        while out_ch % g1 != 0 and g1 > 1:
            g1 -= 1

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(g1, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(g1, out_ch)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = F.gelu(self.norm1(self.conv1(x)))
        x = self.drop(x)
        x = self.norm2(self.conv2(x))
        return F.gelu(x + residual)


class _TemporalStem(nn.Module):
    """Encode a `(B, C, T, H, W)` tensor into a 2D feature map."""

    def __init__(self, in_ch: int, stem_ch: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, stem_ch, kernel_size=(3, 3, 3), padding=1, bias=False)
        self.norm1 = nn.GroupNorm(min(8, stem_ch), stem_ch)
        self.conv2 = nn.Conv3d(stem_ch, stem_ch, kernel_size=(3, 3, 3), padding=1, bias=False)
        self.norm2 = nn.GroupNorm(min(8, stem_ch), stem_ch)
        self.proj = nn.Conv3d(in_ch, stem_ch, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.proj(x)
        x = F.gelu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        x = F.gelu(x + skip)
        return x.mean(dim=2)  # temporal aggregation -> (B, C, H, W)


class _ResDown(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = _ResBlock2d(in_ch, out_ch, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.pool(x))


class _ResUp(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.block = _ResBlock2d(in_ch + skip_ch, out_ch, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class ResidualSTUNet(nn.Module):
    """
    Dual-branch spatiotemporal U-Net with persistence residual head.

    Branch A: past state encoder (all channels, first 10 hrs)
    Branch B: future forcing encoder (all channels except cpm25, hrs 10:26)
    Head    : predicts residual over last observed cpm25 map.
    """

    def __init__(
        self,
        in_channels: int,
        base_feature_channels: int,
        t_in_cpm: int = 10,
        out_steps: int = 16,
        base_ch: int = 64,
        stem_ch: int = 48,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.t_in_cpm = t_in_cpm
        self.out_steps = out_steps
        self.base_feature_channels = base_feature_channels

        past_in = in_channels
        future_in = in_channels - 1  # exclude future cpm25 channel

        self.past_stem = _TemporalStem(past_in, stem_ch)
        self.future_stem = _TemporalStem(future_in, stem_ch)
        self.fuse = _ResBlock2d(stem_ch * 2 + 1, base_ch, dropout=dropout)

        self.e1 = _ResDown(base_ch, base_ch * 2, dropout=dropout)
        self.e2 = _ResDown(base_ch * 2, base_ch * 4, dropout=dropout)
        self.e3 = _ResDown(base_ch * 4, base_ch * 8, dropout=dropout)
        self.bot = _ResDown(base_ch * 8, base_ch * 16, dropout=dropout)

        self.u3 = _ResUp(base_ch * 16, base_ch * 8, base_ch * 8, dropout=dropout)
        self.u2 = _ResUp(base_ch * 8, base_ch * 4, base_ch * 4, dropout=dropout)
        self.u1 = _ResUp(base_ch * 4, base_ch * 2, base_ch * 2, dropout=dropout)
        self.u0 = _ResUp(base_ch * 2, base_ch, base_ch, dropout=dropout)

        self.delta_head = nn.Sequential(
            _ResBlock2d(base_ch, base_ch, dropout=dropout),
            nn.Conv2d(base_ch, out_steps, 1),
        )
        self.lead_bias = nn.Parameter(torch.zeros(1, out_steps, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        x, H0, W0 = _pad_to_multiple(x, 16)

        # Past branch sees all channels for the first observed 10 hours.
        x_past = x[:, :, :self.t_in_cpm]

        # Future branch sees exogenous channels only from the forcing window.
        # Channel 0 is cpm25 and is masked/unavailable beyond t=10.
        x_future = x[:, 1:, self.t_in_cpm:]

        past_feat = self.past_stem(x_past)
        future_feat = self.future_stem(x_future)

        last_cpm = x[:, 0, self.t_in_cpm - 1]
        z0 = self.fuse(torch.cat([past_feat, future_feat, last_cpm[:, None]], dim=1))
        z1 = self.e1(z0)
        z2 = self.e2(z1)
        z3 = self.e3(z2)
        zb = self.bot(z3)

        y3 = self.u3(zb, z3)
        y2 = self.u2(y3, z2)
        y1 = self.u1(y2, z1)
        y0 = self.u0(y1, z0)

        delta = self.delta_head(y0) + self.lead_bias
        persistence = last_cpm[:, None].repeat(1, self.out_steps, 1, 1)
        pred = torch.clamp(persistence + delta, 0.0, 1.0)
        pred = pred[:, :, :H0, :W0]
        return pred.permute(0, 2, 3, 1)


# ─────────────────────────────────────────────────────────────────────────────
#  TFNO2D (Fourier baseline)
# ─────────────────────────────────────────────────────────────────────────────

class _SpectralConv2d(nn.Module):
    """2D complex-valued spectral convolution via FFT."""

    def __init__(self, in_ch: int, out_ch: int, modes1: int, modes2: int):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1 / max(1, in_ch * out_ch)
        # Store spectral kernels as real-imag pairs and convert with view_as_complex
        # so the rest of the pipeline remains real-valued and PyTorch-friendly.
        self.weight_pos_ri = nn.Parameter(
            scale * torch.randn(in_ch, out_ch, modes1, modes2, 2, dtype=torch.float32)
        )
        self.weight_neg_ri = nn.Parameter(
            scale * torch.randn(in_ch, out_ch, modes1, modes2, 2, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x, norm='ortho')

        # Complex kernels for explicit phase-shift learning (plume advection).
        w_pos = torch.view_as_complex(self.weight_pos_ri.contiguous()).to(x_ft.dtype)
        w_neg = torch.view_as_complex(self.weight_neg_ri.contiguous()).to(x_ft.dtype)

        out_ft = torch.zeros(
            B, w_pos.shape[1], H, W // 2 + 1,
            dtype=x_ft.dtype, device=x.device,
        )
        out_ft_pos = torch.einsum(
            'bixy,ioxy->boxy',
            x_ft[:, :, :self.modes1, :self.modes2],
            w_pos,
        )
        out_ft_neg = torch.einsum(
            'bixy,ioxy->boxy',
            x_ft[:, :, -self.modes1:, :self.modes2],
            w_neg,
        )

        # Keep explicit real/imag views for numerical transparency and compatibility.
        out_ft[:, :, :self.modes1, :self.modes2] = torch.view_as_complex(
            torch.view_as_real(out_ft_pos).contiguous()
        )
        out_ft[:, :, -self.modes1:, :self.modes2] = torch.view_as_complex(
            torch.view_as_real(out_ft_neg).contiguous()
        )
        return torch.fft.irfft2(out_ft, s=(H, W), norm='ortho')


class _FNOBlock(nn.Module):
    def __init__(self, width: int, modes: int):
        super().__init__()
        self.spectral = _SpectralConv2d(width, width, modes, modes)
        self.bypass   = nn.Conv2d(width, width, 1)
        g = min(8, width)
        while width % g != 0:
            g -= 1
        self.norm = nn.GroupNorm(g, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.norm(self.spectral(x) + self.bypass(x)))


class _LocalConvBlock(nn.Module):
    """Lightweight local spatial block for U-FNO high-frequency detail path."""

    def __init__(self, width: int):
        super().__init__()
        g = min(8, width)
        while width % g != 0 and g > 1:
            g -= 1
        self.conv1 = nn.Conv2d(width, width, 3, padding=1)
        self.norm1 = nn.GroupNorm(g, width)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1)
        self.norm2 = nn.GroupNorm(g, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.gelu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return F.gelu(x + residual)


class TFNO2D(nn.Module):
    """Tucker-factorized FNO2D.  Same interface as UNet.

    Enhancements vs baseline:
    - Reflective boundary padding (topic 7) pads from 140×124 → 144×128 before
      spectral layers then crops back.  This prevents the FFT periodic assumption
      from introducing spectral leakage at the edges of India.
    - ``clamp_output=False`` when residual prediction is used (topic 8).
    """

    def __init__(self, in_channels: int, out_steps: int = 16,
                 width: int = 64, modes: int = 24, depth: int = 6,
                 padding_mode: str = 'circular',
                 clamp_output: bool = True):
        super().__init__()
        self.clamp_output = clamp_output
        # Padding sizes to bring 140→144 and 124→128 (multiples of 8 / power-of-2)
        self._pad = (2, 2, 2, 2)   # (left, right, top, bottom) for F.pad
        self.lift   = nn.Conv2d(in_channels, width, 1)
        self.local_lift = nn.Conv2d(in_channels, width, 3, padding=1)
        self.blocks = nn.ModuleList([_FNOBlock(width, modes) for _ in range(depth)])
        self.local_blocks = nn.ModuleList([_LocalConvBlock(width) for _ in range(depth)])
        self.fuse_blocks = nn.ModuleList([nn.Conv2d(width * 2, width, 1) for _ in range(depth)])
        self.proj   = nn.Sequential(
            nn.Conv2d(width * 2, width * 2, 1), nn.GELU(),
            nn.Conv2d(width * 2, out_steps, 1),
        )

    def freeze_non_spectral(self) -> None:
        """Freeze everything except spectral kernels for PFT phase-2."""
        for p in self.parameters():
            p.requires_grad = False
        for block in self.blocks:
            for p in block.spectral.parameters():
                p.requires_grad = True

    def unfreeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        x = x.reshape(B, C * T, H, W)
        # ── Reflective padding (Topic 7): smooth boundary for FFT periodicity ──
        x = F.pad(x, self._pad, mode='reflect')        # (B, C*T, H+4, W+4)
        global_feat = self.lift(x)
        local_feat = F.gelu(self.local_lift(x))

        # U-FNO hybrid: Fourier branch captures global transport, CNN branch
        # preserves local/high-frequency hotspot structure, fused via skip-concat.
        for fno_block, local_block, fuse in zip(self.blocks, self.local_blocks, self.fuse_blocks):
            global_feat = fno_block(global_feat)
            local_feat = local_block(local_feat)
            global_feat = F.gelu(fuse(torch.cat([global_feat, local_feat], dim=1)))

        x = self.proj(torch.cat([global_feat, local_feat], dim=1))  # (B, T_out, H+4, W+4)
        # Crop back to original spatial dims
        x = x[:, :, 2:2+H, 2:2+W]                     # (B, T_out, H, W)
        if self.clamp_output:
            x = torch.clamp(x, 0.0, 1.0)
        return x.permute(0, 2, 3, 1)                   # (B, H, W, T_out)


# ─────────────────────────────────────────────────────────────────────────────
#  Registry / Factory
# ─────────────────────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    'unet': UNet,
    'tfno2d': TFNO2D,
    'fno2d': TFNO2D,
    'res_stunet': ResidualSTUNet,
}


def register_model(name: str, model_cls) -> None:
    """Register a future model class in one place for config-driven swapping."""
    MODEL_REGISTRY[name.lower()] = model_cls


def get_available_models():
    """Return sorted list of available model names."""
    return sorted(MODEL_REGISTRY)

def build_model(cfg) -> nn.Module:
    """Instantiate and move model to device based on cfg['model']['type']."""
    # Dynamically set input_channels based on tensor shape
    # Expect cfg['tensor_channels'] to be set before calling build_model
    input_channels = cfg.get('tensor_channels', cfg['features']['input_channels'])
    t_out      = cfg['time']['t_out']
    t_in_cpm   = cfg['time'].get('t_in_cpm', 10)
    n_features = cfg['features'].get('input_channels', cfg['features'].get('n_features', input_channels))
    mtype      = cfg['model'].get('type', 'tfno2d').lower()
    in_channels = input_channels

    if mtype not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type: {mtype!r}. Available: {', '.join(get_available_models())}."
        )

    if mtype == 'unet':
        model = MODEL_REGISTRY[mtype](
            in_channels = in_channels,
            out_steps   = t_out,
            base_ch     = cfg['model'].get('base_ch', 64),
        )
    elif mtype in ('tfno2d', 'fno2d'):
        # Disable output clamp when residual prediction is used (z-scores / deltas
        # are unbounded; clamping to [0,1] would cut negative residuals).
        residual_mode = cfg.get('training', {}).get('residual_target', False)
        model = MODEL_REGISTRY[mtype](
            in_channels  = in_channels,
            out_steps    = t_out,
            width        = cfg['model']['width'],
            modes        = cfg['model']['modes'],
            depth        = cfg['model']['depth'],
            clamp_output = not residual_mode,
        )
    elif mtype == 'res_stunet':
        model = MODEL_REGISTRY[mtype](
            in_channels = input_channels,
            base_feature_channels = n_features,
            t_in_cpm = t_in_cpm,
            out_steps = t_out,
            base_ch = cfg['model'].get('base_ch', 64),
            stem_ch = cfg['model'].get('stem_ch', 48),
            dropout = cfg['model'].get('dropout', 0.05),
        )

    return model.to(cfg['device'])

