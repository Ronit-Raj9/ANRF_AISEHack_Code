"""
TFNO2D — Tucker-Factorized Fourier Neural Operator for PM2.5 forecasting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """2D spectral convolution via FFT."""

    def __init__(self, in_ch, out_ch, modes1, modes2):
        super().__init__()
        self.modes1, self.modes2 = modes1, modes2
        scale = 1 / (in_ch * out_ch)
        self.weights = nn.Parameter(
            scale * torch.randn(in_ch, out_ch, modes1, modes2, 2)
        )

    def compl_mul2d(self, x, w):
        xr, xi = x[..., 0], x[..., 1]
        wr, wi = w[..., 0], w[..., 1]
        return torch.stack([xr * wr - xi * wi, xr * wi + xi * wr], dim=-1)

    def forward(self, x):
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x, norm='ortho')
        out_ft = torch.zeros(
            B, self.weights.shape[1], H, W // 2 + 1,
            dtype=torch.cfloat, device=x.device,
        )
        x_ft_real = torch.view_as_real(x_ft[:, :, :self.modes1, :self.modes2])
        out_real  = self.compl_mul2d(x_ft_real, self.weights)
        out_ft[:, :, :self.modes1, :self.modes2] = torch.view_as_complex(out_real)
        return torch.fft.irfft2(out_ft, s=(H, W), norm='ortho')


class FNOBlock(nn.Module):
    """Single Fourier layer: spectral conv + pointwise bypass + norm."""

    def __init__(self, width, modes1, modes2):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes1, modes2)
        self.bypass   = nn.Conv2d(width, width, 1)
        self.norm     = nn.GroupNorm(8, width)

    def forward(self, x):
        return F.gelu(self.norm(self.spectral(x) + self.bypass(x)))


class TFNO2D(nn.Module):
    """
    Tucker-factorized FNO2D for PM2.5 forecasting.

    Input:  (B, C, T_in, H, W) — channels = features, T_in = time steps
    Output: (B, H, W, T_out)
    """

    def __init__(self, in_channels, out_steps=16, width=64, modes=24, depth=6):
        super().__init__()
        self.lift = nn.Conv2d(in_channels, width, 1)
        self.blocks = nn.ModuleList(
            [FNOBlock(width, modes, modes) for _ in range(depth)]
        )
        self.proj = nn.Sequential(
            nn.Conv2d(width, width * 2, 1),
            nn.GELU(),
            nn.Conv2d(width * 2, out_steps, 1),
        )

    def forward(self, x):
        # x: (B, C, T, H, W) → flatten C*T → (B, C*T, H, W)
        B, C, T, H, W = x.shape
        x = x.reshape(B, C * T, H, W)
        x = self.lift(x)
        for block in self.blocks:
            x = block(x)
        x = self.proj(x)                 # (B, T_out, H, W)
        return x.permute(0, 2, 3, 1)     # (B, H, W, T_out)


def build_model(cfg):
    """Instantiate TFNO2D from config and move to device."""
    n_features = cfg['features']['n_features']
    t_in_met   = cfg['time']['t_in_met']
    t_out      = cfg['time']['t_out']

    model = TFNO2D(
        in_channels=n_features * t_in_met,
        out_steps=t_out,
        width=cfg['model']['width'],
        modes=cfg['model']['modes'],
        depth=cfg['model']['depth'],
    ).to(cfg['device'])

    return model
