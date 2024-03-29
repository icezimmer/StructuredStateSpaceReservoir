"""
Minimal version of S4D with extra options and features stripped out, for pedagogical purposes.
Source: https://github.com/state-spaces/s4/blob/main/models/s4/s4d.py
"""

import math
import torch
import torch.nn as nn
from einops import repeat

from src.models.nn.dropout import DropoutNd


class S4DKernel(nn.Module):
    """Generate convolution kernel from diagonal SSM parameters."""

    def __init__(self, d_input, d_state=64, dt_min=0.001, dt_max=0.1, lr=None):
        super().__init__()

        # Generate dt
        H = d_input
        log_dt = torch.rand(H) * (
                math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)  # (H)

        C = torch.randn(H, d_state // 2, dtype=torch.cfloat)  # (H, N//2)
        self.C = nn.Parameter(torch.view_as_real(C))  # (H, N//2, 2)
        self.register("log_dt", log_dt, lr)

        log_A_real = torch.log(0.5 * torch.ones(H, d_state // 2))  # (H, N//2)
        A_imag = math.pi * repeat(torch.arange(d_state // 2), 'n -> h n', h=H)  # (H, N//2)

        self.register("log_A_real", log_A_real, lr)
        self.register("A_imag", A_imag, lr)

    def forward(self, L):
        """
        returns: (..., c, L) where c is number of channels (default 1)
        """
        # Materialize parameters
        dt = torch.exp(self.log_dt)  # (H)
        C = torch.view_as_complex(self.C)  # (H, N//2)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag  # (H, N//2) N//2 copies of A on dim=1

        # Vandermonde multiplication
        dtA = A * dt.unsqueeze(-1)  # (H, N//2)
        K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device)  # (H, N//2, L)
        C = C * (torch.exp(dtA) - 1.) / A  # (H, N//2)
        K = 2 * torch.einsum('hn, hnl -> hl', C, torch.exp(K)).real  # (H, L)

        return K

    def register(self, name, tensor, lr=None):
        """
        Register a tensor with a configurable learning rate and 0 weight decay
        """

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


class S4D(nn.Module):
    def __init__(self, d_input, d_state=64, dropout=0.0, transposed=True, **kernel_args):
        super().__init__()

        self.d_input = d_input
        self.d_state = d_state
        self.d_output = self.d_input
        self.transposed = transposed

        self.D = nn.Parameter(torch.randn(self.d_input))  # (H)

        # SSM Kernel
        self.kernel = S4DKernel(self.d_input, d_state=self.d_state, **kernel_args)

        # Pointwise
        self.activation = nn.GELU()
        # dropout_fn = nn.Dropout2d # NOTE: bugged in PyTorch 1.11
        dropout_fn = DropoutNd
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        # position-wise output transform to mix features
        self.output_linear = nn.Sequential(
            nn.Conv1d(self.d_input, 2 * self.d_input, kernel_size=1),
            nn.GLU(dim=-2),
        )

    def forward(self, u, **kwargs):  # absorbs return_output and transformer src mask
        """ Input and output shape (B, H, L) """
        if not self.transposed: u = u.transpose(-1, -2)
        L = u.size(-1)

        # Compute SSM Kernel
        k = self.kernel(L=L)  # (H, L)

        # Convolution
        k_f = torch.fft.rfft(k, n=2 * L)  # (H, L)
        u_f = torch.fft.rfft(u, n=2 * L)  # (B, H, L)
        y = torch.fft.irfft(u_f * k_f, n=2 * L)[..., :L]  # (B, H, L)

        # Compute D term in state space equation - essentially a skip connection
        y = y + u * self.D.unsqueeze(-1)  # (B, H, L), self.D.unsqueeze(-1) is (H, 1)

        y = self.dropout(self.activation(y))
        y = self.output_linear(y)
        if not self.transposed: y = y.transpose(-1, -2)
        return y, None  # Return a dummy state to satisfy this repo's interface, but this can be modified
