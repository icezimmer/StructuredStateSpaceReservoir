"""Minimal version of S4D with extra options and features stripped out, for pedagogical purposes."""

import math
import torch
import torch.nn as nn

from src.models.nn.dropout import DropoutNd


class S4RKernel(nn.Module):
    """Generate convolution kernel from diagonal SSM parameters."""

    @staticmethod
    def _discrete_state_matrix(d_state, min_radius, max_radius):
        """
        Create a state matrix Lambda_bar for the discrete dynamics;
        lambda = radius * (cos(theta) + i * sin(theta)):
        radius in [min_radius, max_radius),
        theta in [0, 2pi).
        :param d_state: latent state dimension
        :return: Lambda_bar
        """

        radius = min_radius + (max_radius - min_radius) * torch.rand(d_state, dtype=torch.float32)
        theta = 2 * torch.pi * torch.rand(d_state, dtype=torch.float32)
        alpha_tensor = radius * torch.cos(theta)
        omega_tensor = radius * torch.sin(theta)

        Lambda_bar = torch.complex(alpha_tensor, omega_tensor)
        return Lambda_bar.view(-1, 1)

    def __init__(self, d_input, d_state=64, min_radius=0.9, max_radius=1, dt_min=0.001, dt_max=0.1, lr=None):
        super().__init__()
        # Generate dt
        log_dt = torch.rand(d_input, dtype=torch.float32) * (
                math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)

        d_output = d_input
        self.C = nn.Parameter(torch.randn(d_output, d_state, dtype=torch.complex64))
        self.register("log_dt", log_dt, lr)

        # Generate A
        self.A = self._discrete_state_matrix(d_state, min_radius, max_radius)

    def forward(self, input_length):
        """
        returns: (..., c, L) where c is number of channels (default 1) and L is the length of the input sequence
        """

        # Materialize parameters
        dt = torch.exp(self.log_dt)  # (H)

        # Vandermonde multiplication
        dtA = self.A * dt.unsqueeze(-1)  # (H N)
        K = dtA.unsqueeze(-1) * torch.arange(input_length, device=self.A.device)  # (H N L)
        C = self.C * (torch.exp(dtA) - 1.) / self.A
        K = 2 * torch.einsum('hn, hnl -> hl', C, torch.exp(K)).real

        return K

    def register(self, name, tensor, lr=None):
        """
        Register a tensor as a parameter or buffer, with an associated learning rate.
        :param name:
        :param tensor:
        :param lr:
        :return:
        """

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


class S4R(nn.Module):
    def __init__(self, d_input, d_state=64, dropout=0.0, **kernel_args):
        super().__init__()

        self.d_input = d_input
        self.d_state = d_state
        self.d_output = d_input

        self.D = nn.Parameter(torch.randn(self.d_output, self.d_input, dtype=torch.float32))

        # SSM Kernel
        self.kernel = S4RKernel(self.d_input, d_state=self.d_state, **kernel_args)

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

    def forward(self, u):  # absorbs return_output and transformer src mask
        """ Input and output shape (B, H, L) """
        input_length = u.size(-1)

        # Compute SSM Kernel
        k = self.kernel(input_length=input_length)  # (H L)

        # Convolution
        k_f = torch.fft.fft(k, n=input_length)  # (H L)
        u_f = torch.fft.fft(u, n=input_length)  # (B H L)
        y = torch.fft.irfft(u_f * k_f, n=input_length)[..., :input_length]  # (B H L)

        # Compute D term in state space equation - essentially a skip connection
        y = y + u * self.D.unsqueeze(-1)

        y = self.dropout(self.activation(y))
        y = self.output_linear(y)
        return y, None  # Return a dummy state to satisfy this repo's interface, but this can be modified
