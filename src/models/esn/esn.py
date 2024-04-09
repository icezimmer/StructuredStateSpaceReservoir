import torch
import torch.nn as nn
from src.reservoir.state_reservoir import DiscreteStateReservoir
from src.reservoir.reservoir import Reservoir


class ESN(nn.Module):
    """Class of Echo State Network model using PyTorch."""

    def __init__(self, d_model, input_scaling=1.0, spectral_radius=0.9, leakage_rate=1, drop_kernel=0.0, dropout=0.0):
        """
        Constructor of ESN model.
        """
        super().__init__()
        self.d_model = d_model
        self.leakage_rate = leakage_rate

        reservoir = Reservoir(d_in=d_model, d_out=d_model)
        w_in = reservoir.uniform_matrix(scaling=input_scaling, field='real')
        self.w_in = nn.Parameter(w_in, requires_grad=False)

        state_reservoir = DiscreteStateReservoir(self.d_state)
        w_hh = state_reservoir.echo_state_matrix(max_radius=spectral_radius)
        self.w_hh = nn.Parameter(w_hh, requires_grad=False)

        self.bias = nn.Parameter(torch.rand(self.d_model) * 2 - 1, requires_grad=False)

        self.drop_kernel = nn.Dropout(drop_kernel) if drop_kernel > 0 else nn.Identity()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.nl = nn.Tanh()

    def forward(self, u: torch.Tensor):
        """
        Forward pass of ESN model.
        """
        x = torch.zeros(u.shape[0], self.d_model, u.shape[-1], dtype=torch.float32, device=u.device)

        x_prev = torch.zeros(u.shape[0], self.d_model, device=u.device)
        for k in range(u.shape[-1]):
            u_k = u[:, :, k]
            preactivation = (torch.einsum('ph, bh -> bp', self.w_in, u_k)
                             + torch.einsum('pp, bp -> bp', self.drop_kernel(self.w_hh), x_prev)
                             + self.bias.squeeze(0))
            x_curr = (1 - self.leakage_rate) * x_prev + self.leakage_rate * self.nl(preactivation)
            x[:, :, k] = x_curr
            x_prev = x_curr

        x = self.drop(x)
        return x, None
