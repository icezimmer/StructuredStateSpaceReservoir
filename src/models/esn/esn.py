import torch
import torch.nn as nn
from src.reservoir.state import DiscreteStateReservoir
from src.reservoir.matrices import Reservoir


class ESN(nn.Module):
    """Class of Echo State Network model using PyTorch."""

    def __init__(self, d_input, d_state, input_scaling=1.0, spectral_radius=1.0, leakage_rate=0.5):
        """
        Constructor of ESN model.
        """
        super().__init__()
        self.d_input = d_input
        self.d_state = d_state
        self.leakage_rate = leakage_rate

        input2state_reservoir = Reservoir(d_in=self.d_input, d_out=self.d_state)
        w_in = input2state_reservoir.uniform_disk_matrix(radius=input_scaling, field='real')
        self.w_in = nn.Parameter(w_in, requires_grad=False)

        self.register_buffer('x0', torch.zeros(self.d_state, dtype=torch.float32))

        state_reservoir = DiscreteStateReservoir(self.d_state)
        w_hh = state_reservoir.echo_state_matrix(max_radius=spectral_radius)
        self.w_hh = nn.Parameter(w_hh, requires_grad=False)

        self.bias = nn.Parameter(torch.rand(self.d_state) * 2 - 1, requires_grad=False)
        self.nl = nn.Tanh()

    def step(self, u, x=None):
        if x is None:
            x = self.x0.unsqueeze(0).expand(u.shape[0], -1)
        preactivation = (torch.einsum('ph, bh -> bp', self.w_in, u)
                         + torch.einsum('pp, bp -> bp', self.w_hh, x)
                         + self.bias.squeeze(0))
        x = (1 - self.leakage_rate) * x + self.leakage_rate * self.nl(preactivation)
        return x

    def forward(self, u: torch.Tensor):
        """
        Forward pass of ESN model.
        """
        x_prev = self.x0.unsqueeze(0).expand(u.shape[0], -1)
        x_list = []
        for k in range(u.shape[-1]):
            u_k = u[:, :, k]
            preactivation = (torch.einsum('ph, bh -> bp', self.w_in, u_k)
                             + torch.einsum('pp, bp -> bp', self.w_hh, x_prev)
                             + self.bias.squeeze(0))
            x_curr = (1 - self.leakage_rate) * x_prev + self.leakage_rate * self.nl(preactivation)
            x_list.append(x_curr.unsqueeze(-1))
            x_prev = x_curr
        x = torch.cat(tensors=x_list, dim=-1)

        return x, None
